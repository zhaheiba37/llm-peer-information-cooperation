"""Run the v7 comprehension gate, then the 72-trial pilot, using localhost only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request

import design_v7
import prompts_v7 as p

ROOT = Path(__file__).resolve().parent
OPTIONS = dict(temperature=0.7, top_p=0.95, top_k=20, num_ctx=4096, num_predict=256)


def api(path, payload=None):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request('http://127.0.0.1:11434/api/' + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as response:
        return json.load(response)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON number')


def parse_object(content):
    try:
        value = json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)
        return (value, 'valid') if isinstance(value, dict) else (None, 'not_object')
    except (ValueError, TypeError):
        return None, 'invalid_json'


def same_answer(value, expected):
    if isinstance(expected, dict):
        return (isinstance(value, dict) and set(value) == set(expected)
                and all(same_answer(value[k], v) for k, v in expected.items()))
    if expected is None or type(expected) is bool:
        return value is expected
    if type(expected) is int:
        return type(value) is int and value == expected
    if type(expected) is float:
        return type(value) in (int, float) and math.isfinite(value) and abs(value-expected) < 1e-8
    return type(value) is type(expected) and value == expected


def parse_decision(content):
    obj, status = parse_object(content)
    if obj is None:
        return None, status
    if set(obj) != {'contribution'}:
        return None, 'wrong_keys'
    value = obj['contribution']
    if type(value) is not int:
        return None, 'not_integer'
    if not 0 <= value <= 10:
        return None, 'out_of_range'
    return value, 'valid'


def response_status(record):
    if record.get('transport_status') != 'ok':
        return 'transport_error'
    response = record.get('response', {})
    if not response.get('done') or response.get('done_reason') == 'length':
        return 'incomplete'
    return 'complete'


def call(outdir, phase, row, item, model, seed):
    request = dict(model=model, stream=False, keep_alive='10m',
                   messages=item['messages'], options=dict(OPTIONS, seed=seed))
    record = dict(phase=phase, trial=row, prompt_id=item['prompt_id'],
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), request=request,
                  messages_sha256=hashlib.sha256(json.dumps(item['messages'], sort_keys=True).encode()).hexdigest())
    if 'expected_not_for_model' in item:
        record['expected_not_for_model'] = item['expected_not_for_model']
    start = time.perf_counter()
    try:
        record.update(response=api('chat', request), transport_status='ok')
    except Exception as exc:
        record.update(transport_status='error', error=f'{type(exc).__name__}: {exc}')
    record['wall_seconds'] = time.perf_counter() - start
    with (outdir/'raw.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def assess_check(record):
    content = record.get('response', {}).get('message', {}).get('content', '')
    obj, parse_status = parse_object(content)
    status = response_status(record)
    expected = record['expected_not_for_model']
    return dict(check_id=record['trial']['check_id'], condition=record['trial']['condition'],
                kind=record['trial']['kind'], prompt_id=record['prompt_id'],
                correct=status == 'complete' and same_answer(obj, expected),
                response_status=status, parse_status=parse_status,
                expected=expected, parsed=obj, raw_output=content,
                seconds=record['wall_seconds'])


def gate(checks):
    correct = Counter(x['kind'] for x in checks if x['correct'])
    total = Counter(x['kind'] for x in checks)
    return dict(passed=all(total[k] == 18 and correct[k] >= 17 for k in ('payoff', 'structure', 'information')),
                correct_by_kind=dict(correct), total_by_kind=dict(total),
                rule='54 independent checks, 18 per kind; at least 17/18 per kind',
                sampling_options=OPTIONS)


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (dict, list))
                             else row.get(k) for k in fields})


def summarize(outdir):
    raw = [json.loads(line) for line in (outdir/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    checks = [assess_check(record) for record in raw if record['phase'] == 'comprehension']
    decisions = []
    for record in raw:
        if record['phase'] != 'pilot':
            continue
        content = record.get('response', {}).get('message', {}).get('content', '')
        value, status = parse_decision(content)
        if response_status(record) != 'complete':
            value, status = None, response_status(record)
        item = dict(record['trial'], contribution=value, status=status,
                    raw_output=content, seconds=record['wall_seconds'])
        decisions.append(item)
    by_condition = []
    for condition in p.CONDITIONS:
        subset = [x for x in decisions if x['condition'] == condition]
        valid = [x['contribution'] for x in subset if x['status'] == 'valid']
        by_condition.append(dict(condition=condition, attempts=len(subset), valid=len(valid),
                                 mean=statistics.mean(valid) if valid else None,
                                 stdev=statistics.stdev(valid) if len(valid)>1 else None,
                                 counts=dict(Counter(valid))))
    result = dict(version=p.VERSION, status='pilot_complete' if len(decisions)==72 else 'comprehension_only',
                  gate=gate(checks), comprehension_correct=sum(x['correct'] for x in checks),
                  comprehension_total=len(checks), pilot_attempts=len(decisions),
                  pilot_valid=sum(x['status']=='valid' for x in decisions), conditions=by_condition,
                  wall_seconds=sum(x['wall_seconds'] for x in raw),
                  transport_errors=sum(x['transport_status']!='ok' for x in raw),
                  evaluation_tokens=sum(x.get('response',{}).get('eval_count',0) for x in raw),
                  interpretation='Local quantized model and pipeline diagnostics only; not confirmatory evidence.')
    write_json(outdir/'summary.json', result)
    write_csv(outdir/'understanding.csv', checks,
              ['check_id','condition','kind','correct','expected','parsed','raw_output','response_status','parse_status','seconds','prompt_id'])
    write_csv(outdir/'decisions.csv', decisions,
              ['trial_id','condition','prompt_id','replicate','contribution','status','raw_output','seconds'])
    write_csv(outdir/'condition_summary.csv', by_condition,
              ['condition','attempts','valid','mean','stdev','counts'])
    lines = ['# v7本地理解检查与流程预实验', '',
             '仅用于本地模型能力与流程调试，不作为正式实验或创新点有效性的证据。', '',
             f"理解检查：{result['comprehension_correct']}/{len(checks)}；门槛通过：{result['gate']['passed']}。", '',
             '|题目类别|正确/总数|', '|---|---|']
    for kind in ('payoff','structure','information'):
        lines.append(f"|{kind}|{result['gate']['correct_by_kind'].get(kind,0)}/{result['gate']['total_by_kind'].get(kind,0)}|")
    if not result['gate']['passed']:
        lines += ['', '未通过理解门槛，因此未启动72次决策预实验。需要先区分模型能力、格式与连接问题，不能将此解释为不合作或课题无效。']
    else:
        lines += ['', f"决策预实验：{len(decisions)}次，合法输出{result['pilot_valid']}次。", '',
                  '|条件|有效/尝试|均值|样本标准差|', '|---|---|---|---|']
        for row in by_condition:
            lines.append(f"|{row['condition']}|{row['valid']}/{row['attempts']}|{row['mean']}|{row['stdev']}|")
        lines += ['', '六条件均固定A历史贡献5；T1；含N条件两种信息顺序各6次。均值仅作调试描述，不计算显著性、不据此挑选正式模型。']
    lines += ['', '完整请求、原始响应与研究者答案保存在raw.jsonl；标准答案不进入模型请求。'
              '每次使用新的两消息上下文；无格式强制、无输出修复、无自动重问，无效值不记为0。', '',
              'metadata.json保存模型、摘要、运行参数和文件哈希；prompt_bank_snapshot.json与schedule_snapshot.json保存完整输入及安排。']
    (outdir/'运行报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3-vl-4b-instruct:latest')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.summarize)))
        return
    tags, version = api('tags'), api('version')
    installed = next((x for x in tags['models'] if x['name']==args.model), None)
    if installed is None:
        raise RuntimeError('Requested model not installed; no automatic download.')
    shown = api('show', {'model':args.model})
    outdir = ROOT/'runs'/('v7_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    outdir.mkdir(parents=True, exist_ok=False)
    schedule, bank = design_v7.build_schedule(), p.make_bank()
    lookup = {x['prompt_id']: x for value in bank.values() if isinstance(value,list)
              for x in value if isinstance(x,dict) and 'prompt_id' in x}
    source_names = ['prompts_v7.py','design_v7.py','run_local_v7.py']
    metadata = dict(purpose='v7_local_comprehension_gate_and_pipeline_pilot', version=p.VERSION,
                    started_utc=datetime.now(timezone.utc).isoformat(), model=args.model,
                    installed_model=installed, ollama_version=version, model_show=shown,
                    options=OPTIONS, checks_only=args.checks_only,
                    qualification_model_slot=design_v7.MODEL_SLOTS[0],
                    slot_note='Use one family slot only to select 54 unique check contexts; actual model is the installed local model, not GPT.',
                    source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
                    notes='Localhost only. No paid calls, downloads, retries, schema enforcement or automatic output repair. Independent contexts.')
    write_json(outdir/'metadata.json',metadata)
    write_json(outdir/'prompt_bank_snapshot.json',bank)
    write_json(outdir/'schedule_snapshot.json',schedule)
    print('OUTPUT '+json.dumps(str(outdir)), flush=True)
    checks = [x for x in schedule['eligibility_checks'] if x['model_slot']==design_v7.MODEL_SLOTS[0]]
    outcomes=[]
    for index,row in enumerate(checks):
        record=call(outdir,'comprehension',row,lookup[row['prompt_id']],args.model,70000+index)
        outcome=assess_check(record)
        outcomes.append(outcome)
        print(f"CHECK {index+1}/54 {row['kind']} correct={outcome['correct']} output={json.dumps(outcome['raw_output'])}",flush=True)
        if record['transport_status']!='ok':
            break
    summary=summarize(outdir)
    write_json(outdir/'gate.json',gate(outcomes))
    if summary['gate']['passed'] and not args.checks_only:
        for index,row in enumerate(schedule['pilot_one_debug_model']):
            record=call(outdir,'pilot',row,lookup[row['prompt_id']],args.model,80000+index)
            print(f"PILOT {index+1}/72 {row['condition']} status={response_status(record)}",flush=True)
            if record['transport_status']!='ok':
                break
        summary=summarize(outdir)
    try:
        write_json(outdir/'gpu_state.json',api('ps'))
    except Exception as exc:
        write_json(outdir/'gpu_state.json',{'error':str(exc)})
    print('SUMMARY '+json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
