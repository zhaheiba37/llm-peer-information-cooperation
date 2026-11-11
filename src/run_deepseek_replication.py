"""Capability-qualified DeepSeek replication only; never starts formal collection."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import check_revision_r1 as c
import deepseek_api as d
import qualify_flash_thinking as old_checks
import run_current_multiround as r
from run_local_v7 import parse_object,same_answer


def check_tasks():
    rows=old_checks.tasks()
    for task in rows:
        condition=task['condition'];old=c.context(condition)
        user=task['messages'][1]['content'];assert user.startswith(old+'\n\n')
        context=r.messages(condition,r.p.initial_history(condition))[1]['content'].rsplit('\n\n',1)[0]
        suffix=user[len(old):].replace('contribute per round?','contribute in round 2?')
        task['messages'][1]['content']=context+suffix
    random.Random(2026110513).shuffle(rows)
    # The first actual paid request is also one of the 61 scored checks.
    first=next(x for x in rows if x['task_id']=='round_count')
    return [first]+[x for x in rows if x is not first]


def do_check(task,out,effort,api_client=None):
    api=d if api_client is None else api_client
    record=None
    for attempt in range(3):
        if r.STOP.is_set():
            return dict(task=task,status='not_started',correct=False)
        r.pace();record=dict(task=task,attempt=attempt+1)
        try:
            actual=api.call(task['messages'],f'qualification/{effort}/'+task['task_id'])
            choice=actual['response']['choices'][0];message=choice['message']
            parsed,status=parse_object(message.get('content',''))
            record.update(status='complete',actual=actual,parsed=parsed,parse_status=status,
                finish_reason=choice.get('finish_reason'),reasoning_present=bool(message.get('reasoning_content')),
                correct=choice.get('finish_reason')=='stop' and same_answer(parsed,task['expected_not_for_model']))
        except Exception as exc:
            record.update(status='transport_error',error=str(exc),correct=False)
            if r.fatal_error(str(exc)):r.stop(str(exc))
        r.append(out/'raw.jsonl',record)
        if record['status']!='transport_error' or attempt==2 or not r.transient(record.get('error','')) or r.STOP.is_set():break
        time.sleep(3*(attempt+1))
    return record


def qualify(root,plan,effort):
    d.CONFIG_PATH=d.ROOT/f'deepseek_pilot_{effort}_config.json'
    out=root/('qualification_'+effort);out.mkdir(exist_ok=False)
    begin=time.perf_counter();tasks=plan['checks'];results=[]
    first=do_check(tasks[0],out,effort);results.append(first)
    print(json.dumps(dict(qualification=effort,completed=1,total=61,status=first['status'],correct=first['correct'],answer=first.get('parsed'))),flush=True)
    if first['status']=='complete':
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(do_check,task,out,effort) for task in tasks[1:]]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results)%10==0 or len(results)==61:
                    print(json.dumps(dict(qualification=effort,completed=len(results),total=61,
                        correct=sum(x['correct'] for x in results),statuses=dict(Counter(x['status'] for x in results)))),flush=True)
    groups=defaultdict(list)
    for row in results:groups[row['task']['category']].append(row)
    by_category={key:dict(correct=sum(x['correct'] for x in value),total=len(value)) for key,value in groups.items()}
    payoff=groups['payoff'];facts=[x for x in results if x['task']['category']!='payoff']
    result=dict(effort=effort,completed=len(results),planned=61,correct=sum(x['correct'] for x in results),
        passed=len(results)==61 and sum(x['correct'] for x in payoff)>=34 and sum(x['correct'] for x in facts)==25 and all(x.get('finish_reason')=='stop' for x in results),
        categories=by_category,statuses=dict(Counter(x['status'] for x in results)),wall_seconds=time.perf_counter()-begin,
        note='Same provisional engineering gate as Qwen; current-round wording and history-first. No cooperation decisions in qualification.')
    attempts=r.read(out/'raw.jsonl');paid=[x['actual'] for x in attempts if 'actual' in x]
    result.update(attempts=len(attempts),cost_cny=sum(x['estimated_list_price_cny'] for x in paid),
        cache_aware_estimate_cny=sum(x['estimated_cache_aware_cost_cny'] for x in paid),
        thinking_present=sum(bool(x['response']['choices'][0]['message'].get('reasoning_content')) for x in paid))
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('QUALIFICATION '+json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    if not args.run:
        balance=json.loads((d.ROOT/'deepseek_balance_before_pilot_20261105.json').read_text(encoding='utf-8'))
        assert balance['response']['is_available']
        cny=next(float(x['total_balance']) for x in balance['response']['balance_infos'] if x['currency']=='CNY')
        cap=min(3.0,cny*.9)
        configs={}
        for effort,max_tokens in [('low',4096),('high',8192)]:
            cfg=dict(base_url='https://api.deepseek.com',model='deepseek-flash',thinking={'type':'enabled'},
                reasoning_effort=effort,top_p=.95,max_tokens=max_tokens,response_format={'type':'json_object'},
                timeout_seconds=120,max_input_bytes_per_request=20000,local_spend_cap_cny=cap,
                budget_ledger_file='deepseek_replication_budget.jsonl',
                price_cny_per_million={'peak':dict(input_cache_miss=2,input_cache_hit=.04,output=8),
                    'off_peak':dict(input_cache_miss=1,input_cache_hit=.02,output=4)},
                note='temperature intentionally omitted; no Qwen thinking_budget. Peak-priced reservations, dynamic time-band estimates. No formal sampling.')
            configs[effort]=cfg
            (d.ROOT/f'deepseek_pilot_{effort}_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        out=d.ROOT/'runs'/('deepseek_replication_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
        out.mkdir(exist_ok=False)
        plan=dict(purpose='cross_family_small_sample_replication_not_formal',configs=configs,config=configs['low'],
            workers=5,checks=check_tasks(),pilot_jobs=r.make_jobs('pilot',5,2026110514),
            effort_policy='Low first. Only if all 61 low checks finish but gate fails, test high on same frozen 61 questions. Choose first passing config; never select by cooperation outcome. Stop if both fail.',
            check_gate='Payoff >=34/36, other facts25/25, all61 complete and normal stop. Engineering threshold, not literature standard.',
            formal_jobs=[],formal_auto_start=False,transport_retry_limit=2,retry_delays=[3,6],no_answer_repair=True,
            balance_before=balance,norm_history_policy='Identical to Qwen v9: full past actions/payoffs, current-round statement only, all history-first, no reasoning feedback.',
            source_sha256={name:hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest() for name in
                ('run_deepseek_replication.py','deepseek_api.py','run_current_multiround.py','run_pilot_v8.py','prompts_v7.py','check_revision_r1.py','qualify_flash_thinking.py','recheck_plus_r1.py')})
        (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'round2_prompts.json').write_text(json.dumps({condition:r.messages(condition,r.p.initial_history(condition)) for condition in r.p.CONDITIONS},ensure_ascii=False,indent=2),encoding='utf-8')
        for name in plan['source_sha256']:(out/name).write_bytes((d.ROOT/name).read_bytes())
        print('PREPARED '+json.dumps(str(out)),flush=True);return
    root=args.run;plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    if (root/'qualification_low').exists() or (root/'pilot').exists():raise RuntimeError('Attempts exist; refuse duplicate or selective rerun.')
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest()==digest
    chosen=None;qualifications=[]
    for effort in ('low','high'):
        assert plan['configs'][effort]==json.loads((d.ROOT/f'deepseek_pilot_{effort}_config.json').read_text(encoding='utf-8'))
        outcome=qualify(root,plan,effort);qualifications.append(outcome)
        if outcome['passed']:chosen=effort;break
        if outcome['completed']!=61 or any(key!='complete' for key in outcome['statuses']) or r.STOP.is_set():break
    if chosen:
        active=dict(config=plan['configs'][chosen],chosen_effort=chosen,qualification_summaries=qualifications,
            norm_history_policy=plan['norm_history_policy'],pilot_jobs=plan['pilot_jobs'],formal_auto_start=False)
        (root/'active_plan.json').write_text(json.dumps(active,ensure_ascii=False,indent=2),encoding='utf-8')
        # The generic analysis reads config from plan.json. Keep the original
        # frozen plan unchanged; export a separate analysis view of this config.
        print('PILOT_START '+json.dumps(dict(chosen_effort=chosen,episodes=30,calls=270)),flush=True)
        r.run_stage(root/'pilot',plan['pilot_jobs'],'pilot',d)
        integrity=r.audit(root/'pilot',active)
        (root/'pilot'/'integrity_check.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    status=dict(chosen_effort=chosen,qualifications=qualifications,formal_started=False,
        pilot_started=chosen is not None,stop_reasons=r.STOP_REASON)
    try:
        after=d.get_balance();(root/'balance_after.json').write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:status['balance_after_error']=str(exc)
    (root/'final_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE '+json.dumps(dict(folder=str(root),**status)),flush=True)


if __name__=='__main__':main()
