"""v10 pilot: true round 1, current statements only, 6x10 independent games."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import check_revision_r1 as c
import prompts_v7 as p
import qwen_api as q
import run_current_multiround as control
from run_local_v7 import parse_decision
from run_pilot_v8 import append, pace, transient


ENDING='Choose your contribution for this round. Return only {"contribution": n}, where n is an integer from 0 to 10. Do not include an explanation.'


def messages(condition,history,rules_text=None):
    if condition not in p.CONDITIONS:raise ValueError('Unknown condition')
    if not isinstance(history,list) or not 0<=len(history)<10:raise ValueError('Expect zero to nine completed rounds')
    peer=10 if p.CONDITIONS[condition][0]=='high' else 0
    for row in history:
        p.round_points(row)
        if any(row[name]!=peer for name in p.PEERS):raise ValueError('Partner history does not match the assigned script')
    current=len(history)+1
    parts=[c.RULES if rules_text is None else rules_text]
    if history:parts.append(p.history_block(history))
    norm=p.CONDITIONS[condition][1]
    if norm!='none':
        amount=10 if norm=='high' else 0
        parts.append(f'Before the contribution decisions in round {current}, B, C, and D each stated: '
            f'"In round {current}, each participant should contribute {amount} points."')
    parts.extend([f'It is now round {current}.',ENDING])
    return [dict(role='system',content=p.SYSTEM),dict(role='user',content='\n\n'.join(parts))]


def jobs():
    rows=[dict(episode_id=f'pilot/{condition}/{index:03}',phase='pilot',condition=condition,
        replicate=index,order='history_first',first_round=1,final_round=10)
        for condition in p.CONDITIONS for index in range(10)]
    random.Random(2026110515).shuffle(rows)
    return rows


def check_stop(out):
    if (out/'STOP_REQUESTED').exists():control.stop('Stop requested via saved flag.')
    return control.STOP.is_set()


def episode(job,out,api_client=None,initial_history=None,rules_text=None):
    api=q if api_client is None else api_client
    begin=time.perf_counter();history=[dict(x) for x in (initial_history or [])];status='complete'
    messages(job['condition'],history,rules_text)
    for current in range(len(history)+1,11):
        if check_stop(out):status='operational_stop';break
        sent=messages(job['condition'],history,rules_text);record=None
        for attempt in range(3):
            if check_stop(out):status='operational_stop';break
            pace()
            if check_stop(out):status='operational_stop';break
            record=dict(job=job,round=current,attempt=attempt+1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
                history_before=[dict(x) for x in history],
                messages_sha256=hashlib.sha256(json.dumps(sent,sort_keys=True).encode()).hexdigest())
            started=time.perf_counter()
            try:
                actual=api.call(sent,f"v10/{job['episode_id']}/round_{current}/attempt_{attempt+1}")
                choice=actual['response']['choices'][0]
                value,parse_status=parse_decision(choice['message'].get('content',''))
                answer_status=parse_status if choice.get('finish_reason')=='stop' else 'incomplete'
                record.update(actual=actual,status=answer_status,parse_status=parse_status,
                    contribution=value if answer_status=='valid' else None,finish_reason=choice.get('finish_reason'))
            except Exception as exc:
                record.update(status='transport_error',error=str(exc))
                if control.fatal_error(str(exc)):control.stop(str(exc))
            record['attempt_wall_seconds']=time.perf_counter()-started
            append(out/'raw.jsonl',record)
            if record['status']!='transport_error' or attempt==2 or not transient(record.get('error','')) or control.STOP.is_set():break
            time.sleep(3*(attempt+1))
        if status=='operational_stop':break
        if record is None or record['status']!='valid':
            status=record['status'] if record else 'operational_stop';break
        peer=10 if p.CONDITIONS[job['condition']][0]=='high' else 0
        row=dict(A=record['contribution'],B=peer,C=peer,D=peer)
        history.append(row)
        append(out/'decisions.jsonl',dict(job=job,round=current,contribution=row['A'],contributions=row,
            payoff=p.round_points(row),cumulative_payoff=sum(p.round_points(x) for x in history)))
    result=dict(job=job,status=status,completed_decisions=len(history),final_history=history,
        episode_wall_seconds=time.perf_counter()-begin)
    append(out/'episodes.jsonl',result)
    return result


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def audit(out,plan):
    raw,decisions,episodes=[read(out/name) for name in ('raw.jsonl','decisions.jsonl','episodes.jsonl')]
    assert len({(x['job']['episode_id'],x['round']) for x in decisions})==len(decisions)
    for item in decisions:
        condition=item['job']['condition'];peer=10 if p.CONDITIONS[condition][0]=='high' else 0
        assert item['contributions']==dict(A=item['contribution'],B=peer,C=peer,D=peer)
        assert item['payoff']==p.round_points(item['contributions'])
        previous=[x for x in decisions if x['job']['episode_id']==item['job']['episode_id'] and x['round']<=item['round']]
        assert item['cumulative_payoff']==sum(x['payoff'] for x in previous)
    for item in raw:
        condition=item['job']['condition']
        previous=sorted([x for x in decisions if x['job']['episode_id']==item['job']['episode_id'] and x['round']<item['round']],key=lambda x:x['round'])
        expected=[x['contributions'] for x in previous]
        assert item['history_before']==expected and len(expected)==item['round']-1
        if 'actual' in item:
            req=item['actual']['request']
            assert req['messages']==messages(condition,expected,plan.get('rules_text'))
            config=plan['continuation_config'] if item.get('collection_stage')=='continuation' else plan['config']
            for key in ('model','enable_thinking','thinking_budget','temperature','thinking','reasoning_effort','top_p','max_tokens','response_format'):
                if key in config:assert req[key]==config[key]
                else:assert key not in req
            text=req['messages'][1]['content'];norm=p.CONDITIONS[condition][1]
            assert text.count('each stated:')==(0 if norm=='none' else 1)
            assert f'Round {item["round"]} record:' not in text
            assert f'It is now round {item["round"]}.' in text
    return dict(passed=True,attempts_checked=len(raw),decisions_checked=len(decisions),episodes_recorded=len(episodes),
        checks=['true observed round1','no fabricated initial A contribution','current-round statements only',
            'full action/payoff history','no current partner-action leak',
            'inference settings match their logged collection stage','correct per-round and cumulative payoffs'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    q.CONFIG_PATH=q.ROOT/'qwen_api_v10_round1_config.json'
    if not args.run:
        cfg=json.loads((q.ROOT/'qwen_api_current_repeat_config.json').read_text(encoding='utf-8'))
        cfg.update(local_spend_cap_cny=2.5,budget_ledger_file='qwen_v10_round1_budget.jsonl',
            note='Only 60 true-round1 ten-round pilot episodes, max600 decisions. Current opinion only. No formal auto-start. Budget estimates exclude cache/free credits.')
        q.CONFIG_PATH.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        out=q.ROOT/'runs'/('v10_qwen_round1_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
        out.mkdir(exist_ok=False)
        schedule=jobs()
        assert len(schedule)==60 and Counter(x['condition'] for x in schedule)=={x:10 for x in p.CONDITIONS}
        first_prompts={condition:messages(condition,[]) for condition in p.CONDITIONS}
        assert len({json.dumps(x,sort_keys=True) for x in first_prompts.values()})==3
        plan=dict(version='v10_true_round1_current_opinion_only',config=cfg,workers=5,jobs=schedule,
            purpose='exploratory_pilot_not_formal',max_decision_calls=600,formal_auto_start=False,
            history_policy='All prior actual actions and own payoffs; only the current-round peer opinion. No earlier opinions or reasoning returned.',
            first_round_policy='No past behavior is supplied; assigned future partner script is hidden. Three unique opinion prompts with20 independent first-round requests each. No response cloning.',
            analysis_policy='Round1 pool by3 opinion conditions; round2+ by6 conditions. Describe round1,round2,episode means3-9 and round10 separately; relative N+ minus absent effects by behavior; no p-value gate or optional effect-based stopping.',
            limitations='Opinion acts from round1; later history is a mediator. Estimate entire intervention/feedback effects, not a fixed-history direct opinion effect. Do not adjust away own round1 as if pretreatment.',
            retries=2,retry_delays=[3,6],no_answer_repair=True,
            source_sha256={name:hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest() for name in
                ('run_from_round1.py','qwen_api.py','run_current_multiround.py','run_pilot_v8.py','prompts_v7.py','check_revision_r1.py')})
        (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'round1_prompts.json').write_text(json.dumps(first_prompts,ensure_ascii=False,indent=2),encoding='utf-8')
        for name in plan['source_sha256']:(out/name).write_bytes((q.ROOT/name).read_bytes())
        print('PREPARED '+json.dumps(str(out)),flush=True);return
    out=args.run
    if (out/'raw.jsonl').exists():raise RuntimeError('Attempts exist; refusing duplicate or selective rerun.')
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    assert json.loads(q.CONFIG_PATH.read_text(encoding='utf-8'))==plan['config']
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest()==digest
    started=time.perf_counter();results=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(episode,job,out) for job in plan['jobs']]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%5==0 or results[-1]['status']!='complete':
                print(json.dumps(dict(finished=len(results),planned=60,statuses=dict(Counter(x['status'] for x in results)),
                    seconds=round(time.perf_counter()-started,2))),flush=True)
    summary=dict(wall_seconds=time.perf_counter()-started,planned_episodes=60,
        episode_statuses=dict(Counter(x['status'] for x in results)),stop_reasons=control.STOP_REASON,formal_started=False)
    (out/'execution_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    integrity=audit(out,plan)
    (out/'integrity_check.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE '+json.dumps(dict(folder=str(out),**summary,integrity=integrity)),flush=True)


if __name__=='__main__':main()
