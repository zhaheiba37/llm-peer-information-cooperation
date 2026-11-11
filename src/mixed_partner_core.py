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
import prompts_v11_public_observation as v
RULES=v.RULES

def rewrite(text):return text

def script(condition,minority_actor):
    assert minority_actor in p.PEERS
    majority=10 if p.CONDITIONS[condition][0]=='high' else 0
    return {name:(10-majority if name==minority_actor else majority) for name in p.PEERS}

import run_current_multiround as control
from run_local_v7 import parse_decision
from run_pilot_v8 import append, pace, transient


ENDING='Choose your contribution for this round. Return only {"contribution": n}, where n is an integer from 0 to 10. Do not include an explanation.'


def messages(condition,history,rules_text=None,minority_actor=None):
    if condition not in p.CONDITIONS:raise ValueError('Unknown condition')
    if not isinstance(history,list) or not 0<=len(history)<10:raise ValueError('Expect zero to nine completed rounds')
    if history and minority_actor is None:
        majority=10 if p.CONDITIONS[condition][0]=='high' else 0
        minority_actor=next(name for name in p.PEERS if history[0][name]!=majority)
    peers=script(condition,minority_actor or 'B')
    for row in history:
        p.round_points(row)
        if any(row[name]!=peers[name] for name in p.PEERS):raise ValueError('Partner history does not match the assigned script')
    current=len(history)+1
    parts=[RULES if rules_text is None else rules_text]
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
    messages(job['condition'],history,rules_text,job['minority_actor'])
    for current in range(len(history)+1,11):
        if check_stop(out):status='operational_stop';break
        sent=messages(job['condition'],history,rules_text,job['minority_actor']);record=None
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
        row=dict(A=record['contribution'],**script(job['condition'],job['minority_actor']))
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
        condition=item['job']['condition'];peers=script(condition,item['job']['minority_actor'])
        assert item['contributions']==dict(A=item['contribution'],**peers)
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
            assert req['messages']==messages(condition,expected,plan.get('rules_text'),item['job']['minority_actor'])
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

