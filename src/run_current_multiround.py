"""v9: fixed history-first, current-round peer norms; conditional formal sampling.

Pilot 6x5 independent episodes, formal 6x50 NEW episodes if the frozen gate
passes. One runner process, five threads, no answer repair or optional stopping
on formal effects. Formal collection stops only for operational/financial limits.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time

import check_revision_r1 as c
import prompts_v7 as p
import qwen_api as q
from run_local_v7 import parse_decision
from run_pilot_v8 import append, pace, transient

STOP = threading.Event()
STOP_REASON = []
STOP_LOCK = threading.Lock()


def messages(condition, history):
    user = p.decision(condition, history=history, order='history_first')
    assert user.count(p.RULES['T1']) == 1
    user = user.replace(p.RULES['T1'], c.RULES, 1)
    norm = p.CONDITIONS[condition][1]
    current = len(history)+1
    if norm != 'none':
        old = p.norm_block(condition)
        assert user.count(old) == 1
        amount = 10 if norm == 'high' else 0
        statement = (f'Before the contribution decisions in round {current}, B, C, and D each stated: '
                     f'"In round {current}, each participant should contribute {amount} points."')
        user = user.replace(old, statement, 1)
    return [dict(role='system', content=p.SYSTEM), dict(role='user', content=user)]


def make_jobs(phase, count, seed):
    rows = [dict(episode_id=f'{phase}/{condition}/{index:03}', phase=phase, condition=condition,
                 replicate=index, order='history_first', final_round=10)
            for condition in p.CONDITIONS for index in range(count)]
    random.Random(seed).shuffle(rows)
    return rows


def fatal_error(message):
    lower = message.lower()
    return any(word in lower for word in ('spend cap', 'http 401', 'http 403', 'insufficient', 'arrear',
        'overdue', 'quota', 'invalidapikey', 'not enough balance', 'balance', 'accessdenied'))


def stop(reason):
    with STOP_LOCK:
        if not STOP_REASON:
            STOP_REASON.append(reason)
        STOP.set()


def episode(job, out, api_client=None):
    client = api_client or q
    started = time.perf_counter()
    condition = job['condition']
    history = p.initial_history(condition)
    n_valid = 0
    status = 'complete'
    for current in range(2, 11):
        if STOP.is_set():
            status = 'operational_stop'
            break
        sent = messages(condition, history)
        record = None
        for attempt in range(3):
            if STOP.is_set():
                status = 'operational_stop'
                break
            pace()
            record = dict(job=job, round=current, attempt=attempt+1,
                history_before=[dict(x) for x in history], timestamp_utc=datetime.now(timezone.utc).isoformat(),
                messages_sha256=hashlib.sha256(json.dumps(sent, sort_keys=True).encode()).hexdigest())
            call_start = time.perf_counter()
            try:
                actual = client.call(sent, f"v9/{job['episode_id']}/round_{current}/attempt_{attempt+1}")
                choice = actual['response']['choices'][0]
                value, parsed_status = parse_decision(choice['message'].get('content', ''))
                answer_status = parsed_status if choice.get('finish_reason') == 'stop' else 'incomplete'
                record.update(actual=actual, status=answer_status, parse_status=parsed_status,
                    contribution=value if answer_status == 'valid' else None, finish_reason=choice.get('finish_reason'))
            except Exception as exc:
                record.update(status='transport_error', error=str(exc))
                if fatal_error(str(exc)):
                    stop(str(exc))
            record['attempt_wall_seconds'] = time.perf_counter()-call_start
            append(out/'raw.jsonl', record)
            if record['status'] != 'transport_error' or attempt == 2 or not transient(record.get('error', '')) or STOP.is_set():
                break
            time.sleep(3*(attempt+1))
        if status == 'operational_stop':
            break
        if record is None or record['status'] != 'valid':
            status = record['status'] if record else 'operational_stop'
            break
        peer = 10 if p.CONDITIONS[condition][0] == 'high' else 0
        row = dict(A=record['contribution'], B=peer, C=peer, D=peer)
        history.append(row)
        n_valid += 1
        append(out/'decisions.jsonl', dict(job=job, round=current, contribution=row['A'], contributions=row,
            payoff=p.round_points(row), cumulative_payoff=sum(p.round_points(x) for x in history)))
    result = dict(job=job, status=status, completed_decisions=n_valid, final_history=history,
        episode_wall_seconds=time.perf_counter()-started)
    append(out/'episodes.jsonl', result)
    return result


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def audit(out, plan):
    raw, decisions, episodes = [read(out/name) for name in ('raw.jsonl','decisions.jsonl','episodes.jsonl')]
    assert len({(x['job']['episode_id'],x['round']) for x in decisions}) == len(decisions)
    for item in decisions:
        job = item['job']
        peer = 10 if p.CONDITIONS[job['condition']][0] == 'high' else 0
        assert item['contributions'] == dict(A=item['contribution'],B=peer,C=peer,D=peer)
        assert item['payoff'] == p.round_points(item['contributions'])
        prior = [x for x in decisions if x['job']['episode_id']==job['episode_id'] and x['round']<=item['round']]
        assert item['cumulative_payoff'] == p.round_points(p.initial_history(job['condition'])[0])+sum(x['payoff'] for x in prior)
    for item in raw:
        job = item['job']
        prior = sorted([x for x in decisions if x['job']['episode_id']==job['episode_id'] and x['round']<item['round']],key=lambda x:x['round'])
        expected = p.initial_history(job['condition'])+[x['contributions'] for x in prior]
        assert item['history_before'] == expected and len(expected)==item['round']-1
        if 'actual' in item:
            req = item['actual']['request']
            assert req['messages'] == messages(job['condition'],expected)
            for key in ('model','enable_thinking','thinking_budget','temperature','top_p','max_tokens','response_format','thinking','reasoning_effort'):
                if key in plan['config']:
                    assert req[key] == plan['config'][key]
                else:
                    assert key not in req
    return dict(passed=True, attempts_checked=len(raw), decisions_checked=len(decisions), episodes=len(episodes))


def assess_gate(out, plan):
    integrity = audit(out, plan)
    episodes, decisions, raw = [read(out/name) for name in ('episodes.jsonl','decisions.jsonl','raw.jsonl')]
    complete = [x for x in episodes if x['status']=='complete']
    min_complete = min(sum(x['job']['condition']==condition for x in complete) for condition in p.CONDITIONS)
    last = {(x['job']['episode_id'],x['round']):x for x in raw}
    valid_rate = sum(x['status']=='valid' for x in last.values())/len(last) if last else 0
    def mean_round(condition, current):
        values = [x['contribution'] for x in decisions if x['job']['condition']==condition and x['round']==current]
        return statistics.mean(values) if values else None
    per_round=[]
    for current in range(2,11):
        high=mean_round('D_LOW_N_HIGH',current); absent=mean_round('D_LOW_N_ABSENT',current)
        per_round.append(dict(round=current, high=high, absent=absent, difference=high-absent if high is not None and absent is not None else None))
    def later_episode_means(condition):
        values=[]
        for item in episodes:
            if item['job']['condition']!=condition:continue
            selected=[x['contribution'] for x in decisions if x['job']['episode_id']==item['job']['episode_id'] and 3<=x['round']<=9]
            if len(selected)==7:values.append(statistics.mean(selected))
        return values
    hi=later_episode_means('D_LOW_N_HIGH'); no=later_episode_means('D_LOW_N_ABSENT')
    later=statistics.mean(hi)-statistics.mean(no) if hi and no else None
    instantaneous=per_round[0]['difference']
    positive=sum(x['difference'] is not None and x['difference']>0 for x in per_round if 3<=x['round']<=9)
    checks=dict(integrity=integrity['passed'],all_30_episodes_accounted=len(episodes)==30,
        completed_at_least_29=len(complete)>=29,at_least_4_complete_per_condition=min_complete>=4,
        returned_decision_valid_rate_at_least_99pct=valid_rate>=.99,
        round2_advantage_at_least_one=instantaneous is not None and instantaneous>=1,
        nonterminal_advantage_at_least_one=later is not None and later>=1,
        positive_in_at_least_four_nonterminal_rounds=positive>=4,
        no_operational_stop=not STOP.is_set())
    result=dict(passed=all(checks.values()),checks=checks,completed_episodes=len(complete),valid_rate=valid_rate,
        round2_difference=instantaneous,nonterminal_episode_mean_difference=later,
        positive_nonterminal_rounds=positive,per_round=per_round,integrity=integrity,
        note='Predeclared go/no-go engineering rule, not a significance test or proof. Pilot never enters formal data.')
    (out/'gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def run_stage(out, jobs, phase, api_client=None):
    out.mkdir(exist_ok=False)
    begin=time.perf_counter();results=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(episode,job,out,api_client) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%5==0 or results[-1]['status']!='complete':
                print(json.dumps(dict(phase=phase,finished=len(results),planned=len(jobs),
                    statuses=dict(Counter(x['status'] for x in results)),seconds=round(time.perf_counter()-begin,2))),flush=True)
    metadata=dict(phase=phase,wall_seconds=time.perf_counter()-begin,planned_episodes=len(jobs),
        status_counts=dict(Counter(x['status'] for x in results)),stop_reasons=STOP_REASON)
    (out/'execution_summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    q.CONFIG_PATH=q.ROOT/'qwen_api_v9_multiround_config.json'
    if not args.run:
        config=json.loads((q.ROOT/'qwen_api_current_repeat_config.json').read_text(encoding='utf-8'))
        config.update(local_spend_cap_cny=6.0,budget_ledger_file='qwen_v9_multiround_budget.jsonl',
            note='Conditional pilot plus single-Qwen formal sampling authorized 2026-11-05. No top-up or account limit change. Stop on financial/auth errors. Local total committed cap CNY6.')
        q.CONFIG_PATH.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        out=q.ROOT/'runs'/('v9_current_multiround_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
        out.mkdir(exist_ok=False)
        plan=dict(version='v9_current_history_first_20261105',config=config,workers=5,
            pilot_jobs=make_jobs('pilot',5,2026110510),formal_jobs=make_jobs('formal',50,2026110511),
            gate_rule=dict(min_complete=29,min_per_condition=4,min_valid_rate=.99,min_round2_difference=1,
                          min_round3_to9_episode_mean_difference=1,min_positive_rounds3_to9=4),
            transport_retry_limit=2,retry_delays=[3,6],no_answer_repair=True,
            norm_history_policy='All past contributions and own per-round payoffs; only current-round peer statement. Earlier statements not repeated in history. No reasoning carried forward.',
            primary_outcomes=['round2: D_LOW_N_HIGH minus D_LOW_N_ABSENT','round2: (N_HIGH minus N_LOW) at D_HIGH minus the same at D_LOW'],
            secondary_outcomes=['episode mean rounds3-9','round10 separately','round2-10 trajectories, episode is sampling unit'],
            inference_policy='Report effects and episode-bootstrap intervals; if significance is later evaluated, retain planned three-family six-test Holm family. Single-model report alone is not complete cross-model confirmation.',
            formal_stopping='Fixed 50 attempted episodes per condition regardless of effect. Never replace invalid-answer episodes or add samples until significant. Financial/operational stop leaves durable partial data; no filling.',
            source_sha256={name:hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest() for name in
                ('run_current_multiround.py','qwen_api.py','prompts_v7.py','check_revision_r1.py','run_pilot_v8.py')})
        (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'round2_prompts.json').write_text(json.dumps({condition:messages(condition,p.initial_history(condition)) for condition in p.CONDITIONS},ensure_ascii=False,indent=2),encoding='utf-8')
        for name in plan['source_sha256']:(out/name).write_bytes((q.ROOT/name).read_bytes())
        print('PREPARED '+json.dumps(str(out)),flush=True);return
    out=args.run
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    assert plan['config']==json.loads(q.CONFIG_PATH.read_text(encoding='utf-8'))
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest()==digest
    if (out/'pilot').exists() or (out/'formal').exists():raise RuntimeError('Existing data: inspect checkpoints; no automatic duplicate rerun.')
    run_stage(out/'pilot',plan['pilot_jobs'],'pilot')
    gate=assess_gate(out/'pilot',plan)
    print('GATE '+json.dumps(gate),flush=True)
    if gate['passed']:
        print('FORMAL_START: new fixed-sample batch; pilot excluded.',flush=True)
        run_stage(out/'formal',plan['formal_jobs'],'formal')
        integrity=audit(out/'formal',plan)
        (out/'formal'/'integrity_check.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    else:
        print('FORMAL_NOT_STARTED: frozen pilot gate failed; stop for user discussion.',flush=True)
    print('DONE '+json.dumps(str(out)),flush=True)


if __name__=='__main__':main()
