"""Frozen 300 round-2 pilot plus 30 independent 10-round pipeline trajectories.

One process, five threads. Only transient transport errors are retried (twice).
No answer repair, answer-based retry, or sharing reasoning between calls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import threading
import time

import check_revision_r1 as c
import prompts_v7 as p
import qwen_api as q
from run_local_v7 import parse_decision

WRITE_LOCK = threading.Lock()
START_LOCK = threading.Lock()
LAST_START = 0.0


def messages(condition, history, order):
    user = p.decision(condition, history=history, order=order)
    assert user.count(p.RULES['T1']) == 1
    user = user.replace(p.RULES['T1'], c.RULES, 1)
    return [dict(role='system', content=p.SYSTEM), dict(role='user', content=user)]


def jobs():
    rows = []
    for phase, count, final_round in [('single', 50, 2), ('continuation', 5, 10)]:
        for condition, (behavior, norm) in p.CONDITIONS.items():
            for index in range(count):
                order = 'statement_first' if norm != 'none' and index % 2 else 'history_first'
                rows.append(dict(episode_id=f'{phase}/{condition}/{index:03}', phase=phase,
                    condition=condition, replicate=index, order=order, final_round=final_round))
    rng = random.Random(202611055)
    rng.shuffle(rows)
    return rows


def append(path, value):
    with WRITE_LOCK:
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')


def transient(message):
    return any(x in message for x in ('network error', 'connection error', 'HTTP 429', 'HTTP 500', 'HTTP 502', 'HTTP 503', 'HTTP 504'))


def pace():
    global LAST_START
    with START_LOCK:
        delay = .25-(time.perf_counter()-LAST_START)
        if delay > 0:
            time.sleep(delay)
        LAST_START = time.perf_counter()


def episode(job, out):
    started = time.perf_counter()
    condition = job['condition']
    history = p.initial_history(condition)
    n_valid = 0
    final_status = 'complete'
    for round_no in range(2, job['final_round']+1):
        request_messages = messages(condition, history, job['order'])
        assert len(history) == round_no-1
        for attempt in range(3):
            record = dict(job=job, round=round_no, attempt=attempt+1,
                timestamp_utc=datetime.now(timezone.utc).isoformat(), history_before=[dict(x) for x in history],
                messages_sha256=hashlib.sha256(json.dumps(request_messages, sort_keys=True).encode()).hexdigest())
            pace()
            call_start = time.perf_counter()
            try:
                actual = q.call(request_messages, f"v8_pilot/{job['episode_id']}/round_{round_no}/attempt_{attempt+1}")
                choice = actual['response']['choices'][0]
                value, parsed_status = parse_decision(choice['message'].get('content', ''))
                status = parsed_status if choice.get('finish_reason') == 'stop' else 'incomplete'
                record.update(status=status, actual=actual, contribution=value if status == 'valid' else None,
                    parse_status=parsed_status, finish_reason=choice.get('finish_reason'))
            except Exception as exc:
                record.update(status='transport_error', error=str(exc))
            record['attempt_wall_seconds'] = time.perf_counter()-call_start
            append(out/'raw.jsonl', record)
            if record['status'] != 'transport_error' or attempt == 2 or not transient(record.get('error', '')):
                break
            time.sleep(3*(attempt+1))
        if record['status'] != 'valid':
            final_status = record['status']
            break
        n_valid += 1
        peer = 10 if p.CONDITIONS[condition][0] == 'high' else 0
        row = dict(A=record['contribution'], B=peer, C=peer, D=peer)
        record_payoff = p.round_points(row)
        history.append(row)
        append(out/'decisions.jsonl', dict(job=job, round=round_no, contribution=row['A'],
            contributions=row, payoff=record_payoff, cumulative_payoff=sum(p.round_points(x) for x in history)))
    result = dict(job=job, status=final_status, completed_decisions=n_valid, final_history=history,
        episode_wall_seconds=time.perf_counter()-started)
    append(out/'episodes.jsonl', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    config = json.loads((q.ROOT/'qwen_api_flash_qualification_config.json').read_text(encoding='utf-8'))
    config.update(local_spend_cap_cny=2.5, budget_ledger_file='qwen_v8_pilot_budget.jsonl',
        settle_usage_budget=True, note='One concurrent runner process. Pilot-only actual settled cost plus outstanding/unknown reservations capped at CNY2.5. No balance query.')
    q.CONFIG_PATH = q.ROOT/'qwen_api_v8_pilot_config.json'
    if not args.run:
        qualifier = q.ROOT/'runs'/'flash_thinking_qualification_20261105_052051_UTC'/'summary.json'
        assert json.loads(qualifier.read_text(encoding='utf-8'))['passed']
        out = q.ROOT/'runs'/('v8_flash_pilot_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
        out.mkdir(parents=True, exist_ok=False)
        q.CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        plan = dict(purpose='exploratory_pilot_not_formal_data', config=config,
            workers=5, minimum_start_interval_seconds=.25, max_transport_retries=2, retry_wait_seconds=[3, 6],
            no_retry_for_invalid_answers=True, single_calls=300, continuation_episodes=30, continuation_calls=270,
            eligibility_summary=str(qualifier), jobs=jobs(),
            source_sha256={name: hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest() for name in
                ['run_pilot_v8.py', 'qwen_api.py', 'prompts_v7.py', 'check_revision_r1.py']},
            note='All calls use fresh system+user context. Continued history uses actual A choices. Original N remains available. Absent N has one order. Continuation is independent of single-round pilot and is for pipeline verification, not dynamic hypothesis testing.')
        (out/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        snapshot = {job['episode_id']: messages(job['condition'], p.initial_history(job['condition']), job['order']) for job in plan['jobs']}
        (out/'round2_prompts.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
        for name in plan['source_sha256']:
            (out/name).write_bytes((q.ROOT/name).read_bytes())
        print('PREPARED '+json.dumps(dict(folder=str(out), calls_sent=0, planned_calls=570)), flush=True)
        return
    out = args.run
    if (out/'raw.jsonl').exists():
        raise RuntimeError('Run already has attempts; refusing to duplicate data. Inspect before any targeted resume.')
    plan = json.loads((out/'plan.json').read_text(encoding='utf-8'))
    assert plan['config'] == json.loads(q.CONFIG_PATH.read_text(encoding='utf-8'))
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest() == digest
    overall_start = time.perf_counter()
    for phase in ('single', 'continuation'):
        stage_jobs = [x for x in plan['jobs'] if x['phase'] == phase]
        stage_start = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = [pool.submit(episode, job, out) for job in stage_jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                count = len(results)
                if count % 10 == 0 or result['status'] != 'complete' or count == len(stage_jobs):
                    print(json.dumps(dict(phase=phase, completed_episodes=count, planned_episodes=len(stage_jobs),
                        failed_episodes=sum(x['status'] != 'complete' for x in results),
                        elapsed_seconds=round(time.perf_counter()-stage_start, 2))), flush=True)
        append(out/'timings.jsonl', dict(phase=phase, wall_seconds=time.perf_counter()-stage_start,
            episodes=len(results), successful=sum(x['status'] == 'complete' for x in results)))
    (out/'execution_summary.json').write_text(json.dumps(dict(total_wall_seconds=time.perf_counter()-overall_start,
        episode_status_counts=dict(Counter(x['status'] for x in [json.loads(line) for line in (out/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]))), indent=2), encoding='utf-8')
    print('DONE '+json.dumps(str(out)), flush=True)


if __name__ == '__main__':
    main()
