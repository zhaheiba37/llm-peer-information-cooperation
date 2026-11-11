"""Frozen r1 qualification; independent calls, no cooperation sampling."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics

import check_revision_r1 as c
import prompts_v7 as p
import qwen_api as q
from recheck_plus_r1 import tasks as previous_tasks
from run_local_v7 import parse_object, same_answer


def tasks():
    # Fresh odd contributions also exercise fractional payoffs. Historical A stays 5.
    rows = [x for x in previous_tasks() if x['category'] != 'payoff']
    for condition, (behavior, norm) in p.CONDITIONS.items():
        peer = 10 if behavior == 'high' else 0
        for own in (1, 3, 7):
            vector = dict(A=own, B=peer, C=peer, D=peer)
            for rep in range(2):
                rows.append(dict(task_id=f'payoff/{condition}/{own}/{rep}', category='payoff',
                    condition=condition, vector=vector, replicate=rep,
                    messages=[dict(role='system', content=p.SYSTEM), dict(role='user',
                        content=c.context(condition)+'\n\n'+c.payoff_question(vector, True))],
                    expected_not_for_model={'payoff': p.round_points(vector)}))
    extra = [
        ('round_count', 'How many rounds does this game have? Return only {"rounds": n}.', {'rounds': 10}),
        ('group_size', 'How many participants are in your group, including you? Return only {"participants": n}.', {'participants': 4}),
        ('integer_range', 'What contributions are allowed in each round? Return only {"minimum": x, "maximum": y, "integer_only": true} with the correct numbers and Boolean value.', {'minimum': 0, 'maximum': 10, 'integer_only': True}),
        ('fresh_endowment', 'If your payoff in a previous round was 22.5 points, how many new points do you receive at the start of the next round? Return only {"points": n}.', {'points': 10}),
        ('simultaneous', 'Do all four participants choose their contributions simultaneously in each round? Return only {"simultaneous": true} or {"simultaneous": false}.', {'simultaneous': True}),
        ('total_payoff', 'If your payoffs in the first two rounds are 7.5 and 9.5 points, what is your accumulated payoff over these two rounds? Return only {"total": x}.', {'total': 17.0}),
    ]
    for index, (key, question, expected) in enumerate(extra):
        condition = list(p.CONDITIONS)[index]
        rows.append(dict(task_id=key, category='basic_rule', condition=condition,
            messages=[dict(role='system', content=p.SYSTEM), dict(role='user',
                content=c.context(condition)+'\n\n'+question)], expected_not_for_model=expected))
    assert len(rows) == 61 and len({x['task_id'] for x in rows}) == 61
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((q.ROOT/'qwen_api_flash_thinking_probe_config.json').read_text(encoding='utf-8'))
    config.update(local_spend_cap_cny=2.1, note='61-question qualification; shared ledger reserves upper bounds, not actual charges. No formal cooperation sampling.')
    q.CONFIG_PATH = q.ROOT/'qwen_api_flash_qualification_config.json'
    q.CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    out = args.resume or q.ROOT/'runs'/('flash_thinking_qualification_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    if args.resume:
        plan = json.loads((out/'plan.json').read_text(encoding='utf-8'))
        attempts = [json.loads(line) for line in (out/'raw.jsonl').read_text(encoding='utf-8').splitlines()] if (out/'raw.jsonl').exists() else []
    else:
        rows = tasks()
        random.Random(202611054).shuffle(rows)
        plan = dict(planned=61, config=config, candidate='r1', purpose='comprehension_qualification_not_cooperation_results',
            provisional_engineering_gate='61 completed; payoff >=34/36; all 25 factual checks correct; all responses finish normally. Engineering criterion, not a literature standard.',
            note='Frozen before responses. Independent calls. No answer-based retries. Fresh own contributions 1,3,7; historical A=5. Reasoning not inserted into later prompts.', tasks=rows)
        out.mkdir(parents=True, exist_ok=False)
        (out/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        attempts = []
    if args.prepare_only:
        print(json.dumps(dict(folder=str(out), planned=61, calls_sent=0)))
        return
    done = {x['task']['task_id'] for x in attempts if x['status'] == 'complete'}
    for index, task in enumerate(plan['tasks']):
        if task['task_id'] in done:
            continue
        record = dict(task=task)
        try:
            actual = q.call(task['messages'], 'flash_thinking_qualification/'+task['task_id'])
            choice = actual['response']['choices'][0]
            message = choice['message']
            parsed, status = parse_object(message.get('content', ''))
            record.update(status='complete', actual=actual, parsed=parsed, parse_status=status,
                finish_reason=choice.get('finish_reason'), reasoning_present=bool(message.get('reasoning_content')),
                correct=choice.get('finish_reason') == 'stop' and same_answer(parsed, task['expected_not_for_model']))
        except Exception as exc:
            record.update(status='error', correct=False, error=str(exc))
        with (out/'raw.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        attempts.append(record)
        print(json.dumps(dict(number=index+1, total=61, task=task['task_id'], status=record['status'], correct=record['correct'], answer=record.get('parsed'))), flush=True)
        if record['status'] == 'error':
            break
    completed = [x for x in attempts if x['status'] == 'complete']
    groups = defaultdict(list)
    for record in completed:
        groups[record['task']['category']].append(record)
    summary = {key: dict(correct=sum(x['correct'] for x in value), total=len(value)) for key, value in groups.items()}
    payoff = groups['payoff']
    facts = [x for x in completed if x['task']['category'] != 'payoff']
    result = dict(planned=61, completed=len(completed), correct=sum(x['correct'] for x in completed), summary=summary,
        passed=len(completed) == 61 and sum(x['correct'] for x in payoff) >= 34 and sum(x['correct'] for x in facts) == 25 and all(x['finish_reason'] == 'stop' for x in completed),
        errors=sum(x['status'] == 'error' for x in attempts),
        reasoning_present=sum(x['reasoning_present'] for x in completed),
        estimated_list_price_cny=sum(x['actual']['estimated_list_price_cny'] for x in completed))
    if completed:
        result.update(mean_seconds=statistics.mean(x['actual']['seconds'] for x in completed),
            total_request_seconds=sum(x['actual']['seconds'] for x in completed),
            prompt_tokens=sum(x['actual']['response']['usage']['prompt_tokens'] for x in completed),
            completion_tokens=sum(x['actual']['response']['usage']['completion_tokens'] for x in completed))
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RESULT '+json.dumps(dict(folder=str(out), **result)), flush=True)


if __name__ == '__main__':
    main()
