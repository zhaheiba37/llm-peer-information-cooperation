"""Frozen fresh-number check of the r1 candidate. No formal decision sampling."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import statistics
from collections import defaultdict
import check_revision_r1 as c
import prompts_v7 as p
import qwen_api as q
from run_local_v7 import parse_object,same_answer


def tasks():
    rows=[]
    def add(key,category,condition,user,expected,**meta):
        rows.append(dict(task_id=key,category=category,condition=condition,**meta,
            messages=[dict(role='system',content=p.SYSTEM),dict(role='user',content=user)],expected_not_for_model=expected))
    # Matches the controlled partner patterns, with fresh own-contribution values.
    for condition,(behavior,norm) in p.CONDITIONS.items():
        peer=10 if behavior=='high' else 0
        for own in (2,6,10):
            vector=dict(A=own,B=peer,C=peer,D=peer)
            for rep in range(2):
                add(f'payoff/{condition}/{own}/{rep}','payoff',condition,
                    c.context(condition)+'\n\n'+c.payoff_question(vector,True),
                    {'payoff':p.round_points(vector)},vector=vector,replicate=rep)
        for item in [x for x in c.tasks() if x['category']=='statement_present' and x['condition']==condition]:
            rows.append(item)
        if norm!='none':
            question=('According to B\'s recorded statement, how many points should each participant contribute per round? '
                      'Return only {"amount": n}, where n is an integer. Do not include an explanation.')
            add('stated_amount/'+condition,'stated_amount',condition,c.context(condition)+'\n\n'+question,
                {'amount':10 if norm=='high' else 0})
        question=('According to the round 1 record, how many points did B contribute in round 1? '
                  'Return only {"amount": n}, where n is an integer. Do not include an explanation.')
        add('past_behavior/'+condition,'past_behavior',condition,c.context(condition)+'\n\n'+question,{'amount':peer})
    rows.extend(x for x in c.tasks() if x['category']=='basic_rule')
    assert len(rows)==55
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);args=parser.parse_args()
    q.CONFIG_PATH=q.ROOT/'qwen_api_config.json'
    out=args.resume or q.ROOT/'runs'/('plus_r1_recheck_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    if args.resume:
        plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
        attempts=[json.loads(line) for line in (out/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    else:
        rows=tasks();random.Random(202611052).shuffle(rows)
        plan=dict(model='qwen3.7-plus-2026-05-26',candidate='check_revision_r1',planned=55,
                  mode='nonthinking',purpose='fresh_number_recheck_not_cooperation_results',
                  provisional_engineering_gate='payoff >=34/36; factual checks 19/19; not a literature standard',
                  note='All 55 planned before responses. Identical r1 rule and question wording. No answer-based retries or question selection.',tasks=rows)
        out.mkdir(parents=True,exist_ok=False)
        (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8');attempts=[]
    done={x['task']['task_id'] for x in attempts if x['status']=='complete'}
    for index,task in enumerate(plan['tasks']):
        if task['task_id'] in done:continue
        record=dict(task=task)
        try:
            actual=q.call(task['messages'],'plus_r1_fresh_recheck/'+task['task_id'])
            choice=actual['response']['choices'][0]
            content=choice['message'].get('content','')
            parsed,status=parse_object(content)
            record.update(status='complete',actual=actual,raw_output=content,parsed=parsed,
                    correct=choice.get('finish_reason')=='stop' and same_answer(parsed,task['expected_not_for_model']))
        except Exception as exc:record.update(status='error',correct=False,error=str(exc))
        with (out/'raw.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        attempts.append(record)
        print(json.dumps(dict(number=index+1,total=55,task=task['task_id'],status=record['status'],correct=record['correct'],output=record.get('raw_output'))),flush=True)
        if record['status']=='error':break
    completed=[x for x in attempts if x['status']=='complete'];groups=defaultdict(list)
    for x in completed:groups[x['task']['category']].append(x)
    summary={k:dict(correct=sum(x['correct'] for x in v),total=len(v)) for k,v in groups.items()}
    payload=dict(planned=55,completed=len(completed),network_errors=len(attempts)-len(completed),summary=summary,
        estimated_list_price_cny=sum(x['actual']['estimated_list_price_cny'] for x in completed),
        seconds_mean=statistics.mean(x['actual']['seconds'] for x in completed) if completed else None,
        seconds_median=statistics.median(x['actual']['seconds'] for x in completed) if completed else None)
    (out/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT '+json.dumps(dict(folder=str(out),**payload)),flush=True)


if __name__=='__main__':main()
