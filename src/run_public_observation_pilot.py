"""Fresh six-condition n5 DeepSeek pilot with explicit public contributions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time
import deepseek_api as d
import qwen_api as q
import run_from_round1 as r
import run_deepseek_from_round1 as prior
import run_deepseek_replication as checks
import prompts_v11_public_observation as v


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def tasks():
    rows=prior.tasks()
    for task in rows:
        text=task['messages'][1]['content'];assert text.count(r.c.RULES)==1
        task['messages'][1]['content']=text.replace(r.c.RULES,v.RULES,1)
    for number,(question,answer) in enumerate([
        ('After each round, can B, C, and D observe A\'s contribution? Return only {"can_observe": true} or {"can_observe": false}.',{'can_observe':True}),
        ('Do the rules state that every participant observes every other participant\'s round payoff? Return only {"stated": true} or {"stated": false}.',{'stated':False})]):
        rows.append(dict(task_id=f'public_visibility/{number}',category='visibility',condition='D_HIGH_N_ABSENT',
            messages=[dict(role='system',content=r.p.SYSTEM),dict(role='user',content=v.RULES+'\n\n'+question)],expected_not_for_model=answer))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--provider',choices=('deepseek','qwen'),default='deepseek');args=parser.parse_args()
    api=q if args.provider=='qwen' else d
    api.CONFIG_PATH=api.ROOT/f'{args.provider}_v11_public_config.json'
    if not args.run:
        config_name='qwen_api_v10_round1_config.json' if args.provider=='qwen' else 'deepseek_v10_continuation_config.json'
        config=json.loads((api.ROOT/config_name).read_text(encoding='utf-8'))
        config.update(max_tokens=8192,local_spend_cap_cny=1.5 if args.provider=='qwen' else 2.5,budget_ledger_file=f'{args.provider}_v11_public_budget.jsonl',
            note='20 new-context comprehension checks then fresh six-condition n5 ten-round pilot only. All calls max_tokens8192, no formal auto-start.')
        save(api.CONFIG_PATH,config)
        jobs=[dict(episode_id=f'public_observation/{c}/{i:03}',phase='pilot',condition=c,replicate=i,order='history_first',first_round=1,final_round=10) for c in r.p.CONDITIONS for i in range(5)]
        random.Random(2026110520).shuffle(jobs)
        out=api.ROOT/'runs'/(f'v11_{args.provider}_public_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'));out.mkdir()
        plan=dict(version='v11_public_contributions_private_own_payoffs',provider=args.provider,config=config,workers=5,jobs=jobs,checks=tasks(),rules_text=v.RULES,
            purpose='fresh_independent_exploratory_pilot',max_decision_calls=300,formal_auto_start=False,
            gate='All20 comprehension checks correct and normal finish. Stop on failure; no outcome-based retries, replacements, or optional early stop.',
            history_policy='True round1, full past actual contributions and own payoffs, history first, only current-round opinions, no reasoning feedback.',
            analysis_policy='Round1 three opinion groups n10; round2 and later six groups n5; whole-game unit; never pool with older pilots.',
            comparison_limit='Historical v10 predominantly used max_tokens4096; v11 uniformly8192. Independent small samples and output-cap difference prevent isolating wording as sole cause. Across models, Qwen thinking_budget1024 and temperature0.7 are not equal compute to DeepSeek effort low without temperature.',
            source_sha256={name:hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest() for name in
                ('run_public_observation_pilot.py','prompts_v11_public_observation.py','run_from_round1.py','run_deepseek_from_round1.py','run_deepseek_replication.py','deepseek_api.py','qwen_api.py','run_pilot_v8.py','prompts_v7.py','check_revision_r1.py')})
        save(out/'plan.json',plan)
        save(out/'round1_prompts.json',{c:r.messages(c,[],v.RULES) for c in r.p.CONDITIONS})
        save(out/'round4_examples.json',{c:r.messages(c,[dict(A=a,B=(10 if r.p.CONDITIONS[c][0]=='high' else 0),C=(10 if r.p.CONDITIONS[c][0]=='high' else 0),D=(10 if r.p.CONDITIONS[c][0]=='high' else 0)) for a in (10,0,7)],v.RULES) for c in r.p.CONDITIONS})
        for name in plan['source_sha256']:(out/name).write_bytes((d.ROOT/name).read_bytes())
        print('PREPARED '+str(out),flush=True);return
    out=args.run;plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if (out/'qualification').exists() or (out/'raw.jsonl').exists():raise RuntimeError('Existing attempts; refusing duplicate/selective rerun.')
    assert plan.get('provider','deepseek')==args.provider
    assert json.loads(api.CONFIG_PATH.read_text(encoding='utf-8'))==plan['config']
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest()==digest
    started=time.perf_counter();qual=out/'qualification';qual.mkdir()
    with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(lambda t:checks.do_check(t,qual,f'{args.provider}_public_v11',api),plan['checks']))
    gate=dict(planned=20,completed=len(results),correct=sum(x['correct'] for x in results),passed=len(results)==20 and all(x['correct'] for x in results),wall_seconds=time.perf_counter()-started)
    save(qual/'summary.json',gate);print('CHECKS '+json.dumps(gate),flush=True)
    if gate['passed'] and not r.control.STOP.is_set():
        begin=time.perf_counter();results=[]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(r.episode,job,out,api,None,v.RULES) for job in plan['jobs']]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results)%5==0 or results[-1]['status']!='complete':print(json.dumps(dict(finished=len(results),planned=30,statuses=dict(Counter(x['status'] for x in results)),seconds=round(time.perf_counter()-begin,2))),flush=True)
        summary=dict(wall_seconds=time.perf_counter()-begin,planned_episodes=30,episode_statuses=dict(Counter(x['status'] for x in results)),stop_reasons=r.control.STOP_REASON,formal_started=False)
        save(out/'execution_summary.json',summary);save(out/'integrity_check.json',r.audit(out,plan))
    status=dict(qualification=gate,pilot_started=gate['passed'],formal_started=False,total_wall_seconds=time.perf_counter()-started,stop_reasons=r.control.STOP_REASON)
    if args.provider=='deepseek':
        try:save(out/'balance_after.json',d.get_balance())
        except Exception as exc:status['balance_after_error']=str(exc)
    save(out/'final_status.json',status);print('DONE '+json.dumps(status),flush=True)


if __name__=='__main__':main()
