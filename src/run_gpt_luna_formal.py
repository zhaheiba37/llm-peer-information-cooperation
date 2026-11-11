"""Authorized GPT formal v11: 6x50 complete NEW10-round games,3 workers."""
import json,time,random,hashlib,argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import run_from_round1 as r
import run_public_observation_pilot as pilot
import run_deepseek_replication as checks
import prompts_v11_public_observation as v
import gpt_luna_recovery_api as api
ROOT=api.ROOT

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def initial_jobs():
    rows=[dict(episode_id=f'formal/gpt6_luna/{c}/{i:03}/attempt00',phase='formal',condition=c,replicate=i,formal_slot=f'gpt6_luna/{c}/{i:03}',episode_attempt=0,order='history_first',first_round=1,final_round=10) for c in r.p.CONDITIONS for i in range(50)]
    random.Random(202611061650).shuffle(rows);return rows

def replacement(job):
    attempt=job['episode_attempt']+1
    return dict(job,episode_attempt=attempt,episode_id=f"formal/gpt6_luna/{job['condition']}/{job['replicate']:03}/attempt{attempt:02}",replaces_episode_id=job['episode_id'])

def verify(out,plan):
    integrity=r.audit(out,plan)
    for row in r.read(out/'raw.jsonl'):
        if 'actual' in row:
            req=row['actual']['request']
            assert req['max_completion_tokens']==8192 and req['store'] is False and req['stream'] is False
            assert row['actual']['response']['model']=='gpt-6-luna'
    integrity['openai_output_limit_and_returned_model_checked']=True
    return integrity

def prepare():
    out=ROOT/'runs'/('formal_v11_gpt6_luna_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'));out.mkdir()
    plan=dict(version='formal_v11_gpt6_luna',formal=True,provider='openai',config=api.CONFIG,rules_text=v.RULES,jobs=initial_jobs(),checks=pilot.tasks(),max_decision_calls=3000,workers=3,max_starts_per_second=4,local_cap_usd=.50,primary_analysis='Round1 pooled3 opinions; round2 six conditions; episode means3-9 and round10; opinion differences by behavior and interactions; bootstrap5000 whole episodes. New analysis labeled exploratory.',failure_policy='Same-input transient retries max2, delays3/6seconds. Failed games retained; independent replacement from true round1 same slot, unique attemptID, max10 replacements/condition. No valid-answer retry, no result-based selection. Stop auth/budget/unexpected failures.',independence='All formal calls fresh and separate from pilots; current opinion only, full past actual contributions and own payoffs, history-first, no reasoning feedback.',source_sha256={})
    for name in ['run_gpt_luna_formal.py','gpt_luna_recovery_api.py','run_from_round1.py','run_deepseek_replication.py','run_public_observation_pilot.py','run_deepseek_from_round1.py','prompts_v11_public_observation.py','check_revision_r1.py','prompts_v7.py','run_pilot_v8.py','run_current_multiround.py']:
        data=(ROOT/name).read_bytes();plan['source_sha256'][name]=hashlib.sha256(data).hexdigest();(out/name).write_bytes(data)
    save(out/'plan.json',plan)
    (ROOT/'gpt_luna_formal_latest_path.txt').write_text(str(out),encoding='utf-8')
    print('PREPARED '+str(out),flush=True);return out

def run(out):
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    assert not (out/'qualification').exists(),'Refusing duplicate collection'
    assert api.CONFIG==plan['config']
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    api.OUT=out;api.COMMITTED=0;api.CAP_USD=plan['local_cap_usd']
    start=time.perf_counter();qual=out/'qualification';qual.mkdir();answers=[]
    for task in plan['checks']:
        answer=checks.do_check(task,qual,'gpt6_luna_formal_low',api);answers.append(answer)
        print(json.dumps(dict(stage='qualification',completed=len(answers),correct=sum(x['correct'] for x in answers))),flush=True)
        if not answer['correct']:r.control.stop('Qualification failed: '+task['task_id']);break
    gate=dict(total=20,correct=sum(x['correct'] for x in answers),completed=len(answers),passed=len(answers)==20 and all(x['correct'] for x in answers),wall_seconds=time.perf_counter()-start)
    save(qual/'summary.json',gate)
    if not gate['passed'] or r.control.STOP.is_set():save(out/'final_status.json',dict(complete=False,qualification=gate,stop_reasons=r.control.STOP_REASON));return
    results=[];replacements=Counter();iterator=iter(plan['jobs']);begin=time.perf_counter();last=0
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={}
        def submit(job):
            if r.control.STOP.is_set():return
            r.append(out/'scheduled_jobs.jsonl',job)
            pending[pool.submit(r.episode,job,out,api,None,v.RULES)]=job
        def submit_next():
            job=next(iterator,None)
            if job is not None:submit(job)
        for _ in range(3):submit_next()
        while pending:
            done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try:result=future.result()
                except Exception as exc:r.control.stop('Unexpected runner failure: '+str(exc));raise
                results.append(result)
                if result['status']!='complete' and not r.control.STOP.is_set():
                    c=job['condition'];replacements[c]+=1
                    if replacements[c]>10:r.control.stop('Replacement safety limit: '+c)
                    else:submit(replacement(job))
                else:submit_next()
            if time.perf_counter()-last>=30 or not pending:
                last=time.perf_counter();counts=Counter(x['job']['condition'] for x in results if x['status']=='complete')
                progress=dict(stage='formal',finished=len(results),complete=sum(counts.values()),by_condition=dict(counts),failures=dict(Counter(x['status'] for x in results if x['status']!='complete')),pending=len(pending),seconds=time.perf_counter()-begin,conservative_committed_usd=api.COMMITTED,stop_reasons=r.control.STOP_REASON)
                save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    complete=[x for x in results if x['status']=='complete'];counts=Counter(x['job']['condition'] for x in complete)
    success=all(counts[c]==50 for c in r.p.CONDITIONS)
    save(out/'integrity_check.json',verify(out,plan))
    raw=r.read(out/'raw.jsonl');paid=[x['actual'] for x in raw if 'actual' in x];qualpaid=[x['actual'] for x in r.read(qual/'raw.jsonl') if 'actual' in x]
    summary=dict(provider='openai',complete=success,qualification=gate,completed_episodes=len(complete),attempted_episodes=len(results),by_condition=dict(counts),statuses=dict(Counter(x['status'] for x in results)),requests=len(raw),valid_decisions=len(r.read(out/'decisions.jsonl')),wall_seconds=time.perf_counter()-begin,total_wall_seconds=time.perf_counter()-start,estimated_list_price_usd=sum(x['estimated_list_price_usd'] for x in paid),estimated_cache_aware_cost_usd=sum(x['estimated_cache_aware_cost_usd'] for x in paid),qualification_cost_usd=sum(x['estimated_list_price_usd'] for x in qualpaid),conservative_committed_usd=api.COMMITTED,stop_reasons=r.control.STOP_REASON,formal=True)
    save(out/'execution_summary.json',summary);save(out/'final_status.json',summary)
    if success:
        view=out/'complete50_view';view.mkdir();ids={x['job']['episode_id'] for x in complete}
        save(view/'plan.json',dict(plan,jobs=[x['job'] for x in complete],selection_note='One successful independent game per frozen slot; technical failures retained in parent, no outcome selection.'))
        for name in ['raw.jsonl','decisions.jsonl','episodes.jsonl']:
            for row in r.read(out/name):
                if row['job']['episode_id'] in ids:r.append(view/name,row)
        save(view/'execution_summary.json',summary);save(view/'integrity_check.json',verify(view,plan))
    print('DONE '+json.dumps(dict(root=str(out),**summary)),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.run:run(args.run)
    else:prepare()
