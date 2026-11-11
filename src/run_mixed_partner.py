"""Authorized mixed-partner generalization: three providers, four conditions50x10."""
import argparse,json,random,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import mixed_partner_core as r
import run_public_observation_pilot as old
import run_deepseek_replication as checks
import mixed_partner_core as b
import qwen_api as q
import deepseek_api as d
import gpt_luna_recovery_api as g
ROOT=q.ROOT
CONDITIONS=['D_HIGH_N_HIGH','D_HIGH_N_ABSENT','D_LOW_N_HIGH','D_LOW_N_ABSENT']
APIS={'deepseek':d,'qwen':q,'gpt':g}
WORKERS={'deepseek':8,'qwen':8,'gpt':3}
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def jobs(provider):
    rows=[dict(episode_id=f'mixed21/{provider}/{c}/{i:03}/attempt00',phase='generalization',condition=c,replicate=i,minority_actor=['B','C','D'][i%3],formal_slot=f'{provider}/{c}/{i:03}',episode_attempt=0,order='history_first',first_round=1,final_round=10) for c in CONDITIONS for i in range(50)]
    random.Random(202611082100).shuffle(rows);return rows

def prepare():
    root=ROOT/'runs'/('mixed21_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'));root.mkdir()
    tasks=old.tasks()
    for t in tasks:t['messages']=[dict(t['messages'][0]),dict(role='user',content=b.rewrite(t['messages'][1]['content']))]
    configs={}
    for provider in ['deepseek','qwen']:
        cfg=json.loads((ROOT/f'{provider}_formal_v11_config.json').read_text(encoding='utf-8'))
        cfg.update(local_spend_cap_cny=6,budget_ledger_file=f'{provider}_{root.name}_budget.jsonl',note='Authorized mixed21 four conditions50 games10 rounds. Separate from main; fixed parameters; no outcome-based sampling.')
        configs[provider]=cfg;save(ROOT/f'{provider}_mixed21_config.json',cfg)
    configs['gpt']=g.CONFIG
    plan=dict(version='mixed21_v11_20261108',providers=['deepseek','qwen','gpt'],conditions=CONDITIONS,configs=configs,workers=WORKERS,jobs={p:jobs(p) for p in APIS},checks=tasks,rules_text=b.RULES,formal=False,purpose='Mixed partner behavior: 10/10/0 versus 10/0/0, unanimous should10 versus absent. Original A wording; fixed minority identity within game, counterbalanced17/17/16 across games. No claim about pure majority mechanism.',max_decision_calls_per_model=2000,cap_cny_per_chinese_provider=6,cap_usd_gpt=.50,failure_policy='Same-input transient retry max2 at3/6seconds; technical incomplete games retained and independently replaced from round1, max10/condition. No valid-answer retries, no result-based stopping. Stop provider on nontransient transport/auth/budget or failed comprehension.',analysis='Round1 pool two opinions n100 each; round2, mean3-9, final10; should10 minus absent within each composition; stratified whole-game bootstrap5000 preserving minority17/17/16. No outcome selection; intervals pointwise. Compare extremes descriptively with batch and payoff/composition confounding disclosed.',source_sha256={})
    for name in ['run_mixed_partner.py','mixed_partner_core.py','run_from_round1.py','run_public_observation_pilot.py','run_deepseek_replication.py','run_deepseek_from_round1.py','qwen_api.py','deepseek_api.py','gpt_luna_recovery_api.py','prompts_v7.py','check_revision_r1.py','prompts_v11_public_observation.py','run_current_multiround.py','run_pilot_v8.py']:
        content=(ROOT/name).read_bytes();plan['source_sha256'][name]=hashlib.sha256(content).hexdigest();(root/name).write_bytes(content)
    save(root/'plan.json',plan)
    save(root/'example_round1_prompts.json',{c:b.messages(c,[]) for c in CONDITIONS})
    save(root/'example_round4_prompts.json',{c:b.messages(c,[dict(A=a,**r.script(c,'B')) for a in (0,10,3)]) for c in CONDITIONS})
    (ROOT/'mixed21_latest_path.txt').write_text(str(root),encoding='utf-8')
    print('PREPARED '+str(root),flush=True)

def verify(out,plan):
    result=r.audit(out,plan)
    for row in r.read(out/'raw.jsonl'):
        if 'actual' in row:
            request=row['actual']['request']
            if 'max_completion_tokens' in plan['config']:assert request['max_completion_tokens']==8192
    return result

class SafeClient:
    def __init__(self,api):self.api=api
    def call(self,messages,purpose):
        try:return self.api.call(messages,purpose)
        except Exception as e:
            if not r.control.transient(str(e)):r.control.stop(str(e))
            raise

def run(root,provider):
    allplan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    for n,digest in allplan['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==digest
    out=root/provider;out.mkdir(exist_ok=False)
    api=APIS[provider]
    if provider=='gpt':api.OUT=out;api.COMMITTED=0;api.CAP_USD=.50
    else:
        api.CONFIG_PATH=ROOT/f'{provider}_mixed21_config.json'
        assert json.loads(api.CONFIG_PATH.read_text(encoding='utf-8'))==allplan['configs'][provider]
    r.messages=b.messages
    plan=dict(allplan,provider=provider,config=allplan['configs'][provider],jobs=allplan['jobs'][provider],workers=WORKERS[provider],max_decision_calls=2000)
    save(out/'plan.json',plan)
    client=SafeClient(api);qual=out/'qualification';qual.mkdir();start=time.perf_counter();answers=[]
    for task in plan['checks']:
        answer=checks.do_check(task,qual,provider+'_mixed21',client);answers.append(answer)
        print(json.dumps(dict(provider=provider,stage='qualification',completed=len(answers),correct=sum(x['correct'] for x in answers))),flush=True)
        if not answer['correct']:r.control.stop('Qualification failed: '+task['task_id']);break
    gate=dict(total=20,completed=len(answers),correct=sum(x['correct'] for x in answers),passed=len(answers)==20 and all(x['correct'] for x in answers),wall_seconds=time.perf_counter()-start)
    save(qual/'summary.json',gate)
    if not gate['passed'] or r.control.STOP.is_set():save(out/'final_status.json',dict(complete=False,qualification=gate,stop_reasons=r.control.STOP_REASON));return
    results=[];replacements=Counter();iterator=iter(plan['jobs']);begin=time.perf_counter();last=0
    with ThreadPoolExecutor(max_workers=WORKERS[provider]) as pool:
        pending={}
        def submit(job):
            if r.control.STOP.is_set():return
            r.append(out/'scheduled_jobs.jsonl',job);pending[pool.submit(r.episode,job,out,client,None,b.RULES)]=job
        def next_job():
            job=next(iterator,None)
            if job is not None:submit(job)
        for _ in range(WORKERS[provider]):next_job()
        while pending:
            done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try:result=future.result()
                except Exception as e:r.control.stop('Unexpected runner error: '+str(e));raise
                results.append(result)
                if result['status']!='complete' and not r.control.STOP.is_set():
                    c=job['condition'];replacements[c]+=1
                    if replacements[c]>10:r.control.stop('Replacement limit '+c)
                    else:
                        attempt=job['episode_attempt']+1
                        submit(dict(job,episode_attempt=attempt,episode_id=f"mixed21/{provider}/{c}/{job['replicate']:03}/attempt{attempt:02}",replaces_episode_id=job['episode_id']))
                else:next_job()
            if time.perf_counter()-last>30 or not pending:
                last=time.perf_counter();counts=Counter(x['job']['condition'] for x in results if x['status']=='complete')
                progress=dict(provider=provider,complete=sum(counts.values()),finished=len(results),by_condition=dict(counts),failures=dict(Counter(x['status'] for x in results if x['status']!='complete')),seconds=time.perf_counter()-begin,stop_reasons=r.control.STOP_REASON)
                save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    complete=[x for x in results if x['status']=='complete'];counts=Counter(x['job']['condition'] for x in complete)
    success=all(counts[c]==50 for c in CONDITIONS)
    save(out/'integrity_check.json',verify(out,plan))
    paid=[x['actual'] for x in r.read(out/'raw.jsonl') if 'actual' in x];qualpaid=[x['actual'] for x in r.read(qual/'raw.jsonl') if 'actual' in x]
    currency='usd' if provider=='gpt' else 'cny';key='estimated_list_price_'+currency
    summary=dict(provider=provider,complete=success,qualification=gate,completed_episodes=len(complete),attempted_episodes=len(results),by_condition=dict(counts),statuses=dict(Counter(x['status'] for x in results)),requests=len(r.read(out/'raw.jsonl')),valid_decisions=len(r.read(out/'decisions.jsonl')),wall_seconds=time.perf_counter()-begin,total_wall_seconds=time.perf_counter()-start,currency=currency,estimated_list_cost=sum(x[key] for x in paid),qualification_cost=sum(x[key] for x in qualpaid),returned_models=dict(Counter(x['response'].get('model') for x in paid)),stop_reasons=r.control.STOP_REASON)
    aware='estimated_cache_aware_cost_'+currency
    if paid and all(aware in x for x in paid):summary['estimated_cache_aware_cost']=sum(x[aware] for x in paid)
    save(out/'execution_summary.json',summary);save(out/'final_status.json',summary)
    if success:
        view=out/'complete50_view';view.mkdir();ids={x['job']['episode_id'] for x in complete}
        save(view/'plan.json',dict(plan,jobs=[x['job'] for x in complete]))
        for name in ['raw.jsonl','decisions.jsonl','episodes.jsonl']:
            for row in r.read(out/name):
                if row['job']['episode_id'] in ids:r.append(view/name,row)
        save(view/'execution_summary.json',summary);save(view/'integrity_check.json',verify(view,plan))
    print('DONE '+json.dumps(dict(root=str(out),**summary)),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--provider',choices=APIS);args=parser.parse_args()
    if args.run:run(args.run,args.provider)
    else:prepare()
