"""Frozen sequential-provider formal collection. No outcome-dependent replacement."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib,json,random,time
from pathlib import Path
import deepseek_api as d
import qwen_api as q
import run_from_round1 as r
import run_public_observation_pilot as pilot
import run_deepseek_replication as checks
import prompts_v11_public_observation as v

def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def initial_jobs(provider):
    jobs=[dict(episode_id=f'formal/{provider}/{c}/{i:03}/attempt00',phase='formal',condition=c,replicate=i,
        formal_slot=f'{provider}/{c}/{i:03}',episode_attempt=0,order='history_first',first_round=1,final_round=10)
        for c in r.p.CONDITIONS for i in range(50)]
    random.Random(202611061224).shuffle(jobs);return jobs

def stage(root,provider,plan):
    api=d if provider=='deepseek' else q;api.CONFIG_PATH=api.ROOT/f'{provider}_formal_v11_config.json'
    assert json.loads(api.CONFIG_PATH.read_text(encoding='utf-8'))==plan['configs'][provider]
    r.control.STOP.clear();r.control.STOP_REASON.clear()
    out=root/provider;out.mkdir(exist_ok=False)
    stage_plan=dict(version=plan['version'],config=plan['configs'][provider],rules_text=v.RULES,jobs=plan['jobs'][provider],max_decision_calls=3000,workers=20,formal=True)
    save(out/'plan.json',stage_plan)
    qual=out/'qualification';qual.mkdir();started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:qualified=list(pool.map(lambda t:checks.do_check(t,qual,f'{provider}_formal_v11',api),plan['checks']))
    gate=dict(correct=sum(x['correct'] for x in qualified),total=20,passed=all(x['correct'] for x in qualified))
    save(qual/'summary.json',gate);print('QUALIFICATION '+json.dumps(dict(provider=provider,**gate)),flush=True)
    if not gate['passed'] or r.control.STOP.is_set():return dict(provider=provider,complete=False,qualification=gate,reason='qualification_failed_or_operational_stop')
    results=[];replacements=Counter();last=0;begin=time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as pool:
        pending={}
        def submit(job):
            r.append(out/'scheduled_jobs.jsonl',job)
            pending[pool.submit(r.episode,job,out,api,None,v.RULES)]=job
        for job in plan['jobs'][provider]:submit(job)
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try:result=future.result()
                except Exception as exc:
                    r.control.stop('Unexpected runner failure: '+str(exc));raise
                results.append(result)
                if result['status']!='complete' and not r.control.STOP.is_set():
                    condition=job['condition'];replacements[condition]+=1
                    if replacements[condition]>10:r.control.stop('Replacement safety limit reached for '+condition)
                    else:
                        attempt=job['episode_attempt']+1;new=dict(job,episode_attempt=attempt,
                            episode_id=f"formal/{provider}/{condition}/{job['replicate']:03}/attempt{attempt:02}",replaces_episode_id=job['episode_id'])
                        submit(new)
                if len(results)%10==0 or result['status']!='complete':
                    counts=Counter(x['job']['condition'] for x in results if x['status']=='complete')
                    print(json.dumps(dict(provider=provider,finished=len(results),complete=sum(counts.values()),by_condition=dict(counts),failures=dict(Counter(x['status'] for x in results if x['status']!='complete')),seconds=round(time.perf_counter()-begin,1))),flush=True)
            if time.perf_counter()-last>30:
                last=time.perf_counter();save(out/'progress.json',dict(completed_results=len(results),complete=sum(x['status']=='complete' for x in results),pending=len(pending),stop_reasons=r.control.STOP_REASON))
    complete=[x for x in results if x['status']=='complete'];counts=Counter(x['job']['condition'] for x in complete)
    success=all(counts[c]==50 for c in r.p.CONDITIONS)
    integrity=r.audit(out,stage_plan);save(out/'integrity_check.json',integrity)
    raw=r.read(out/'raw.jsonl');paid=[x['actual'] for x in raw if 'actual' in x]
    summary=dict(provider=provider,complete=success,planned_episodes=300,completed_episodes=len(complete),attempted_episodes=len(results),
        by_condition=dict(counts),statuses=dict(Counter(x['status'] for x in results)),requests=len(raw),valid_decisions=len(r.read(out/'decisions.jsonl')),
        wall_seconds=time.perf_counter()-begin,total_wall_seconds=time.perf_counter()-started,
        cost_cny=sum(x['estimated_list_price_cny'] for x in paid),stop_reasons=r.control.STOP_REASON,
        analysis_policy='Complete50/game condition primary view; all failed and partial attempts retained for completion-rate and all-attempt sensitivity analysis.')
    if provider=='deepseek':
        summary['cache_aware_estimate_cny']=sum(x['estimated_cache_aware_cost_cny'] for x in paid)
        try:save(out/'balance_after.json',d.get_balance())
        except Exception as exc:summary['balance_error']=str(exc)
    save(out/'execution_summary.json',summary)
    if success:
        view=out/'complete50_view';view.mkdir();ids={x['job']['episode_id'] for x in complete}
        save(view/'plan.json',dict(stage_plan,jobs=[x['job'] for x in complete],selection_note=summary['analysis_policy']))
        for name in ('raw.jsonl','decisions.jsonl','episodes.jsonl'):
            for row in r.read(out/name):
                if row['job']['episode_id'] in ids:r.append(view/name,row)
        save(view/'execution_summary.json',summary);save(view/'integrity_check.json',r.audit(view,stage_plan))
    print('STAGE_DONE '+json.dumps(summary),flush=True);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    if not args.run:
        configs={}
        for provider,api in [('deepseek',d),('qwen',q)]:
            cfg=json.loads((api.ROOT/f'{provider}_v11_public_config.json').read_text(encoding='utf-8'))
            cfg.update(local_spend_cap_cny=8.0,budget_ledger_file=f'{provider}_formal_v11_budget.jsonl',note='Authorized formal v11: six conditions50 complete independent10-round games. Fixed settings. Only technical failures replaced from round1; all attempts retained. Provider order DeepSeek then Qwen.')
            configs[provider]=cfg;save(api.ROOT/f'{provider}_formal_v11_config.json',cfg)
        out=q.ROOT/'runs'/('formal_v11_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'));out.mkdir()
        plan=dict(version='formal_v11_public_observation',created_utc=datetime.now(timezone.utc).isoformat(),configs=configs,
            providers=['deepseek','qwen'],workers=20,max_starts_per_second=4,jobs={p:initial_jobs(p) for p in configs},checks=pilot.tasks(),rules_text=v.RULES,
            target='50 complete independent games per model per condition, ten rounds each, 6000 valid decisions plus technical-failure attempts.',
            failure_policy='Transient network/429/5xx retried at identical inputs max2 retries. Invalid/truncated/exhausted failures terminate game; replace entire game under same condition from true round1 with unique ID, max10 replacements/condition. Preserve all partial records. Never replace by contribution value. Stop on budget/authentication or unexpected configuration errors; do not change output limits.',
            primary_analysis='Round1 three opinion groups pooled across unobserved futureD, round2 six conditions; episode means3-9 and round10; compare N+ vs absent and N+ vs N- by behavior, differences and whole-episode intervals. Extra post-hoc analyses labeled exploratory. Completion50 view plus all-attempt missingness sensitivity.',
            model_comparison='Same tasks, distinct native thinking controls; no equal-compute claim.',
            source_sha256={name:hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest() for name in ('run_formal_v11.py','run_from_round1.py','run_deepseek_replication.py','run_public_observation_pilot.py','prompts_v11_public_observation.py','deepseek_api.py','qwen_api.py','run_pilot_v8.py','run_current_multiround.py','check_revision_r1.py','prompts_v7.py')})
        save(out/'plan.json',plan)
        for name in plan['source_sha256']:(out/name).write_bytes((q.ROOT/name).read_bytes())
        print('PREPARED '+str(out),flush=True);return
    root=args.run;plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    assert not any((root/p).exists() for p in plan['providers']),'Existing collection; refusing duplicate rerun.'
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((q.ROOT/name).read_bytes()).hexdigest()==digest
    statuses=[]
    for provider in plan['providers']:
        result=stage(root,provider,plan);statuses.append(result);save(root/'status.json',statuses)
        if not result['complete']:break
    print('ALL_DONE '+json.dumps(dict(root=str(root),providers=statuses)),flush=True)

if __name__=='__main__':main()
