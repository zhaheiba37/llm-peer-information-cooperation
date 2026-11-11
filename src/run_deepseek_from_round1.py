"""Frozen DeepSeek true-round1 replication; only pilot, never formal."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import deepseek_api as d
import run_from_round1 as r
import run_deepseek_replication as checks


def tasks():
    rows=[]
    for condition in ('D_HIGH_N_HIGH','D_HIGH_N_LOW','D_HIGH_N_ABSENT'):
        norm=r.p.CONDITIONS[condition][1]
        context=r.messages(condition,[])[1]['content'].rsplit('\n\n',1)[0]
        questions=[('current','What round is it now? Return only {"round": n}.',{'round':1}),
            ('past','Does the information above contain any completed-round contribution records? Return only {"past_records": true} or {"past_records": false}.',{'past_records':False}),
            ('opinion','Did B state how much each participant should contribute in the current round? Return only {"statement_present": true} or {"statement_present": false}.',{'statement_present':norm!='none'}),
            ('payoff','This is an independent payoff calculation question, not a contribution decision. Suppose the round-1 contributions are A=3, B=0, C=0, D=0 points. Under the rules, what is A\'s round payoff? Return only {"payoff": x}.',{'payoff':8.5})]
        for name,question,expected in questions:
            rows.append(dict(task_id=condition+'/'+name,category=name,condition=condition,
                messages=[dict(role='system',content=r.p.SYSTEM),dict(role='user',content=context+'\n\n'+question)],expected_not_for_model=expected))
    for condition,(behavior,norm) in r.p.CONDITIONS.items():
        peer=10 if behavior=='high' else 0
        history=[dict(A=a,B=peer,C=peer,D=peer) for a in (10,0,7)]
        context=r.messages(condition,history)[1]['content'].rsplit('\n\n',1)[0]
        question=('Return only a JSON object with these fields: "current_round" (the current round number), '
            '"last_A_contribution" (A\'s contribution in the last completed round), '
            '"last_B_contribution" (B\'s contribution in that round), "last_A_payoff" (A\'s payoff in that round), '
            '"current_statement_present" (a Boolean indicating whether B stated how much participants should contribute in the current round).')
        rows.append(dict(task_id=condition+'/round4',category='history',condition=condition,
            messages=[dict(role='system',content=r.p.SYSTEM),dict(role='user',content=context+'\n\n'+question)],
            expected_not_for_model=dict(current_round=4,last_A_contribution=7,last_B_contribution=peer,
                last_A_payoff=r.p.round_points(history[-1]),current_statement_present=norm!='none')))
    assert len(rows)==18
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
    d.CONFIG_PATH=d.ROOT/'deepseek_v10_round1_config.json'
    if not args.run:
        account=json.loads((d.ROOT/'deepseek_round1_account_check_20261105.json').read_text(encoding='utf-8'))
        assert account['balance']['response']['is_available']
        balance=next(float(x['total_balance']) for x in account['balance']['response']['balance_infos'] if x['currency']=='CNY')
        cfg=json.loads((d.ROOT/'deepseek_pilot_low_config.json').read_text(encoding='utf-8'))
        cfg.update(local_spend_cap_cny=min(3,balance*.9),budget_ledger_file='deepseek_v10_round1_budget.jsonl',
            note='18 new-context checks, then only 60 true-round1 10-round pilot episodes if all18 correct. No formal auto-start, no model selection by contribution outcomes.')
        d.CONFIG_PATH.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        out=d.ROOT/'runs'/('v10_deepseek_round1_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'));out.mkdir(exist_ok=False)
        plan=dict(version='v10_true_round1_current_opinion_only',config=cfg,workers=5,jobs=r.jobs(),checks=tasks(),
            purpose='cross_model_pilot_not_formal',max_decision_calls=600,formal_auto_start=False,
            gate='All18 new-context checks correct with normal finish; previous same-day same-settings61/61 qualification retained, not rerun. Stop on failure; no answer-based retries or effort escalation.',
            earlier_qualification='runs/deepseek_replication_20261105_091540_UTC/qualification_low/summary.json',
            history_policy='Identical prompt function and schedule to Qwen v10: real round1; all prior actual actions and own payoffs; only current-round opinion; no reasoning feedback.',
            analysis_policy='Round1 pooled by3 opinion groups; round2, episode means3-9 and round10 by6 groups; n10 games each; no outcome-based stop or formal auto-start.',
            comparison_limit='Same tasks, not equal reasoning compute. DeepSeek effort low omits temperature; Qwen uses thinking_budget1024 and temperature0.7.',
            account_check=account,retries=2,no_answer_repair=True,
            source_sha256={name:hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest() for name in
                ('run_deepseek_from_round1.py','run_from_round1.py','run_deepseek_replication.py','deepseek_api.py','qwen_api.py','run_current_multiround.py','run_pilot_v8.py','prompts_v7.py','check_revision_r1.py')})
        (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'round1_prompts.json').write_text(json.dumps({c:r.messages(c,[]) for c in r.p.CONDITIONS},ensure_ascii=False,indent=2),encoding='utf-8')
        for name in plan['source_sha256']:(out/name).write_bytes((d.ROOT/name).read_bytes())
        print('PREPARED '+str(out),flush=True);return
    out=args.run;plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if (out/'qualification').exists() or (out/'raw.jsonl').exists():raise RuntimeError('Existing attempts: refusing duplicate/selective rerun.')
    assert json.loads(d.CONFIG_PATH.read_text(encoding='utf-8'))==plan['config']
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest()==digest
    started=time.perf_counter();qualification=out/'qualification';qualification.mkdir();results=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(checks.do_check,t,qualification,'low_v10') for t in plan['checks']]
        for f in as_completed(futures):results.append(f.result())
    gate=dict(planned=18,completed=len(results),correct=sum(x['correct'] for x in results),
        passed=len(results)==18 and all(x['correct'] for x in results),wall_seconds=time.perf_counter()-started)
    (qualification/'summary.json').write_text(json.dumps(gate,indent=2),encoding='utf-8');print('CHECKS '+json.dumps(gate),flush=True)
    if gate['passed'] and not r.control.STOP.is_set():
        begin=time.perf_counter();results=[]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(r.episode,job,out,d) for job in plan['jobs']]
            for f in as_completed(futures):
                results.append(f.result())
                if len(results)%5==0 or results[-1]['status']!='complete':
                    print(json.dumps(dict(finished=len(results),planned=60,statuses=dict(Counter(x['status'] for x in results)),seconds=round(time.perf_counter()-begin,2))),flush=True)
        summary=dict(wall_seconds=time.perf_counter()-begin,planned_episodes=60,episode_statuses=dict(Counter(x['status'] for x in results)),stop_reasons=r.control.STOP_REASON,formal_started=False)
        (out/'execution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        integrity=r.audit(out,plan);(out/'integrity_check.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    status=dict(qualification=gate,pilot_started=gate['passed'],formal_started=False,total_wall_seconds=time.perf_counter()-started,stop_reasons=r.control.STOP_REASON)
    try:
        balance=d.get_balance();(out/'balance_after.json').write_text(json.dumps(balance,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:status['balance_after_error']=str(exc)
    (out/'final_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('DONE '+json.dumps(status),flush=True)


if __name__=='__main__':main()
