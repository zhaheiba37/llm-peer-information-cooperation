"""Portable analysis of NEW runs; original experimental data are not included."""
import argparse,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path,help='Provider directory with episodes.jsonl');p.add_argument('--output',type=Path,default=Path('analysis_output'));p.add_argument('--seed',type=int,default=20261111);p.add_argument('--no-plot',action='store_true',help='Compute statistics without matplotlib');a=p.parse_args()
 rows=[json.loads(l) for l in (a.run/'episodes.jsonl').read_text('utf-8').splitlines() if l.strip()]
 rows=[r for r in rows if r['status']=='complete'];groups={};slots=set()
 for r in rows:
  job=r['job'];slot=job.get('formal_slot',job['episode_id'])
  if slot in slots:raise ValueError('Duplicate complete formal slot')
  slots.add(slot);assert len(r['final_history'])==10
  groups.setdefault(job['condition'],[]).append(r)
 rng=np.random.default_rng(a.seed);summary={};boots={};means={}
 for c,rs in sorted(groups.items()):
  mat=np.array([[h['A'] for h in r['final_history']] for r in rs]);assert np.all((mat>=0)&(mat<=10))
  if 'minority_actor' in rs[0]['job']:
   total=np.zeros((5000,10))
   for actor in ['B','C','D']:
    sub=mat[[r['job']['minority_actor']==actor for r in rs]]
    if not len(sub):raise ValueError('Missing minority stratum')
    total+=sub[rng.integers(len(sub),size=(5000,len(sub)))].sum(1)
   bs=total/len(mat)
  else:bs=mat[rng.integers(len(mat),size=(5000,len(mat)))].mean(1)
  means[c]=mat.mean(0);boots[c]=bs
  summary[c]={'n':len(mat),'mean':means[c].tolist(),'ci95':np.quantile(bs,[.025,.975],axis=0).T.tolist(),'mean_rounds3_9':float(mat[:,2:9].mean()),'mean_rounds2_10':float(mat[:,1:10].mean())}
 effects={}
 for c in groups:
  absent=c.replace('N_HIGH','N_ABSENT')
  if c.endswith('N_HIGH') and absent in groups:
   diff=boots[c]-boots[absent];effects[c]={'mean':(means[c]-means[absent]).tolist(),'ci95':np.quantile(diff,[.025,.975],axis=0).T.tolist()}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps({'conditions':summary,'cooperative_opinion_effects':effects,'bootstrap_seed':a.seed,'resamples':5000},indent=2),'utf-8')
 if a.no_plot:return
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4))
 for c,x in effects.items():
  lohi=np.array(x['ci95']);ax.plot(range(2,11),x['mean'][1:],label=c);ax.fill_between(range(2,11),lohi[1:,0],lohi[1:,1],alpha=.15)
 ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Round',ylabel='Contribution difference');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'opinion_effects.png',dpi=200)
if __name__=='__main__':main()
