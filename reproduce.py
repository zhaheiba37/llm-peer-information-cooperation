"""Prepare historical study plans without API calls; execution is explicit."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCRIPTS={'main':'run_formal_v11.py','main-gpt':'run_gpt_luna_formal.py','paraphrase':'run_paraphrase_b.py','mixed':'run_mixed_partner.py'}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--study',choices=SCRIPTS,required=True)
 p.add_argument('--execute',action='store_true',help='Authorize paid API calls for an existing plan')
 p.add_argument('--run',type=Path,help='Previously prepared run directory')
 p.add_argument('--provider',choices=['qwen','deepseek','gpt'])
 a=p.parse_args();src=ROOT/'src';(src/'runs').mkdir(exist_ok=True)
 cmd=[sys.executable,str(src/SCRIPTS[a.study])]
 if a.execute:
  if not a.run:p.error('--execute requires --run')
  needed={'main':['DEEPSEEK_API_KEY','DASHSCOPE_API_KEY'],'main-gpt':['OPENAI_API_KEY']}.get(a.study)
  if needed is None:
   if not a.provider:p.error('extension execution requires --provider')
   needed=[{'qwen':'DASHSCOPE_API_KEY','deepseek':'DEEPSEEK_API_KEY','gpt':'OPENAI_API_KEY'}[a.provider]]
  if any(not os.environ.get(k) for k in needed):p.error('Missing environment variable(s): '+', '.join(needed))
  if not (a.run/'plan.json').is_file():p.error('No plan.json in --run')
  cmd+=['--run',str(a.run.resolve())]
  if a.study in ('paraphrase','mixed'):cmd+=['--provider',a.provider]
 else:
  if a.run or a.provider:p.error('--run/--provider are only used with --execute')
  print('PLAN ONLY: no API requests. Use the printed PREPARED directory for a later explicit execution.',flush=True)
 subprocess.run(cmd,cwd=src,check=True)
if __name__=='__main__':main()
