"""Offline regression checks. Networking is blocked even if keys are present."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
(ROOT/'tmp').mkdir(exist_ok=True)
import run_from_round1 as r
import prompts_v11_public_observation as v
import prompts_paraphrase_b as b
import mixed_partner_core as m
import run_public_observation_pilot as pilot
import run_deepseek_replication as checks
import run_formal_v11 as formal
import run_mixed_partner as mixed
class Tests(unittest.TestCase):
 def test_schedules(self):
  rows=formal.initial_jobs('qwen');self.assertEqual(len(rows),300);self.assertEqual(len({j['formal_slot'] for j in rows}),300)
  rows=mixed.jobs('qwen')
  for c in mixed.CONDITIONS:
   from collections import Counter
   self.assertEqual(Counter(j['minority_actor'] for j in rows if j['condition']==c),{'B':17,'C':17,'D':16})
 def test_payoff(self):
  self.assertEqual(r.p.round_points(dict(A=3,B=0,C=0,D=0)),8.5)
  self.assertEqual(r.p.round_points(dict(A=10,B=10,C=10,D=10)),20)
 def test_information(self):
  for c in r.p.CONDITIONS:
   msg=r.messages(c,[],v.RULES)[1]['content'];self.assertNotIn('Round 1 record:',msg)
   self.assertEqual(msg.count('each stated:'),0 if c.endswith('ABSENT') else 1)
   peer=10 if r.p.CONDITIONS[c][0]=='high' else 0
   msg=r.messages(c,[dict(A=4,B=peer,C=peer,D=peer)]*3,v.RULES)[1]['content']
   self.assertIn('Round 3 record:',msg);self.assertNotIn('Round 4 record:',msg)
   self.assertEqual(msg.count('each stated:'),0 if c.endswith('ABSENT') else 1)
 def test_wording_and_checks(self):
  self.assertIn('common account in round 1',b.messages('D_HIGH_N_HIGH',[])[1]['content'])
  tasks=pilot.tasks();self.assertEqual(len(tasks),20)
  for task in tasks:self.assertIn('expected_not_for_model',task)
 def test_episode(self):
  class Mock:
   def call(self,messages,purpose):
    return {'request':{'messages':messages},'response':{'choices':[{'finish_reason':'stop','message':{'content':'{"contribution": 4}'}}]}}
  r.control.STOP.clear();r.control.STOP_REASON.clear()
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as t,patch.object(r,'pace',lambda:None):
   job=dict(episode_id='mock',condition='D_HIGH_N_HIGH',replicate=0,first_round=1,final_round=10)
   result=r.episode(job,Path(t),Mock(),None,v.RULES)
   self.assertEqual(result['status'],'complete');self.assertEqual(result['completed_decisions'],10)
   rows=r.read(Path(t)/'decisions.jsonl');self.assertEqual(len(rows),10)
   self.assertEqual(rows[-1]['payoff'],23)
 def test_invalid_output_not_repaired(self):
  class Mock:
   calls=0
   def call(self,*args):
    self.calls+=1
    return {'response':{'choices':[{'finish_reason':'stop','message':{'content':'{"contribution": 11}'}}]}}
  mock=Mock();r.control.STOP.clear()
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as t,patch.object(r,'pace',lambda:None):
   result=r.episode(dict(episode_id='invalid',condition='D_HIGH_N_HIGH'),Path(t),mock,None,v.RULES)
   self.assertNotEqual(result['status'],'complete');self.assertEqual(mock.calls,1)
if __name__=='__main__':
 with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('Offline check attempted networking')):
  unittest.main()

