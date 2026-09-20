"""Blind AB/BA outcome grading and per-transition quality rewards; node only."""
import json,sys,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from compare import compare
from render_demos import atomic
from reward import transition_reward
P=Path(__file__).resolve().parent

def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert sys.platform=='linux';tasks=json.loads((P/'evaluation-tasks.json').read_text());jobs=[];rewards=[];outcomes=[]
 for task in tasks:
  for k in range(2):
   summaries={}
   for model in ['parent','trained']:
    f=P/'evaluation'/model/task['id']/f'sample-{k:02d}/summary.json'
    if f.exists():summaries[model]=json.loads(f.read_text())
   identity=task['id']+f'-s{k}'
   if all(summaries.get(m,{}).get('canvas') for m in ['parent','trained']):
    jobs.append(dict(id=identity,kind='outcome',reference=str(P/task['reference']),candidate=summaries['trained']['canvas'],anchor=summaries['parent']['canvas'],task='Match the reference faithfully. Task: '+task['kind']))
   else:outcomes.append(dict(id=identity,mask=True,reward=None,reason='missing_painting',execution_valid={m:summaries.get(m,{}).get('valid_canvas',False) for m in ['parent','trained']}))
   if k!=0 or 'trained' not in summaries:continue
   previous=str(P/task['canvas']) if task.get('canvas') else str(P/'data/blank.png')
   for t in summaries['trained']['turns']:
    key=identity+f'-t{t["turn"]}';action=t.get('action','paint');canvas=t.get('canvas')
    if not t.get('valid'):
     rewards.append(dict(id=key,task_id=task['id'],turn=t['turn'],**transition_reward(valid=False)));continue
    if action=='paint' and t.get('no_change') and not t.get('finished'):
     rewards.append(dict(id=key,task_id=task['id'],turn=t['turn'],**transition_reward(valid=True,no_change=True)))
    else:jobs.append(dict(id=key,kind='transition',reference=str(P/task['reference']),candidate=canvas,anchor=previous,action=action,no_change=t.get('no_change',False),terminal=t.get('finished',False),turn=t['turn'],task_id=task['id'],task='Judge only whether the painting faithfully matches the reference; no speed or token budget.'))
    previous=canvas
 out=P/'judging';out.mkdir(exist_ok=True);atomic(out/'pairs.json',jobs)
 def one(row):
  if file_hash(row['candidate'])==file_hash(row['anchor'])==file_hash(row['reference']):
   return dict(id=row['id'],reward=.5,mask=False,candidate_acceptable=True,anchor_acceptable=True,acceptability_consistent=True,evidence='Both paintings are byte-identical to the reference')
  return compare(row,'http://127.0.0.1:8000/v1','judge',out/'raw')
 with ThreadPoolExecutor(max_workers=4) as pool:
  future={pool.submit(one,row):row for row in jobs}
  for f in as_completed(future):
   row=future[f]
   try:result=f.result()
   except Exception as exc:result=dict(reward=None,mask=True,error=f'{type(exc).__name__}: {exc}')
   if row['kind']=='outcome':outcomes.append(dict(id=row['id'],**{k:v for k,v in result.items() if k!='id'}))
   else:rewards.append(dict(id=row['id'],task_id=row['task_id'],turn=row['turn'],judgment=result,**transition_reward(valid=True,action=row['action'],judgment=result,no_change=row['no_change'],terminal=row['terminal'])))
   atomic(out/'progress.json',dict(completed_outcomes=len(outcomes),completed_transitions=len(rewards),expected_judge_pairs=len(jobs)))
 atomic(out/'outcomes.json',outcomes);atomic(out/'transition-rewards.json',rewards)
 eligible=[r for r in outcomes if not r.get('mask')]
 atomic(out/'summary.json',dict(outcomes=len(outcomes),eligible=len(eligible),masked=len(outcomes)-len(eligible),wins=sum(r['reward']==1 for r in eligible),ties=sum(r['reward']==.5 for r in eligible),losses=sum(r['reward']==0 for r in eligible),transition_rewards=len(rewards),unmasked_transitions=sum(not r['mask'] for r in rewards),rl_optimizer_updates=0,note='Model-judged development preferences; no human-qualified win-rate claim. Reward ledger is prepared for RL, not evidence of policy-gradient training.'))

if __name__=='__main__':main()
