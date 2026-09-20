"""Resumable student episodes, scheduling reference families as renders arrive."""
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from rollout import run
from render_demos import atomic
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',default='base');p.add_argument('--samples',type=int,default=8);p.add_argument('--concurrency',type=int,default=8);p.add_argument('--wait-references',action='store_true');a=p.parse_args()
rows=json.loads((P/'evaluation-references.json').read_text());config=json.loads((P/'campaign.json').read_text());out=P/'evaluation'/a.model;out.mkdir(parents=True,exist_ok=True)
jobs=[(r,k) for k in range(a.samples) for r in rows];results=[];pending={};deadline=time.monotonic()+7200

def one(row,k):
 folder=out/row['id']/f'sample-{k:02d}';done=folder/'summary.json'
 if done.exists():return dict(reference_id=row['id'],family=row['family'],sample=k,**json.loads(done.read_text()),resumed=True)
 result=run(P/'data'/row['reference'],folder,'http://127.0.0.1:8000/v1',a.model,P/'vendor/integrations/watercolour/renderer.py',str(P/'renderer-env/bin/python'),config,seed=918000+k)
 return dict(reference_id=row['id'],family=row['family'],sample=k,**result)
with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
 while jobs or pending:
  if time.monotonic()>deadline:raise TimeoutError('Evaluation phase exceeded two hours')
  for row,k in list(jobs):
   if len(pending)>=a.concurrency:break
   ref=P/'data'/row['reference'];receipt=ref.with_suffix('.json')
   if not ref.exists() or not receipt.exists():continue
   if not json.loads(receipt.read_text()).get('valid'):continue
   pending[pool.submit(one,row,k)]=(row,k);jobs.remove((row,k))
  if not pending:
   if not a.wait_references and jobs:raise RuntimeError('Some selected references have no valid render')
   time.sleep(3);continue
  complete,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
  for f in complete:
   row,k=pending.pop(f)
   try:r=f.result()
   except Exception as exc:r=dict(reference_id=row['id'],family=row['family'],sample=k,finished=False,valid_canvas=False,error=f'{type(exc).__name__}: {exc}')
   results.append(r)
   atomic(out/'progress.json',dict(model=a.model,completed=len(results),expected=len(rows)*a.samples,pending=len(pending),waiting=len(jobs),valid=sum(x.get('valid_canvas',False) for x in results),finished=sum(x.get('finished',False) for x in results)))
atomic(out/'summary.json',dict(model=a.model,expected=len(rows)*a.samples,completed=len(results),quality_judged=False,results=results))
print(json.dumps(dict(model=a.model,completed=len(results),valid=sum(x.get('valid_canvas',False) for x in results),finished=sum(x.get('finished',False) for x in results))))
