"""Matched fresh-reference drawing, repair and stopping evaluation on the node."""
import argparse,json,sys,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from rollout import run
from render_demos import atomic
P=Path(__file__).resolve().parent

def main():
 assert sys.platform=='linux'
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tasks',default='evaluation-tasks.json');p.add_argument('--output',default='evaluation');p.add_argument('--workers',type=int,default=16);p.add_argument('--samples',type=int,default=2);a=p.parse_args()
 tasks=json.loads((P/a.tasks).read_text());config=json.loads((P/'campaign.json').read_text());out=P/a.output/a.model;out.mkdir(parents=True,exist_ok=True);rows=[]
 # A manual early launch and the pipeline may both request this model. Only
 # one producer can generate its cases; the follower reuses saved summaries.
 lock=(out/'producer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 def one(task,k):
  folder=out/task['id']/f'sample-{k:02d}';summary=folder/'summary.json'
  if summary.exists():return dict(task=task,sample=k,**json.loads(summary.read_text()))
  result=run(P/task['reference'],folder,'http://127.0.0.1:8000/v1',a.model,P/'vendor/integrations/watercolour/renderer.py',P/'renderer-env/bin/python',config,seed=918200+k,
      initial_canvas=P/task['canvas'] if task.get('canvas') else None,initial_program=P/task['program'] if task.get('program') else None)
  return dict(task=task,sample=k,**result)
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  jobs={pool.submit(one,t,k):(t,k) for k in range(a.samples) for t in tasks}
  for f in as_completed(jobs):
   task,k=jobs[f]
   try:row=f.result()
   except Exception as exc:row=dict(task=task,sample=k,valid_canvas=False,finished=False,error=f'{type(exc).__name__}: {exc}')
   rows.append(row);atomic(out/'progress.json',dict(model=a.model,completed=len(rows),expected=len(jobs),valid=sum(x.get('valid_canvas',False) for x in rows),finished=sum(x.get('finished',False) for x in rows)))
 atomic(out/'summary.json',dict(model=a.model,completed=len(rows),expected=len(jobs),results=rows))

if __name__=='__main__':main()
