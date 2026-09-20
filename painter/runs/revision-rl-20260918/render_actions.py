"""Render sampled actions on Linux, preserving source and canvas hashes."""
import sys,json,re,subprocess,hashlib,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
P=Path(__file__).resolve().parent

def one(f):
 from render_demos import atomic
 from rollout import sketch
 folder=f.parent
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 s=json.loads(f.read_text());row=dict(id=s['id'],state_id=s['state']['id'],valid=False,action=None,canvas=None,no_change=False)
 try:
  if s['finish_reason']!='stop':
   row['failure_kind']='cutoff';raise ValueError('Operational token cutoff')
  raw=s['text'];stop=bool(re.search(r'^FINISHED\s*$',re.sub(r'```.*?```','',raw,flags=re.S),re.M))
  initial=P/s['state']['canvas'];initial_program=P/s['state']['program']
  if stop and '```' not in raw:row.update(valid=True,action='finish',canvas=str(initial.relative_to(P)),no_change=True)
  else:
   row['failure_kind']='program'
   code=sketch(raw)
   if stop:raise ValueError('Paint and finish must be separate actions')
   src=folder/'program.js';src.write_text(code);os.chmod(src,0o644);os.chmod(folder,0o777)
   if code.strip()==initial_program.read_text().strip():row.update(valid=True,action='paint',canvas=str(initial.relative_to(P)),no_change=True)
   else:
    png=folder/'painting.png'
    row['failure_kind']='infrastructure'
    subprocess.run(['/usr/sbin/runuser','-u','painter','--',str(P/'renderer-env/bin/python'),str(P/'vendor/integrations/watercolour/renderer.py'),str(src),'--output',str(png),'--timeout','120','--backend','swiftshader'],capture_output=True,timeout=140)
    receipt=json.loads(png.with_suffix('.json').read_text())
    if not receipt['valid']:
     row['failure_kind']='program';raise ValueError(str(receipt.get('errors')))
    assert receipt['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
    assert receipt['png_sha256']==hashlib.sha256(png.read_bytes()).hexdigest()
    row.update(valid=True,action='paint',canvas=str(png.relative_to(P)),no_change=png.read_bytes()==initial.read_bytes())
 except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'[:1500]
 if row['valid']:row.pop('failure_kind',None)
 atomic(folder/'result.json',row);return row

def main():
 assert sys.platform=='linux','Rendering only runs on the node'
 from render_demos import atomic
 files=sorted((P/'rl-rollouts').glob('*/sample.json'));rows=[]
 with ThreadPoolExecutor(max_workers=12) as pool:
  for f in as_completed([pool.submit(one,f) for f in files]):
   rows.append(f.result());atomic(P/'rl-render-progress.json',dict(completed=len(rows),expected=len(files),valid=sum(r['valid'] for r in rows)))
 atomic(P/'rl-rollouts/summary.json',dict(results=rows,total=len(rows),valid=sum(r['valid'] for r in rows)))
if __name__=='__main__':main()
