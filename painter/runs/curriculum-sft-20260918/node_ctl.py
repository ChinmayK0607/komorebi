"""Small owned-node status/export client. No model computation on the controller."""
import argparse,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('action',choices=['status','export']);a=p.parse_args()
pod=json.loads((P/'pod.json').read_text());r=subprocess.run(['lium','ps',pod['id'],'--format','json'],capture_output=True,text=True,check=True)
current,_=json.JSONDecoder().raw_decode(r.stdout.lstrip());assert len(current)==1 and current[0]['id']==pod['id'];pod=current[0]
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(P/'known_hosts'),'-p',str(pod['ports']['22']),'root@'+pod['ip']]
code='''from pathlib import Path
import json,subprocess
p=Path('/root/painter');result={}
for n in ['phase.json','data/render-progress.json','preparation-progress.json','preflight.json','evaluation/base/progress.json','evaluation/trained/progress.json','pipeline.exit','training.exit','training-setup.exit']:
 f=p/n
 if f.exists():
  try:result[n]=json.loads(f.read_text())
  except ValueError:result[n]=f.read_text()[-1000:]
for n in ['logs/server-base.log','logs/preparation.log','source/brush-rl/runs/curriculum-sft-20260918-setup/train.log','pipeline.log']:
 f=p/n
 if f.exists():result[n]=f.read_text()[-1800:]
for name in ['optimizer-update-audit.jsonl','group-validation.jsonl']:
 f=p/'train-output/curriculum-sft-20260918/artifacts'/name
 if f.exists():result[name]=[json.loads(s) for s in f.read_text().splitlines()[-2:]]
root=p/'train-output/curriculum-sft-20260918/artifacts/adapters'
result['checkpoints']=[json.loads(f.read_text()) for f in sorted(root.glob('step_*/hf-upload.json'))]
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader'],text=True).strip()
print(json.dumps(result))
'''
if a.action=='export':code="import subprocess;subprocess.run(['python3','/root/painter/export_runtime.py'],check=True)"
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True)
if a.action=='status':
 state=json.loads(r.stdout);state['provider']={k:pod.get(k) for k in ['id','status','spent_usd','price_per_hour','uptime','removal_scheduled_at']};(P/'latest-status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
else:print(r.stdout)
