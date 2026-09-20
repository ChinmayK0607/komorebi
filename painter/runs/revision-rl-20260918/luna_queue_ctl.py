"""Move blinded image jobs and Luna receipts between controller and GPU node."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

import reward_queue

ROOT = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['pull','push','status'])
    p.add_argument('batch',nargs='?',default='')
    p.add_argument('--host',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--known-hosts',type=Path,default=ROOT/'known_hosts')
    p.add_argument('--run-id',help='Remote output run id for status inspection.')
    p.add_argument('--root',type=Path,default=ROOT/'luna-live')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--identity-mode',action='store_true',
                   help='Stage and verify unique-asset rolemap labels.')
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    if args.action!='status' and not re.fullmatch(r'[a-zA-Z0-9_-]+',args.batch):
        raise ValueError('Simple batch name required')
    common=['-i','/Users/chinmay/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=20',
            '-o','UserKnownHostsFile='+str(args.known_hosts)]
    ssh=['ssh',*common,'-p',str(args.port),'root@'+args.host]
    scp=['scp',*common,'-P',str(args.port)]
    def remote(code):
        return subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True).stdout
    if args.action=='pull':
        result=subprocess.run(ssh+[f'cd /root/painter && python3 reward_queue.py export {args.batch} --limit {args.limit}'],
                              capture_output=True,text=True,check=True).stdout.strip()
        if result=='No unassigned rewards':
            print(result);return
        metadata=json.loads(result)
        archive=args.root/(args.batch+'.tar.gz')
        subprocess.run(scp+['root@'+args.host+':'+metadata['archive'],str(archive)],check=True)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=metadata['sha256']:
            raise ValueError('Transferred image archive hash mismatch')
        dest=args.root/'reward-batches';dest.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path=Path(member.name)
                if path.is_absolute() or '..' in path.parts or path.parts[0]!=args.batch or not (member.isfile() or member.isdir()):
                    raise ValueError('Unexpected archive member')
            tar.extractall(dest,filter='data')
        reward_queue.stage(args.root,args.batch,args.identity_mode)
        print(json.dumps(dict(batch=args.batch,root=str(args.root),**metadata)))
    elif args.action=='push':
        reward_queue.receipts(args.root,args.batch,args.identity_mode)
        folder=args.root/'reward-batches'/args.batch/'receipts'
        receipts=[json.loads(f.read_text()) for f in sorted(folder.glob('*/receipt.json'))]
        payload=json.dumps(receipts)
        code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/root/painter')
from quality_signal import reconcile
from taskset_impl import atomic
root=Path('/root/painter');receipts=json.loads(PAYLOAD);identity_mode=IDENTITY_MODE;installed=0
for receipt in receipts:
 ident=receipt['id']
 assert len(ident)==64 and all(c in '0123456789abcdef' for c in ident)
 folder=root/'judge-queue'/ident
 request=json.loads((folder/'request.json').read_text())
 assert receipt['judge_model']=='gpt-5.6-luna' and receipt.get('reasoning_effort')=='low'
 assert receipt['image_sha256']==request['image_sha256']
 for role,path in request['images'].items():
  assert hashlib.sha256((root/path).read_bytes()).hexdigest()==request['image_sha256'][role]
 if identity_mode:
  assert receipt.get('visual_provenance',{}).get('mode')=='content_addressed_unique_assets'
  expected_equal=request['image_sha256']['candidate']==request['image_sha256']['anchor']
  deterministic=receipt.get('deterministic_identity',{})
  assert deterministic.get('candidate_bytes_equal')==expected_equal
  if expected_equal: assert deterministic.get('relative_winner')=='tie'
 assert receipt['judgment']==reconcile(receipt['candidate_A'],receipt['candidate_B'])
 atomic(folder/'receipt.json',receipt);installed+=1
print(json.dumps(dict(installed=installed)))
'''.replace('PAYLOAD',repr(payload)).replace('IDENTITY_MODE',repr(args.identity_mode))
        # Use Prime's environment only for its taskset dependency; no ML runs.
        code=code.replace('from taskset_impl import atomic','from supervise_async import atomic')
        print(remote(code).strip())
    else:
        code='''import json,subprocess
from pathlib import Path
p=Path('/root/painter');out={}
for name in ['training-ready.json','training-setup.exit','supervisor-status.json','judge-service-status.json','async-launch.exit','async-activation.exit']:
 f=p/name
 if f.exists():
  try:out[name]=json.loads(f.read_text())
  except ValueError:out[name]=f.read_text()[-200:]
q=list((p/'judge-queue').glob('*/request.json'))
out['queue']={'total':len(q),'unassigned':sum(not f.with_name('assigned.json').exists() and not f.with_name('receipt.json').exists() for f in q),'receipts':sum(f.with_name('receipt.json').exists() for f in q)}
run_id=RUN_ID
if not run_id:
    experiment=json.loads((p/'EXPERIMENT.json').read_text())
    run_id=experiment.get('restart_run_name') or experiment.get('run_name')
run=p/'async-output'/run_id
for f in run.glob('**/optimizer-update-audit.jsonl'):
 rows=f.read_text().splitlines()
 if rows:
  row=json.loads(rows[-1]);out['optimizer']={k:v for k,v in row.items() if k!='tensors'}
out['checkpoints']=[json.loads(f.read_text()) for f in run.glob('artifacts/adapters/step_*/hf-upload.json')]
for name in ['logs/activation.log','logs/setup-controller.log','logs/download.log','logs/prime-install.log']:
 f=p/name
 if f.exists():out[name]=f.read_text()[-1200:]
for name in ['trainer.log','orchestrator.log','inference.log']:
 files=sorted(run.glob('logs/attempt_*/'+name))
 if files:out[name]=files[-1].read_text()[-2000:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print(json.dumps(out,indent=2))
'''
        code=code.replace('RUN_ID',repr(args.run_id)).replace('IDENTITY_MODE',repr(args.identity_mode))
        output=remote(code);(args.root/'latest-status.json').write_text(output);print(output)


if __name__=='__main__':main()
