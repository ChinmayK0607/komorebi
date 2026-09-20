"""Preserve code, paintings, receipts and logs; exclude all weights and secrets."""
import hashlib,json,tarfile,time,io
from pathlib import Path
P=Path(__file__).resolve().parent;files=[]
for folder in ['data','previous-evaluation','evaluation','judging','logs','source','vendor','train-output']:
 for f in (P/folder).rglob('*'):
  if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts:continue
  if f.suffix not in ('.json','.jsonl','.js','.png','.log','.txt','.py','.sh','.toml','.csv'):continue
  if f.name in ('train.jsonl','dev.jsonl','validation.jsonl'):continue
  files.append(f)
files += [f for f in P.iterdir() if f.is_file() and not f.is_symlink() and ((f.suffix in ('.py','.sh','.csv') and f.name!='env.sh') or f.name in ('campaign.json','EXPERIMENT.json','phase.json','preflight.json','preprocessing-rejections.json','token-audit.json','serving-parity.json','training-ready.json','hardware-plan.json','model-receipt.json','model-staging-receipt.json','evaluation-references.json','evaluation-tasks.json','recovery-pools.json','recovery-data-receipt.json','initial-adapter-receipt.json','judge-receipt.json','training.exit','pipeline.exit','training-setup.exit','pipeline.log','bootstrap.log','dataset-receipt.json'))]
files += [P/'sft-data/mixture.json'] if (P/'sft-data/mixture.json').exists() else []
# Snapshot each file once: live training logs may grow while collection runs.
manifest={}
with tarfile.open(P/'evidence.tar.gz.partial','w:gz') as t:
 for f in files:
  data=f.read_bytes();name=str(f.relative_to(P));manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
 data=(json.dumps(manifest,indent=2)+'\n').encode();(P/'evidence-manifest.json').write_bytes(data)
 info=tarfile.TarInfo('evidence-manifest.json');info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
files.append(P/'evidence-manifest.json')
(P/'evidence.tar.gz.partial').replace(P/'evidence.tar.gz')
r=dict(file='evidence.tar.gz',files=len(files),bytes=(P/'evidence.tar.gz').stat().st_size,sha256=hashlib.sha256((P/'evidence.tar.gz').read_bytes()).hexdigest(),contains_weights=False,at=time.time())
(P/'export-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
