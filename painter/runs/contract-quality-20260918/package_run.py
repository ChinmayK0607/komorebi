import tarfile,hashlib,json
from pathlib import Path
from foundation_lock import check
check();p=Path(__file__).resolve().parent
allow={'campaign.json','EXPERIMENT.json','retention.jsonl','recovery-pools.json','recovery-data-receipt.json','evaluation-tasks.json','evaluation-references.json','legacy-evaluation-tasks.json','training-state-tasks.json','foundation-lock.json','foundation-validation.json','selected-checkpoint.json','CONTRACT.md'}
files=[f for f in p.iterdir() if f.is_file() and (f.suffix in ('.py','.sh','.cjs') or f.name in allow)]
for name in ['source','vendor','data','diversity','previous-evaluation']:
 files += [f for f in (p/name).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc',)]
assert all(not f.is_symlink() for f in files)
manifest={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(p/'run-input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(p/'run-input-manifest.json')
with tarfile.open(p/'run-inputs.tar.gz','w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)),recursive=False)
print(json.dumps({'files':len(files),'bytes':(p/'run-inputs.tar.gz').stat().st_size,'sha256':hashlib.sha256((p/'run-inputs.tar.gz').read_bytes()).hexdigest()}))
