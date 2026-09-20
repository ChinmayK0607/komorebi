"""Node evidence export: data and logs only, never environments or weights."""
import hashlib,json,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
files=[p for folder in ('data','logs') for p in (P/folder).rglob('*') if p.is_file() and (p.suffix in ('.json','.jsonl','.js','.png','.log','.txt') or ('.previous-' in p.name and Path(p.name.split('.previous-')[0]).suffix in ('.png','.json')))]
files += [p for p in P.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.cjs')]
files += [p for p in (P/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [P/name for name in ('README.md','NEXT-RUN.md','CONTRACT.md','foundation-lock.json','foundation-check.json','foundation-tests.log','foundation-validation.json','selected-checkpoint.json','campaign.json','hardware-plan.json','ready.json','setup.exit','demonstrations.exit','payload-manifest.json') if (P/name).exists()]
assert all(not f.is_symlink() for f in files)
manifest={str(f.relative_to(P)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files}
(P/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(P/'evidence-manifest.json')
with tarfile.open(P/'evidence.tar.gz','w:gz') as t:
    for f in files:t.add(f,arcname=str(f.relative_to(P)),recursive=False)
receipt=dict(file='evidence.tar.gz',files=len(files),bytes=(P/'evidence.tar.gz').stat().st_size,sha256=hashlib.sha256((P/'evidence.tar.gz').read_bytes()).hexdigest(),contains_weights=False)
(P/'export-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
