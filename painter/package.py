"""Build an input-only portable node bundle, excluding access details and outputs."""
import hashlib,json,tarfile
from pathlib import Path
from foundation_lock import check
check()
P=Path(__file__).resolve().parent
files=[p for p in P.iterdir() if p.suffix in ('.py','.sh','.cjs') or p.name in ('campaign.json','README.md','NEXT-RUN.md','CONTRACT.md','foundation-lock.json','foundation-validation.json','selected-checkpoint.json')]
files += [p for p in (P/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [p for p in (P/'data').rglob('*') if p.is_file() and (p.suffix=='.js' or p.name=='manifest.jsonl')]
manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(P/'payload-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(P/'payload-manifest.json')
with tarfile.open(P/'node-inputs.tar.gz','w:gz') as tar:
    for f in files:tar.add(f,arcname=str(f.relative_to(P)),recursive=False)
print(json.dumps(dict(files=len(files),bytes=(P/'node-inputs.tar.gz').stat().st_size,sha256=hashlib.sha256((P/'node-inputs.tar.gz').read_bytes()).hexdigest())))
