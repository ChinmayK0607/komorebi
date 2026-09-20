import hashlib,json,tarfile
from pathlib import Path
p=Path('/root/painter');files=sorted(f for f in (p/'logs').glob('*') if f.is_file())
files += [p/x for x in ['hardware-plan.json','foundation-check.json','setup.exit','ready.json'] if (p/x).is_file()]
a=p/'h100-setup-preserved.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)),recursive=False)
print(json.dumps(dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest(),bytes=a.stat().st_size,files=len(files),optimizer_updates=0)))
