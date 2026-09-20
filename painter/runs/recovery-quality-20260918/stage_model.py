"""Stage verified model bytes off encrypted FUSE before memory-mapped loading."""
import hashlib,json,shutil,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
P=Path(__file__).resolve().parent
assert sys.platform=='linux','Model files stay on the compute node.'
pointer=P/'model-path.txt';source=Path(pointer.read_text().strip());receipt=json.loads((P/'model-receipt.json').read_text());dest=Path('/tmp/painter-models')/source.name
if source==dest:raise SystemExit(0)
dest.mkdir(parents=True,exist_ok=True)
needed=sum(v['bytes'] for v in receipt['files'].values())
assert shutil.disk_usage(dest).free>needed+5*1024**3,'Insufficient node scratch space'
def one(item):
 name,meta=item;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+'.partial')
 shutil.copyfile(source/name,tmp)
 with tmp.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==meta['sha256'],name
 tmp.replace(target);return name
complete=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for f in as_completed([pool.submit(one,x) for x in receipt['files'].items()]):
  complete.append(f.result());(P/'model-staging-progress.json').write_text(json.dumps({'completed':len(complete),'expected':len(receipt['files'])})+'\n')
(P/'model-encrypted-path.txt').write_text(str(source)+'\n');pointer.write_text(str(dest)+'\n')
(P/'model-staging-receipt.json').write_text(json.dumps(dict(source=str(source),destination=str(dest),verified=True,files=len(complete),bytes=needed,completed_at=time.time()),indent=2)+'\n')
