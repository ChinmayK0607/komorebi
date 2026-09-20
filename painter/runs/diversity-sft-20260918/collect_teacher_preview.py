"""Retrieve only the selected, already rendered teacher images and receipts."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
pod=json.loads((P/'pod.json').read_text())
q=subprocess.run(['lium','ps',pod['id'],'--format','json'],capture_output=True,text=True,check=True)
pods,_=json.JSONDecoder().raw_decode(q.stdout.lstrip());assert len(pods)==1 and pods[0]['id']==pod['id'];pod=pods[0]
code='''from pathlib import Path
import json,tarfile,io,sys,hashlib
p=Path('/root/painter/diversity');selection=json.loads((p/'review-ready.json').read_text());ids=set(selection['ids'])
rows=[json.loads(s) for s in (p/'materialized-manifest.jsonl').read_text().splitlines() if json.loads(s)['id'] in ids]
files=[]
for row in rows:
 for relative in row['programs'].values():
  source=p/relative
  for suffix in ['.png','.json']:
   file=source.with_suffix(suffix)
   if file.exists():files.append(file)
manifest={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for f in files:
  b=f.read_bytes();name=str(f.relative_to(p));manifest[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  info=tarfile.TarInfo(name);info.size=len(b);archive.addfile(info,io.BytesIO(b))
 for name,value in [('metadata.json',rows),('manifest.json',manifest),('selection.json',selection)]:
  b=json.dumps(value).encode();info=tarfile.TarInfo(name);info.size=len(b);archive.addfile(info,io.BytesIO(b))
'''
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','UserKnownHostsFile='+str(P/'known_hosts'),'-p',str(pod['ports']['22']),'root@'+pod['ip'],'python3 -']
q=subprocess.run(ssh,input=code.encode(),capture_output=True,check=True);assert len(q.stdout)<100*1024**2
out=P/'teacher-preview';out.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(q.stdout)) as archive:
 for m in archive.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  f=out/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(archive.extractfile(m).read())
for name,meta in json.loads((out/'manifest.json').read_text()).items():
 b=(out/name).read_bytes();assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256'],name
rows=json.loads((out/'metadata.json').read_text());review=[]
for row in rows:
 reference=out/Path(row['programs']['reference']).with_suffix('.png')
 if reference.exists():review.append({k:row[k] for k in ['id','primary_family','layout','object_count','objects']}|{'image':str(reference.resolve())})
(out/'review-input.json').write_text(json.dumps(review,indent=2)+'\n')
(out/'verified.json').write_text(json.dumps(dict(images=sum(1 for p in out.rglob('*.png')),files=len(json.loads((out/'manifest.json').read_text())),verified=True),indent=2)+'\n')
print(json.dumps(dict(review_targets=len(review),folder=str(out))))
