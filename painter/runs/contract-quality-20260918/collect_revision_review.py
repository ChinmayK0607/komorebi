from pathlib import Path
import json,hashlib,tarfile,random,argparse
a=argparse.ArgumentParser();a.add_argument("--model",choices=["parent","midpoint","trained"],required=True);args=a.parse_args()
model=args.model
p=Path('/root/painter');out=p/(model+'-revision-all');out.mkdir(exist_ok=True)
assert not(out/'pairs.json').exists(),'Existing full review; do not regenerate'
summary=json.loads((p/'evaluation'/model/'summary.json').read_text());assert summary['completed']==summary['expected']==128
pairs=[];key=[];missing=[];rng=random.Random(918128)
rows=list(summary['results']);rng.shuffle(rows)
for row in rows:
 task=row['task'];ident=f'pair-{len(key)+1:03d}';paints=[t for t in row.get('turns',[]) if t.get('valid') and t.get('action')=='paint' and t.get('canvas')]
 before=p/task['canvas'] if task.get('canvas') else Path(paints[0]['canvas']) if paints else None
 after=Path(row['canvas']) if row.get('canvas') else None
 flip=rng.random()<.5;A,B=(after,before) if flip else (before,after)
 item=dict(id=ident,task_id=task['id'],sample=row['sample'],family=task['family'],kind=task['kind'],A='final' if flip else 'initial',B='initial' if flip else 'final',comparison='initial_vs_final',source_group=task['source_group'],finished=row['finished'],operational_cutoff=row['operational_cutoff'],image_sha256={})
 if before is None or after is None or not before.exists() or not after.exists():
  missing.append(dict(id=ident,status='missing_canvas'));key.append(item);continue
 folder=out/ident;folder.mkdir();paths={'reference':p/task['reference'],'A':A,'B':B}
 for side,src in paths.items():
  raw=src.read_bytes();(folder/(side+'.png')).write_bytes(raw);item['image_sha256'][side]=hashlib.sha256(raw).hexdigest()
 pairs.append(dict(id=ident,**{side:ident+'/'+side+'.png' for side in paths}));key.append(item)
for name,data in [('pairs.json',pairs),('private-key.json',key),('mechanical-failures.json',missing)]:
 (out/name).write_text(json.dumps(data,indent=2)+'\n')
files=[f for f in out.rglob('*') if f.is_file()];manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(out/'manifest.json')
with tarfile.open(p/(model+'-revision-all.tar.gz'),'w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(out)),recursive=False)
print(json.dumps(dict(pairs=len(pairs),total=len(key),missing=len(missing),archive_sha256=hashlib.sha256((p/(model+'-revision-all.tar.gz')).read_bytes()).hexdigest())))
