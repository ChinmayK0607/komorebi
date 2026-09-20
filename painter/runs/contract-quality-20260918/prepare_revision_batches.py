"""Verify a revision archive, reuse hash-identical prior judgments, blind batches."""
import argparse,json,hashlib,tarfile
from pathlib import Path,PurePosixPath
p=Path(__file__).resolve().parent

def prepare(model):
 out=p/(model+'-revision-all');archive=p/(model+'-revision-all.tar.gz')
 receipt=json.loads((p/(model+'-all-collection.json')).read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
 with tarfile.open(archive) as t:
  assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in t.getmembers())
  assert sum(m.size for m in t.getmembers())<1024**3
  t.extractall(out,filter='data')
 for name,sha in json.loads((out/'manifest.json').read_text()).items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
 pairs=json.loads((out/'pairs.json').read_text());key=json.loads((out/'private-key.json').read_text());byid={r['id']:r for r in key}
 for row in pairs:
  for k in ['reference','A','B']:row[k]=str((out/row[k]).resolve())
 reused=[]
 if model=='parent':
  old=p/'parent-revision-probe';oldkey={r['id']:r for r in json.loads((old/'private-key.json').read_text())};oldlabels=json.loads((old/'labels.json').read_text())
  lookup={(r['task_id'],r['sample']):r for r in key}
  for label in oldlabels['judgments']:
   source=oldkey[label['id']];target=lookup[(source['task_id'],0)];a=source['image_sha256'];b=target['image_sha256'];assert a['reference']==b['reference']
   swap=a['A']!=b['A']
   assert (a['A'],a['B'])==((b['B'],b['A']) if swap else (b['A'],b['B']))
   mapped=dict(label,id=target['id'],provenance=dict(file=str(old/'labels.json'),original_id=label['id'],swapped_AB=swap,method='Exact reference/A/B image SHA-256 match; original raw label unchanged'))
   if swap:
    mapped['reason']='Imported prior review with A/B reversed in this layout. Original reason: '+label['reason']
    mapped['winner']={'A':'B','B':'A','tie':'tie','uncertain':'uncertain'}[label['winner']]
    mapped['acceptable_A'],mapped['acceptable_B']=label['acceptable_B'],label['acceptable_A']
    mapped['reviewed_images']=[label['reviewed_images'][0],label['reviewed_images'][2],label['reviewed_images'][1]]
   for side,path in zip(['reference','A','B'],mapped['reviewed_images']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==b[side]
   reused.append(mapped)
  (out/'labels-reused.json').write_text(json.dumps(dict(judge_model='gpt-5.6-luna',judgments=reused),indent=2)+'\n')
 (out/'review-input.json').write_text(json.dumps(pairs,indent=2)+'\n')
 done={r['id'] for r in reused};pending=[r for r in pairs if r['id'] not in done]
 for index,start in enumerate(range(0,len(pending),24),1):
  (out/f'batch-{index:02d}.json').write_text(json.dumps(pending[start:start+24],indent=2)+'\n')
 report=dict(total_episodes=len(key),visual_pairs=len(pairs),reused_labels=len(reused),pending_pairs=len(pending),batches=(len(pending)+23)//24,all_task_kinds_included=True)
 (out/'review-status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('model',choices=['parent','midpoint','trained']);prepare(a.parse_args().model)
