"""Validate actual image review provenance; report full-episode revision coverage."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from aggregate_luna import summarize

def aggregate(folder):
 folder=Path(folder).resolve();key=json.loads((folder/'private-key.json').read_text());lookup={r['id']:r for r in key};labels={}
 for file in sorted(folder.glob('labels-*.json')):
  block=json.loads(file.read_text());assert block['judge_model']=='gpt-5.6-luna'
  for row in block['judgments']:
   ident=row['id'];assert ident in lookup and ident not in labels,ident
   assert row['winner'] in ['A','B','tie','uncertain']
   assert row['confidence'] in ['high','medium','low']
   assert all(row['acceptable_'+side] in ['yes','no','uncertain'] for side in ['A','B'])
   assert row['reason'].strip() and len(row['reviewed_images'])==3
   for side,path in zip(['reference','A','B'],row['reviewed_images']):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==lookup[ident]['image_sha256'][side],(ident,side)
   labels[ident]=row
 decoded=[]
 for r in key:
  label=labels.get(r['id']);outcome='unreviewed'
  if label:
   if label['winner']=='uncertain' or label['confidence']=='low':outcome='masked'
   elif label['winner']=='tie':outcome='tie'
   else:outcome='win' if r[label['winner']]=='final' else 'loss'
  decoded.append(dict(r,label=label,outcome=outcome))
 reviewed=[r for r in decoded if r['label']]
 def absolute(rows):
  result={}
  for version in ['initial','final']:
   grades=[]
   for row in rows:
    side=next(side for side in ['A','B'] if row[side]==version)
    grades.append('uncertain' if row['label']['confidence']=='low' else row['label']['acceptable_'+side])
   result[version]=dict(Counter(grades),total=len(grades))
  return result
 report=dict(judge_model='gpt-5.6-luna',total_episodes=len(key),reviewed=len(reviewed),pending=len(key)-len(reviewed),
   complete=len(reviewed)==len(key),interpretation='Wins favor final canvas. Painting compares first valid canvas to final; repair/finish compare provided initial canvas to final. Results are agent judgments, not human ground truth.',
   by_kind={kind:dict(preference=summarize([r for r in reviewed if r['kind']==kind]),absolute=absolute([r for r in reviewed if r['kind']==kind])) for kind in ['paint','repair','finish']},
   by_family={family:summarize([r for r in reviewed if r['family']==family and r['kind']=='paint']) for family in sorted({r['family'] for r in key})})
 (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(folder/'decoded-judgments.json').write_text(json.dumps(decoded,indent=2)+'\n');return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();print(json.dumps(aggregate(args.folder),indent=2))
