"""Validate actual Luna reviews, mask disagreement, preserve on-policy samples."""
import json,hashlib,collections
from pathlib import Path
from quality_signal import reconcile,reward,advantages
P=Path(__file__).resolve().parent

def main():
 root=P/'reward-review';key=json.loads((root/'private-key.json').read_text());lookup={r['id']:r for r in key};pairs={r['id']:r for r in json.loads((root/'review-input.json').read_text())};labels={}
 for f in sorted(root.glob('labels-*.json')):
  doc=json.loads(f.read_text());assert doc['judge_model']=='gpt-5.6-luna'
  for row in doc['judgments']:
   ident=row['id'];assert ident in lookup and ident not in labels
   assert row['reason'].strip() and row['winner'] in ['A','B','tie','uncertain']
   expected=[pairs[ident][k] for k in ['reference','A','B']];assert row['reviewed_images']==expected
   for side,path in zip(['reference','A','B'],expected):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==lookup[ident]['image_sha256'][side]
   labels[ident]=row
 assert set(labels)==set(lookup),f'{len(lookup)-len(labels)} visual reviews pending'
 keyed=collections.defaultdict(dict)
 for ident,k in lookup.items():keyed[k['sample_id']][k['candidate_side']]=labels[ident]
 groups=collections.defaultdict(list)
 for f in sorted((P/'rl-rollouts').glob('*/sample.json')):
  sample=json.loads(f.read_text());result=json.loads(f.with_name('result.json').read_text());views=keyed[sample['id']]
  judgment=reconcile(views.get('A'),views.get('B'));score=reward(result,judgment)
  groups[sample['state']['id']].append(dict(id=sample['id'],reward=score,judgment=judgment,result=result))
 selected=[];reports=[]
 for state,rows in groups.items():
  assert len(rows)==8
  adv=advantages([r['reward'] for r in rows]);reports.append(dict(state_id=state,rewards=[r['reward'] for r in rows],advantages=adv,eligible=any(v is not None for v in adv)))
  for row,a in zip(rows,adv):
   row['advantage']=a
   if a is not None:selected.append(row)
 assert selected,'No confidently graded within-state reward variation; no fake optimizer update'
 (P/'reward-audit.json').write_text(json.dumps(dict(groups=reports,total_samples=192,train_samples=len(selected),judge_model='gpt-5.6-luna',speed_reward=0),indent=2)+'\n')
 (P/'rl-credit.json').write_text(json.dumps(selected,indent=2)+'\n');print(json.dumps(dict(eligible_groups=sum(r['eligible'] for r in reports),samples=len(selected))))
if __name__=='__main__':main()
