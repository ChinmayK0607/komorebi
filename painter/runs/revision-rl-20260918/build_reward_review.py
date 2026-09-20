"""Blind every valid training action twice with reversed A/B orientation."""
import json,hashlib,shutil,random,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 root=P/'reward-review';assert not root.exists(),'Preserve existing review';root.mkdir()
 draws=[]
 for file in sorted((P/'rl-rollouts').glob('*/result.json')):
  result=json.loads(file.read_text());sample=json.loads(file.with_name('sample.json').read_text())
  if result['valid']:
   for orientation in [0,1]:draws.append((sample,result,orientation))
 random.Random(918818).shuffle(draws)
 pairs=[];key=[]
 for index,(sample,result,orientation) in enumerate(draws,1):
  ident=f'pair-{index:03d}';out=root/ident;out.mkdir()
  paths=dict(reference=P/sample['state']['reference'],A=P/(sample['state']['canvas'] if orientation else result['canvas']),B=P/(result['canvas'] if orientation else sample['state']['canvas']))
  hashes={}
  for side,path in paths.items():shutil.copyfile(path,out/(side+'.png'));hashes[side]=hashlib.sha256(path.read_bytes()).hexdigest()
  pairs.append(dict(id=ident,**{side:ident+'/'+side+'.png' for side in paths}))
  key.append(dict(id=ident,sample_id=sample['id'],state_id=sample['state']['id'],candidate_side='B' if orientation else 'A',image_sha256=hashes))
 (root/'pairs.json').write_text(json.dumps(pairs,indent=2)+'\n');(root/'private-key.json').write_text(json.dumps(key,indent=2)+'\n')
 # Include raw on-policy records and training states for later exact-token audit.
 files=[f for folder in ['reward-review','rl-rollouts','rl-states'] for f in (P/folder).rglob('*') if f.is_file()]+[P/'EXPERIMENT.json',P/'rl-states.json']
 manifest={str(f.relative_to(P)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files}
 (P/'reward-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(P/'reward-manifest.json')
 with tarfile.open(P/'reward-evidence.tar.gz','w:gz') as t:
  for f in files:t.add(f,arcname=str(f.relative_to(P)),recursive=False)
 archive=P/'reward-evidence.tar.gz';receipt=dict(pairs=len(pairs),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
 (P/'reward-export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
