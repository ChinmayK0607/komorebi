"""Export blinded Luna work from the node; validate labels into immutable rewards."""
import argparse
import hashlib
import json
import random
import shutil
import tarfile
from pathlib import Path
from quality_signal import reconcile

JUDGE_MODELS = {'gpt-5.6-luna'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')

def export(root, batch, limit):
    dest=root/'reward-batches'/batch
    if dest.exists():raise ValueError('Batch already exists: '+str(dest))
    requests=[]
    for f in sorted((root/'judge-queue').glob('*/request.json')):
        if f.with_name('receipt.json').exists() or f.with_name('assigned.json').exists():continue
        requests.append((f,json.loads(f.read_text())))
        if len(requests)>=limit:break
    if not requests:print('No unassigned rewards');return
    dest.mkdir(parents=True)
    jobs=[(f,r,side) for f,r in requests for side in ['A','B']]
    random.Random(batch).shuffle(jobs)
    pairs=[];key=[]
    for index,(f,r,side) in enumerate(jobs):
        ident=f'{batch}-{index:04d}';folder=dest/ident;folder.mkdir()
        order={'reference':'reference','A':'candidate' if side=='A' else 'anchor',
               'B':'anchor' if side=='A' else 'candidate'}
        images={};hashes={}
        for k,src in order.items():
            path=root/r['images'][src]
            if sha(path)!=r['image_sha256'][src]:raise ValueError('Changed request image')
            shutil.copyfile(path,folder/(k+'.png'));images[k]=ident+'/'+k+'.png';hashes[k]=sha(path)
        pairs.append(dict(id=ident,**images))
        key.append(dict(id=ident,request_id=r['id'],candidate_side=side,image_sha256=hashes,request=r))
    write(dest/'pairs.json',pairs);write(dest/'private-key.json',key)
    for f,r in requests:write(f.with_name('assigned.json'),dict(batch=batch))
    archive=dest.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=batch)
    write(archive.with_suffix('.receipt.json'),dict(sha256=sha(archive),requests=len(requests),pairs=len(pairs)))
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),requests=len(requests),pairs=len(pairs))))

def stage(root,batch):
    dest=root/'reward-batches'/batch
    pairs=json.loads((dest/'pairs.json').read_text())
    for p in pairs:
        for k in ['reference','A','B']:p[k]=str((dest/p[k]).resolve())
    write(dest/'review-input.json',pairs)
    # Put reversed views on different agents without exposing orientation.
    key={r['id']:r for r in json.loads((dest/'private-key.json').read_text())}
    request_ids=list(dict.fromkeys(key[p['id']]['request_id'] for p in pairs))
    groups=[[],[],[]]
    for p in pairs:
        k=key[p['id']];index=request_ids.index(k['request_id'])
        groups[(index+(k['candidate_side']=='B'))%3].append(p)
    for i,group in enumerate(groups,1):
        if group:write(dest/f'judge-{i:02d}.json',group)
    print(json.dumps(dict(pairs=len(pairs),batches=sum(bool(g) for g in groups))))

def receipts(root,batch):
    dest=root/'reward-batches'/batch
    keys={r['id']:r for r in json.loads((dest/'private-key.json').read_text())}
    pairs={r['id']:r for r in json.loads((dest/'review-input.json').read_text())}
    labels={}
    judge_models=set()
    for f in sorted(dest.glob('labels-*.json')):
        doc=json.loads(f.read_text());assert doc['judge_model'] in JUDGE_MODELS
        assert doc.get('reasoning_effort')=='low', 'Luna low provenance required'
        judge_models.add(doc['judge_model'])
        for row in doc['judgments']:
            ident=row['id'];assert ident in keys and ident not in labels
            assert row['winner'] in ['A','B','tie','uncertain']
            assert row['confidence'] in ['high','medium','low']
            assert all(row['acceptable_'+s] in ['yes','no','uncertain'] for s in ['A','B'])
            assert row['reason'].strip()
            assert row['reviewed_images']==[pairs[ident][s] for s in ['reference','A','B']]
            for s in ['reference','A','B']:assert sha(Path(pairs[ident][s]))==keys[ident]['image_sha256'][s]
            labels[ident]=row
    grouped={}
    for ident,row in labels.items():
        k=keys[ident];grouped.setdefault(k['request_id'],{})[k['candidate_side']]=row
    out=dest/'receipts';out.mkdir(exist_ok=True);ready=0
    for k in keys.values():
        r=k['request'];v=grouped.get(r['id'],{})
        if set(v)!= {'A','B'}:continue
        # Preserve the actual judge identity in the receipt; do not relabel a
        # third-party vision model as Luna.
        assert len(judge_models)==1
        result=dict(id=r['id'],image_sha256=r['image_sha256'],judge_model=next(iter(judge_models)),
                    reasoning_effort='low',
                    candidate_A=v['A'],candidate_B=v['B'],judgment=reconcile(v['A'],v['B']))
        path=out/r['id'];path.mkdir(exist_ok=True);write(path/'receipt.json',result);ready+=1
    # Each request appears in two key rows.
    print(json.dumps(dict(labels=len(labels),expected=len(keys),ready_requests=ready//2)))

def install(root,batch):
    """Install only reconciled, provenance-checked receipts into the live queue."""
    dest=root/'reward-batches'/batch
    keys={r['request']['id']:r for r in json.loads((dest/'private-key.json').read_text())}
    installed=0
    for folder in sorted((dest/'receipts').glob('*/receipt.json')):
        receipt=json.loads(folder.read_text()); ident=receipt.get('id')
        if (ident not in keys or receipt.get('judge_model') not in JUDGE_MODELS
                or receipt.get('reasoning_effort') != 'low'):
            raise ValueError(f'Invalid receipt provenance: {folder}')
        request=keys[ident]['request']
        if receipt.get('image_sha256')!=request.get('image_sha256'):
            raise ValueError(f'Image hash mismatch: {folder}')
        target=root/'judge-queue'/request['id']/'receipt.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(receipt,indent=2)+'\n');installed+=1
    print(json.dumps(dict(batch=batch,installed=installed)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','stage','receipts','install']);p.add_argument('batch');p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    if a.action=='export':export(a.root,a.batch,a.limit)
    elif a.action=='stage':stage(a.root,a.batch)
    elif a.action=='receipts':receipts(a.root,a.batch)
    else:install(a.root,a.batch)
