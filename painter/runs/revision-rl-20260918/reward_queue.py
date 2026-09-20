"""Export blinded Luna work from the node; validate labels into immutable rewards."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
from quality_signal import reconcile

JUDGE_MODELS = {'gpt-5.6-luna'}
IDENTITY_ROLES = ('reference', 'A', 'B')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def _request_sort_key(root, path, request):
    """Order pending requests oldest-first with deterministic tie-breaks.

    Requests from before ``created_at`` was recorded are treated as timestamp
    zero so legacy work remains eligible ahead of newly created requests.
    Present timestamps must be finite JSON numbers; silently accepting an
    unusable value would make queue order depend on an implementation detail.
    """
    if 'created_at' not in request:
        created_at = 0.0
    else:
        value = request['created_at']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'Invalid created_at in request: {path}')
        try:
            created_at = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f'Invalid created_at in request: {path}') from None
        if not math.isfinite(created_at):
            raise ValueError(f'Invalid created_at in request: {path}')
    return (created_at, str(request.get('id', '')), str(path.relative_to(root)))


def _materialize_identity_asset(dest, digest, source):
    """Install one immutable, neutral-name copy of an identity image."""
    assets_dir = dest / 'assets'
    if assets_dir.is_symlink() or (assets_dir.exists() and not assets_dir.is_dir()):
        raise ValueError(f'Identity asset directory conflicts: {assets_dir}')
    assets_dir.mkdir(exist_ok=True)
    destination = assets_dir / f'{digest}.png'
    if destination.is_symlink():
        raise ValueError(f'Identity asset destination is a symlink: {destination}')
    if destination.exists():
        if not destination.is_file() or sha(destination) != digest:
            raise ValueError(f'Existing identity asset differs: {destination}')
        return destination.resolve()

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{digest}.', suffix='.tmp',
                                         dir=assets_dir, delete=False) as output:
            temporary = Path(output.name)
        shutil.copyfile(source, temporary)
        if sha(temporary) != digest:
            raise ValueError(f'Copied identity asset differs: {source}')
        # Link into place so a concurrent stage cannot be silently overwritten.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or sha(destination) != digest:
                raise ValueError(f'Concurrent identity asset differs: {destination}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination.resolve()

def export(root, batch, limit):
    dest=root/'reward-batches'/batch
    if dest.exists():raise ValueError('Batch already exists: '+str(dest))
    pending=[]
    for f in sorted((root/'judge-queue').glob('*/request.json')):
        if f.with_name('receipt.json').exists() or f.with_name('assigned.json').exists():continue
        pending.append((f, json.loads(f.read_text())))
    requests=sorted(pending, key=lambda item: _request_sort_key(root, *item))[:limit]
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

def _identity_stage(dest, pairs, key):
    """Build one content-addressed asset table and opaque rolemap views."""
    source_paths = {}
    pair_hashes = {}
    for pair in pairs:
        private = key[pair['id']]
        hashes = private['image_sha256']
        actual = {}
        for role in IDENTITY_ROLES:
            path = (dest / pair[role]).resolve()
            if not path.is_relative_to(dest.resolve()) or not path.is_file() or sha(path) != hashes[role]:
                raise ValueError(f'Changed identity-mode image: {pair["id"]}:{role}')
            actual[role] = sha(path)
            source_paths.setdefault(actual[role], set()).add(str(path))
        pair_hashes[pair['id']] = actual
    digests = sorted(source_paths)
    # Asset IDs must be stable across batches. Ordinal IDs allow a stale public
    # label to bind an asset-0003 from one batch to different bytes in another.
    asset_ids = {digest: f'asset-{digest}' for digest in digests}
    canonical_paths = {
        digest: _materialize_identity_asset(
            dest, digest, Path(sorted(source_paths[digest])[0]))
        for digest in digests
    }
    assets = [{
        'asset_id': asset_ids[digest],
        'bytes': canonical_paths[digest].stat().st_size,
        'path': str(canonical_paths[digest]),
        'sha256': digest,
    } for digest in digests]
    items, bindings = [], {}
    for index, pair in enumerate(pairs, 1):
        rolemap = {role: asset_ids[pair_hashes[pair['id']][role]] for role in IDENTITY_ROLES}
        item = {
            'id': pair['id'],
            'ordinal': f'ordinal-{index:04d}',
            'rolemap': rolemap,
            'asset_ids': list(dict.fromkeys(rolemap[role] for role in IDENTITY_ROLES)),
            'display_order': list(IDENTITY_ROLES),
        }
        items.append(item)
        request = key[pair['id']]['request']
        request_hashes = request['image_sha256']
        bindings[pair['id']] = {
            'request_id': key[pair['id']]['request_id'],
            'candidate_side': key[pair['id']]['candidate_side'],
            'rolemap': rolemap,
            'image_sha256': pair_hashes[pair['id']],
            'deterministic_identity': {
                'candidate_bytes_equal': request_hashes['candidate'] == request_hashes['anchor'],
                'reference_matches': [
                    side for side in ('candidate', 'anchor')
                    if request_hashes[side] == request_hashes['reference']
                ],
            },
        }
    return assets, items, bindings


def stage(root,batch,identity_mode=False):
    dest=root/'reward-batches'/batch
    pairs=json.loads((dest/'pairs.json').read_text())
    key={r['id']:r for r in json.loads((dest/'private-key.json').read_text())}
    if identity_mode:
        assets, items, bindings = _identity_stage(dest, pairs, key)
        write(dest/'assets.json', {'version': 1, 'content_address': 'sha256', 'assets': assets})
        write(dest/'identity-review-input.json', items)
        write(dest/'identity-key.json', bindings)
        request_ids=list(dict.fromkeys(key[p['id']]['request_id'] for p in pairs))
        groups=[[],[],[]]
        item_by_id={item['id']:item for item in items}
        for p in pairs:
            k=key[p['id']];index=request_ids.index(k['request_id'])
            groups[(index+(k['candidate_side']=='B'))%3].append(item_by_id[p['id']])
        for i,group in enumerate(groups,1):
            if group:write(dest/f'judge-{i:02d}-assets.json',group)
        print(json.dumps(dict(pairs=len(pairs),batches=sum(bool(g) for g in groups),
                              identity_mode=True,assets=len(assets))))
        return
    for p in pairs:
        for k in ['reference','A','B']:p[k]=str((dest/p[k]).resolve())
    write(dest/'review-input.json',pairs)
    # Put reversed views on different agents without exposing orientation.
    request_ids=list(dict.fromkeys(key[p['id']]['request_id'] for p in pairs))
    groups=[[],[],[]]
    for p in pairs:
        k=key[p['id']];index=request_ids.index(k['request_id'])
        groups[(index+(k['candidate_side']=='B'))%3].append(p)
    for i,group in enumerate(groups,1):
        if group:write(dest/f'judge-{i:02d}.json',group)
    print(json.dumps(dict(pairs=len(pairs),batches=sum(bool(g) for g in groups))))

def _identity_receipts(root, batch):
    dest = root/'reward-batches'/batch
    assets_path = dest/'assets.json'
    items_path = dest/'identity-review-input.json'
    if not assets_path.exists() or not items_path.exists():
        raise AssertionError('identity-mode stage artifacts are missing')
    assets_doc = json.loads(assets_path.read_text())
    assert assets_doc.get('content_address') == 'sha256'
    assets = {row['asset_id']: row for row in assets_doc.get('assets', [])}
    assert len(assets) == len(assets_doc.get('assets', []))
    for asset_id, asset in assets.items():
        assert asset_id == f'asset-{asset["sha256"]}', f'Asset ID is not content addressed: {asset_id}'
        path = Path(asset['path'])
        assert path.is_file() and path.stat().st_size == asset['bytes']
        assert sha(path) == asset['sha256'], f'Asset hash mismatch: {asset_id}'
    items = json.loads(items_path.read_text())
    item_by_id = {item['id']: item for item in items}
    assert len(item_by_id) == len(items)
    keys = {r['id']: r for r in json.loads((dest/'private-key.json').read_text())}
    assert set(item_by_id) == set(keys)
    for item in items:
        assert set(item) == {'id', 'ordinal', 'rolemap', 'asset_ids', 'display_order'}
        assert item['display_order'] == list(IDENTITY_ROLES)
        assert set(item['rolemap']) == set(IDENTITY_ROLES)
        assert item['asset_ids'] == list(dict.fromkeys(item['rolemap'][role] for role in IDENTITY_ROLES))
        assert all(asset_id in assets for asset_id in item['asset_ids'])
        key = keys[item['id']]
        for role in IDENTITY_ROLES:
            assert item['rolemap'][role] in assets
            asset = assets[item['rolemap'][role]]
            assert asset['sha256'] == key['image_sha256'][role]
            assert sha(Path(asset['path'])) == asset['sha256']

    label_fields = {
        'id', 'rolemap', 'observations', 'winner', 'confidence',
        'acceptable_A', 'acceptable_B', 'reason', 'displayed_asset_ids',
    }
    labels = {}
    judge_models = set()
    for path in sorted(dest.glob('labels-*.json')):
        doc = json.loads(path.read_text())
        assert set(doc) == {'judge_model', 'reasoning_effort', 'judgments'}
        assert doc['judge_model'] in JUDGE_MODELS
        assert doc.get('reasoning_effort') == 'low'
        judge_models.add(doc['judge_model'])
        for row in doc['judgments']:
            ident = row.get('id')
            assert ident in item_by_id and ident not in labels
            assert set(row) == label_fields
            item = item_by_id[ident]
            assert row['rolemap'] == item['rolemap']
            assert row['displayed_asset_ids'] == item['asset_ids']
            assert isinstance(row['observations'], dict)
            assert set(row['observations']) == set(item['asset_ids'])
            assert all(isinstance(text, str) and text.strip() for text in row['observations'].values())
            assert row['winner'] in {'A', 'B', 'tie', 'uncertain'}
            assert row['confidence'] in {'high', 'medium', 'low'}
            assert row['acceptable_A'] in {'yes', 'no', 'uncertain'}
            assert row['acceptable_B'] in {'yes', 'no', 'uncertain'}
            assert str(row['reason']).strip()
            # Identical candidates have a deterministic relative tie, but
            # their absolute acceptability remains supplied by this label.
            private = keys[ident]
            request = private['request']
            if request['image_sha256']['candidate'] == request['image_sha256']['anchor']:
                assert row['winner'] == 'tie', f'identity candidate was not tied: {ident}'
                assert row['acceptable_A'] == row['acceptable_B'], (
                    f'identical candidates have conflicting absolute quality: {ident}')
            labels[ident] = row

    grouped = {}
    for ident, row in labels.items():
        private = keys[ident]
        grouped.setdefault(private['request_id'], {})[private['candidate_side']] = (ident, row)
    out = dest/'receipts'
    out.mkdir(exist_ok=True)
    ready = 0
    requests = {private['request']['id']: private['request'] for private in keys.values()}
    for request in requests.values():
        views = grouped.get(request['id'], {})
        if set(views) != {'A', 'B'}:
            continue
        assert len(judge_models) == 1
        first_id, first = views['A']
        reverse_id, reverse = views['B']
        deterministic = {
            'candidate_bytes_equal': request['image_sha256']['candidate'] == request['image_sha256']['anchor'],
            'relative_winner': 'tie' if request['image_sha256']['candidate'] == request['image_sha256']['anchor'] else None,
            'reference_matches': [
                side for side in ('candidate', 'anchor')
                if request['image_sha256'][side] == request['image_sha256']['reference']
            ],
            'source': 'sha256',
        }
        result = dict(
            id=request['id'],
            image_sha256=request['image_sha256'],
            judge_model=next(iter(judge_models)),
            reasoning_effort='low',
            candidate_A=first,
            candidate_B=reverse,
            judgment=reconcile(first, reverse),
            deterministic_identity=deterministic,
            visual_provenance={
                'mode': 'content_addressed_unique_assets',
                'asset_manifest_sha256': sha(assets_path),
                'views': [
                    {'id': first_id, 'rolemap': first['rolemap'],
                     'displayed_asset_ids': first['displayed_asset_ids']},
                    {'id': reverse_id, 'rolemap': reverse['rolemap'],
                     'displayed_asset_ids': reverse['displayed_asset_ids']},
                ],
            },
        )
        path = out/request['id']
        path.mkdir(exist_ok=True)
        write(path/'receipt.json', result)
        ready += 1
    print(json.dumps(dict(labels=len(labels), expected=len(keys), ready_requests=ready,
                          identity_mode=True)))


def receipts(root,batch,identity_mode=False):
    if identity_mode:
        return _identity_receipts(root, batch)
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

def install(root,batch,identity_mode=False):
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
        if identity_mode:
            if receipt.get('visual_provenance', {}).get('mode') != 'content_addressed_unique_assets':
                raise ValueError(f'Invalid identity receipt provenance: {folder}')
            assets_path = dest/'assets.json'
            if not assets_path.is_file() or receipt['visual_provenance'].get(
                    'asset_manifest_sha256') != sha(assets_path):
                raise ValueError(f'Identity asset manifest mismatch: {folder}')
            deterministic = receipt.get('deterministic_identity', {})
            expected_equal = request['image_sha256']['candidate'] == request['image_sha256']['anchor']
            if deterministic.get('candidate_bytes_equal') != expected_equal:
                raise ValueError(f'Identity metadata mismatch: {folder}')
            if expected_equal and deterministic.get('relative_winner') != 'tie':
                raise ValueError(f'Identity tie metadata mismatch: {folder}')
        target=root/'judge-queue'/request['id']/'receipt.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(receipt,indent=2)+'\n');installed+=1
    print(json.dumps(dict(batch=batch,installed=installed)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','stage','receipts','install']);p.add_argument('batch');p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--limit',type=int,default=24);p.add_argument('--identity-mode',action='store_true',help='Use unique asset IDs and rolemaps for stage/receipts/install');a=p.parse_args()
    if a.action=='export':export(a.root,a.batch,a.limit)
    elif a.action=='stage':stage(a.root,a.batch,a.identity_mode)
    elif a.action=='receipts':receipts(a.root,a.batch,a.identity_mode)
    else:install(a.root,a.batch,a.identity_mode)
