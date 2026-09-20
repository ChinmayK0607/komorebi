"""Actual file-queue invariants; no image rendering or model calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import reward_queue as queue


class LunaQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.request_ids=[]
        for n in range(6):
            paths={};hashes={}
            for role in ('reference','anchor','candidate'):
                f=self.root/f'{n}-{role}.png'
                f.write_bytes(f'{n}-anchor'.encode() if n == 0 and role == 'candidate'
                              else f'{n}-{role}'.encode())
                paths[role]=f.name;hashes[role]=queue.sha(f)
            ident=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
            self.request_ids.append(ident)
            folder=self.root/'judge-queue'/ident;folder.mkdir(parents=True)
            queue.write(folder/'request.json',dict(id=ident,images=paths,image_sha256=hashes))
        queue.export(self.root,'test',6);queue.stage(self.root,'test')
        self.batch=self.root/'reward-batches/test'

    def labels(self):
        keys={v['id']:v for v in json.loads((self.batch/'private-key.json').read_text())}
        for source in self.batch.glob('judge-*.json'):
            judgments=[]
            for p in json.loads(source.read_text()):
                side=keys[p['id']]['candidate_side']
                judgments.append(dict(id=p['id'],winner=side,confidence='high',
                    acceptable_A='yes' if side=='A' else 'no',acceptable_B='yes' if side=='B' else 'no',
                    reason='Candidate matches the silhouette',reviewed_images=[p[s] for s in ('reference','A','B')]))
            queue.write(source.with_name(source.name.replace('judge-','labels-')),
                        dict(judge_model='gpt-5.6-luna',reasoning_effort='low',judgments=judgments))

    def make_identity_batch(self, batch, contents):
        dest=self.root/'reward-batches'/batch
        ident=f'{batch}-0000';folder=dest/ident;folder.mkdir(parents=True)
        paths={};hashes={}
        for role,data in contents.items():
            path=folder/(role+'.png');path.write_bytes(data)
            paths[role]=ident+'/'+role+'.png';hashes[role]=queue.sha(path)
        request=dict(id=f'request-{batch}',images=paths,image_sha256={
            'reference':hashes['reference'],'anchor':hashes['B'],'candidate':hashes['A']})
        queue.write(dest/'pairs.json',[dict(id=ident,**paths)])
        queue.write(dest/'private-key.json',[dict(
            id=ident,request_id=request['id'],candidate_side='A',
            image_sha256=hashes,request=request)])
        queue.stage(self.root,batch,identity_mode=True)
        return json.loads((dest/'assets.json').read_text()), json.loads(
            (dest/'identity-review-input.json').read_text())

    def test_reversed_views_go_to_different_agents(self):
        keys={v['id']:v for v in json.loads((self.batch/'private-key.json').read_text())}
        for source in self.batch.glob('judge-*.json'):
            ids=[keys[v['id']]['request_id'] for v in json.loads(source.read_text())]
            self.assertEqual(len(ids),len(set(ids)))

    def test_receipts_preserve_low_effort_and_hashes(self):
        self.labels();queue.receipts(self.root,'test');queue.install(self.root,'test')
        for ident in self.request_ids:
            receipt=json.loads((self.root/'judge-queue'/ident/'receipt.json').read_text())
            self.assertEqual(receipt['reasoning_effort'],'low')
            self.assertEqual(receipt['judgment']['delta'],1)

    def test_wrong_judge_cannot_enter_training(self):
        self.labels()
        source=next(self.batch.glob('labels-*.json'))
        doc=json.loads(source.read_text());doc['judge_model']='another-model';queue.write(source,doc)
        with self.assertRaises(AssertionError):queue.receipts(self.root,'test')

    def identity_labels(self):
        queue.stage(self.root, 'test', identity_mode=True)
        keys={v['id']:v for v in json.loads((self.batch/'private-key.json').read_text())}
        for index, source in enumerate(sorted(self.batch.glob('judge-*-assets.json')), 1):
            judgments=[]
            for item in json.loads(source.read_text()):
                private=keys[item['id']]
                request=private['request']
                equal=(request['image_sha256']['candidate'] ==
                       request['image_sha256']['anchor'])
                side=private['candidate_side']
                winner='tie' if equal else side
                absolute='no' if equal else None
                judgments.append(dict(
                    id=item['id'], rolemap=item['rolemap'],
                    observations={asset_id: 'visible image'
                                  for asset_id in item['asset_ids']},
                    winner=winner, confidence='high',
                    acceptable_A=absolute or ('yes' if side == 'A' else 'no'),
                    acceptable_B=absolute or ('yes' if side == 'B' else 'no'),
                    reason='Unique assets displayed and compared',
                    displayed_asset_ids=item['asset_ids'],
                ))
            queue.write(self.batch/f'labels-identity-{index:02d}.json',
                        dict(judge_model='gpt-5.6-luna',reasoning_effort='low',
                             judgments=judgments))

    def test_identity_stage_deduplicates_assets_and_hides_paths(self):
        queue.stage(self.root, 'test', identity_mode=True)
        assets=json.loads((self.batch/'assets.json').read_text())['assets']
        self.assertEqual(len(assets), len({asset['sha256'] for asset in assets}))
        for asset in assets:
            self.assertEqual(asset['asset_id'], 'asset-'+asset['sha256'])
            canonical=Path(asset['path'])
            self.assertTrue(canonical.is_absolute())
            self.assertEqual(canonical, (self.batch/'assets'/(asset['sha256']+'.png')).resolve())
            self.assertEqual(queue.sha(canonical), asset['sha256'])
        for role in ('reference', 'A', 'B'):
            self.assertTrue((self.batch/'test-0000'/(role+'.png')).is_file())
        manifests=sorted(self.batch.glob('judge-*-assets.json'))
        self.assertTrue(manifests)
        for manifest in manifests:
            for item in json.loads(manifest.read_text()):
                self.assertEqual(set(item), {'id','ordinal','rolemap','asset_ids','display_order'})
                self.assertEqual(item['asset_ids'],
                                 list(dict.fromkeys(item['rolemap'][role]
                                                    for role in ('reference','A','B'))))
                self.assertNotIn('path', item)

    def test_identity_paths_are_stable_across_orientations(self):
        batch='oriented'
        dest=self.root/'reward-batches'/batch
        pairs=[];keys=[]
        for index, (a_bytes, b_bytes) in enumerate(
                ((b'candidate', b'anchor'), (b'anchor', b'candidate'))):
            ident=f'{batch}-{index:04d}';folder=dest/ident;folder.mkdir(parents=True)
            paths={};hashes={}
            for role,data in (('reference', b'reference'), ('A', a_bytes), ('B', b_bytes)):
                path=folder/(role+'.png');path.write_bytes(data)
                paths[role]=ident+'/'+role+'.png';hashes[role]=queue.sha(path)
            request=dict(id=f'request-{index}', images=paths,
                         image_sha256={'reference':hashes['reference'],
                                       'candidate':hashes['A'], 'anchor':hashes['B']})
            pairs.append(dict(id=ident, **paths))
            keys.append(dict(id=ident, request_id=request['id'], candidate_side='A',
                             image_sha256=hashes, request=request))
        queue.write(dest/'pairs.json', pairs)
        queue.write(dest/'private-key.json', keys)
        queue.stage(self.root, batch, identity_mode=True)

        assets={row['sha256']: row for row in
                json.loads((dest/'assets.json').read_text())['assets']}
        items={row['id']: row for row in
               json.loads((dest/'identity-review-input.json').read_text())}
        candidate_digest=queue.sha(dest/f'{batch}-0000/A.png')
        anchor_digest=queue.sha(dest/f'{batch}-0000/B.png')
        self.assertEqual(items[f'{batch}-0000']['rolemap']['A'],
                         items[f'{batch}-0001']['rolemap']['B'])
        self.assertEqual(items[f'{batch}-0000']['rolemap']['B'],
                         items[f'{batch}-0001']['rolemap']['A'])
        self.assertEqual(Path(assets[candidate_digest]['path']),
                         (dest/'assets'/(candidate_digest+'.png')).resolve())
        self.assertEqual(Path(assets[anchor_digest]['path']),
                         (dest/'assets'/(anchor_digest+'.png')).resolve())

    def test_identity_stage_rejects_conflicting_existing_asset(self):
        digest=queue.sha(self.batch/'test-0000/A.png')
        assets_dir=self.batch/'assets';assets_dir.mkdir()
        (assets_dir/(digest+'.png')).write_bytes(b'wrong bytes')
        with self.assertRaisesRegex(ValueError, 'Existing identity asset differs'):
            queue.stage(self.root, 'test', identity_mode=True)

    def test_content_asset_ids_are_global_across_batches(self):
        shared=b'shared bytes'
        assets_a, items_a=self.make_identity_batch('batch-a', {
            'reference':shared, 'A':b'batch-a candidate', 'B':b'batch-a anchor'})
        assets_b, items_b=self.make_identity_batch('batch-b', {
            'reference':shared, 'A':b'batch-b candidate', 'B':b'batch-a anchor'})
        by_hash_a={row['sha256']:row for row in assets_a['assets']}
        by_hash_b={row['sha256']:row for row in assets_b['assets']}
        self.assertEqual(by_hash_a[queue.sha(self.root/'reward-batches/batch-a/batch-a-0000/reference.png')]['asset_id'],
                         by_hash_b[queue.sha(self.root/'reward-batches/batch-b/batch-b-0000/reference.png')]['asset_id'])
        candidate_a=items_a[0]['rolemap']['A']
        candidate_b=items_b[0]['rolemap']['A']
        self.assertNotEqual(candidate_a, candidate_b)
        self.assertEqual(items_a[0]['ordinal'], items_b[0]['ordinal'])
        for assets in (assets_a, assets_b):
            for asset in assets['assets']:
                self.assertEqual(asset['asset_id'], 'asset-'+asset['sha256'])

    def test_identity_receipts_keep_absolute_quality_and_deterministic_tie_separate(self):
        self.identity_labels()
        queue.receipts(self.root, 'test', identity_mode=True)
        keys={v['request']['id']:v for v in
              json.loads((self.batch/'private-key.json').read_text())}
        receipts=[json.loads(path.read_text())
                  for path in (self.batch/'receipts').glob('*/receipt.json')]
        receipt=next(item for item in receipts
                     if item['deterministic_identity']['candidate_bytes_equal'])
        self.assertEqual(receipt['visual_provenance']['mode'],
                         'content_addressed_unique_assets')
        request=keys[receipt['id']]['request']
        self.assertTrue(request['image_sha256']['candidate'] ==
                        request['image_sha256']['anchor'])
        self.assertEqual(receipt['deterministic_identity']['relative_winner'], 'tie')
        self.assertEqual(receipt['candidate_A']['acceptable_A'], 'no')
        self.assertEqual(receipt['candidate_B']['acceptable_B'], 'no')
        self.assertFalse(receipt['judgment']['candidate_acceptable'])
        queue.install(self.root, 'test', identity_mode=True)

    def test_identity_rolemap_tamper_and_hash_mismatch_fail(self):
        self.identity_labels()
        label=next(self.batch.glob('labels-identity-*.json'))
        doc=json.loads(label.read_text())
        doc['judgments'][0]['rolemap']['A']='asset-tampered'
        queue.write(label, doc)
        with self.assertRaises(AssertionError):
            queue.receipts(self.root, 'test', identity_mode=True)

    def test_identity_role_hash_binding_rejects_valid_asset_swap(self):
        self.identity_labels()
        items_path=self.batch/'identity-review-input.json'
        items=json.loads(items_path.read_text())
        item=items[0]
        original_a=item['rolemap']['A']
        item['rolemap']['A']=item['rolemap']['B']
        item['asset_ids']=list(dict.fromkeys(item['rolemap'][role]
                                            for role in ('reference','A','B')))
        queue.write(items_path, items)
        for label in self.batch.glob('labels-identity-*.json'):
            doc=json.loads(label.read_text())
            for row in doc['judgments']:
                if row['id'] == item['id']:
                    row['rolemap']=item['rolemap']
                    row['displayed_asset_ids']=item['asset_ids']
            queue.write(label, doc)
        self.assertNotEqual(original_a, item['rolemap']['A'])
        with self.assertRaises(AssertionError):
            queue.receipts(self.root, 'test', identity_mode=True)

    def test_identity_acceptability_conflict_is_not_fabricated(self):
        self.identity_labels()
        keys={v['id']:v for v in json.loads((self.batch/'private-key.json').read_text())}
        target=None
        for label in self.batch.glob('labels-identity-*.json'):
            doc=json.loads(label.read_text())
            for row in doc['judgments']:
                if (keys[row['id']]['request']['image_sha256']['candidate'] ==
                        keys[row['id']]['request']['image_sha256']['anchor']):
                    row['acceptable_B']='yes'
                    target=label
                    break
            if target:
                queue.write(label, doc)
                break
        self.assertIsNotNone(target)
        with self.assertRaises(AssertionError):
            queue.receipts(self.root, 'test', identity_mode=True)

        # Re-stage a clean diagnostic, then mutate the content-addressed file.
        label.unlink()
        for path in self.batch.glob('labels-identity-*.json'):
            path.unlink()
        queue.stage(self.root, 'test', identity_mode=True)
        asset=json.loads((self.batch/'assets.json').read_text())['assets'][0]
        path=Path(asset['path'])
        path.write_bytes(path.read_bytes()+b'tampered')
        with self.assertRaises(AssertionError):
            queue.receipts(self.root, 'test', identity_mode=True)


if __name__=='__main__':unittest.main()
