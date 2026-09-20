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
        manifests=sorted(self.batch.glob('judge-*-assets.json'))
        self.assertTrue(manifests)
        for manifest in manifests:
            for item in json.loads(manifest.read_text()):
                self.assertEqual(set(item), {'id','ordinal','rolemap','asset_ids','display_order'})
                self.assertEqual(item['asset_ids'],
                                 list(dict.fromkeys(item['rolemap'][role]
                                                    for role in ('reference','A','B'))))
                self.assertNotIn('path', item)

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
