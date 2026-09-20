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
                f=self.root/f'{n}-{role}.png';f.write_bytes(f'{n}-{role}'.encode())
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


if __name__=='__main__':unittest.main()
