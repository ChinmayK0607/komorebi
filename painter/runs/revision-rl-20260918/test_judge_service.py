import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from judge_service import process_request
from openrouter_judge import parse_json


class JudgeServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        images, hashes = {}, {}
        for role in ('reference', 'anchor', 'candidate'):
            p = self.root / (role + '.png')
            p.write_bytes(role.encode())
            images[role] = p.name
            hashes[role] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.request = dict(id=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                            images=images, image_sha256=hashes)
        self.folder = self.root / 'judge-queue' / self.request['id']
        self.folder.mkdir(parents=True)
        (self.folder/'request.json').write_text(json.dumps(self.request))

    def compare(self, pair, key, model, attempts):
        winner = 'A' if pair['A'].endswith('candidate.png') else 'B'
        return dict(winner=winner, confidence='high', acceptable_A='yes' if winner=='A' else 'no',
                    acceptable_B='yes' if winner=='B' else 'no', reason='Candidate preserves structure')

    def test_receipt_is_reconciled_and_cached(self):
        r = process_request(self.root, self.folder, 'test', compare=self.compare)
        self.assertEqual(r['judgment']['delta'], 1)
        def no_call(*args, **kwargs):
            self.fail('Already completed receipt was rejudged')
        self.assertEqual(process_request(self.root, self.folder, 'test', compare=no_call), r)

    def test_first_orientation_survives_second_api_failure(self):
        calls = []
        def flaky(*args, **kwargs):
            calls.append(args[0]['id'])
            if len(calls) == 2:
                raise RuntimeError('Provider unavailable')
            return self.compare(*args, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'Provider unavailable'):
            process_request(self.root, self.folder, 'test', compare=flaky)
        self.assertFalse((self.folder/'receipt.json').exists())
        self.assertEqual(len(list(self.folder.glob('judge-*.json'))), 1)
        resumed = []
        def retry(*args, **kwargs):
            resumed.append(args[0]['id'])
            return self.compare(*args, **kwargs)
        process_request(self.root, self.folder, 'test', compare=retry)
        self.assertEqual(resumed, calls[1:])

    def test_changed_image_is_rejected_even_with_receipt(self):
        process_request(self.root, self.folder, 'test', compare=self.compare)
        (self.root/'candidate.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'image changed'):
            process_request(self.root, self.folder, 'test', compare=self.compare)

    def test_missing_field_with_extra_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            parse_json(json.dumps(dict(winner='A', confidence='high', acceptable_A='yes',
                                       acceptable_B='no', extra='does not replace reason')))


if __name__ == '__main__':
    unittest.main()
