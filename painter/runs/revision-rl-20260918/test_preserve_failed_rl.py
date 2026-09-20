"""Preservation allowlist tests; no uploads, weights, or node operations."""
from pathlib import Path
import tempfile
import unittest

import preserve_failed_rl as preserve


class PreserveEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'preservation'

    def tearDown(self):
        self.temp.cleanup()

    def test_evaluation_and_rl_state_references_are_collected_safely(self):
        evaluation = self.root / 'evaluation'
        evaluation.mkdir()
        (evaluation / 'summary.json').write_text('{"status":"matched"}\n')
        (evaluation / 'reference.png').write_bytes(b'reference evidence')
        (evaluation / 'hf-token.txt').write_bytes(b'credential-like data')
        (evaluation / 'weights').mkdir()
        (evaluation / 'weights' / 'model.bin').write_bytes(b'weight bytes')

        states = self.root / 'rl-states' / 'state-00'
        states.mkdir(parents=True)
        (states / 'reference.png').write_bytes(b'rl reference')
        (states / 'program.js').write_text('setup();\n')
        (states / 'weights').mkdir()
        (states / 'weights' / 'adapter.safetensors').write_bytes(b'weights')
        (self.root / 'rl-states.json').write_text('[{"id":"state-00"}]\n')
        (self.root / 'evaluation-manifest.json').write_text('{"schema":1}\n')
        (self.root / 'evaluation-campaign.json').write_text('{"seed":917}\n')

        symlink = evaluation / 'linked-reference.png'
        try:
            symlink.symlink_to(evaluation / 'reference.png')
        except (OSError, NotImplementedError):
            symlink = None

        files = preserve.collect_evidence(self.root, 'unused-run', self.output)
        names = set(files)
        self.assertIn('evaluation/summary.json', names)
        self.assertIn('evaluation/reference.png', names)
        self.assertIn('rl-states/state-00/reference.png', names)
        self.assertIn('rl-states/state-00/program.js', names)
        self.assertIn('rl-states.json', names)
        self.assertIn('evaluation-manifest.json', names)
        self.assertIn('evaluation-campaign.json', names)
        self.assertNotIn('evaluation/hf-token.txt', names)
        self.assertNotIn('rl-states/state-00/weights/adapter.safetensors', names)
        if symlink is not None:
            self.assertNotIn('evaluation/linked-reference.png', names)


if __name__ == '__main__':
    unittest.main()
