"""Pure CLI tests for matched evaluation plumbing; no model or renderer calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

run_dir = Path(__file__).parent
sys.path.insert(0, str(run_dir / 'inputs'))
sys.path.insert(0, str(run_dir))
import evaluate_tasks
import package_async
sys.path.pop(0)
sys.path.pop(0)


class EvaluateTasksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_default_and_absolute_input_resolution(self):
        old_root = evaluate_tasks.P
        evaluate_tasks.P = self.root
        try:
            self.assertEqual(evaluate_tasks.resolve_input('campaign.json'),
                             self.root / 'campaign.json')
            absolute = self.root / 'fresh-campaign.json'
            self.assertEqual(evaluate_tasks.resolve_input(str(absolute)), absolute)
        finally:
            evaluate_tasks.P = old_root

    def test_base_url_and_immutable_config_are_propagated(self):
        (self.root / 'evaluation-tasks.json').write_text(json.dumps([
            {'id': 'heldout-000', 'reference': 'reference.png'}]))
        (self.root / 'reference.png').write_bytes(b'reference')
        config = self.root / 'fresh-campaign.json'
        config.write_text(json.dumps({'operational_limits': {'max_turns': 1}}))
        original_config = config.read_bytes()
        calls = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return {'valid_canvas': True, 'finished': True}

        old_root = evaluate_tasks.P
        evaluate_tasks.P = self.root
        argv = ['evaluate_tasks.py', '--model', 'candidate', '--samples', '1',
                '--workers', '1', '--config', str(config),
                '--base-url', 'http://127.0.0.1:8001/v1']
        try:
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(evaluate_tasks.sys, 'platform', 'linux'), \
                    mock.patch.object(evaluate_tasks, 'run', side_effect=fake_run), \
                    mock.patch.object(evaluate_tasks, 'atomic'):
                evaluate_tasks.main()
        finally:
            evaluate_tasks.P = old_root

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][2], 'http://127.0.0.1:8001/v1')
        self.assertEqual(calls[0][0][6], json.loads(original_config))
        self.assertEqual(config.read_bytes(), original_config)

    def test_evaluation_overlay_is_packaged_for_future_bundles(self):
        self.assertIn('evaluate_tasks.py', package_async.OVERLAYS)


if __name__ == '__main__':
    unittest.main()
