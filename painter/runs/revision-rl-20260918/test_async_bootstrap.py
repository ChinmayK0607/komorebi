"""Pure tests for fresh async configs and bounded bootstrap transport."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import deploy_async
import package_async
import prepare_async


class AsyncBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_fresh_experiment_and_bounded_training_limits_are_honored(self):
        config = self.root / 'fresh-experiment.json'
        config.write_text(json.dumps({
            'run_name': 'fresh-run',
            'restart_run_name': 'fresh-run',
            'max_steps': 2,
            'checkpoint_interval': 1,
        }))
        experiment = package_async.load_experiment(config)
        self.assertEqual(prepare_async.training_limits(experiment), (2, 1))
        self.assertEqual(package_async.load_experiment(config)['max_steps'], 2)

    def test_stream_file_uses_authenticated_stdin_and_bounded_calls(self):
        source = self.root / 'input.bin'
        source.write_bytes(b'bounded transfer data')
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            stream = kwargs.get('stdin')
            if stream is not None:
                self.assertEqual(stream.read(), source.read_bytes())
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(deploy_async.subprocess, 'run', side_effect=fake_run):
            deploy_async._stream_file(
                ['ssh', '-i', 'key', 'root@test'], source,
                '/root/painter/private/hf-token',
                '/root/painter/private/.hf-token.part')

        self.assertEqual(len(calls), 4)
        self.assertTrue(all('scp' not in call[0] for call in calls))
        self.assertEqual(calls[0][0][-3:], ['tee', '--',
                                            '/root/painter/private/.hf-token.part'])
        self.assertEqual(calls[0][1]['timeout'], deploy_async.FILE_TRANSFER_TIMEOUT_SECONDS)
        self.assertIs(calls[0][1]['stdout'], deploy_async.subprocess.DEVNULL)
        self.assertEqual(calls[1][0][-3:], ['chmod', '600',
                                            '/root/painter/private/.hf-token.part'])
        self.assertEqual(calls[2][0][-5:], ['mv', '-f', '--',
                                            '/root/painter/private/.hf-token.part',
                                            '/root/painter/private/hf-token'])
        self.assertEqual(calls[3][0][-3:], ['chmod', '600',
                                            '/root/painter/private/hf-token'])
        self.assertTrue(all(call[1]['timeout'] <= deploy_async.FILE_TRANSFER_TIMEOUT_SECONDS
                            for call in calls))

    def test_deploy_passes_fresh_experiment_when_building_bundle(self):
        experiment = self.root / 'fresh-experiment.json'
        experiment.write_text(json.dumps({
            'run_name': 'fresh-run',
            'restart_run_name': 'fresh-run',
            'max_steps': 2,
            'checkpoint_interval': 1,
        }))
        token = self.root / 'input.bin'
        token.write_bytes(b'bounded transfer data')
        bundle = self.root / 'async-restart-fresh-run.tar.gz'
        manifest = bundle.with_suffix('.manifest.json')

        old_root = deploy_async.ROOT
        deploy_async.ROOT = self.root
        try:
            def fake_run(command, **kwargs):
                if command and command[0] == 'python3':
                    bundle.write_bytes(b'bundle')
                    manifest.write_text(json.dumps({
                        'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    }))
                return subprocess.CompletedProcess(command, 0)

            argv = [
                'deploy_async.py', '--host', 'test', '--port', '1',
                '--hf-token-file', str(token), '--experiment', str(experiment),
                '--setup-only',
            ]
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(deploy_async.subprocess, 'run', side_effect=fake_run) as run, \
                    mock.patch.object(deploy_async, '_stream_file') as stream:
                deploy_async.main()
            package_call = next(call.args[0] for call in run.call_args_list
                                if call.args[0] and call.args[0][0] == 'python3')
        finally:
            deploy_async.ROOT = old_root

        self.assertIn('--experiment', package_call)
        self.assertEqual(package_call[package_call.index('--experiment') + 1],
                         str(experiment.resolve()))
        self.assertEqual(stream.call_count, 2)


if __name__ == '__main__':
    unittest.main()
