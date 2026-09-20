"""Bounded, resumable controller transport tests; no network or credentials."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import luna_queue_ctl as ctl


def archive_bytes(batch='batch'):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        data = b'{"pairs": []}\n'
        member = tarfile.TarInfo(f'{batch}/pairs.json')
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


class LunaQueueTransportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.batch = 'batch'
        self.payload = archive_bytes(self.batch)
        self.metadata = {
            'archive': '/root/painter/reward-batches/batch.tar.gz',
            'sha256': hashlib.sha256(self.payload).hexdigest(),
            'requests': 1,
            'pairs': 2,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_streams_to_part_then_atomically_installs_verified_archive(self):
        def fake_run(command, **kwargs):
            self.assertEqual(kwargs['timeout'], ctl.ARCHIVE_TRANSFER_TIMEOUT_SECONDS)
            kwargs['stdout'].write(self.payload)
            return subprocess.CompletedProcess(command, 0, b'', b'')

        with mock.patch.object(ctl.subprocess, 'run', side_effect=fake_run):
            archive = ctl._download_archive(['ssh', 'test'], self.metadata,
                                             self.root, self.batch)
        self.assertEqual(archive.read_bytes(), self.payload)
        self.assertEqual(list(self.root.glob('*.tar.gz.part')), [])

    def test_bad_transfer_is_removed_and_never_installed(self):
        def fake_run(command, **kwargs):
            kwargs['stdout'].write(b'corrupt')
            return subprocess.CompletedProcess(command, 0, b'', b'')

        with mock.patch.object(ctl.subprocess, 'run', side_effect=fake_run):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                ctl._download_archive(['ssh', 'test'], self.metadata,
                                      self.root, self.batch)
        self.assertFalse((self.root / 'batch.tar.gz').exists())
        self.assertEqual(list(self.root.glob('*.tar.gz.part')), [])

    def test_existing_different_archive_is_not_overwritten(self):
        archive = self.root / 'batch.tar.gz'
        archive.write_bytes(b'old evidence')
        with mock.patch.object(ctl.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'differs'):
                ctl._download_archive(['ssh', 'test'], self.metadata,
                                      self.root, self.batch)
        run.assert_not_called()
        self.assertEqual(archive.read_bytes(), b'old evidence')

    def test_existing_remote_export_is_reused_without_exporting_again(self):
        with mock.patch.object(ctl, '_remote_metadata', return_value=self.metadata), \
             mock.patch.object(ctl, '_remote_export') as export:
            result = ctl._metadata_for_pull(['ssh', 'test'], self.batch, 8)
        self.assertEqual(result['sha256'], self.metadata['sha256'])
        export.assert_not_called()

    def test_remote_metadata_binds_existing_archive_and_receipt(self):
        archive_path = '/root/painter/reward-batches/batch.tar.gz'
        receipt_path = '/root/painter/reward-batches/batch.tar.receipt.json'
        receipt = json.dumps({
            'sha256': self.metadata['sha256'], 'requests': 1, 'pairs': 2,
        }).encode()

        def fake_run(command, **kwargs):
            if command[-2:] in (['-f', archive_path], ['-f', receipt_path]):
                return subprocess.CompletedProcess(command, 0, b'', b'')
            if command[-2:] == ['--', receipt_path]:
                return subprocess.CompletedProcess(command, 0, receipt, b'')
            self.fail(f'unexpected transport command: {command}')

        with mock.patch.object(ctl.subprocess, 'run', side_effect=fake_run):
            result = ctl._remote_metadata(['ssh', 'test'], self.batch)
        self.assertEqual(result['archive'], archive_path)
        self.assertEqual(result['pairs'], 2)

    def test_existing_batch_directory_is_not_overwritten(self):
        archive = self.root / 'batch.tar.gz'
        archive.write_bytes(self.payload)
        destination = self.root / 'reward-batches' / self.batch
        destination.mkdir(parents=True)
        (destination / 'pairs.json').write_text('different\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            ctl._extract_archive(archive, self.root, self.batch)
        self.assertEqual((destination / 'pairs.json').read_text(), 'different\n')

    def test_remote_python_has_a_hard_timeout_for_push_and_status(self):
        completed = subprocess.CompletedProcess(['ssh'], 0, stdout='ok', stderr='')
        with mock.patch.object(ctl.subprocess, 'run', return_value=completed) as run:
            self.assertEqual(ctl._remote_python(['ssh', 'test'], 'pass'), 'ok')
        self.assertEqual(run.call_args.kwargs['timeout'], ctl.REMOTE_COMMAND_TIMEOUT_SECONDS)

    def test_pull_cli_prints_success_metadata_once(self):
        local_archive = self.root / 'batch.tar.gz'
        local_archive.write_bytes(self.payload)
        output = io.StringIO()
        argv = ['luna_queue_ctl.py', 'pull', self.batch, '--host', 'test', '--port', '1',
                '--root', str(self.root)]
        with mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(ctl, '_metadata_for_pull', return_value=self.metadata), \
             mock.patch.object(ctl, '_download_archive', return_value=local_archive), \
             mock.patch.object(ctl, '_extract_archive'), \
             mock.patch.object(ctl.reward_queue, 'stage'), \
             redirect_stdout(output):
            ctl.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['batch'], self.batch)
        self.assertEqual(result['sha256'], self.metadata['sha256'])


if __name__ == '__main__':
    unittest.main()
