"""Deploy the pinned bundle to an already rented node and start supervised RL.

The provider owns the TTL; this script never rents a node or changes a lease.
Credentials are transferred separately and are excluded from the bundle.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE_COMMAND_TIMEOUT_SECONDS = 120
FILE_TRANSFER_TIMEOUT_SECONDS = 300
REMOTE_BUNDLE_PATH = '/root/painter/input-bundle.tar.gz'
REMOTE_BUNDLE_PART = '/root/painter/.input-bundle.tar.gz.part'
REMOTE_TOKEN_PATH = '/root/painter/private/hf-token'
REMOTE_TOKEN_PART = '/root/painter/private/.hf-token.part'


def _stream_file(ssh, source, destination, temporary, *, timeout=FILE_TRANSFER_TIMEOUT_SECONDS):
    """Stream a local file over authenticated SSH, then atomically install it."""
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Transfer source is not a regular file: ' + str(source))
    try:
        with source.open('rb') as stream:
            # tee consumes stdin on the remote host; suppress its stdout so a
            # credential can never be echoed into the controller log.
            subprocess.run(ssh + ['tee', '--', temporary], stdin=stream,
                           stdout=subprocess.DEVNULL, timeout=timeout, check=True)
        subprocess.run(ssh + ['chmod', '600', temporary],
                       timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(ssh + ['mv', '-f', '--', temporary, destination],
                       timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(ssh + ['chmod', '600', destination],
                       timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True,
                       stdout=subprocess.DEVNULL)
    except Exception:
        # Do not leave a partially transferred credential behind after a
        # timeout or authentication failure. Cleanup is best effort and bounded.
        try:
            subprocess.run(ssh + ['rm', '-f', '--', temporary],
                           timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--hf-token-file', type=Path, required=True)
    p.add_argument('--run-id', help='Fresh run prefix; defaults to EXPERIMENT restart_run_name.')
    p.add_argument('--experiment', type=Path,
                   help='Fresh experiment config; defaults to the historical EXPERIMENT.json.')
    p.add_argument('--source-id', help='Immutable source baseline identifier to record in the bundle.')
    p.add_argument('--bundle', type=Path,
                   help='Existing bundle; otherwise build one with package_async.py.')
    p.add_argument('--identity-file', type=Path, default=Path('/Users/chinmay/.ssh/id_ed25519'))
    p.add_argument('--known-hosts', type=Path,
                   help='Dedicated known-hosts file for this pod.')
    p.add_argument('--setup-only', action='store_true',
                   help='Install/render/download only; leave activation/training stopped.')
    a = p.parse_args()
    if not a.hf_token_file.is_file():
        raise ValueError('Checkpoint credential required')
    experiment_path = (a.experiment or ROOT/'EXPERIMENT.json').resolve()
    if not experiment_path.is_file():
        raise ValueError('Experiment config not found: ' + str(experiment_path))
    experiment = json.loads(experiment_path.read_text())
    run_id = a.run_id or experiment.get('restart_run_name') or experiment.get('run_name')
    if not run_id or '/' in run_id or '\\' in run_id:
        raise ValueError('A simple --run-id is required')
    archive = (a.bundle or ROOT/f'async-restart-{run_id}.tar.gz').resolve()
    if a.bundle is None:
        package_command = ['python3', str(ROOT/'package_async.py'), '--run-id', run_id,
                           '--experiment', str(experiment_path), '--output', str(archive)]
        if a.source_id:
            package_command.extend(['--source-id', a.source_id])
        subprocess.run(package_command, check=True)
    if not archive.is_file():
        raise ValueError('Bundle not found: '+str(archive))
    known_hosts = a.known_hosts or (ROOT/f'known_hosts-{run_id}')
    common = ['-i', str(a.identity_file),
              '-o', 'BatchMode=yes',
              '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20',
              '-o', 'UserKnownHostsFile='+str(known_hosts)]
    ssh = ['ssh', *common, '-p', str(a.port), 'root@'+a.host]
    subprocess.run(ssh + ['test ! -e /root/painter/training-ready.json && mkdir -p /root/painter/private /root/painter/logs && chmod 700 /root/painter/private'],
                   timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True)
    _stream_file(ssh, archive, REMOTE_BUNDLE_PATH, REMOTE_BUNDLE_PART)
    receipt = json.loads(archive.with_suffix('.manifest.json').read_text())
    code = '''import hashlib,tarfile
from pathlib import Path
p=Path('/root/painter/input-bundle.tar.gz')
assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED
with tarfile.open(p) as archive:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in archive.getmembers())
 archive.extractall('/root/painter',filter='data')
print('Bundle hash verified and extracted',flush=True)
'''.replace('EXPECTED', repr(receipt['sha256']))
    subprocess.run(ssh+['python3 -'], input=code, text=True,
                   timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True)
    _stream_file(ssh, a.hf_token_file, REMOTE_TOKEN_PATH, REMOTE_TOKEN_PART)
    if a.setup_only:
        launch = '''cd /root/painter
nohup bash setup_async_node.sh /root/painter > logs/setup-controller.log 2>&1 < /dev/null &
echo 'Setup watcher started; activation is separate'
'''
    else:
        launch = '''cd /root/painter
nohup bash setup_async_node.sh /root/painter > logs/setup-controller.log 2>&1 < /dev/null &
nohup bash activate_async.sh /root/painter > logs/activation.log 2>&1 < /dev/null &
echo 'Setup and activation watchers started'
'''
    subprocess.run(ssh+[launch], timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True)


if __name__ == '__main__':
    main()
