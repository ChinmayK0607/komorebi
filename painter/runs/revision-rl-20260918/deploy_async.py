"""Deploy the pinned bundle to an already rented node and start supervised RL.

The provider owns the TTL; this script never rents a node or changes a lease.
Credentials are transferred separately and are excluded from the bundle.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--hf-token-file', type=Path, required=True)
    p.add_argument('--known-hosts', type=Path, default=ROOT/'known_hosts-luna-20260920')
    a = p.parse_args()
    if not a.hf_token_file.is_file():
        raise ValueError('Checkpoint credential required')
    subprocess.run(['python3', str(ROOT/'package_async.py')], check=True)
    common = ['-i', '/Users/chinmay/.ssh/id_ed25519', '-o', 'BatchMode=yes',
              '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20',
              '-o', 'UserKnownHostsFile='+str(a.known_hosts)]
    ssh = ['ssh', *common, '-p', str(a.port), 'root@'+a.host]
    scp = ['scp', *common, '-P', str(a.port)]
    subprocess.run(ssh + ['test ! -e /root/painter/training-ready.json && mkdir -p /root/painter/private /root/painter/logs && chmod 700 /root/painter/private'], check=True)
    archive = ROOT/'async-restart-20260920.tar.gz'
    subprocess.run(scp + [str(archive), 'root@'+a.host+':/root/painter/input-bundle.tar.gz'], check=True)
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
    subprocess.run(ssh+['python3 -'], input=code, text=True, check=True)
    subprocess.run(scp+[str(a.hf_token_file), 'root@'+a.host+':/root/painter/private/hf-token'], check=True)
    subprocess.run(ssh+['chmod 600 /root/painter/private/hf-token'], check=True)
    launch = '''cd /root/painter
nohup bash setup_async_node.sh /root/painter > logs/setup-controller.log 2>&1 < /dev/null &
nohup bash activate_async.sh /root/painter > logs/activation.log 2>&1 < /dev/null &
echo 'Setup and activation watcher started'
'''
    subprocess.run(ssh+[launch], check=True)


if __name__ == '__main__':
    main()
