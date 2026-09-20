"""Freeze/check the small shared foundation, independently of experiment knobs.

One automatic checksum check on setup; no manual approval gate. Historical run
snapshots are never rewritten. This covers the painter interface and renderer,
not a claim that distributed training has been packaged or validated here.
"""
import argparse
import hashlib
import json
from pathlib import Path
from contract import identity

ROOT=Path(__file__).resolve().parent
FILES=('contract.py','client.py','rollout.py','render_demos.py','recovery_data.py',
       'compile_teacher.cjs','pairwise.py','reward.py','foundation_lock.py')


def snapshot(root=ROOT):
    root=Path(root)
    files=[root/name for name in FILES]
    files.extend(sorted(p for p in (root/'vendor/integrations/watercolour').rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.js','.html')))
    return dict(schema=1,contract=identity(),files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def check(path=None):
    path=Path(path) if path else ROOT/'foundation-lock.json'
    expected=json.loads(path.read_text());actual=snapshot()
    names=set(expected['files'])|set(actual['files'])
    changed=sorted(n for n in names if expected['files'].get(n)!=actual['files'].get(n))
    if expected.get('contract')!=actual['contract'] or changed:
        raise ValueError('Foundation differs from its snapshot: '+', '.join(changed)+'. Version the change and run matched baselines; do not silently reuse an old run.')
    return dict(verified=True,contract=actual['contract'],files=len(actual['files']),
                lock_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true',help='Explicitly record the current tested foundation');args=p.parse_args()
    if args.snapshot:
        (ROOT/'foundation-lock.json').write_text(json.dumps(snapshot(),indent=2)+'\n')
    print(json.dumps(check(),indent=2))
