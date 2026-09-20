"""Move blinded image jobs and Luna receipts between controller and GPU node."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

import reward_queue

ROOT = Path(__file__).resolve().parent
REMOTE_COMMAND_TIMEOUT_SECONDS = 120
ARCHIVE_TRANSFER_TIMEOUT_SECONDS = 300
REMOTE_ROOT = Path('/root/painter')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command, *, timeout, **kwargs):
    """Run a transport command with a hard upper bound."""
    return subprocess.run(command, timeout=timeout, **kwargs)


def _remote_paths(batch):
    base = REMOTE_ROOT / 'reward-batches' / batch
    return str(base.with_suffix('.tar.gz')), str(base.with_suffix('.tar.receipt.json'))


def _validate_metadata(batch, metadata):
    archive, _ = _remote_paths(batch)
    if metadata.get('archive') not in (None, archive):
        raise ValueError('Remote archive path is not bound to the requested batch')
    digest = metadata.get('sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Remote archive receipt has an invalid SHA-256')
    for key in ('requests', 'pairs'):
        if not isinstance(metadata.get(key), int) or metadata[key] < 1:
            raise ValueError('Remote archive receipt has invalid counts')
    if metadata['pairs'] != metadata['requests'] * 2:
        raise ValueError('Remote archive receipt has incoherent request/pair counts')
    result = dict(metadata)
    result['archive'] = archive
    return result


def _remote_file_exists(ssh, path):
    result = _run(ssh + ['test', '-f', path], timeout=REMOTE_COMMAND_TIMEOUT_SECONDS,
                  capture_output=True)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise subprocess.CalledProcessError(result.returncode, result.args,
                                        result.stdout, result.stderr)


def _remote_metadata(ssh, batch):
    archive, receipt_path = _remote_paths(batch)
    if not (_remote_file_exists(ssh, archive) and _remote_file_exists(ssh, receipt_path)):
        return None
    result = _run(ssh + ['cat', '--', receipt_path], timeout=REMOTE_COMMAND_TIMEOUT_SECONDS,
                  capture_output=True, check=True)
    try:
        metadata = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Remote archive receipt is not JSON') from exc
    metadata['archive'] = archive
    return _validate_metadata(batch, metadata)


def _remote_export(ssh, batch, limit):
    result = _run(ssh + ['python3', str(REMOTE_ROOT / 'reward_queue.py'), 'export', batch,
                         '--limit', str(limit)], timeout=REMOTE_COMMAND_TIMEOUT_SECONDS,
                  capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    if output == 'No unassigned rewards':
        return None
    try:
        metadata = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError('Remote export did not return JSON metadata') from exc
    return _validate_metadata(batch, metadata)


def _metadata_for_pull(ssh, batch, limit):
    """Reuse a remote export when present; export only a genuinely new batch."""
    existing = _remote_metadata(ssh, batch)
    if existing is not None:
        return existing
    try:
        metadata = _remote_export(ssh, batch, limit)
    except subprocess.CalledProcessError:
        # Another controller may have completed the export between the probe and
        # export call. Reuse it only after independently verifying its receipt.
        existing = _remote_metadata(ssh, batch)
        if existing is None:
            raise
        return existing
    if metadata is None:
        return None
    receipt = _remote_metadata(ssh, batch)
    if receipt is None or any(receipt[key] != metadata[key] for key in ('sha256', 'requests', 'pairs')):
        raise ValueError('Remote export is missing a bound archive receipt')
    return receipt


def _download_archive(ssh, metadata, root, batch):
    """Stream a verified archive to a temporary file without clobbering evidence."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f'{batch}.tar.gz'
    expected = metadata['sha256']
    if archive.is_symlink() or archive.exists():
        if not archive.is_file() or sha(archive) != expected:
            raise ValueError('Existing local archive differs from remote evidence')
        return archive
    fd, temporary = tempfile.mkstemp(prefix=f'.{batch}.', suffix='.tar.gz.part', dir=root)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open('wb') as output:
            _run(ssh + ['cat', '--', metadata['archive']], timeout=ARCHIVE_TRANSFER_TIMEOUT_SECONDS,
                 stdout=output, stderr=subprocess.PIPE, check=True)
        if sha(temporary_path) != expected:
            raise ValueError('Transferred image archive hash mismatch')
        # An atomic hard-link install cannot clobber a concurrent successful
        # pull. The temporary file is on the same filesystem as the target.
        try:
            os.link(temporary_path, archive)
        except FileExistsError:
            if archive.is_file() and not archive.is_symlink() and sha(archive) == expected:
                return archive
            raise ValueError('Concurrent local archive differs from remote evidence')
        temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive


def _validate_members(members, batch):
    for member in members:
        path = Path(member.name)
        if (not path.parts or path.is_absolute() or '..' in path.parts
                or path.parts[0] != batch or not (member.isfile() or member.isdir())):
            raise ValueError('Unexpected archive member')


def _existing_members_match(source, destination):
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            if target.is_symlink() or not target.is_dir():
                raise ValueError('Existing batch evidence has a conflicting path')
        elif not target.is_file() or target.is_symlink() or sha(path) != sha(target):
            raise ValueError('Existing batch evidence differs from remote archive')


def _extract_archive(archive, root, batch):
    """Extract into a sibling temporary directory, then atomically install it."""
    destination_root = root / 'reward-batches'
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / batch
    with tempfile.TemporaryDirectory(prefix=f'.{batch}.extract-', dir=destination_root) as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            _validate_members(members, batch)
            tar.extractall(temporary_root, filter='data')
        extracted = temporary_root / batch
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError('Existing batch path is not a directory')
            _existing_members_match(extracted, destination)
            return destination
        os.replace(extracted, destination)
    return destination


def _remote_python(ssh, code):
    result = _run(ssh + ['python3', '-'], input=code, text=True, capture_output=True,
                  timeout=REMOTE_COMMAND_TIMEOUT_SECONDS, check=True)
    return result.stdout


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['pull','push','status'])
    p.add_argument('batch',nargs='?',default='')
    p.add_argument('--host',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--known-hosts',type=Path,default=ROOT/'known_hosts')
    p.add_argument('--run-id',help='Remote output run id for status inspection.')
    p.add_argument('--root',type=Path,default=ROOT/'luna-live')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--identity-mode',action='store_true',
                   help='Stage and verify unique-asset rolemap labels.')
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    if args.action!='status' and not re.fullmatch(r'[a-zA-Z0-9_-]+',args.batch):
        raise ValueError('Simple batch name required')
    common=['-i','/Users/chinmay/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=20',
            '-o','UserKnownHostsFile='+str(args.known_hosts)]
    ssh=['ssh',*common,'-p',str(args.port),'root@'+args.host]
    if args.action=='pull':
        metadata=_metadata_for_pull(ssh,args.batch,args.limit)
        if metadata is None:
            print('No unassigned rewards');return
        archive=_download_archive(ssh,metadata,args.root,args.batch)
        _extract_archive(archive,args.root,args.batch)
        reward_queue.stage(args.root,args.batch,args.identity_mode)
        print(json.dumps(dict(batch=args.batch,root=str(args.root),**metadata)))
    elif args.action=='push':
        reward_queue.receipts(args.root,args.batch,args.identity_mode)
        folder=args.root/'reward-batches'/args.batch/'receipts'
        receipts=[json.loads(f.read_text()) for f in sorted(folder.glob('*/receipt.json'))]
        payload=json.dumps(receipts)
        code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/root/painter')
from quality_signal import reconcile
from taskset_impl import atomic
root=Path('/root/painter');receipts=json.loads(PAYLOAD);identity_mode=IDENTITY_MODE;installed=0
for receipt in receipts:
 ident=receipt['id']
 assert len(ident)==64 and all(c in '0123456789abcdef' for c in ident)
 folder=root/'judge-queue'/ident
 request=json.loads((folder/'request.json').read_text())
 assert receipt['judge_model']=='gpt-5.6-luna' and receipt.get('reasoning_effort')=='low'
 assert receipt['image_sha256']==request['image_sha256']
 for role,path in request['images'].items():
  assert hashlib.sha256((root/path).read_bytes()).hexdigest()==request['image_sha256'][role]
 if identity_mode:
  assert receipt.get('visual_provenance',{}).get('mode')=='content_addressed_unique_assets'
  expected_equal=request['image_sha256']['candidate']==request['image_sha256']['anchor']
  deterministic=receipt.get('deterministic_identity',{})
  assert deterministic.get('candidate_bytes_equal')==expected_equal
  if expected_equal: assert deterministic.get('relative_winner')=='tie'
 assert receipt['judgment']==reconcile(receipt['candidate_A'],receipt['candidate_B'])
 atomic(folder/'receipt.json',receipt);installed+=1
print(json.dumps(dict(installed=installed)))
        '''.replace('PAYLOAD',repr(payload)).replace('IDENTITY_MODE',repr(args.identity_mode))
        # Use Prime's environment only for its taskset dependency; no ML runs.
        code=code.replace('from taskset_impl import atomic','from supervise_async import atomic')
        print(_remote_python(ssh,code).strip())
    else:
        code='''import json,subprocess
from pathlib import Path
p=Path('/root/painter');out={}
for name in ['training-ready.json','training-setup.exit','supervisor-status.json','judge-service-status.json','async-launch.exit','async-activation.exit']:
 f=p/name
 if f.exists():
  try:out[name]=json.loads(f.read_text())
  except ValueError:out[name]=f.read_text()[-200:]
q=list((p/'judge-queue').glob('*/request.json'))
out['queue']={'total':len(q),'unassigned':sum(not f.with_name('assigned.json').exists() and not f.with_name('receipt.json').exists() for f in q),'receipts':sum(f.with_name('receipt.json').exists() for f in q)}
run_id=RUN_ID
if not run_id:
    experiment=json.loads((p/'EXPERIMENT.json').read_text())
    run_id=experiment.get('restart_run_name') or experiment.get('run_name')
run=p/'async-output'/run_id
for f in run.glob('**/optimizer-update-audit.jsonl'):
 rows=f.read_text().splitlines()
 if rows:
  row=json.loads(rows[-1]);out['optimizer']={k:v for k,v in row.items() if k!='tensors'}
out['checkpoints']=[json.loads(f.read_text()) for f in run.glob('artifacts/adapters/step_*/hf-upload.json')]
for name in ['logs/activation.log','logs/setup-controller.log','logs/download.log','logs/prime-install.log']:
 f=p/name
 if f.exists():out[name]=f.read_text()[-1200:]
for name in ['trainer.log','orchestrator.log','inference.log']:
 files=sorted(run.glob('logs/attempt_*/'+name))
 if files:out[name]=files[-1].read_text()[-2000:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print(json.dumps(out,indent=2))
'''
        code=code.replace('RUN_ID',repr(args.run_id)).replace('IDENTITY_MODE',repr(args.identity_mode))
        output=_remote_python(ssh,code);(args.root/'latest-status.json').write_text(output);print(output)


if __name__=='__main__':main()
