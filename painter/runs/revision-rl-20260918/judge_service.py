"""Continuously grade the node's immutable image requests with the selected VLM.

No model inference or rendering runs in this service. Keys stay in the process
environment; successful orientations survive API failures and service restarts.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from openrouter_judge import MODEL, judge_one, parse_json
from quality_signal import reconcile


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def request_images(root, request):
    hashes = request['image_sha256']
    expected = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if request['id'] != expected:
        raise ValueError('Request id does not match image hashes')
    images = {}
    for role in ('reference', 'anchor', 'candidate'):
        path = (root / request['images'][role]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Image outside node workspace')
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[role]:
            raise ValueError('Request image changed: ' + role)
        images[role] = str(path)
    return images


def process_request(root, folder, key, model=MODEL, compare=judge_one):
    request = json.loads((folder / 'request.json').read_text())
    images = request_images(root, request)
    existing = folder / 'receipt.json'
    if existing.exists():
        receipt = json.loads(existing.read_text())
        if (receipt.get('id') != request['id'] or receipt.get('judge_model') != model
                or receipt.get('image_sha256') != request['image_sha256']):
            raise ValueError('Existing receipt provenance mismatch')
        return receipt
    labels = {}
    # Vary which view is sent first without exposing candidate identity to judge.
    order = ('A', 'B') if int(request['id'][0], 16) % 2 else ('B', 'A')
    for side in order:
        pair = {'id': request['id'] + '-' + side, 'reference': images['reference'],
                'A': images['candidate' if side == 'A' else 'anchor'],
                'B': images['anchor' if side == 'A' else 'candidate']}
        saved = folder / ('judge-' + side + '.json')
        if saved.exists():
            cached = json.loads(saved.read_text())
            if cached.get('model') != model or cached.get('hashes') != request['image_sha256']:
                raise ValueError('Cached orientation provenance mismatch')
            labels[side] = parse_json(json.dumps(cached['label']))
        else:
            labels[side] = compare(pair, key, model, attempts=3)
            atomic(saved, {'model': model, 'hashes': request['image_sha256'], 'label': labels[side]})
    # Recheck before accepting labels in case a file changed during the request.
    request_images(root, request)
    receipt = dict(id=request['id'], image_sha256=request['image_sha256'], judge_model=model,
                   candidate_A=labels['A'], candidate_B=labels['B'],
                   judgment=reconcile(labels['A'], labels['B']))
    atomic(existing, receipt)
    return receipt


def serve(root, key, workers, model):
    lock = (root / 'judge-service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inflight, retry_at = {}, {}
    completed = usable = failures = 0
    last_success = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            now = time.time()
            for future, folder in list(inflight.items()):
                if not future.done():
                    continue
                del inflight[future]
                try:
                    receipt = future.result()
                    completed += 1
                    usable += receipt['judgment'] is not None
                    last_success = now
                    (folder / 'judge-error.json').unlink(missing_ok=True)
                except Exception as exc:
                    failures += 1
                    retry_at[folder] = now + 60
                    atomic(folder / 'judge-error.json', {'at': now, 'error': str(exc)[:1000]})
            pending = [f.parent for f in sorted((root / 'judge-queue').glob('*/request.json'))
                       if not f.with_name('receipt.json').exists()]
            for folder in pending:
                if len(inflight) >= workers:
                    break
                if folder not in inflight.values() and retry_at.get(folder, 0) <= now:
                    inflight[pool.submit(process_request, root, folder, key, model)] = folder
            status = dict(at=now, pid=os.getpid(), judge_model=model, pending=len(pending),
                          inflight=len(inflight), completed=completed, usable=usable,
                          failures=failures, last_success=last_success)
            atomic(root / 'judge-service-status.json', status)
            time.sleep(2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--model', default=MODEL)
    args = p.parse_args()
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise SystemExit('OPENROUTER_API_KEY is required')
    serve(args.root.resolve(), key, max(1, args.workers), args.model)
