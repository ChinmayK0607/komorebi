"""Build the portable restart bundle without credentials or old optimizer state."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
OVERLAYS = ('setup_async_node.sh', 'activate_async.sh', 'launch_async.sh', 'prepare_async.py',
            'run_from_sft.py', 'train_from_sft.py', 'render_actions.py', 'taskset_impl.py',
            'quality_signal.py', 'reward_queue.py',
            'supervise_async.py', 'test_async_contract.py', 'test_quality_signal.py',
            'test_luna_queue.py')


def main():
    inputs = ROOT / 'inputs'
    files = {str(p.relative_to(inputs)): p for p in inputs.rglob('*')
             if p.is_file() and not p.is_symlink()
             and not {'__pycache__', '.git', 'private'} & set(p.relative_to(inputs).parts)
             and p.name not in {'pipeline.py', 'sample_phase.sh', 'input-manifest.json'}}
    files.update({name: ROOT / name for name in OVERLAYS})
    files['EXPERIMENT.json'] = ROOT / 'EXPERIMENT.json'
    # Reuse only image-hash-conditioned receipts, never stale training traces.
    for p in (ROOT/'judge-queue').glob('*/receipt.json'):
        receipt = json.loads(p.read_text())
        if receipt.get('judge_model') == 'gpt-5.6-luna' and receipt.get('reasoning_effort') == 'low':
            files[str(p.relative_to(ROOT))] = p
    target = ROOT/'async-restart-20260920.tar.gz'
    manifest = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                for name, path in files.items()}
    with tarfile.open(target, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    receipt = dict(path=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   bytes=target.stat().st_size, files=manifest)
    target.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}))


if __name__ == '__main__':
    main()
