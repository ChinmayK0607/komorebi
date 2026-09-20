"""Build the portable restart bundle without credentials or old optimizer state."""
import hashlib
import json
import argparse
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
OVERLAYS = ('setup_async_node.sh', 'activate_async.sh', 'launch_async.sh', 'prepare_async.py',
            'run_from_sft.py', 'train_from_sft.py', 'render_actions.py', 'taskset_impl.py',
            'quality_signal.py', 'reward_queue.py',
            'supervise_async.py', 'test_async_contract.py', 'test_quality_signal.py',
            'test_luna_queue.py')


def main():
    parser = argparse.ArgumentParser(description='Build a portable async RL bundle.')
    parser.add_argument('--run-id', help='Fresh output/Hugging Face run prefix.')
    parser.add_argument('--source-id', help='Immutable source baseline identifier to record in the override.')
    parser.add_argument('--output', type=Path,
                        help='Bundle path; defaults to async-restart-<run-id>.tar.gz.')
    args = parser.parse_args()
    inputs = ROOT / 'inputs'
    files = {str(p.relative_to(inputs)): p for p in inputs.rglob('*')
             if p.is_file() and not p.is_symlink()
             and not {'__pycache__', '.git', 'private'} & set(p.relative_to(inputs).parts)
             and p.name not in {'pipeline.py', 'sample_phase.sh', 'input-manifest.json'}}
    files.update({name: ROOT / name for name in OVERLAYS})
    experiment = json.loads((ROOT/'EXPERIMENT.json').read_text())
    run_id = args.run_id or experiment.get('restart_run_name') or experiment.get('run_name')
    if not run_id or '/' in run_id or '\\' in run_id or run_id in {'.', '..'}:
        raise ValueError('A simple run id is required')
    # The override is bundled only; the historical source manifest remains untouched.
    experiment['run_name'] = run_id
    experiment['restart_run_name'] = run_id
    if args.source_id:
        experiment['source_baseline'] = args.source_id
    with tempfile.TemporaryDirectory(prefix='painter-package-') as temp:
        override = Path(temp) / 'EXPERIMENT.json'
        override.write_text(json.dumps(experiment, indent=2) + '\n')
        files['EXPERIMENT.json'] = override
        # A fresh campaign must never inherit old rewards or optimizer traces.
        target = (args.output or ROOT/f'async-restart-{run_id}.tar.gz').resolve()
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
