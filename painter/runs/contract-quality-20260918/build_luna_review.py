"""Blind all matched painting pairs using preserved node PNGs, without rendering."""
import argparse
import hashlib
import itertools
import json
import random
import shutil
from pathlib import Path


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    assert not (output / 'pairs.json').exists(), 'Preserve an existing review unchanged'
    key_path = output.parent / 'luna-review-key.json'
    assert not key_path.exists(), 'Existing private comparison key'
    tasks = json.loads((root / 'evaluation-tasks.json').read_text())
    assert len(tasks) == 32
    models = ['parent', 'midpoint', 'trained']
    outcomes = {}
    for model in models:
        summary = json.loads((root / 'evaluation' / model / 'summary.json').read_text())
        assert summary['completed'] == summary['expected'] == 128
        outcomes[model] = {(r['task']['id'], r['sample']): r for r in summary['results']}
        assert len(outcomes[model]) == 128

    def local(path):
        f = (root / str(path).removeprefix('/root/painter/')).resolve()
        assert f.is_relative_to(root) and f.is_file(), path
        return f

    rng = random.Random(918384)
    specifications = []
    for left, right in itertools.combinations(models, 2):
        for group in ['previous', 'new_composition']:
            cases = [(task, k) for task in tasks for k in range(4)
                     if ('new_composition' if task['id'].startswith('v3-') else 'previous') == group]
            assert len(cases) % 2 == 0
            flips = [False] * (len(cases)//2) + [True] * (len(cases)//2)
            rng.shuffle(flips)
            for (task, sample), flip in zip(cases, flips):
                a, b = (right, left) if flip else (left, right)
                specifications.append(dict(task=task, sample=sample, group=group,
                                           comparison=left + '_vs_' + right, A=a, B=b))
    rng.shuffle(specifications)
    output.mkdir(parents=True, exist_ok=True)
    pairs, key, mechanical = [], [], []
    for number, spec in enumerate(specifications, 1):
        ident = f'pair-{number:03d}'
        folder = output / ident
        folder.mkdir()
        task, sample = spec['task'], spec['sample']
        paths = {'reference': local(task['reference'])}
        available = {}
        for side in ['A', 'B']:
            row = outcomes[spec[side]][(task['id'], sample)]
            available[side] = bool(row.get('valid_canvas') and row.get('canvas'))
            if available[side]:
                paths[side] = local(row['canvas'])
        files, hashes = {}, {}
        for side, source in paths.items():
            dest = folder / (side + '.png')
            shutil.copyfile(source, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
            files[side], hashes[side] = str(dest), digest
        key.append(dict(id=ident, task_id=task['id'], sample=sample, group=spec['group'],
                        family=task['family'],kind=task['kind'],source_group=task['source_group'], comparison=spec['comparison'], A=spec['A'], B=spec['B'],
                        image_sha256=hashes, available=available))
        if all(available.values()):
            pairs.append(dict(id=ident, **files))
        else:
            # Missing/failed canvases remain explicit failures, never silently
            # omitted from denominators or replaced with invented images.
            mechanical.append(dict(id=ident, available=available, status='missing_or_invalid_canvas',
                                   images=files, manual_resolution_required=True))
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')
    write(output / 'pairs.json', pairs)
    write(output / 'mechanical-failures.json', mechanical)
    write(key_path, key)
    for start in range(0, len(pairs), 24):
        write(output / f'batch-{start // 24 + 1:02d}.json', pairs[start:start + 24])
    report = dict(total_pairs=len(specifications), ready_visual_pairs=len(pairs),
                  failed_canvas_pairs=len(mechanical), reference_groups=24, tasks=32, samples_per_reference=4,
                  comparisons=['parent_vs_midpoint', 'parent_vs_trained', 'midpoint_vs_trained'],
                  all_painting_samples_included=True,all_repair_and_finish_samples_included=True, images_copied_without_transformation=True,
                  review_status='pending actual Luna inspection')
    write(output / 'manifest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output)))
