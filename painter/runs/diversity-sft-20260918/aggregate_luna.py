"""Validate blinded visual labels and summarize matched reference comparisons."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def summarize(rows):
    counts = collections.Counter(r['outcome'] for r in rows)
    grouped = collections.defaultdict(list)
    for row in rows:
        if row['outcome'] != 'masked':
            grouped[row['task_id']].append({'win': 1, 'tie': .5, 'loss': 0}[row['outcome']])
    scores = [sum(v) / len(v) for v in grouped.values()]
    interval = None
    if scores:
        rng = random.Random(918384)
        boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(10000))
        interval = [boot[249], boot[9749]]
    return dict(total=len(rows), wins=counts['win'], ties=counts['tie'],
                losses=counts['loss'], masked=counts['masked'],
                reference_groups=len(grouped),
                mean_reference_preference=sum(scores)/len(scores) if scores else None,
                reference_bootstrap_95=interval)


def aggregate(root):
    root = Path(root).resolve()
    pairs = json.loads((root/'luna-review/pairs.json').read_text())
    pair_map = {r['id']: r for r in pairs}
    key = json.loads((root/'luna-review-key.json').read_text())
    key_map = {r['id']: r for r in key}
    assert len(key_map) == 144
    labels = {}
    files = sorted((root/'luna-review').glob('labels-*.json'))
    for file in files:
        review = json.loads(file.read_text())
        assert review['judge_model'] == 'gpt-5.6-luna', file
        for row in review['judgments']:
            ident = row['id']
            assert ident in pair_map and ident not in labels, ident
            assert row['winner'] in ['A', 'B', 'tie', 'uncertain']
            assert row['acceptable_A'] in ['yes', 'no', 'uncertain']
            assert row['acceptable_B'] in ['yes', 'no', 'uncertain']
            assert row['confidence'] in ['high', 'medium', 'low']
            assert row['reason'].strip()
            expected = [pair_map[ident][k] for k in ['reference', 'A', 'B']]
            assert sorted(row['reviewed_images']) == sorted(expected), ident
            for side, path in zip(['reference', 'A', 'B'], expected):
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == key_map[ident]['image_sha256'][side]
            labels[ident] = row
    assert set(labels) == set(pair_map), f'Missing labels: {set(pair_map)-set(labels)}'
    rows = []
    for item in key:
        row = dict(item)
        label = labels.get(item['id'])
        left, right = item['comparison'].split('_vs_')
        row['preferred_model'] = item[label['winner']] if label and label['winner'] in ['A','B'] else None
        if label is None or label['winner'] == 'uncertain' or label['confidence'] == 'low':
            row['outcome'] = 'masked'
        elif label['winner'] == 'tie':
            row['outcome'] = 'tie'
        else:
            row['outcome'] = 'win' if row['preferred_model'] == right else 'loss'
        row['label'] = label
        rows.append(row)
    comparisons = {}
    for comparison in ['parent_vs_midpoint','parent_vs_trained','midpoint_vs_trained']:
        subset = [r for r in rows if r['comparison'] == comparison]
        comparisons[comparison] = dict(all=summarize(subset), **{
            g: summarize([r for r in subset if r['group'] == g])
            for g in ['previous','new_composition']})
    # Each painting appears in two independently blinded comparisons. Require
    # agreement for an absolute yes/no; expose inconsistent grades as uncertain.
    votes = collections.defaultdict(list)
    for row in rows:
        if not row['label']:
            continue
        for side in ['A', 'B']:
            grade = row['label']['acceptable_' + side]
            if row['label']['confidence'] == 'low':
                grade = 'uncertain'
            votes[(row[side], row['group'], row['task_id'], row['sample'])].append(grade)
    absolute = {}
    for model in ['parent', 'midpoint', 'trained']:
        absolute[model] = {}
        for group in ['all', 'previous', 'new_composition']:
            selected = [v for (m,g,_,_),v in votes.items() if m == model and (group == 'all' or g == group)]
            grades = ['yes' if v == ['yes','yes'] else 'no' if v == ['no','no'] else 'uncertain' for v in selected]
            absolute[model][group] = dict(collections.Counter(grades), total=len(grades))
    summary = dict(judge_model='gpt-5.6-luna', total_pairs=144,
                   visually_reviewed_pairs=len(labels),
                   interpretation='Wins favor the model named on the right of each comparison. Low confidence and uncertainty are masked. Samples are clustered by reference. Agent labels are not human ground truth.',
                   comparisons=comparisons, absolute_acceptability=absolute,
                   **comparisons['parent_vs_trained']['all'])
    (root/'luna-review/decoded-judgments.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'luna-review/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'collected-final/evidence/luna-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    print(json.dumps(aggregate(a.root),indent=2))
