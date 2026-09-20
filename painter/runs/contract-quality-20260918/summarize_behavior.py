"""Summarize preserved rollout metadata; performs no inference or rendering."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(rows):
    turns = [t for r in rows for t in r.get('turns', [])]
    paint = [t for t in turns if t.get('action') == 'paint' and t.get('valid')]
    tokens = [t['tokens'] for t in turns if isinstance(t.get('tokens'), (int, float))]
    paint_tokens = [t['tokens'] for t in paint if isinstance(t.get('tokens'), (int, float))]
    return {
        'episodes': len(rows),
        'valid_canvases': sum(bool(r.get('valid_canvas')) for r in rows),
        'finished': sum(bool(r.get('finished')) for r in rows),
        'operational_cutoffs': sum(bool(r.get('operational_cutoff')) for r in rows),
        'mean_turns': statistics.mean(len(r.get('turns', [])) for r in rows),
        'mean_episode_tokens': statistics.mean(r.get('generated_tokens', 0) for r in rows),
        'mean_tokens_per_turn': statistics.mean(tokens) if tokens else None,
        'median_tokens_per_paint_turn': statistics.median(paint_tokens) if paint_tokens else None,
        'invalid_turns': sum(not t.get('valid', False) for t in turns),
        'successful_paint_turns': len(paint),
        'paint_turns_after_observation': sum(bool(t.get('observed_canvas')) for t in paint),
        'unchanged_paint_turns': sum(bool(t.get('no_change')) for t in paint),
        'deferred_finish_requests': sum(bool(t.get('finish_deferred')) for t in paint),
        'turn_count_histogram': dict(collections.Counter(len(r.get('turns', [])) for r in rows)),
        'errors': dict(collections.Counter(t['error'] for t in turns if t.get('error'))),
    }


def build(root):
    result = {}
    for model in ['parent', 'midpoint', 'trained']:
        source = json.loads((root/'evaluation'/model/'summary.json').read_text())
        assert source['completed'] == source['expected'] == 128
        result[model] = {kind: summarize([r for r in source['results'] if r['task']['kind'] == kind])
                         for kind in ['paint', 'repair', 'finish']}
    (root/'behavior-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps({model: {kind: {k: v for k, v in row.items() if k != 'errors'}
                             for kind, row in kinds.items()} for model, kinds in result.items()}, indent=2))
