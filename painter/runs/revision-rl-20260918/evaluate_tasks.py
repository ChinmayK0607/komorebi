"""Matched fresh-reference drawing, repair and stopping evaluation on the node."""
import argparse
import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rollout import run
from render_demos import atomic

P = Path(__file__).resolve().parent


def resolve_input(value):
    path = Path(value)
    return path if path.is_absolute() else P / path


def main():
    assert sys.platform == 'linux'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--tasks', default='evaluation-tasks.json')
    parser.add_argument('--config', default='campaign.json',
                        help='Immutable evaluation config; defaults to campaign.json.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1',
                        help='OpenAI-compatible server base URL.')
    parser.add_argument('--output', default='evaluation')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--samples', type=int, default=2)
    args = parser.parse_args()

    tasks = json.loads(resolve_input(args.tasks).read_text())
    config = json.loads(resolve_input(args.config).read_text())
    out = P / args.output / args.model
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # A manual early launch and the pipeline may both request this model. Only
    # one producer can generate its cases; the follower reuses saved summaries.
    lock = (out / 'producer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)

    def one(task, sample):
        folder = out / task['id'] / f'sample-{sample:02d}'
        summary = folder / 'summary.json'
        if summary.exists():
            return dict(task=task, sample=sample, **json.loads(summary.read_text()))
        result = run(
            P / task['reference'], folder, args.base_url, args.model,
            P / 'vendor/integrations/watercolour/renderer.py',
            P / 'renderer-env/bin/python', config, seed=918200 + sample,
            initial_canvas=P / task['canvas'] if task.get('canvas') else None,
            initial_program=P / task['program'] if task.get('program') else None)
        return dict(task=task, sample=sample, **result)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one, task, sample): (task, sample)
                for sample in range(args.samples) for task in tasks}
        for future in as_completed(jobs):
            task, sample = jobs[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(task=task, sample=sample, valid_canvas=False,
                           finished=False,
                           error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
            atomic(out / 'progress.json', dict(
                model=args.model, completed=len(rows), expected=len(jobs),
                valid=sum(row.get('valid_canvas', False) for row in rows),
                finished=sum(row.get('finished', False) for row in rows)))
    atomic(out / 'summary.json', dict(model=args.model, completed=len(rows),
                                      expected=len(jobs), results=rows))
    lock.close()


if __name__ == '__main__':
    main()
