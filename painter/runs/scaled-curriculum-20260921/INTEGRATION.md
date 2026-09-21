# Native RL integration status

The shared native RL task and reward integration is implemented in
`painter/native_painting_rl.py`. It reuses the corrected native interaction
loop, accepts finite scaled training manifests, keeps source order interleaved
for Prime's pinned sampler, and scores only the terminal reference-conditioned
Luna-low pairwise judgment. The evaluation helper now derives `num_tasks` and
`expected_count` from each manifest; its default remains 12 for controlled
comparison callers.

The RL loader rotates group positions across families before taking a second
row from any group. This keeps a large family from consuming the sampler
prefix; the pure test covers 12 families × 200 groups with two rows per group
and checks the first 512 rows for family balance and uniqueness.

Local pure validation completed:

```text
python -m unittest painter.test_native_painting_rl painter.test_native_painting_eval painter.test_native_painting
17 tests passed
python -m unittest painter.test_luna_cli_judge
8 tests passed
python -m py_compile painter/native_painting.py painter/native_painting_rl.py painter/native_painting_eval.py painter/luna_cli_judge.py
```

The finite `painter/luna_cli_judge.py` worker is pure-tested with a fake runner
against two staged orientation inputs. It sends the approved prompt on stdin,
uses an argument vector, attaches exactly `reference`, `A`, and `B` in that
order, applies the strict four-field schema, bounds concurrency at two, and
rejects missing, changed, failed, or invalid inputs without writing a label.
The bounded invocation is:

```text
python -m painter.luna_cli_judge \
  --batch-dir <one-staged-reward-batch> \
  --protocol-dir painter/runs/aesthetic-rl-20260921 \
  --max-concurrency 2
```

It requires the installed `codex` executable, authenticated account, and
available `gpt-5.6-luna` model; those are runtime prerequisites rather than
credentials stored by this workspace. Each call has a finite 1800-second
timeout. Raw stdout, stderr, JSON event stdout, and last-message evidence are
retained, and `turn.completed.usage` is parsed when present.
Each pair now gets an isolated neutral scratch directory containing only
hash-verified role copies such as `reference.jpg`, `A.png`, and `B.png`, with
suffixes derived from image magic; execution records retain both the original
staged paths and the attachment paths/hashes.
This is a hypothesis-driven mitigation for one reversed live view, not a
claim that the root cause has been established.
Its label document is accepted by the existing `judge_queue._label_documents`
shape, but the worker does not build receipts or install them. The execution
record says `attachment_status: attached`; it deliberately does not claim that
the CLI model tool inspected the images and does not write `inspected: true`.
The parent subsequently ran six views across three subjects with explicit neutral filenames: all were schema-valid, cylinder/cup preferences were order-consistent, and box preferences differed. The cylinder two-view batch took 20.29 seconds at concurrency two. This does not establish sustained throughput or broad judge accuracy; see CLI_BATCH_REVIEW.json.

This does not validate the Linux node, Prime configuration, renderer runtime,
or native optimizer execution. Before scaled RL, the node still needs distinct
fixed opponents rendered for each training rollout group, the
pinned `aesthetic_judge.py`/`judge_queue.py` consumer must be active, and the
native runtime configuration must be validated against the installed pins.
Missing judgments remain missing; no durable judge-throughput observer or
optimizer-admission observer is claimed here.
