# Codex cloud teacher benchmark

This branch carries the benchmark source and pinned Watercolour renderer in Git.
The 40 COCO128 reference JPEGs live in the public Hugging Face dataset
[`CK0607/komorebi-painter-teachers`](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers),
at immutable revision `811564c415aac98601754ce6133de33bf25cd69d`.
`cloud/fetch_references.py` verifies all 40 hashes before setup continues.
The dataset card and `refs.json` retain provenance; per-image licensing has not
been established by those records.

Create a Codex cloud environment for the `codex/painter-teacher-cloud` branch
of `ChinmayK0607/komorebi`. Set the environment setup command to:

```bash
bash painter/benchmarks/openrouter-teachers-20260922/cloud/setup.sh
```

The setup installs the Linux renderer, Chromium, Python and Node dependencies,
and runs a fixed renderer smoke plus a one-episode key-free dry run. It does
not contact the paid model API. The Codex cloud container must permit root
package installation and `runuser`; `setup_benchmark_node.sh` checks this.
Use a fresh cloud task if the setup fails. The bundled references and any
`episodes/` evidence stay out of Git.

For the task phase, enable outbound **POST** access to Vercel AI Gateway and
outbound **GET/POST** access to Hugging Face. Codex cloud disables task-phase
internet by default. A narrow domain/method allowlist is preferable if its
proxy supports all redirect hosts; test one paid episode before widening it.
The setup phase already has internet access for dependencies and public input
downloads.

The run phase needs `AI_GATEWAY_API_KEY`, `HF_TOKEN` with dataset-write access,
and a stable `PAINTER_RUN_ID` as environment variables. Codex cloud's
**Secrets** are only exposed to setup scripts, so they cannot directly supply
keys to a task-phase benchmark. Environment variables are accessible to the
agent and its shell: use limited, revocable keys; never place values in a
prompt, Git, logs, screenshots, or a committed environment file. The runner
passes the Gateway key to its transport through the process environment and
never prints it.

Run one paid episode first from the cloud task's terminal:

```bash
bash painter/benchmarks/openrouter-teachers-20260922/cloud/run.sh \
  --track screen --limit-episodes 1 \
  --models stepfun/step-5-preview
```

Use a unique `PAINTER_RUN_ID` for each independent task. The wrapper packages
raw episodes, progress, events, summary and gallery if present when the runner
exits, uploads the archive to `runs/<PAINTER_RUN_ID>/` in the public HF dataset,
downloads it anonymously, and verifies SHA-256. Its receipt includes the Git
commit and dataset commit. On a hard container termination, the exit hook may
not execute, so scale only after one cloud task completes and the public
receipt is verified. Full benchmark runs are many-hour jobs at the observed
provider latency; Codex cloud execution lifetime is not established here.

The ongoing rented-node screen is a separate run. Leave it running to
completion, collect its evidence, and release that node only after its archive
is hash-verified. Do not merge its artifacts into a cloud run receipt.
