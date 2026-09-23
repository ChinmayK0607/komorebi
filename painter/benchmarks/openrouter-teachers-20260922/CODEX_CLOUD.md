# Codex cloud teacher benchmark

This branch carries the benchmark source and pinned Watercolour renderer in Git.
The 40 COCO128 reference JPEGs live in the public Hugging Face dataset
[`CK0607/komorebi-painter-teachers`](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers),
at immutable revision `811564c415aac98601754ce6133de33bf25cd69d`.
`cloud/fetch_references.py` verifies all 40 hashes before setup continues.
The dataset card and `refs.json` retain provenance; per-image licensing has not
been established by those records.

The `komorebi-painter-teachers` Codex cloud environment is configured for
`ChinmayK0607/komorebi`. Select branch `codex/painter-teacher-cloud`. Its
environment setup command is:

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

The rented-node 12-episode screen completed and its archive was hash-verified
from the public dataset at `runs/rented-screen-12-20260923/`. The rented node
was released. Do not merge its artifacts into a cloud run receipt.

The first Codex cloud setup attempt failed because the universal image put its
Node 20 ahead of the installed Node 22. Commit `9ccdb83` made the cloud
wrappers select `/usr/bin` first. The next setup verified all 40 reference
hashes, a sandboxed SwiftShader render, and a one-episode dry run with zero
paid calls. The corresponding cloud task is
`task_e_6ab3bb481770832ba6c23462c7a5995d`.

The first paid cloud episode (`task_e_6ab3c5b9ec2c832b87d636515a30d0f2`,
run ID `cloud-teacher-smoke-20260923`) produced an `api_error` before any
model tokens. Its five-file evidence archive was published and its public
SHA-256 verified at dataset revision `3ddd64cc3e7458c507dc1e25c1a0fb123e2834e2`.
Unauthenticated GET and POST probes reached AI Gateway (HTTP 308 and 400), and
one minimal authenticated request to its OpenAI-compatible `/v1` endpoint
returned HTTP 200. A minimal call through the default AI SDK Gateway route
reproduced the original `GatewayResponseError: Invalid error response format`.
The transport therefore uses the AI SDK OpenAI-compatible provider at the
Gateway's documented `/v1` endpoint. A paid painting rerun is required before
claiming this change fixed generation; keep the failed archive as evidence.

The `/v1` rerun (`task_e_6ab3cb061454832b9e456f8e7c8dc77f`, run ID
`cloud-teacher-v1-smoke-20260923`) also produced `api_error`, now with
`AI_APICallError: Cannot connect to API`, zero tokens, and no canvas. Its
five-file archive was hash-verified at dataset revision
`4eb905e7dd6675aa931a4843da9422f12f7cd089`. In that same cloud
environment, both HTTP and HTTPS proxy variables were present (their values
were not inspected), and a credential-free Node 22.23.2 `fetch` with
`NODE_USE_ENV_PROXY=1` reached the allowlisted Gateway (HTTP 308). Node's
documented proxy switch is now set in `cloud/run.sh` before the Python runner
spawns the AI SDK transport. A paid painting rerun is still required before
claiming the cloud benchmark works end to end.
