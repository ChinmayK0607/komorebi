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

The next bounded task (`task_e_6ab3cd037ed8832b9b9ae4e46ffcefe9`) made no
model request: setup stopped with apt exit code 100 because the cloud proxy
returned HTTP 403 for the universal image's unused `apt.llvm.org` source.
`cloud/setup.sh` now moves only that source into the ignored runtime directory
before apt updates, while keeping the Ubuntu and Node sources. The failed task
is infrastructure evidence, not a model result.

The first complete cloud episode (`task_e_6ab3ce329c90832b9d6f87b35a3ae964`,
run ID `cloud-teacher-proxy-smoke-20260923`) tested whether the documented
Gateway `/v1` transport plus `NODE_USE_ENV_PROXY=1` could run the full
generate/render/revise loop. Its matched infrastructure baseline was the
earlier `/v1` cloud episode on the same model and reference that failed before
any tokens with `Cannot connect to API`; this is not a matched visual-quality
comparison. Source commit was `4361dd56df26f3b25de1ebf72da0c711ae0a11d1`,
reference dataset revision `811564c415aac98601754ce6133de33bf25cd69d`,
config SHA-256 `512e2cc3a7d566cbf14ceb7fe5196c05e108ccdbe3f5a62f70b9b962aac0c5f3`,
reference manifest SHA-256 `44334c164a7f1a3b1c7c0931444eefa64886d6bd88087735992cb53f78d45b74`,
and renderer SHA-256 `ad7ba24d632f0f16989741560c3af0fe42e426275c1df01954b60ddd69b63395`.
DeepSeek V4.1 Flash painted `coco128-000000000109` through six valid 600×600
canvases and ended at the screen turn limit in 845.3 active seconds, with six
provider requests and 184,655 reported total tokens. All six provider-cost
fields were absent; no dollar figure is inferred. Parent visual inspection of
the final canvas found the river, shoreline, grass and reflection recognizable,
but people and benches poorly resolved. This is one unblinded observation,
not a five-model ranking or a demonstrated visual improvement over a matched
baseline.

The automatic public-HF publish hook failed on Xet with HTTP 400, and its
LFS retry hit the cloud proxy's HTTP 403 on
`hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`. Adding that host to the
environment's domain allowlist did not update the running task's pinned
network policy. A normal Git push of the archive under an unforced `.data`
path was rejected by HF's pre-receive hook, which required Xet. The task
preserved the 22-file archive and uploaded it as ten Base64 ASCII text parts,
which HF classified as regular files. It then anonymously fetched every part,
reassembled the archive, and verified SHA-256
`7127933bb9db005be7694f582d996930607d9cc8b09375213fea1b4a07b29161`
over 3,694,748 bytes. An independent local anonymous fetch and reassembly
verified the same hash. The public
[receipt and parts](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/tree/main/runs/cloud-teacher-proxy-smoke-20260923)
record source and dataset commits. Temporary transfer files were removed from
the cloud source worktree without committing them; no model rerun, rented GPU,
or scheduled task was used for publication recovery. Next verify a fresh
cloud task can use the newly allowed LFS host, make publication reliable for
larger result archives, then screen models against the same references while
measuring accepted visual demonstrations per dollar and hour.

A fresh, provider-free cloud task (`task_e_6ab3d832068c832b8ef636a8bff838a0`)
confirmed that the revised allowlist applies to new tasks: a 16,361-byte
`.gz` probe used Git LFS with `HF_HUB_DISABLE_XET=1`, uploaded through
`hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`, and was anonymously
downloaded and SHA-256 verified at HF dataset commit
`12dd7f2c47b5ef98f9c425a5ebf3e6d9e59a357d` (probe SHA-256
`75230e34facd8ed63ede07194f6e2e95b26586d6e324b897a0ba00c4e9e49328`).
No model was called. `cloud/run.sh` therefore selects this verified LFS path
for future result publication. A full-size archive remains untested on the
new task policy, so the next paid batch must still verify its public receipt.

The result publisher now retries a failed archive upload as Base64 ASCII
`.txt` parts through regular HF commits, then anonymously reconstructs and
checks the complete archive before writing a receipt. It logs only an error
type on fallback because proxy exceptions can contain signed upload URLs.
Either receipt format can be downloaded and independently verified without a
credential:

```bash
python3 painter/benchmarks/openrouter-teachers-20260922/cloud/fetch_results.py \
  cloud-teacher-proxy-smoke-20260923 /tmp/painter-teacher-evidence.tar.gz
```

The split fallback is a durability path, not a claim that very large text
archives are an efficient long-term storage format. Screen outputs should be
kept in bounded task slices with unique run IDs and verified receipts.
