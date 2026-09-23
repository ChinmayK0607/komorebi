# Cloud teacher benchmark recovery, 2026-09-23

## Decision before launch

The initial quality benchmark cannot yet support a model-quality ranking. Four
anonymously downloaded, hash-verified public shards (`full-quality-02`, `04`,
`07`, `09`) contain 80 episodes: 42 ended at the 180-second renderer deadline,
28 ended in provider connection/header errors, and 10 reached another terminal
status. The 28 API errors report either `Headers Timeout Error` (22) or
`other side closed` (6) after the AI SDK's three attempts. This is censored
infrastructure evidence, not 70 poor paintings.

**Hypothesis.** Streaming a response from the Gateway and allowing a longer
bounded renderer deadline will recover usable teacher paintings and turns from
the same reference/model pairs. A saved generated sketch that timed out at
180 seconds should be replayed before another paid model request. Streamed
provider calls should reduce the observed header/connection failures; this
remains unverified until a paid end-to-end sample completes.

**Matched baseline.** The unchanged initial shards used source
`10f047ac91204bf1260e9325d4003f9a7d4e57a3`, config SHA-256
`512e2cc3a7d566cbf14ceb7fe5196c05e108ccdbe3f5a62f70b9b962aac0c5f3`,
renderer SHA-256
`ad7ba24d632f0f16989741560c3af0fe42e426275c1df01954b60ddd69b63395`,
40-reference manifest SHA-256
`44334c164a7f1a3b1c7c0931444eefa64886d6bd88087735992cb53f78d45b74`,
and public reference revision `811564c415aac98601754ce6133de33bf25cd69d`.
The same public archives and original per-turn receipts are retained; recovery
outputs will have separate public run IDs and immutable source hashes.

**Recovery method.** First, replay one saved timeout program on a prepared
Linux Codex cloud environment at a 600-second hard renderer deadline, with
zero model calls. Publish the replay evidence publicly and verify its archive
hash anonymously. Then restore a complete verified quality shard in a fresh
cloud workspace and resume one censored episode. The existing completed model
response must be reused; newly needed later turns may issue paid requests.
The recovery's source run ID, override deadline, renderer SHA, reused response
count, public dataset revision, final-canvas validity, elapsed time, errors,
and missing cost information must be reported. Do not conflate an offline
replay painting with a teacher continuation that saw the recovered canvas.

Scale to the remaining censored shards only if the smoke run verifies the
renderer and transport; compare quality on the same references afterward.
No scheduled tasks or recurring monitors are part of this recovery.

## Implementation and verification

The Gateway transport now consumes a streamed response while preserving the
one-response-per-turn JSONL protocol. The SDK no longer performs hidden
retries; Python retains its bounded two-retry policy and treats connection or
timeout failures as retryable. The renderer supports an explicit deadline up
to 900 seconds. Recovery scripts download and hash-verify public HF archives,
restore saved episodes without overwriting files, and retain original response
bindings when only the renderer identity changed. A separate replay script
rerenders saved timeout programs without model calls.

Local verification: TypeScript typecheck; 19 Python tests; Python compile and
shell syntax checks; anonymous source archive verification and restoration of
`full-quality-02` (157 episode files); replay dry run identified one selected
timeout program. No local Mac rendering or paid model calls were performed.
## Zero-call replay smoke

The [cloud replay task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab40a4f8910832bb8e9cc8b622d7cc8)
ran source `ab2d0a169808e7abe3d60e3f996701679aa69375` against one saved
DeepSeek program on reference `coco128-000000000064` from quality shard `02`.
The original episode ended at the 180-second renderer deadline. The replay
produced a valid 600×600 canvas in **172.351 seconds** under the explicit
600-second deadline, with **zero paid model calls**. This difference may be
runtime variance or reduced concurrent contention; one replay does not show
that increasing the deadline alone fixes all censored cases.

The [public replay receipt](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/main/runs/renderer-replay-quality02-smoke-20260923/receipt.json)
records dataset commit `754040be93dd241b9e6cfd78ab5b4d3e96b89f6b`, archive size
627,988 bytes and SHA-256
`31821fbe6e2041c897b96971093646be454fc8bec6b519fbf2052d569c694239`.
The archive was independently reassembled from its public Base64 parts at
that immutable revision; size and hash matched. Direct visual inspection
against the reference found the foreground clock, central tree, and street
layout recognizable, with simplified car, lettering and scene detail. This
is one unblinded observation, not a model-quality ranking.

The next [single-episode continuation](https://chatgpt.com/codex/cloud/tasks/task_e_6ab40c7b6ae4832bbf22e07e50700d6b)
is `quality-recovery-02-e43-20260923`, selecting deterministic offset `43`
only. Its Xiaomi Flash episode ended at a turn-11 render timeout, so a
successful replay needs at most the one remaining turn-12 model response.
A local fake-render check on the anonymously restored shard confirmed its
saved turn-11 response is reused, the 600-second override is passed, and
zero paid requests occur before that render. The actual cloud result,
streamed provider behavior and public evidence hash remain pending.
