# Cloud teacher benchmark recovery, 2026-09-23

## Decision before launch

The initial quality benchmark cannot yet support a model-quality ranking. The
first four anonymously downloaded, hash-verified public shards
(`full-quality-02`, `04`, `07`, `09`) contained 80 episodes: 42 ended at the
180-second renderer deadline, 28 ended in provider connection/header errors,
and 10 reached another terminal status. The 28 API errors reported either
`Headers Timeout Error` (22) or `other side closed` (6) after the AI SDK's
three attempts. This is censored infrastructure evidence, not 70 poor paintings.

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
zero paid requests occur before that render.

## Single-episode continuation result

The continuation ran source
`ab2d0a169808e7abe3d60e3f996701679aa69375` and ended `turn_limit` after
12 turns with a valid final canvas. Turn 11 reused its original response and
rendered successfully in 208.821 seconds. Exactly one new model response was
generated for turn 12 (90,010 reported tokens: 82,137 prompt and 7,873
completion), and its canvas rendered in 205.428 seconds. The full episode
records 611,184 tokens, including the original ten turns. Provider cost is
missing on all 12 turns; the reported known $0.00 is not total spend. The
final painting clarifies individual bird bodies and wings compared with the
retained turn-10 canvas, but remains a simplified rendering of the reference.
This is an unblinded single-case observation, not a general model preference.

The [public recovery receipt](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/main/runs/quality-recovery-02-e43-20260923/receipt.json)
pins HF dataset commit `672e4d89178255940d0a2d11e4fa129c16187df4`, archive size
8,873,196 bytes and SHA-256
`2411cc38915ca9a17a98f5b054bbb9f4ebf7077947115d30dbca01d8191e9c53`.
I independently reassembled its public parts at that revision and verified
both size and hash. The original archive remains separate and unchanged.

**Accounting correction:** This run's `run-summary.json` reports 11
`network_requests_made_this_invocation` by summing archived original turns
whose `api_reused` flag was false in the *old* invocation. Raw turn receipts
show ten old responses, one reused turn-11 response and one new turn-12
response with one attempt. Thus this recovery submitted **one** Gateway
request. The counting code was corrected at source `64b9258`; future runs
count requests submitted by the current transport process, not saved turns.
Do not use the old summary's 11 as a spend or throughput measurement.

**Next decision:** Continue bounded, provider-free replays across verified
quality shards to learn which saved timeout programs are recoverable. Resume
paid turns only where the canvas is valid, then compare visual quality on
matched references. Limit simultaneous Gateway recovery shards while the
original quality tasks finish; streaming is proven for one Xiaomi case, not
for all five providers.

## Expanded verified scope and finite reruns

Three more original quality shards (`01`, `03`, `08`) were independently
downloaded and SHA-256 verified against their public receipts. Their archive
hashes are, respectively, `0bba7173dcfa27b85064d890d59a8f6329739bdc22a940f561cfb7016b5da21e`,
`94a5f4c06845497373f33b68aa3e64a4380c49cb213131b3ce2364ee9588d995`,
and `09a9085ffe70ce2f8747ca9e7f7a99991c720d42ed992c8cfd2377c72f15ab39`.
The eight verified original quality shards (`00`–`04`, `07`–`09`) now cover
160/200 planned episodes: 74 `renderer_error`, 55 `api_error`, 21 `turn_limit`,
9 `complete`, and 1 `invalid`. Original shards `05` and `06` have no public
verified receipt yet. These status counts do not rank visual quality.

At source `454a29b`, eight finite, provider-free cloud tasks were dispatched
to replay all saved render-timeout programs from those verified shards at a
600-second deadline. They use zero model calls, leave the originals unchanged,
and publish separate hash-verifiable public results. The source-to-task map is:

| Original quality shard | Replay cloud task |
| --- | --- |
| `00` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab4112087dc832bbc3b2473fc4633a6) |
| `01` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab412c5f2d4832ba5a0e49c52e35046) |
| `02` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab4114bd680832bbbc9703b93ad933b) |
| `03` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab412d9a938832baf6e5a082cea7455) |
| `04` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab411350aa0832bb140941ccf6f263f) |
| `07` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab4115e0248832ba10fe81815cda5cc) |
| `08` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab412ebf0c4832bb27745f2eb931b05) |
| `09` | [task](https://chatgpt.com/codex/cloud/tasks/task_e_6ab41178c01c832b884538623e035423) |

In parallel, three one-episode corrected continuations were dispatched to
test streaming and saved-turn reuse across providers: [DeepSeek ref368,
global episode 16](https://chatgpt.com/codex/cloud/tasks/task_e_6ab41365e950832b9c6a3a1cb577acd6),
[Step-5 ref136, episode 37](https://chatgpt.com/codex/cloud/tasks/task_e_6ab41379ca4c832b8a8a4b9093fcb50d),
and [MiMo Pro ref349, episode 194](https://chatgpt.com/codex/cloud/tasks/task_e_6ab413909550832bb101477a8d442c4e).
Xiaomi Flash has the verified episode-43 continuation above. The additional
tasks are **in progress**; dispatch is not a successful render, valid painting,
or provider ranking. Only newly needed turns can issue paid Gateway requests.
No scheduled task or watcher was created.

## Gateway credit block and throughput correction

All three additional one-episode continuations published separate archives,
which were anonymously reassembled and SHA-256 verified. Each selected episode
ended with `api_error` because the AI Gateway returned HTTP **402**:
"A positive credit balance is required for all requests, including BYOK".
The three archive hashes are DeepSeek
`0173108669d8bf71bbb6f99ad61eac0bd2b1fe425d94e4ae878c588278f382ba`
(44,446,563 bytes), Step-5
`87440d0f8894403a4df4da9544f819fcb134c96ebe2b8b07490c10ad015d555b`
(39,663,191 bytes), and MiMo Pro
`d5bf6b2bf7e336c45eada7aa07f3381b1b0969d3d43608d174496e9daf4ae43e`
(26,387,421 bytes). No provider/model comparison is possible from these
attempts. The successful Xiaomi Flash continuation happened before this
balance block; it must not be generalized to the other providers.

These three run summaries also incorrectly reported zero requests submitted
this invocation. The Python pool was cleared by `close()` before the counter
was read, while each selected turn's receipt contains one nonretryable 402
response with `attempts: 1`. Source now snapshots transport counters **before**
closing the pool; the regression test simulates a clearing `close()`.
The 402 is an external balance condition, not a resume-path failure. Do not
retry paid Gateway calls until a positive balance is confirmed.

The initial provider-free replay script runs one render at a time within each
cloud shard; eight shards run concurrently on separate VMs. This is a script
choice, not a benchmark constraint. The replay command now accepts
`--workers 2` (bounded 1–8) and uses a fresh isolated browser profile for
each concurrent program. A focused overlap test and all 19 Python tests pass.
No claim of measured wall-time speedup is made until a Linux replay compares
the same programs at one versus two workers. The already-running tasks remain
on their original source commit and retain their serial timing baseline.

## User selection gallery

The user requested a visual choice of teacher before further paid generation.
A just-submitted GLM-only continuation was [cancelled during cloud setup](https://chatgpt.com/codex/cloud/tasks/task_e_6ab418aeec20832b81130abaa8cc0a8f),
before its benchmark command began. No new paid teacher generation was
recorded for that task. The provisional GLM preference is not a final winner.

`build_public_selection_gallery.py` assembles an offline, interactive view of
reference images beside five candidate models. It independently checks every
local archive's SHA-256 and size against its public HF receipt and checks each
reference image hash. Its first local build contains **360 distinct original
episodes from 18 verified archives** (160 quality, 200 speed), 40 references,
and 205 image assets. Quality has 30 terminal valid canvases and 69 episodes
with any retained valid canvas; speed has 75 terminal valid canvases and 96
with any retained valid canvas. Terminal, partial, missing, and future offline
replay paintings are displayed with distinct badges. These are coverage
figures, not pairwise quality scores. The generated HTML, manifest, and image
assets are ignored by Git; the repeatable builder and instructions are tracked.
The local gallery path is
`painter/benchmarks/openrouter-teachers-20260922/results-ai-gateway-20260923/selection-gallery/index.html`.
The page's inline JavaScript passed a syntax check, and all referenced local
assets exist. Rebuild with `--replay-dir` as the cloud replays publish verified
archives; the user will make the teacher choice from the paintings.

The final two original quality shards subsequently published verified public
receipts and passed independent anonymous archive reconstruction. Shard `05`
is 47,997,201 bytes with SHA-256
`f588c70577542f0968bda9b29386d59464de7fd2cdf0bf7d7181159c93ee70cf`;
shard `06` is 10,538,204 bytes with SHA-256
`18cd416612ea42d640a2d8b57ae05798ca1a4697fb53c5aff683c7a1690264ac`.
The original quality track now has all **200/200 episodes** archived; 89
ended `renderer_error`, 73 `api_error`, 25 `turn_limit`, 11 `complete`, and 2
`invalid`. The gallery was rebuilt from all 20 original quality and speed shards
and now has 400 candidate episodes: quality has 36 terminal valid canvases
and 86 with any retained valid canvas; speed has 75 terminal valid canvases
and 96 with any retained valid canvas. All 40 reference images and 222 asset
files exist locally, and the inline JavaScript passed syntax checking. No
replay archive was included yet.

The remaining already-paid timeout sketches were dispatched for finite,
provider-free Linux replay using source `e85ce59`: [speed shards 00–04](https://chatgpt.com/codex/cloud/tasks/task_e_6ab41b5fedac832b9cbd10c018dd40c4)
and [speed shards 05–09](https://chatgpt.com/codex/cloud/tasks/task_e_6ab41b75c904832bb2841b49c32a3f3d)
contain 19 timeout programs each and use a 360-second bound with two isolated
render workers; [quality shards 05–06](https://chatgpt.com/codex/cloud/tasks/task_e_6ab41b8841b8832bb5a035edd9094b8c)
contain 15 timeout programs and use 600 seconds with two workers. The eight
earlier quality replay tasks cover the other 74 timeouts, giving coverage for
all **127** originally timed-out quality/speed programs. All replay tasks are
in progress; no outcome or throughput gain is claimed from dispatch. They
make zero model calls and publish per-shard public hash-verifiable archives.
