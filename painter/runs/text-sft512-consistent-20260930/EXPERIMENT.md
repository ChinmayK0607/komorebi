# SFT-512 initialization with consistent joint-view RL

## Hypothesis and matched comparison

Initialize directly from the publicly hash-verified SFT step-512 adapter.
Test whether conservative RL under the revised aesthetic/readability prompt
and identity-keyed, balanced pairwise supervision improves held-out painting
quality relative to that same SFT checkpoint. This avoids inheriting the
previous RL-step-16 judge's optimization history. Keep the step-512 canvases
as immutable per-prompt reward anchors. A rising admitted-batch reward alone
is not a held-out result or a guarantee of better aesthetics.

Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Initializer: `CK0607/qwen3.8-27b-brush-painting@adf687e5b321514eda7be6eab050434fdc8e44ff`,
prefix `photo-curriculum-sft-20260922-balanced/step_512`, adapter SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Prime: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
Judge: `text-paint-sft-pairwise-consistent-v1`, source `1b9ac1f`,
`xiaomi/mimo-v2.6-flash` through Vercel AI Gateway. The provider exposes no
immutable weight revision. Both paintings are always presented together.

The existing interleaved 40-brief training split (12 simple, 10 intermediate,
18 complex) and 47 held-out briefs remain fixed. Training semantic manifest
SHA-256: `13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`.
The source public archive SHA-256 is
`ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3`.
No train brief is drawn from the selected frozen judge-calibration panel.
The 47 held-out cases contain 8 simple, 8 intermediate and 31 complex briefs;
report by family and include failed/censored outcomes. Original SFT paintings
are reused with their bytes/hashes, not regenerated as a moving opponent.

## Finite execution limits

Maximum 50 optimizer updates, LoRA LR `1e-6`, three visual-feedback turns,
8,192 completion tokens per turn, group eight and batch sixteen
(two briefs per admitted optimizer update). GPU topology is selected after
availability, memory and whole-node price checks; do not assume inexpensive
24/32/48-GB cards fit independent BF16 trainer and inference roles. Public
checkpoints are planned every four updates, with finite hash-verification
and retained raw rollouts, renders, metrics, gradients and judge receipts.

The user's hard ceiling is **$20 total GPU rental per run**, including setup,
evaluation and evidence preservation; Vercel cost is separate and logged.
Register actual rental start, hourly price and training/removal deadlines
before launch. Use a finite process supervisor, not a scheduled Codex task,
recurring automation or heartbeat. No node has been rented at registration.

At updates 15, 20 and then every five updates through 50, compare the latest
five effective optimizer-batch reward means with the preceding five under
the fixed sampler. If the latest mean does not exceed the prior mean, stop
training, preserve the latest public checkpoint, evaluate it and release the
pod. This is an operational early-stop rule, not a statistical significance
test. Do not reset the rule or cherry-pick a better starting window after
observing rewards. Missing metrics, stalled admission, crashes or the cost
deadline must terminate finitely rather than silently continue spending.

Unresolved judge choices return no reward. Genuine invalid model programs
retain a separately labelled zero; operational renderer/judge failures are
not visual failures. Before optimizer start, verify how the pinned Prime
version filters uncertainty across rollout groups and records it. Never
coerce `None` to zero or 0.5. Log group admission/censoring and zero-output
fractions; group-eight admission could be inefficient if uncertainty is high.

## Pending launch evidence

Deployment source: `2ecf9acd2c476b3027a5d57f3ea6fbc81df6ba21`, transferred
as a complete verified Git bundle. The inspected topology is two RTX PRO
6000 Blackwell Server Edition GPUs, each reporting 97,887 MiB, at $2.38 per
whole-node hour. The conservative rental clock began 2026-09-30 11:10:10 UTC;
training must stop by 18:49:22 UTC, and the finite parent controller begins
exact-pod removal by 19:31:22 UTC (three minutes before the $20 boundary).
Linux preparation has started; no optimizer update or judge call has started.
Gateway credential egress was rejected by automatic approval review because
HF-token authorization did not cover this separate credential. Gateway
transfer approval is pending, and that key has not been copied.

Preparation completed: Prime accepted the generated config, the pinned
initializer was verified, and the 40 training/47 held-out fixed anchors were
verified. Config SHA-256:
`5b80e84d1cd4f6687b28efa7a13cc91ee3799618b71e219004f57ee4d3648fb9`.
The nine-file preparation archive is public at dataset revision
`3ee857f9274da0b20a2320a2da9d825857a8be9d`, path
`runs/text-sft512-consistent-20260930/preparation-evidence.tar.gz`, anonymously
verified SHA-256 `22317cc8b8842c83c109a11f3a8d529b7b262656831db787f333ba1fdd3c7343`.
The temporary node HF token was deleted, and the finite controller verified
exact-pod removal at 11:20:10 UTC rather than leaving GPUs idle pending
authorization. `lium ps` showed zero pods. Conservative clock-based GPU spend
was $0.396 (not an invoice), leaving at most $19.604 of this same run's cap;
use additional billing margin when relaunching. Judge calls and optimizer
updates were both zero. Actual Prime-child activation receipt, first real
scored rollout, gradient verification and visual/reward results remain
pending; successful configuration preparation is not a training result.

Record reviewed source commit, frozen config/manifest bytes, initializer and
renderer hashes, actual GPU model/topology/price, rental start/deadlines,
credential-free launch receipt, and native serializer/scorer/gradient checks.
Afterward report matched held-out preferences and visual examples, admitted
and all-completed reward curves, group losses, turn validity, parser and
position conflicts, timing/utilization windows and denominators, public
checkpoint/archive hashes, actual GPU/API cost, and the next decision.

## Approved relaunch and actual startup

The user explicitly approved Gateway-key transfer after the preceding
preparation-only node had been released. Both credentials were mechanically
transferred to an owner-only node directory; no values entered source or
research records. The second rental began at 12:31:56 UTC at $2.38/hour.
Its finite allowance is $19.50, so the prior $0.396 setup estimate plus this
allowance remains below the $20 total ceiling. Estimates are clock based,
not provider invoices. Training launched at 12:39:57 UTC using source
`877584e2a758b1ba9908f3d8568b0955ca072dba`. Preparation, real installed-Prime
guard activation and run preflight passed.

The trainer initialization audit reports 992/992 exact adapter tensor
matches, 496 nonzero B tensors, 116,727,808 adapter parameters and all finite
values, matching adapter SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Optimizer state is fresh; the trainer broadcasts policy v0 and the
orchestrator waits for that broadcast before issuing training episodes.
Inference is physical GPU 0 and trainer physical GPU 1. The first 16
episodes were submitted after synchronization at 12:45:40 UTC. An optimizer
update, gradient evidence and measured pipeline busy fraction remain pending
at this startup record.

## Before-first-update parser repair

The live initial judge responses exposed an infrastructure failure: identical
`preference` and `_preference` fields were rejected as conflicting aliases.
This is not an aesthetic disagreement. The reviewed parser now deduplicates
exactly equal values and records the lossless normalization; unequal values,
missing decisions and truncated/unknown image IDs remain censored. The prompt,
model, pair presentation, winner mapping and reward scale are unchanged.
Original raw receipts and censored group accounting remain preserved.
The parser source SHA-256 after repair is
`f9089e3e34e89f274d9f1ac0282fd3461997f35a049c6b16bb22e0f0f791a48c`.
Deployment time and first optimizer evidence will be recorded separately.

The reviewed completion pipeline performs generation-only evaluation on the
47 held-out cases, screens and uploads evidence publicly, verifies anonymous
archive bytes and signals the finite parent controller for exact-pod release.
It reserves bounded evaluation/upload/release windows and still archives
failure evidence. This is one finite process chain, not a scheduled task.

## Attempt outcome and next decision

Training was stopped by the operator at 13:00:32 UTC, before any optimizer
update. The supervisor records zero updates and child exit 143. Sixty-two
training canvases were saved across turns. Nineteen comparisons finished:
12 scored, five failed initial parsing, and two lacked three-of-four order
consensus. Complete-group censoring prevented an admitted training batch.
The reviewed alias fix recovers two of 16 failed calls and one of the five
failed views; remaining errors involve missing fields, incorrect/truncated
identities, missing weaknesses or conflicting preferences. They were not
converted into rewards. This attempt says nothing about achievable RL
improvement because the optimizer never ran.

The finite completion relay exposed a separate bug: application exit code
143 inside a valid supervisor payload was mistaken for provider-command
failure. It has a regression fix. The parent invoked screened archival
directly, verified public preservation, and signalled exact-pod removal.
Deletion completed at 13:04:55 UTC; provider inventory returned zero pods.
Current-rental cost is estimated at $1.308 by clock and listed price. With
the earlier $0.396 setup, aggregate estimate is $1.704, below the same $20
ceiling. These are not invoices. Recorded receipts contain 57 judge calls
with $0.0225593 Gateway cost and median call duration 28.36 seconds; canceled
or in-flight provider calls may be absent from this tally.

A finite ten-minute utilization capture yielded 55 samples per GPU between
12:54:25 and 13:03:25 UTC, including time after training stopped. GPU 0
(inference) averaged 67.27% reported utilization, with 37/55 nonzero samples.
GPU 1 (trainer) averaged 0%, with 0/55 nonzero samples. This is neither a
run-wide average nor evidence of trainer saturation.

The 80,639,550-byte, 683-file evidence archive is public in
`CK0607/komorebi-painter-experiments` at revision
`00261695802fece720d5bdff470d96053c9d642f`, path
`runs/text-sft512-consistent-20260930/evidence.tar.gz`. Both node and parent
anonymous downloads match SHA-256
`142dfe8d16932d1c1bb50c4f5308c55426f247968cd8ab18a2453120afb0d6b8`.
It includes the pre-upload terminal receipt; the final successful upload
receipt remains separately preserved. No trained adapter exists for this
attempt.

Next decision: replace the requirement to copy long opaque IDs with fixed
LEFT/RIGHT response fields, mapping sides back to content identities in
code before balanced consensus. Keep the aesthetic criteria and reward
scale. Test the actual provider on saved images using CPU/API only. New
GPU launch remains subject to the original frugal spending and trend policy;
do not call offline parser tests proof of real provider reliability or
visual judge accuracy.

## Registered CPU-only transport probe

Hypothesis: using fixed LEFT/RIGHT output fields and mapping them to opaque
content identities in code avoids malformed dynamic identity keys without
changing the requested aesthetic judgment. Select four failed-initial pairs
deterministically from this failed attempt's immutable public archive. Keep
the same briefs, exact PNG bytes, MiMo Flash model, temperature, token limit
and joint two-image layout. Send eight initial views, both orders per pair,
with no retries or rechecks. Preserve all raw outputs, source/prompt/image
hashes, timing and provider usage. This costs no rented GPU.

Success for this smoke test means every initial view parses conservatively;
order agreement is reported separately. Four pairs selected because the old
format failed cannot establish production reliability or visual accuracy.
No reward is assigned to unknown identities, missing required assessments or
conflicting output fields. The probe does not train any model.

Probe result: the side-keyed format parsed all 8/8 initial views and produced
agreement on all 4/4 pair orders at an estimated Gateway cost of $0.002964.
The next attempt may select it explicitly with
`PAINTER_JUDGE_FORMAT=identity-joint-side-schema-v1`; the native RL selector
maps that value to `painter.text_pairwise_consistent_side_judge` and rejects
unregistered values. The selected source-format SHA-256 is
`f94cfc9e880c8423f89a091486a0cf174cf1818c13d9acdf18fd21ace79aa069`.

## Registered corrected-format retry

The retry starts fresh from the same pinned SFT-512 adapter, with the same
40 briefs, group eight, batch 16, three-turn budget, temperature 1.0 and
learning rate 1e-6. The intended change is the tested side-keyed judge
transport. The visual rubric, model, balanced consensus and reward scale
remain fixed. No prior RL adapter or optimizer state is resumed.

The first two setup/failed-attempt rentals used an estimated $1.704. The
corrected retry's additional whole-node GPU ceiling is $18.20, making the
combined estimate at most $19.904 plus no added uncontrolled GPU spend.
The same maximum 50 updates and frozen 15/20/every-five trend checks apply.
Actual rental identity, clock and source/config hashes will be written
before launch. Vercel cost remains separate.

Also record that the prior attempt produced 87 renderer receipts, of which
37 explicitly reported valid rendering; 62 PNG files existed, so PNG file
count is not validity count. Ten actions reached the 180-second renderer
limit. Median elapsed time across all recorded attempts was 10.35 seconds;
p95 reached 180.018 seconds. Syntax/API errors and timeouts are distinct
from malformed judge responses. The transport retry cannot be claimed to
fix model-generated invalid programs or all slow renders.

## Corrected-format retry: preparation failure and preservation

Source `8074ba2` prepared the pinned base model, public SFT-512 adapter and
40/47 fixed-anchor manifests on a fresh two-GPU Linux node. The guard child
then failed with `ModuleNotFoundError: prime_rl`: the outer pipeline chose
system Python before preparation created the Prime environment and failed
to refresh that interpreter afterward. No training process, optimizer update
or trained adapter resulted. The side-keyed judge was not exercised online.

Rental started 2026-09-30 13:14:51 UTC and ended 13:21:16 UTC (384.776 seconds),
an estimated $0.254379 at $2.38/hour. All setup/failed attempts together cost
$1.958576 estimated GPU spend; these are elapsed-clock estimates, not invoices.
There is no measured utilization interval for this preparation attempt.

The finite pipeline preserved 405 files (61,247,021 bytes) in public dataset
`CK0607/komorebi-painter-experiments`, commit
`d26379be261b91e440173afb444a951e0ba8abfd`, path
`runs/text-sft512-consistent-20260930/side-retry/evidence.tar.gz`.
SHA-256 `726b44c397ab12109af5fd1741d5ce0a64219b844e1ba5a195327700ae01bcc8`
matches anonymous downloads on the node and parent. Final upload and terminal
receipts were retrieved before the parent released the exact pod. Lium then
reported zero pods. No scheduled task was created.

The CPU/API transport proof is also public at revision
`faa1a5f4fbff6a2f1a003b4d876b945c8b85ee63`, path
`runs/text-sft512-consistent-20260930/side-transport-probe.tar.gz`,
SHA-256 `ad65e55202ff0a3523285fb95ee528eac8d6a30a08cde727717415fcdc30bf06`.

Next decision: refresh the actual Prime Python after preparation, test the
fresh-node interpreter transition without a rental, then repeat the same
registered run with at most $18.00 additional GPU spend. The combined ceiling
is $19.958576. The same data, SFT initializer, side-keyed judge, gradient
settings, 50-update maximum and 15/20/every-five reward stop policy apply.

## Interpreter-corrected launch

Reviewed source `0772bb60baa524616e25aa6fd439b740d63004cc` refreshes the
Prime interpreter after preparation. The executable shell regression and
embedded-Python compilation checks passed as part of 38 focused run tests.
The real Linux preparation then completed, the orchestrator guard activation
receipt was written, and Prime accepted the roles and native multimodal task
serialization. The finite supervisor started at 2026-09-30 13:31:16 UTC.
This records launch only; no optimizer update or visual gain is established
by these checks.

Rental started 13:26:43 UTC at $2.38/hour, with $18 additional GPU allowance.
Including the prior $1.958576 estimate, the combined ceiling is $19.958576.
Training deadline is 20:15:30 UTC, retaining 45 minutes for evaluation and
preservation; parent release deadline is 20:57:30 UTC. The earlier 50-update
and frozen reward-trend policy still controls earlier stopping. A finite
ten-minute GPU utilization capture was started during model loading; its
measurement window includes startup and must not be called run-wide
saturation. The archive destination is the public dataset path
`runs/text-sft512-consistent-20260930/interpreter-retry/evidence.tar.gz`.

Live source inspection corrected a documentation error: pinned Prime's
`SamplingConfig.max_completion_tokens` is a per-turn limit, not a total
trajectory cap. The actual configuration has always allowed up to 8,192
output tokens per turn across at most three turns (up to 24,576 output tokens,
subject to context and stopping). No sampling configuration was changed.
This affects latency and token accounting; do not attribute a long trajectory
to a violated 8,192-token total cap.

## First optimizer update: live verification

The first admitted batch dispatched at 2026-09-30 13:50:21 UTC with mean
reward 0.1458333333, 16/16 trainable trajectories, zero recorded rollout
errors/cancellations and zero output truncation. Trainer update 1 completed
at 13:55:28 UTC with gradient norm 0.01019287109375, LR 1e-6 and finite
weights. Dispatch and optimizer completion are distinct timestamps in this
asynchronous pipeline; reward-trend checks use admitted batch reward steps.
Completed optimizer counts must come from trainer records, not dispatch alone.

CPU comparison against the pinned SFT-512 adapter found all 992 tensor keys
retained, 992 tensors changed and 37,205,002 changed values. All updated values
are finite. Step-1 broadcast weight SHA-256 is
`f1f056435ef0968f77389b94fa1115436a3d679e742ac855f3c50177bba88970`;
the [small weight audit](first-update-audit.json) records the immutable baseline
hash. Inference logged a successful step-1 LoRA reload at 13:55:28 UTC.
One update verifies the learning path, not visual improvement or reward scaling.

At 13:59 UTC, 30 saved live judge comparisons contain 29 scored decisions
and one unresolved three-of-four vote, which remains censored. Twenty-eight
agreed initially; one resolved after balanced rechecks. These are transport
and consensus counts, not measured visual accuracy. No rubric was weakened.

A completed five-minute capture covers 13:53:42–13:58:32 UTC, 30 ten-second
samples per card. Inference GPU 0 averaged 100% reported utilization, 30/30
nonzero samples; trainer GPU 1 averaged 17.33%, 8/30 nonzero samples. The
window includes the end of update 1 and later batch waiting; it is not
run-wide utilization and the full pipeline is not saturated. The earlier
startup window (13:31:50–13:41:40, 60 samples per card) averaged 46.13% on
inference with 31/60 nonzero samples, versus 0.117% on trainer with 2/60
nonzero samples. Do not report Prime's 37.6% MFU as measured hardware
utilization: its log explicitly substituted A100 peak FLOPS for this GPU.

Next decision remains the registered finite reward checks and matched 47-case
evaluation. Current evidence supports continuing the bounded run, not
promoting the model. Public checkpoint and screened evidence verification
precede the exact-pod release. No scheduled task or recurring watcher exists.

## Operational snapshot: 2026-09-30 14:37 UTC

Trainer updates 1–3 completed, with gradient norms 0.01019287109375,
0.005523681640625 and 0.013671875. Batch 4 dispatched at 14:36:24 UTC;
its optimizer completion was not yet present in the subsequent check.
Admitted-batch rewards are 0.1458333, 0.09375, 0.2395833 and 0.0520833.
These fluctuate across different curriculum briefs and do not establish an
upward reward trend, regression, or a held-out visual-quality gain.

Update 2 waited 4m11s for a batch versus 2m51s active training; update 3
waited 16m50s versus 3m11s active. Rollout collection limits throughput.
Allocator segment-mapping OOM warnings appeared before completed updates
2 and 3, and during batch 4. They are not a demonstrated fatal trainer OOM;
peak memory is near 92.7 GiB and memory pressure remains an unresolved risk.

Eighty-four saved visual comparisons contain 78 scored judgments and six
censored cases: five unresolved three-of-four consensuses and one failed
initial call. Seventy-two agreed initially and six resolved on balanced
rechecks. These are coverage/consensus counts, not judge accuracy. Complete
groups retain the same admission policy; uncertain judgments are not zeros.

At this inspection no public step-4 checkpoint was yet verified. Current
rental-clock estimate is $2.78, approximately $4.74 with all prior setup
and failed attempts; neither is an invoice. The under-$20 combined cap and
15/20/every-five admitted-batch reward checks remain active. No new node,
sampling change, or scheduled task was introduced.

## Operational snapshot: 2026-09-30 15:37 UTC

Nine optimizer updates are complete; every recorded gradient norm is nonzero.
Rewards for admitted batches 1–9 are 0.1458333, 0.09375, 0.2395833,
0.0520833, 0.1041667, 0.125, 0.28125, 0.15625 and 0.1041667. Overall mean
is 0.1446759. First-five mean is 0.1270833 versus latest-four mean 0.1666667;
these unequal windows contain different curriculum briefs and must not be
presented as a matched learning result. The first registered check is at 15.

Step 8 is public in `CK0607/qwen3.8-27b-brush-painting`, revision
`70b9000fcc035e08a540523ee04b3f379f8574c5`, path
`text-sft512-consistent-20260930/step_8`. The node's public upload receipt
and local adapter agree; anonymous parent HF metadata independently matches
weight SHA-256 `00a9416cf7ce019e718595dc753ae0109fba1a21ac467eb3cb79eb78ea50eaa8`.
Matched painting evaluation has not run, so the checkpoint is not promoted.

There are 168 saved visual comparison receipts: 157 scored and 11 uncertain.
Update 7 waited 11m15s versus 2m34s active training; update 9 waited 9m09s
versus 2m18s active. Rollout collection remains the main throughput limit.
Allocator mapping warnings recur but these updates completed, with peak
memory near 92.8 GiB. No new interval utilization measurement is claimed.

The elapsed-clock GPU estimate is $5.162 for the current node plus $1.958576
for earlier attempts: $7.121 total, not an invoice. Registered reward checks,
the combined under-$20 GPU ceiling, public preservation and exact-pod release
remain active. No additional machine, scheduling action or hyperparameter
change occurred in this status check.

## Throughput diagnosis: 2026-09-30 15:58 UTC

Ten optimizer updates have completed. Update 10 took 13m37s: 10m46s waiting
for its batch versus 2m50s active training. Other recent waits range from
8m42s to 16m50s, with roughly 2–3 minutes of active training. This is an
inference/environment supply imbalance, not a slow optimizer alone.

The latest admitted batch contains 96,387 generated tokens across 16
trajectories, averaging 6,024 output tokens per trajectory and three turns.
The last 30 ten-second inference log reports ending 15:57:51 UTC average
236.3 aggregate output tokens/s (median 243.05). At that observed rate, the
latest batch's tokens alone imply approximately 6.8 minutes of decode time;
this is an indicative calculation, not a measured matched batch breakdown.
Rendering, sequential inspection, judgment, group stragglers and discarded
work add latency. The same batch reports packing 0.082 seconds and explicit
policy-barrier waiting 0.0 seconds, so neither explains its collection time.
These throughput logs are not a new GPU utilization measurement.

A live census of 960 render receipts finds median 8.64s, p90 84.25s and
maximum 180.03s; 57 attempts exceed 170s. Valid attempts have median 20.38s;
invalid attempts median 2.339s. In a following census, 55 of the near-timeout
attempts are invalid and two valid. The render errors include syntax failures,
undefined helper variables, missing brush/color setup, and no canvas. Error
messages may occur twice per receipt; their raw counts are not distinct
failure counts. Intermediate-turn invalidity is not terminal-episode failure:
an episode may retain a valid earlier canvas, and in-flight episodes are
included in this census. The 622 invalid receipts in the first census must
not be described as 622 failed trajectories.

There are 203 saved judge comparisons: 189 scored and 14 uncertain. Across
473 saved API-call receipts, median latency is 22.10s, p90 45.97s and maximum
240.12s. Order views run in parallel; adding their times is not episode wall
time. Rendering and provider tails still matter despite reasonable medians.
The orchestrator recorded one 14:19:22 UTC collection window discarding
40/56 episodes: stale=16, errored=3, no_signal=21. This is that window only,
not the run-wide discard rate. Whole-group guard counters were not visible
in the inspected logs; no rejection percentage is inferred from uncertain
judge counts.

The immediate conclusion is that useful rollouts per dollar, rather than
raw trainer speed or larger batches alone, needs improvement. Before scaling
the next run: measure cumulative admission/discard reasons, distinguish model
syntax errors from renderer timeouts, make deterministic syntax/runtime
failures return promptly, and benchmark inference-heavy allocation against
the present one-inference/one-trainer layout. Raising off-policy allowance or
changing reward admission solely to increase throughput could change the
learning algorithm and needs a separate comparison. Do not silently shorten
quality budgets or alter the current registered judge/stopping rule. The
current finite run remains under its existing combined $20 cap; no additional
rental, hyperparameter change or recurring automation was made during this
inspection. Update-10 reward is 0.1979167; no matched visual gain is established.

## Operational snapshot: 2026-09-30 16:52 UTC

Fourteen optimizer updates have completed. Latest completed gradient norms
(12–14) are 0.0072, 0.0110 and 0.0120. Admitted rewards 11–14 are 0.25,
0.0729167, 0.2395833 and 0.1666667. First-five mean is 0.1270833 versus
steps 6–10 mean 0.1729167; these are different curriculum briefs and not
matched evidence of a visual-quality gain. The first registered rolling
five-versus-five decision remains step 15. No rubric or sampling change.

Step 12 is public at revision `57cff68221b225ab448a2fdc5042ba7cc6db5fd2`,
path `text-sft512-consistent-20260930/step_12`, in
`CK0607/qwen3.8-27b-brush-painting`. The verified upload receipt reports
weight SHA-256 `5002613fc1757911e66fad9a2b70bdcb907041609a1f3a4b015abdd6a8db35d8`;
an anonymous parent HF tree API check independently matches that LFS OID.
Matched evaluation remains pending. Update 13 took 24m50s, update 14 6m30s;
rollout supply remains variable and insufficient for continuous training.
Estimated combined elapsed-clock GPU spend is $10.09 including earlier
attempts, not an invoice. Supervisor remains running and release-ready is
absent; the finite cap, reward stopping, preservation and exact-pod release
continue. No new rental or scheduled task was created.

## Step-15 trend decision: 2026-09-30 17:25 UTC

The registered step-15 admitted-batch check continued training: steps 6–10
mean 0.1729167, steps 11–15 mean 0.1854167, delta +0.0125. This is a small
positive rolling training-reward change, not a matched held-out gain or
proof of judge accuracy. The next registered decision is at admitted batch
20. Sixteen optimizer updates are complete; batch 17 dispatched with reward
0.2395833 but its optimizer completion was not yet present. Update 16's
reward is 0.1666667 and gradient norm 0.0072.

The node reports a verified public step-16 checkpoint at revision
`cd43b7eb3aecc03d461b1d65816616b356f1bf58`, path
`text-sft512-consistent-20260930/step_16` in
`CK0607/qwen3.8-27b-brush-painting`. No new anonymous parent hash check was
performed in this snapshot; the existing node verification receipt is the
evidence for this publication. Matched visual evaluation has not started.
Supervisor is still running; release-ready is absent. Estimated combined
elapsed-clock GPU spend is $11.40 including prior attempts, not an invoice.
The combined under-$20 ceiling, finite preservation/release and existing
reward rules remain unchanged. No new rental or scheduling action occurred.


## Reward stop and finite profiling — 2026-09-30 18:24 UTC

The registered batch-20 check stopped training: batches 11–15 averaged
0.1854167 versus 16–20 at 0.1770833 (delta −0.0083333). The native trainer
completed **19 optimizer updates**; admitted batch 20 was dispatched but did
not complete an update. Supervisor field `last_optimizer_step=20` reflects
its admitted-metric counter, not completed optimizer evidence. Preserve this
scope distinction in curves and future supervisor naming.

The last completed broadcast adapter (step 19) was separately validated and
published, with 992 finite tensors and 116,727,808 elements. Public revision
`d3542ce4ff03f9be0c8bfaf4a25e60ea58b66219`, path
`text-sft512-consistent-20260930/step_19`, weight SHA-256
`9481e1aa32e33a9e8eced203656168dcfaabce995bfab337c86f386b7d7d4aa0`,
233,590,600 bytes. Anonymous parent metadata independently matches that hash.
This is an adapter-only recovery, without optimizer state, and is **unevaluated**.
The already-started matched evaluation stays bound to public step 16.

Reviewed profiling separates 19 matched trainer timing rows: wait
12,140.049 seconds and active training 2,931.4469 seconds, denominator
15,071.4959 seconds (80.55% waiting). This timing fraction is not GPU
utilization. The independent 17:41:52–17:44:52 UTC telemetry window contains
19 ten-second samples per GPU: inference mean 99.95%, 19/19 nonzero;
trainer mean 0%, 0/19 nonzero. The pipeline is not saturated.

There are 1,773 observed render receipts: 623 valid and 1,150 invalid,
including intermediate attempts; these are not trajectory success rates.
Render median 8.86 seconds, p95 180.013 seconds, and 123 timeout receipts.
Twenty-eight structured failed traces are observed. Repeated syntax, undefined
helper and brush-method errors are overlapping error mentions, not unique
failures. The 365 judge receipts contain 337 scored and 28 uncertain outcomes;
861 actual HTTP attempts have median 21.93 seconds and p95 53.94 seconds.
Logged discard warnings total stale 40, error 14 and no_signal 90 across four
threshold-warning windows only; they are not a complete run census. Prime's
residual no_signal can include guard rejection and is not evidence of equal
rewards. Guard counters and the generation-token telemetry counter remain
unavailable; do not infer them.

The matched 47-case evaluation began with a hardcoded concurrency of four,
despite a 16-sequence inference server. Ten complete, error-free episode
records existed at 18:21 UTC; visual validity and preference scoring are
still pending. Its existing finite generation allowance is 30 minutes,
so missing cases may be censored. Exact verifier resume compares every
resolved config field, including concurrency. No live config provenance was
rewritten and no risky parent-process pause/restart was performed. The next
launch explicitly configures concurrency rather than inheriting this hidden
four-episode default. It must include a completion-time estimate for the full
held-out split within the rental allowance; a partial panel cannot be called
47-case validation.

Profiling source `painter/profile_rl_evidence.py` excludes validation,
anchors, teacher candidates, credentials and weights. The accepted collected
profile is `painter/collected/spatial-readability-20260930/profile/profile-final.json`
and its Markdown companion; earlier live/reviewed drafts contained corrected
scope/timing errors and are superseded, never results. Raw utilization CSV
is retained alongside them. Snapshot GPU cost including earlier attempts is
approximately $13.75 at 18:24 UTC, clock-based rather than an invoice.
No new rental, training recipe, reward prompt, sampling budget or recurring
watcher was introduced.

Next decision: finish finite evaluation and evidence preservation, then
release the node. Do not scale this recipe from online reward alone. Prioritize
safe fast syntax/API rejection, corrected helper conventions and actual
inspect–repair demonstrations, followed by measured valid-rollout throughput
and explicit evaluation concurrency before another paid training run.


## Preservation audit and release — 2026-09-30 18:36 UTC

The registered training stop was followed by a **partial evaluation failure**:
15/47 episode records completed before the finite 30-minute evaluation timeout
(exit124); the other32 are unobserved/censored, not model failures. All15 completed cases are from the complex family; this non-random prefix
contains no simple/intermediate validation cases. All15
completed episode wrappers report ok=true, which is operational completion,
not visual success. Public step16 remained the evaluation policy; step19 is
preserved independently and unevaluated.

The screened weight-free evidence archive is public at
`CK0607/komorebi-painter-experiments@c4f05445e70442dbab465b667c1854e4f9d206bf`,
path `runs/text-sft512-consistent-20260930/interpreter-retry/evidence.tar.gz`,
507,241,220 bytes, SHA-256
`4ca075b3794e2434b8b6eaec0bc2a2884eeaead20bc92d6001466d3ac9cdfbcf`.
The parent anonymously downloaded and verified the entire archive, then
retained selected small evidence and deleted its temporary large copy. All
eight new teachers' selected program/PNG pairs independently match both local
review hashes and the public archive's member hashes. Training renders,
raw traces, metrics, judge receipts and teacher trajectories are preserved.

**A backup coverage bug remains material:** evaluation final canvases were
written by the old native configuration under the repository-level relative
`evaluation-rollouts` directory, outside the owned run subtree. Consequently
none of the13 recorded valid final evaluation PNGs entered this archive. Saved assistant programs,
turn metadata and target canvas hashes remain recoverable from the traces;
these are not a replacement for the missing pictures. No pairwise visual
result may be reported until a Linux CPU replay reconstructs and verifies the
images (or explicitly labels non-identical replays). This failure also
invalidates any blanket statement that all evaluation renders were backed up.
The new configuration places evaluation artifacts inside the archived output
root and adds artifact checks before successful generation receipts. Code
changes are reviewed/tested separately and were not part of the stopped run.

The finite relay verified public upload before writing its release marker.
The owned node was released at **18:29:23.703 UTC**; a subsequent provider
listing confirms zero active pods. Current-node elapsed-clock estimate is
approximately $12.006, plus $1.9586 earlier failed/setup attempts: **$13.9647
combined GPU estimate**, below the $20 ceiling. It is not a provider invoice.
Gateway API charges and account agent usage are separate. No new GPU run or
scheduled task was started. Source `0772bb6` remains the training identity;
profiling/completion repairs receive their own reviewed Git commit.

Next decision: recover the saved evaluation programs on CPU before interpreting
quality, and validate output-root/backup coverage plus explicit concurrency
before another GPU launch. Do not spend GPU time to recreate deterministic
render artifacts or count missing files as poor model paintings.


## Reviewed repairs and teacher publication — 2026-09-30 18:47 UTC

The native config now writes an absolute evaluation artifact directory beneath
its archived output, and the launch script explicitly defaults to16 concurrent
episodes. Successful generation receipts require adjacent hash-matching PNGs
for valid renderer receipts; symlinks and paths escaping the artifact root
are rejected. These are output-integrity checks, not new aesthetic reward
surfaces. Thirty-four focused local tests and shell syntax checks pass; no
Linux model/render run was launched with this repaired source, so actual
throughput improvement remains unmeasured.

CPU replay preparation recovered13 source programs corresponding to recorded
valid final canvas hashes; two of15 trace records had no valid final canvas.
The other32 cases did not complete. The ignored inputs and manifest are under
`collected-final-partial/replay-inputs-cpu/`; seed0 and pinned renderer/assets
are recorded. FINISHED/no-change/failed-last-turn cases resolve the actual
prior valid painting rather than blindly replaying the last response. Source
preparation is idempotent and does not render on the Mac. Pixel-hash agreement
must be checked after Linux CPU replay; reconstruction is not yet complete.

The node archive's snapshot predates the last teacher inspection metadata,
so complete finalized teacher data was independently screened and published
from the local authored/render evidence, without another GPU rental. Dataset
`CK0607/komorebi-painter-teachers@13c5ea1fc9221eaee7eb06d755b218660f26f4e0`,
path `spatial-readability-20260930/teacher-eight.tar.gz`, has139 files,
3,049,378 bytes and anonymously verified archive SHA-256
`f5628221255f51860613555a79a6e80c584f49f47c2766687987c4a2f5011401`.
It contains the final authored metadata, actual critiques, source programs,
render receipts/images and side-by-side review. There are no human labels or
SFT admission/training-benefit claims. This supplemental data upload corrects
the incomplete metadata snapshot; it does not restore missing RL eval PNGs.

## CPU recovery and reward audit (2026-10-01)

The user approved recovery of the missing evaluation canvases and requested an explanation of inconsistent reward improvement. No new GPU run is launched. Replay the 13 saved code-bearing final programs from the step-16 partial evaluation on Codex Cloud Linux CPU with the original seed 0 and pinned renderer/assets. Compare recovered PNG hashes with the original recorded hashes before assigning provenance. Evaluate against the existing frozen SFT-512 anchors using the same consistent joint-view judge. The scope is 13 recoverable complex-family cases, two completed episodes without a valid final canvas, and 32 unobserved/censored cases; it is not a representative result on all 47 held-out briefs.

Separately join archived raw episodes to admitted-batch and trainer annotations. Test whether the small recent reward decrease is explained by changing prompt mix, invalid programs, judge disagreement and whole-group rejection, policy lag, or missing learning signal. Preserve the previous operational stop decision. Future changes must distinguish fixed-panel quality progress from noisy reward on changing training prompts, and optimizer-completed from admitted batches. Rendering and model execution remain on Linux; local work is source extraction, metadata analysis and API review only.

### Verified reward evidence and implementation changes

The archived stream contains 584 completed episodes in 76 group IDs, including 66 complete eight-member groups and ten partially observed groups. Final states: 337 judged canvases, 28 uncertain judgments, and 219 episodes without a valid final canvas. The orchestrator shipped 320 episodes in twenty batches; these contain 197 judged valid canvases and 123 model-invalid zeros. Native trainer completion logs cover only nineteen updates. Twenty admitted batches must not be relabeled as twenty completed optimizer updates.

Of the 66 complete observed groups, 23 contain at least one unresolved judgment. The whole-group guard excludes these groups. The raw evidence has one complete equal-reward group and no complete all-invalid group; this is not the Prime log's broader `no_signal` discard definition. Forty groups supplied the twenty shipped batches. Unknown or partial group outcomes remain separate.

In 144/365 episodes with a valid retained final canvas, the last paint attempt failed (131 program failures and 13 render deadlines). Ninety-three of these were shipped for learning. Fifty-eight received positive scalar advantage relative to their group. For 47 of those, the full token arrays aligned exactly and 102,987 tokens of the final failed assistant revision carried positive advantage. Overall 293/320 shipped trajectories align exactly; there are zero nonzero advantages on prompt tokens in this aligned subset. Twenty-seven unaligned trajectories do not establish token-level credit. This is coarse terminal-return credit under a rollback environment, not evidence that the prompt loss mask is wrong or that the optimizer did not update.

Among the saved judge receipts, 48/365 initial order pairs disagree; 26 resolve by three-of-four voting, 22 remain unresolved, and six other initial comparisons fail parsing. There are 337 scored receipts with discrete scores: 49 at zero, 129 at one-sixth, 98 at one-third, 45 at two-thirds, 14 at five-sixths and two at one. No scored ties occur in this sample. Subjective consistency and the whole-group exclusion cost both require attention; no unresolved judgment has been converted to an invented score.

A five-batch reward window contains only ten prompt groups, with changing prompt IDs and unequal validity. The observed decline of 0.00833 is small relative to group variation (prior-window group standard deviation 0.09892; recent-window 0.11547). An exploratory independent prompt-group bootstrap gives a broad interval approximately [-0.09375, +0.08333]; it is descriptive only, not a matched-policy or significance result. The registered stop still stands. Future continuation decisions should use a fixed matched validation panel with per-prompt comparisons and report overall reward, valid-canvas quality, validity and censorship separately. A monotonic online mean on different prompts is not a reliable acceptance criterion.

The provider-free 13-case CPU replay was dispatched to the existing signed-in Codex Cloud task. Its public, anonymously verified source bundle is `CK0607/komorebi-painter-experiments@3ebe8003ce1980cba8255cad188750c0e487ca13`, path `runs/text-sft512-consistent-20260930/cpu-recovery/input-bundle-v1.txt` (Base64 encoding of 345,879 archive bytes, decoded SHA-256 `076ee406f059163c6a1a682cc46d54749ff068974117875b6a3c34640be064bb`). The standalone runner SHA is `d1b262181c9597e56b938a267ba9b1d323d0ec6fd290686493a1f92d6c2f1ff7`. It uses three CPU renderer workers and compares exact original target hashes. Results and visual judging are pending; no extra GPU spend or scheduled task.

An opt-in adapter syntax check now uses bounded nonexecuting `node --check` before Chromium. It preserves the same program-error feedback and previous canvas; missing Node/startup/timeouts are operational failures. Eighteen native painting/renderer tests passed with subprocess mocks. The pinned renderer is unchanged. This can save browser work on malformed source but does not repair p5.brush APIs, expensive geometry, vendor errors or aesthetic quality. It is off by default until a Linux pilot verifies it.

### Vendor defect, reproducibility and next decision

The pinned p5.brush 2.2.1 asset contains an undefined identifier in `function xn(t)`: its zero-signed-area fallback for polygons with at least eight vertices reads `v[0]` instead of `t[0]`. The stream records 41 `v is not defined` failed turns. Static inspection and a pure Node regression establish this vendor defect; they do not prove that every one of those 41 failures had this cause. A versioned derivative helper verifies all original hashes and performs one guarded identifier replacement, preserving renderer/assets/licenses and writing before/after provenance. Original asset and renderer bytes remain unchanged for the thirteen-case recovery. The Node fixture reproduces the original ReferenceError and verifies fixed empty, short, zero-area and nondegenerate cases. Thirty focused native-loop, renderer, centroid and reward-audit tests pass, including the real Node fixture; five replay-runner tests also pass. No Linux end-to-end speedup or policy improvement is established by these tests.

The reviewed reward audit is public at `CK0607/komorebi-painter-experiments@2c11aa158f75ff854e7954bf58df12b1b3146a9a`, path `runs/text-sft512-consistent-20260930/reward-audit-v1`. Anonymous bytes match local JSON SHA `9a63f8df6a557e6b0076d01327179b487a6f09d2485cc1387e87dcd74872b575` (48,773 bytes) and HTML SHA `04a9fc52a94547745072888bbcf0121260ecbcc5b86bcff5a6417866f567dfc1` (5,525 bytes). Twelve briefs occur in multiple shipped batches. Their within-run first/last comparisons are useful diagnostics, not fixed held-out evaluation; changing validity can decrease overall return even when valid-canvas quality rises.

The next RL design should retain the frozen SFT-512 baseline and a single pairwise aesthetic reward, with no added detail/geometry bonus. Separate validity, valid-canvas score, judge censorship and fixed matched validation in reporting. Resolve the terminal-credit contract before training: a rejected revision must not receive positive credit merely because an earlier retained canvas won. Any changed final-submission or turn-credit rule is a new registered objective, not a retroactive correction of this run. Use real inspect/correct demonstrations with simpler spatial structure, then scale subject diversity; eight new complete demonstrations are available, but more data alone does not fix contradictory credit or vendor failures.

No GPU has been rented or RL restarted for this audit. The existing $20 per-run GPU ceiling and historical stop decision remain in force. Future stopping should be discussed against a matched panel rather than interpreting monotonicity in two changing briefs per batch as visual progress.

### CPU recovery and same-judge comparison completed

Codex Cloud CPU replay completed 13 cases with three renderer workers in 508 seconds, using Playwright 1.58.0 and Chromium 145.0.7632.6. Eight verified PNGs rendered; all eight differ from the original recorded pixel hashes and are classified as **new CPU replays**, not exact recovered original canvases. Five attempts exceeded the 180-second renderer deadline. These five programs were recorded valid on the original evaluation, so CPU replay failure must not be relabeled as original model failure. Successful CPU cases took 39.981–110.100 seconds (median 68.591); this selected complex-family CPU workload is not a comparable throughput benchmark to the previous GPU renderer timing sample. Long-tail expensive brush rendering remains unresolved.

Public recovery dataset data commit is `2909ea737799f298369f4b14162694734a078c66`, receipt commit `e8ccdbf765c185292d8f33cd245aedab62c0d1fa`, under `runs/text-sft512-consistent-20260930/cpu-recovery/results-v1`. Eight Base64 parts reconstruct the anonymously verified 2,842,335-byte archive with SHA `ed3c6bb1659ec32d520450572a4542ceaafb8ddf5be01b3d80ff0e96ac73abc0`. Root independently decoded and hash-verified those public bytes before local extraction.

All eight verified new replay PNGs were compared against the existing frozen SFT512 anchors using the unchanged side-keyed joint-sheet MiMo Flash judge, exact source SHA `f94cfc9e880c8423f89a091486a0cf174cf1818c13d9acdf18fd21ace79aa069`. Seven scored: three RL wins and four SFT wins, no ties, scored-only mean 0.476190. Pantry `t600-375` remained uncertain after four balanced judgments and is censored, not scored as a tie. Root visual inspection sees clearer jars/placement in this RL pantry, but the two crane paintings remain faint and incomplete; these are qualitative observations, not replacement labels. Twenty-two API calls report total Gateway cost $0.0081835376 with no missing cost fields. GPU spend this audit: $0 incremental. Scope remains eight new replays from thirteen saved complex programs, five CPU recovery timeouts, two original no-canvas episodes and thirty-two unobserved cases; there is no full held-out improvement claim.

After preserving the reviewed audit publicly, the 476,166,521-byte temporary raw-evidence download was removed. The original public hash-verified archive and compact source-hash/report/receipt evidence remain available. No original unique render, code, checkpoint or annotation was deleted.

### Reviewed recovery bundle and optional credit repair

The complete reviewed recovery bundle contains 87 screened files, 9,195,377 archive bytes and SHA `e9afb51bc0f3f8a8667dc51854ced0f62fe62e33c9afcf211919c989f6bf6d91`. Public dataset commit `077ac67f201f186bbedf1711be57c2208214b6c1`, path `runs/text-sft512-consistent-20260930/cpu-recovery/reviewed-results-v1.tar.gz`, was independently downloaded anonymously and byte-hash verified. It includes the new replay PNGs, original-target hashes, failures, frozen SFT anchors, briefs, all eight raw judge receipts, API usage summary and a simple two-image gallery. Source hashes are collapsible and preference labels are directional; 0.5 means relative tie, not absolute painting quality.

An optional `PAINTER_REQUIRE_VALID_FINAL_ACTION=1` contract now returns zero reward for a recorded invalid last action while retaining earlier-canvas provenance, and leaves valid finish/recovery/no-change outcomes eligible for the unchanged judge. Service failures or missing/malformed last-turn evidence remain operational errors. Default-off behavior is preserved and no active run configuration was changed. This is a proposed new terminal objective for a registered future pilot; it does not provide finer credit assignment for intermediate failed attempts. Twenty-four focused helper/RL integration tests pass; renderer/judge/active training configuration is unchanged. All thirteen saved valid programs also pass a real nonexecuting Node syntax check; this is parser compatibility evidence, not a renderer throughput result. No RL restart during this audit.
