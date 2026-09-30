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
8,192 completion tokens per trajectory, group eight and batch sixteen
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
