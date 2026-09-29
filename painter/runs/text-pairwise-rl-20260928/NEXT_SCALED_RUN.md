# Scaled text-paint RL attempt (planned; no GPU run launched yet)

## Question and matched control

Can the pinned Qwen3.8-27B step-512 SFT painter improve its **visual
preference against a freshly rendered same-brief SFT painting** when trained
with the corrected code-only action contract, a wider prompt curriculum, and
the two-order MiMo-v2.6-Flash graded pairwise signal? The last 20-update run
had a weak positive original-validation result but an exact tie on independent
reserve; it is not a winning baseline. Every comparison in this attempt must
use a newly sampled step-0 SFT control under the *same* repaired action and
renderer protocol (`native-painting-eval-v3`, `native-painting-rl-v2`). The
older validation scores cannot be spliced into the learning curve.

## Frozen inputs

- Base model: `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Initial adapter: public step-512 SFT, revision
  `adf687e5b321514eda7be6eab050434fdc8e44ff`, weight SHA-256
  `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
- Prompt pool: [public expanded v3 archive](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/ff7f9877379cb538557911ffe7496830ca9391c3/pools/text-paint-expanded-v3-20260929.tar.gz),
  archive SHA-256 `482aa0d700ba42d5ae6b4dd8f09c4836863d38ccd24c4a287bbbb753c0c05c92`,
  manifest SHA-256 `712349be327ca2f5cc800aa42eb5529d0d9b9d54a3508a3aa3ed8c8c9c26b12d`.
  This preserves 40 original train, 16 original validation, 16 original
  reserve; adds 186 complex train and 32 complex validation prompts. No new
  validation or reserve prompt may enter training or prompt tuning.
- This rental pre-renders a deterministic [**184-brief subset**](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/e2a3aa2a3d9f9ecbddd67f7c5b356b5096c1a8bc/pools/text-paint-run184-20260929.tar.gz) to reserve GPU
  time for gradients: all 72 original foundation/validation/reserve briefs,
  all 32 complex validation briefs, and 80 of the 186 complex training briefs
  ranked by SHA-256 of `scaled-text-baseline-20260929:<id>`. The run-pool
  manifest SHA-256 is
  `494e4d9d76313faaae12fc7676d355c5041d138886d9ae1375f8f537fb8bcea6`.
  The public run-pool archive SHA-256 is
  `18a6bfd0cf38c0bb7b02625282220702ae8e7a45ec417c7c63a119d0bcad08ec`,
  verified by anonymous download.
  This selection is independent of model outputs; 106 other complex training
  briefs remain in the public inventory for later runs.
- Frozen opponent: first valid step-512 SFT painting for each brief sampled
  at seed 918200, then one retry at 918201 for missing/invalid canvases.
  Freeze canvas hashes and exclusions before training. Invalid SFT outputs
  are not quietly treated as weak visual anchors.
- Reward: one two-order, seven-level overall visual preference against the
  frozen same-brief SFT image. No additive structure/aesthetic/length reward.
  Judge disagreements and service failures are logged; service failures are
  censored, not scored as losses.

## Finite execution

Use two 96-GB GPUs: both for baseline rendering; then one for 27B inference
and one for training. The prior 19:34–22:20 UTC training window had inference
mean utilization 99.82% and trainer mean 11.65% across 998 ten-second
samples per GPU. Extra trainer GPUs therefore have no demonstrated value.
Admit 12 simple, 8 intermediate, and 48 complex **training anchors** from
the frozen pool; keep their order simple → intermediate → complex. Target
24 updates, batch eight, group four, three turns, 8192 output tokens per
turn, LoRA LR `1e-6`, and checkpoints every eight updates. This exposes at
most 48 distinct prompt groups if the loader traverses once in order; the
68 admitted anchors are a reusable inventory, not a claim that every one
receives gradients in this short run.
The 24-update target was fixed before optimizer launch from the measured
approximately one completed baseline brief per minute: 32 updates projected
too close to the hard 10-hour rental expiry to leave a credible held-out
evaluation, evidence upload, and node-removal window.

One rented node may cost **at most $25 in GPU charges**. Quote the actual
whole-node hourly price before rental, set a TTL whose quoted maximum is
below $25, and reserve time for evaluation, public hash-verified checkpoint
and evidence upload, and node deletion. The Vercel judge is a separate cost
ledger. A single end-to-end canary checks prompt serialization, valid
painting, both judge orientations, and a nonzero trainer gradient; if it
fails, fix the observed fault on the same rental and record spent time.

## Evaluation and decision

Before optimizer updates, independently sample the fixed step-0 SFT policy
on the **same frozen validation anchors**, with evaluation seed `918210` for
both policies at every checkpoint. Freeze a small fixed train probe
from the first four simple and first four complex rows in curriculum order,
then audit that those IDs actually appear in RL rollout traces. This yields
a comparable training-signal curve; log per-brief scores rather than
comparing changing online batch means. Score step 0, 16, and final on the
same prompt IDs and seeds when the budget permits. Evaluate the full usable
16-foundation + 32-complex validation pool at final, with censored and
invalid rates shown separately. Keep original reserve separate for a final
confirmation only if the validation result is positive. Pair and bootstrap
by prompt ID; report both the combined score (validity included) and the
both-valid visual subset by family. Publish the galleries and all judged
cases, including regressions. A larger online reward is *not* evidence of
learning when the prompt mix changes.

Promote only if the fixed train probe rises, full matched held-out validation
improves over fresh step 0, validity does not mask a both-valid visual drop,
and the untouched reserve confirms rather than reverses the result. Record
the observed uncertainty; no RL method can guarantee a positive result in
advance. If this attempt fails, preserve the checkpoint and failure evidence
without buying another automatic rerun.

## Status

Pool v2 was uploaded and anonymously hash-verified, but its rows mistakenly
declared `mode=text_to_image`, which the native environment rejects. The first
two-GPU baseline process failed immediately when both task shards loaded;
no student response was sampled and no optimizer update occurred. The cause
was a pool-mode mismatch, not a model failure. Pool v3 changes only the task
mode to `fresh` (and adds immutable parent/schema metadata); all prompt bytes,
splits, IDs, and hashes remain unchanged. It was anonymously hash-verified,
and all 290 rows loaded through `NativePaintingTaskset` in the pinned node
runtime before restarting baseline rendering. The 2×96-GB rental began at
2026-09-29 00:28:34 UTC at $2.38/hour with automatic removal at
10:28:22 UTC, a quoted $23.80 maximum. At 15 minutes, the provider reported
$0.59 GPU rental. The initial 290-brief v3 baseline was stopped after early
valid canvases showed throughput that projected beyond the time needed for
RL on this rental. The 184-brief run subset loaded through the native taskset
and is now rendering. The corrected action contract has produced valid SFT
canvases. A live two-order MiMo Flash judge canary preferred one valid SFT
painting over a same-size white blank with score 1.0 and order gap 0.0;
this checks transport/orientation only, not reliability on close pairs.
There is **no RL optimizer update or scored RL result** yet.
During an approximately 10-minute baseline window ending 01:23:39 UTC,
`nvidia-smi dmon` collected 60 samples per GPU at 10-second spacing: both
inference GPUs had 100% mean SM utilization and 60/60 samples at or above
90%. This is a sampled baseline interval, not a whole-run utilization claim.
At 01:40 UTC, 55 completed episodes contained 165 render turns. The native
trace timings summed to 16,374 seconds of model work and 4,410 seconds of
harness work across both parallel shards. Among 82 completed valid render
turns with a numeric duration, the median was 23.29 seconds, the 90th
percentile 128.79 seconds, and the maximum 161.58 seconds. These are
cumulative, overlapping task timings, not elapsed wall time. The renderer is
a meaningful contributor with a long tail even though model generation is
larger; the current matched comparison keeps its implementation fixed.
One additional live same-brief SFT-versus-SFT canary used training brief
`t600-310` (tea pickers on a misty hillside) from two independently sampled
canvases. MiMo Flash scored the newer canvas 0.667 with zero order gap. Visual
inspection found more legible curved tea rows and distributed pickers in
that canvas, versus a more abstract hillside in the other.
This single close-pair check supports image delivery and a plausible ordinal
signal; it does not establish judge accuracy across the curriculum.
The complete 184-case first SFT pass exited successfully. Before any
optimizer update, the automatic retry preparation failed because its new
manifest was placed in `prepared/` while its relative prompt paths still
pointed into `prepared/scaled-run-pool/prompts/`. No retry painting or RL
rollout was sampled in that failed preparation. The repair resolves and
hash-checks each prompt against the original pool before writing an absolute
path into the retry manifest; a relocation regression test covers it. The
failed log and manifest remain distinct from the corrected retry.
The corrected retry completed all 11 selected episodes. The immutable anchor
audit admits 68 training cases (12 simple, 8 intermediate, 48 complex), 47
usable held-out validation cases, and 16 reserve cases. Two prompts still
lack a valid SFT canvas after both fixed-seed attempts: `t600-308` (train)
and `t600-566` (validation). They are excluded from visual-anchor comparisons,
not recast as losses. Train manifest SHA-256:
`9a4cd3795cf3cf1e22d1d00eda8399e50bb699ecaaae683edb0d0b1262fa896d`;
validation manifest SHA-256:
`3ea6a74d10e921321671accb8d6aa0532da0b2cbfbced28add6e1d3ce753f14c`;
reserve manifest SHA-256:
`6fdf75f95f0ae7da202957357adf5e261366f8004c05cb1e8a6537ce93fcca39`.
The fixed eight-case train-probe manifest SHA-256 is
`ad927e7cf36952b36e9325cf6db5001a625be8915cc45d458f4a79702a72ab5b`.
The independent step-0 control now samples all 47 usable validation cases
plus that eight-case probe. Node source at this transition is `9d5c0c5`.

## Rollout diagnosis before optimizer launch (2026-09-29)

The previous margin-run evidence archive at revision
`180ea85f472f39dd7506adff4c764a7e5fc6053f` was downloaded again and
matched its recorded SHA-256 before this audit. Its Prime stream contains
226 complete training traces: 115 (50.9%) ended without a valid canvas and
110 reached a scored visual comparison (one remaining valid trace was not
scored). At turn 1, only 40/226 programs rendered; 47/226 rendered at turn 2
and 79/226 at turn 3. Among final-invalid cases, repeated missing JavaScript
blocks, syntax errors, nonexistent brush methods, and renderer timeouts are
visible in the turn receipts. This is an action-contract/programmability
failure, not proof that the visual comparator cannot distinguish paintings.
The old prompt explicitly requested a visual plan as well as JavaScript;
the new `native-painting-eval-v3`/`native-painting-rl-v2` contract removes
that contradiction and accepts a complete unfenced JavaScript sketch.

The same archive has 110 two-order judge receipts. In 15, the judge picked
the same *screen position* in both A/B orders (order gap 1/3) rather than the
same physical painting; their averaged scores became 0.5. Another 36
receipts had smaller nonzero order gaps. Averaging limits this position bias
but loses preference resolution; order disagreement must be reported in the
new run and a close mean win cannot alone establish visual improvement.
The saved validation/reserve evidence still shows a weak step-20 original
validation gain and a 6-win/2-tie/6-loss reserve tie. Visual spot checks
agree with the mixed outcome: the open pocket compass (`t600-450`) gained a
legible lid and dial, while the pinecone (`t600-532`) lost its layered scale
structure. The two chalk sticks (`t600-513`) were clearer in RL but remained
mechanical. These are selected examples, not a substitute for paired scores.

Decision: finish the already-running independent SFT control and run the
predeclared 24-update scaled text-only experiment under the corrected
contract, with no reward or curriculum change after seeing validation.
The first 30 new control traces had 25 valid final canvases; this is an
interim observation, not the final validity rate. Compare the fixed probe
and all usable held-out prompts against this fresh control, report invalid
and judge-order failure rates, and keep the original reserve untouched for
confirmation. After the finite training/evaluation and public hash-verified
uploads, release the rental rather than leave GPU compute running. If the
result is still mixed, do a deeper failure analysis from the complete new
rollouts before buying another run.

Before seeing a scaled RL checkpoint, the final evaluation was extended to
judge the *fresh SFT canvas directly against the RL canvas* for every case
rendered by both policies, in both image orders. This is separate from the
frozen-anchor reward and will report validity, visual-only preference,
order disagreement, and a prompt-bootstrap interval. The direct comparison
uses the same images and judge model, so it adds no model-generation GPU
work. It is intended to make an apparent anchor-relative improvement easier
to confirm or reject; it does not license selecting a different checkpoint
after looking at the held-out results.

The independent control completed all 55 episodes: 49 had valid final
canvases. On the frozen 47-case held-out set, 41 canvases were valid, all
47 cases comparable, none censored; mean preference against the fixed SFT
anchor was 0.395390. Its fixed eight-case train probe had seven comparable
cases, one censored judge case, and mean 0.488095 over those seven. Across
the control's 48 saved two-order judge receipts, 24 had no orientation gap,
18 had gap 1/6, and six had gap 1/3. These are step-zero/control statistics,
not learned performance. Prime preflight passed with `t600-509` first, and
the finite RL launcher began at 2026-09-29 04:35:55 UTC. The first optimizer
update and any RL checkpoint are still pending at this record point.
