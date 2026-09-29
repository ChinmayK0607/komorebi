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
32 updates, batch eight, group four, three turns, 8192 output tokens per
turn, LoRA LR `1e-6`, and checkpoints every eight updates. This exposes at
most 64 distinct prompt groups if the loader traverses once in order; the
68 admitted anchors are a reusable inventory, not a claim that every one
receives gradients in this short run.

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
for a comparable training-signal curve; log per-brief scores rather than
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
