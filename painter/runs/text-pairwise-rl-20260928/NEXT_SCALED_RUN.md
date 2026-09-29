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
- Prompt pool: [public expanded v2 archive](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/07b5852217f40aef7ee6dd165c5187b6c50c3663/pools/text-paint-expanded-v2-20260929.tar.gz),
  archive SHA-256 `532602ac8ecce253499cc6e556b90966db5d7a9d4401303d4d788a8532ad08a4`,
  manifest SHA-256 `87ed249fa39277da0649e695a682701af13f2a96906017c1c9478983a6c57489`.
  This preserves 40 original train, 16 original validation, 16 original
  reserve; adds 186 complex train and 32 complex validation prompts. No new
  validation or reserve prompt may enter training or prompt tuning.
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
Admit 24 simple, 16 intermediate, and 64 complex **training anchors** from
the frozen pool; keep their order simple → intermediate → complex. Target
32 updates, batch eight, group four, three turns, 8192 output tokens per
turn, LoRA LR `1e-6`, and checkpoints every eight updates. This exposes at
most 64 distinct prompt groups if the loader traverses once in order; the
104 admitted anchors are a reusable inventory, not a claim that every one
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
on the **same frozen validation anchors**. Freeze a small fixed train probe
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

Pool v2 was uploaded and anonymously hash-verified. The schema repair and
generalized anchor/config builders are locally tested, but the new action
contract has **not yet been tested on GPU**. No node is currently rented and
no optimizer update in this planned attempt has occurred.
