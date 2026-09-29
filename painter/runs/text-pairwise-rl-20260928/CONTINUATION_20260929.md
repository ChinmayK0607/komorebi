# Broader text-paint RL continuation (prelaunch protocol)

## Question and matched controls

Starting from the public, hash-verified **unmixed step-16 RL LoRA**, can a
larger and more diverse text-brief curriculum improve paintings without
amplifying its known muddiness, broken silhouettes, or mechanical marks?
The controls are the exact Qwen3.8-27B step-512 SFT and the unchanged
step-16 RL policy. The 47 previously frozen held-out text briefs remain
outside training. Compare the new checkpoint with both controls under the
same three-turn, 8,192-token, seed-918210 renderer/evaluation protocol.
The prior step-16 result is mixed, not a demonstrated general gain.

## Immutable starting point and data

- Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Start: public step-16 RL revision
  `ac4d3eda8e5925e2d23992cfdc1a63620ebde573`, adapter SHA-256
  `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`.
- Frozen same-brief opponent: public SFT step-512 revision
  `adf687e5b321514eda7be6eab050434fdc8e44ff`, adapter SHA-256
  `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
- Prime revision: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
- The [public 184-brief run pool](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/e2a3aa2a3d9f9ecbddd67f7c5b356b5096c1a8bc/pools/text-paint-run184-20260929.tar.gz)
  and the [public prior evidence](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/e39e6e58fc830eb7be0a77a8bdde7d906943d1d3/runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz)
  fix the prompts and frozen SFT canvases. Their archive SHA-256 values are
  `18a6bfd0cf38c0bb7b02625282220702ae8e7a45ec417c7c63a119d0bcad08ec`
  and `ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3`.
  Rehydration verifies every reused brief and canvas hash. Exactly 119
  training cases should remain: 24 simple, 16 intermediate, 79 complex;
  one complex training case and one complex validation case have no usable
  frozen SFT canvas. The 47 usable validation and 16 reserve cases do not
  enter training. Keep the first 40 foundation cases, then put the 31
  previously unused complex cases before the 48 old complex cases. This
  increases potential distinct training groups from the old 68-row inventory
  to 119 while retaining a simple-to-complex curriculum.
- The reward is the **same** two-order MiMo-v2.6-Flash seven-level holistic
  comparison against the frozen SFT canvas. No new additive detail, speed,
  or geometry reward is introduced. Its preference for literal detail is a
  known limitation. The amended beauty prompt is not calibrated for use as
  training reward. Invalid programs and service failures remain distinct.

## Finite execution and spend

Use the known-good two-96GB-GPU Prime layout, one inference and one trainer,
instead of an untested 48GB split. Target **50 optimizer updates**, batch 8,
group 4, three turns, 8,192 output-token ceiling, LoRA LR `1e-6`, exact
8/8 in-flight window, and public checkpoints every five updates. This is a
fresh optimizer initialized from the step-16 weights, not a restoration of
the old optimizer state. Preserve the entire generated config hash and the
actual source commit before launch. Do not count speculative generated traces
as admitted gradients. Reusing the 182 verified SFT canvases avoids a new
184-brief baseline render.

The previous explicit ceiling was **$25 GPU rental per run**. Aim to finish
50 steps but enforce a finite 10-hour TTL at a quoted $2.38/hour or cheaper
equivalent, leaving 75 minutes for evaluation, public upload, and removal.
If the spend deadline arrives first, preserve the latest completed public
checkpoint and report the achieved step count. Vercel judge charges are
recorded separately. No recurring watcher or scheduled task is created.
Only technical launch failures, an exhausted budget, or clear inability to
admit nonzero-gradient batches should interrupt training; a noisy reward
dip alone is not an automatic stop.

The committed preparation source is `ed234af61fd475c5bbef849fe90bd068fa7b820e`;
the transferred Git bundle SHA-256 is
`32f28182f34cb502a5a516dbffdf65ea3e8d655238e9a5e1a178c4e1efe3515d`.
Lium provisioned the quoted two-card 96GB node at **$2.38/hour**, created
2026-09-29 16:24:00 UTC with finite removal set for 2026-09-30 02:23:51 UTC.
The quoted maximum is $23.80. No optimizer update had started when this
provisioning entry was recorded.

### Early live audit, 2026-09-29 17:44 UTC

Seven CUDA optimizer updates completed, each with nonzero gradient norm. The
first six admitted-batch mean rewards were 0.156, 0.344, 0.260, 0.146, 0.240,
and 0.302. These are comparisons with a strong, frozen SFT canvas, not an
absolute quality scale; 0.5 is an effective tie. The first 67 saved generated
traces (including speculative or stale traces, **not** the admitted gradient
denominator) all belong to the simple family: 49 finished with valid canvases
and 18 did not. Their aggregate mean was 0.240, while valid canvases averaged
0.328; 22 traces scored zero, including four valid canvases. Final-invalid
traces had syntax, JavaScript runtime, brush API, and render-time failures.
Among 49 visual receipts, 27 had nonzero two-order score disagreement, with
10 gaps of at least 0.333. The run therefore has both a model-validity problem
and a judge-preference/reliability limitation; the early reward is not evidence
of held-out improvement. Keep the finite run active while gradients and public
checkpoints continue, then decide from matched validation and actual paintings.

GPU telemetry over 80 ten-second samples per GPU, 16:43:20–16:56:40 UTC,
showed inference GPU 0 mean utilization 94.97% and busy fraction 95% at the
90%-utilization threshold; trainer GPU 1 mean utilization 10.21% and busy
fraction 8.7% at that threshold. This is an inference-bound, not saturated
two-GPU pipeline. At about 17:44 UTC, Lium reported $3.22 accrued rental.
The step-5 checkpoint is publicly uploaded and anonymously hash-verified at
HF revision `c3337e02b995f3b97e219232f22f882498a4b55f`; its adapter
SHA-256 is `f860a7a32ca731a482bc79428477b4d804c5e71704c48c1f646c027b1b87b3b2`.

At 18:08 UTC a post-run evaluation dry run found that the prior public
archive's native SFT traces retain absolute canvas paths from the released
node. The files are present in the same archive's portable, hash-verified
gallery, but following the stale paths would falsely mark 41 valid SFT cases
invalid. The post-run direct comparison now reads both SFT and step-16
paintings from those portable gallery images, with the original 47-case
status summaries. A gallery-only dry run found all 41 expected images for
each control. This fixes evaluation preparation only; it does not alter the
active trainer, rollout reward, or data split.

## Evaluation and next decision

Read the run-level reward, gradient, invalid/censored, rendering-time,
throughput, and GPU-utilization traces with stated measurement windows and
denominators. Evaluate at least the last completed checkpoint on the same
47 held-out text prompts, with the historical step-0 SFT and step-16 RL
controls at identical seed and settings. Preserve all canvases and turn
programs, including invalid cases, and inspect matched examples of fan,
drill, yarn, lamp, potter, octopus, and unseen complex scenes. Direct pairwise
visual scoring is exploratory until independent human preference agreement
is demonstrated. Report the all-case validity-inclusive result and the
both-valid visual subset separately; judge order conflicts stay visible.
Do not promote merely because online rewards rise.

At the end, if GPU time remains, run a **small held-out image-reference
transfer diagnostic** using the seven already frozen photo references from
the prior mixed-reference evaluation. Sample SFT, step-16 RL, and the final
continuation on identical reference images, seed, and turns. Do not train on
these images in this text curriculum or interpret a seven-case diagnostic as
photo generalization. If time is insufficient, preserve the final checkpoint
and run the image diagnostic separately instead of losing the public upload.

All final adapters and screened evidence must be uploaded to public Hugging
Face repositories and anonymously hash-verified before the rented node is
removed. Record actual GPU cost and confirm `lium ps` is empty afterward.
