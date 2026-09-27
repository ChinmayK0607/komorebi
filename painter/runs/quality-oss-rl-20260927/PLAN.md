# Quality-first OSS-judge RL, 2026-09-27

Status: one-hour node profiling and controls in progress; no optimizer step or
visual-quality claim yet. This is a finite experiment, not a scheduled task.

## Hypothesis and matched baseline

Starting from the visually preferred step-512 SFT adapter, a single holistic
visual preference reward can improve paintings against the same frozen SFT
policy. Speed reward is deferred. Compare identical references, turn limits,
sampling, renderer, and review layout at baseline and each RL checkpoint.
The 28-photo development set has been used previously and is not an untouched
test set. Report wins/ties/losses, invalid/censored cases, turn/token counts,
and category regressions, not only average reward.

Immutable initializer: `Qwen/Qwen3.8-27B` at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` plus public
`CK0607/qwen3.8-27b-brush-painting/photo-curriculum-sft-20260922-balanced/step_512`
at `adf687e5b321514eda7be6eab050434fdc8e44ff`; adapter SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Prime revision `26b3131d2716a4f8210b165df584b83b4bc54f61`.
The training manifest is the previously reviewed 2,486-row photo/procedural
source; its exact byte hash, reference coverage, and source commit are recorded
in the node launch receipt. The judge candidate is public
`XiaomiMiMo/MiMo-VL-7B-RL` at
`460c34be0c6cfe79b6b311647ae9112784f80b73`.

## Profiling decision

Use four RTX PRO 6000 Blackwell GPUs on one Lium node, provisioned at
$4.76/hour with six-hour auto-termination. Tentative roles: trainer 1,
policy inference 2, local visual judge 1. Measure generation tokens/second,
renderer wall time and process startup, judge latency/parse rate, trainer wait,
policy staleness, GPU busy fractions over an interval, and dollars per accepted
episode. Retain the old six-turn setting only if throughput supports it;
otherwise make a documented smaller-turn quality curriculum while keeping a
matched baseline. Preserve p5.brush as the executable medium.

Before training on the new judge, run image-identity, reference/wrong-subject,
and order-swap controls. For preference reward, evaluate both blinded
orientations; parse failures, uncertain views, and service failures remain
unscored. The model must see the reference and both final canvases. Keep raw
model responses and image hashes for audit. Check exact adapter initialization,
nonzero supervised rollout log probabilities, optimizer step, and weight
broadcast on a short end-to-end canary. Publish and hash-verify every retained
checkpoint in the public Hugging Face model repository.

## After the run

Record measured times, utilization windows and denominators, costs, controls,
loss/reward distributions, sample coverage, failures, and matched visual review
here or in a linked experiment record. A runnable setup without a successful
gradient step or a judged canvas is preparation, not improvement.
