# Text-paint SFT/RL interpolation canary (2026-09-29)

## Decision and hypothesis

The step-16 RL model improves some object detail but often muddies color,
crosses silhouettes, or replaces the step-512 SFT's controlled, warm shapes
with mechanical marks. Test whether a **small effective-weight blend** keeps
RL's useful structure while retaining SFT's cleaner composition. This is an
eight-scene development canary, not a population-level validation or a new
training run. Keep the SFT adapter as the default unless matched images and
held-out judgments support promotion.

The judge prompt must value legibility, clear color, clean boundaries, and
precision with warmth. A cooler but precise painting can still win; warmth
alone is not a substitute for accurate shapes. Penalize muddy dark masses and
overblended detail. The v2 and v3 MiMo Flash wording experiments each match
only six of nine user-stated directions in the selected gallery. Neither is
approved as an RL reward.

## Frozen inputs and matched comparison

- Base: `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- SFT control: public step-512 adapter revision
  `adf687e5b321514eda7be6eab050434fdc8e44ff`; adapter weight SHA-256
  `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
- RL control: public step-16 adapter revision
  `ac4d3eda8e5925e2d23992cfdc1a63620ebde573`; adapter weight SHA-256
  `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`.
- Blend math: rank-32 concatenated LoRA representing exact effective-delta
  interpolation, with planned RL fractions 0.25 and 0.5; save hashes of both.
- Development scenes: potter unloading, yarn ball, hand fan, observatory,
  seahorse, table lamp, hand-crank drill, and octopus. These were selected
  from user-discussed examples, so the test measures tradeoffs on known
  failures rather than generalization. The eight text briefs and manifest are
  in ignored `local-evidence/merge-canary-20260929/`; manifest SHA-256
  `0a3cbf955f05f17c8093b6ca494181037537aa9dc319ce4ca6d5a7ab10a6bef8`.

Use identical brief, seed, temperature, turn count, token limit, renderer,
and model revision for each policy. Report invalid/timeout cases as censored,
not losses or wins. Preserve prompts, programs, canvases, render receipts,
token counts, and a side-by-side gallery. Judge preference against SFT is
exploratory until the rubric agrees on separate reviewed examples.

## Pending results

Record source commit, runtime versions, adapter hashes, rendered counts,
program failures, elapsed time, inference and CPU rendering utilization,
GPU cost with measurement window, matched visual findings, and the next
decision after the canary. Do not claim an RL gain from this test alone.
