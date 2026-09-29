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

## Completed canary and decision

The node ran committed source `92606cd` for inference and rendering; reporting
and archive tooling came from `29e5bcf`. The frozen input manifest SHA-256 was
`0a3cbf955f05f17c8093b6ca494181037537aa9dc319ce4ca6d5a7ab10a6bef8`.
The two exact effective-delta adapter hashes are:

| Candidate | Adapter SHA-256 | Public revision |
| --- | --- | --- |
| 25% RL | `8cf74e474f57456bc3128963e32536088f173acab6e3feb28b7db0dff9a68309` | [`1730c2e`](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/1730c2e4a8fd59556bf59e5b29122be9c16743a3/text-pairwise-merge-canary-20260929/mix025) |
| 50% RL | `45f812ddfcd3e88ac3252abd5737617ef68fe9c3dcd18575cdb7f8976b1b0555` | [`be1db8f`](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/be1db8f1ff7b0f9e4ec73c9051bd6d969838ce17/text-pairwise-merge-canary-20260929/mix050) |

All four policies completed eight matched two-turn traces, each using the same
brief, inference settings, seed, renderer, and base revision. Each produced
**seven valid final canvases**. The domed observatory was censored for all
four: hundreds of tiny brush polygons in its generated programs exceeded the
180-second software-render deadline. Valid-turn counts were SFT 13/16, RL
13/16, mix025 12/16, mix050 12/16. These counts do not establish visual
quality. Completion-token totals were 26,693, 28,417, 27,400, and 26,745,
respectively. The comparison is selected development data, not held-out
validation, and no judge score is promoted from the current calibration.

[Review the per-turn gallery](local-evidence/merge-canary-20260929/gallery/index.html).
On visual inspection, SFT keeps the cleaner hand-fan fan-edge and a legible
table lamp; the 25% blend gives the lamp a rounder shade but loses the second
turn there. The 50% blend yields a more recognizable octopus with distinct
arms, yet its yarn needles are scattered and it loses the drill's coherent
mechanism. The RL drill has a clearer large gear than either blend. The
potter scene is dark in all four outputs; none demonstrates a clean solution
to the user's muddy-wash concern. These are qualitative observations on a
known-failure set, not preference labels. **Retain SFT step 512 as the
default.** Neither blend is promoted or used as a new RL initializer.

The [screened 297-file public archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/2cb11d47c68312740ff046fae2d83d7e66f106ca/runs/text-pairwise-rl-20260928/merge-canary-20260929.tar.gz)
contains the per-turn gallery, programs, traces, manifests, and failures.
Anonymous full-download SHA-256 verification passed:
`dc6550832c2449e967df80b5da311f4bd6a95ff6f6eea8702c7330ce2e1f7712`.
Both public blend adapters also passed anonymous full-download hash checks.
The GPU pod existed from 12:26:40 UTC until release at about 14:07 UTC;
the matched inference evaluation ran 13:12:30–13:58:55 UTC. Lium reported
**$2.17** spent for the whole rental at its final reading ($1.29/hour),
below the $25 per-run cap, and `lium ps` then returned no pods. No
time-series GPU or CPU-utilization sample was collected for this evaluation;
the rental is not evidence of full-pipeline saturation. The judge prompt
experiments were offline and did not contribute training gradients here.

The [renderer profile](RENDER_PROFILE_20260929.md) shows approximately 2×
batch throughput from four concurrent software-render workers, but no
single-paint latency fix. Hardware WebGL was not validated on either graphics
probe. Future training should not use a faster-render claim until matched
pixel and timing tests pass; very long programs need a separate design to
avoid 180-second timeouts. The revised [v4 beauty prompt](JUDGE_BEAUTY_V4_DRAFT.md)
explicitly calls out muddy darkness and precision with warmth while allowing
cooler precise paintings to win, but it still fails critical hand-fan/yarn
spot checks and is **not** a production reward. The next curriculum decision
is to collect broader, independent visual preference labels and address the
renderer bottleneck before more RL, with text-to-paint first.
