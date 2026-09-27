# Four-reference Gateway RL overfit, 2026-09-28

Status: finite four-update run completed on a two-GPU Linux node. All four
adapters are public and hash-verified; matched visual evaluation is running.
Visual improvement remains unverified. This record distinguishes preparation
from results; raw rollouts and images live outside Git.

## Hypothesis and matched baseline

On four fixed references (sphere, bus, flower, and one COCO dog photo), RL from
the visually preferred Qwen3.8-27B SFT step-512 adapter should improve final
p5.brush paintings against the same frozen adapter. A real improvement requires
an actual optimizer step, changed adapter weights, and a matched render review
with identical references, generation limits, and renderer. This is an overfit
test, not a generalization claim. Quality is the only reward; speed reward is
absent.

Base model: `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Initial and frozen adapter:
public `CK0607/qwen3.8-27b-brush-painting/photo-curriculum-sft-20260922-balanced/step_512`,
revision `adf687e5b321514eda7be6eab050434fdc8e44ff`, weight SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Prime revision `26b3131d2716a4f8210b165df584b83b4bc54f61`.

The four-row manifest is derived from the 134-row regenerated profile set,
whose 48 procedural reference images were rendered from hash-checked p5.brush
source programs. The immutable profile archive is [public and hash-verified](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/main/runs/quality-oss-rl-20260927/profile-evidence.tar.gz)
(archive SHA-256 `b48659cdedae0121b7a4388658d97d7a760acf5cf7f4e0fe9d9c5d9ce8a028d0`).
Four-row manifest SHA-256: `58a5d1f06910d60d5b3e5726a1e73692a08d8739fc582847bf23e5fd7619c62a`.
Its row IDs are `sc-v1-train-sphere-0099`, `sc-v1-train-bus-0189`,
`sc-v1-train-flower-0199`, and `coco128-000000000490`.

Two RTX PRO 6000 Blackwell GPUs at $2.38/hour are split trainer 1 and inference
1. The node has a seven-hour automatic termination. The former four-GPU
$4.76/hour profiling node was removed after its archive was independently
downloaded and hash-verified from public Hugging Face. Combined GPU spend is
tracked against the user's $25 GPU-only ceiling. Gateway API use is tracked
separately.

The prepared run is four updates, batch 8, group 4, at most four paint/inspect
turns, 8,192 completion tokens, context 32,768, learning rate `1e-6`, and a
public adapter checkpoint requested every update. Config SHA-256 and public
checkpoint identifiers will be appended after verification. The prepared TOML
config SHA-256 is `b1ae9ab26b065d890931c0b707783a4b13c68f27d0da31a9fc68a7156356697b`.
The source commit `0e77820cbf592bc9a6f2fd24a20628bca2bdb128` was fetched
into the node as a Git bundle; active Gateway source file hashes match it.
The executable
medium remains p5.brush.

For a fresh Linux node, stage the HF token and Gateway `.env.local` as
owner-only files under `/root/painter/private`, clone the committed repository,
then run `bash painter/runs/quality-oss-rl-20260927/setup_node_once.sh`. It
downloads and verifies public inputs, installs the pinned runtime, constructs
the fixed manifest and config, and exits after a native preflight. Launch is
the separate finite `bash painter/runs/quality-oss-rl-20260927/run_overfit.sh`.
The private files are not part of the Git or Hugging Face artifacts.

## Judge and renderer checks

The Vercel AI Gateway judge is `xiaomi/mimo-v2.6-flash`. It sees a single
labeled contact sheet containing reference, A, and B, and returns weaknesses
for both paintings. The contact sheet fixed an observed A/B reversal on bus
and ribbons controls. One otherwise correct response used singular
`weakness_A/B` keys; the parser normalizes these aliases before enforcing the
closed preference schema. The queue smoke test scored both orientations,
installed an image-hash-bound receipt, and verified a bus over wrong-subject
control scored 1.0. Failed or uncertain judgments remain unscored. The
format drift and controls are preserved on the node and should be included in
the final evidence archive.

The initial renderer profile found 48/48 valid procedural images, mean
16.69 seconds per image and mean 14.42 seconds inside painting itself, with
12 parallel workers on the old node. On the two-GPU node, 16 images took
39 seconds at both eight and sixteen workers; eight workers lowered average
individual painting time from 18.66 to 15.28 seconds. Chromium is using
software SwiftShader WebGL; ANGLE GL and EGL probes did not obtain a WebGL
context. These are renderer measurements, not full RL throughput. Profile
actual rollout and trainer wait during the run before changing the renderer
contract.

## Decision after the run

Record optimizer steps, gradient/update evidence, checkpoint upload and hash,
accepted/excluded judgments, fixed-reference baseline versus trained wins,
turn/token counts, GPU busy fractions over a stated window, and total GPU
cost. Promote no adapter solely from reward or training loss. If the fixed
set fails to improve, diagnose which of generation, rendering, judging,
credit assignment, or optimizer/broadcast broke before expanding the data.

## Interim run evidence, 2026-09-27 19:38 UTC

The first two eight-episode batches were trainable 8/8, with no reported
episode errors, cancellations, or truncations. Prime recorded optimizer step 1
at 19:30:56 UTC (loss 0.0402, gradient norm 0.1089, 519 train tokens/s) and
step 2 at 19:36:07 UTC (loss 0.0114, gradient norm 0.0613, 956 train tokens/s).
The inference server acknowledged the step-1 adapter broadcast. The online
reward summaries rose from 0.5000 to 0.6875, but these are small, stochastic
training batches and do not establish a matched visual gain.

[Step 1](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/3a3a0e17f71d4225f81133b6559a8a1a80ba1b9e/gateway-overfit-four-20260928/step_1)
is public with weight SHA-256
`f7fff04e68ddaff229da87f4c8115521e0290a3936fff0490400e7313c43ea8f`;
the upload receipt reports remote hash verification. An unauthenticated HTTP
request to the step-1 adapter configuration returned 200. Step 2 upload was
also reported public and verified; its immutable identity will be recorded
after collection. The matched four-reference evaluation launcher passed a
no-GPU preflight against step 1. It will be run against the last checkpoint
after training exits.

Renderer receipts across 73 actual rollout paintings averaged 15.76 seconds
inside the painting stage. The standalone node profile completed 16 paintings
in 39 seconds with eight workers, or about 0.41 paintings/second. The active
policy has produced fewer than that per second while its inference GPU remains
busy; renderer work is not yet demonstrated to be the full-pipeline bottleneck.
The step-1 training interval included several minutes of idle trainer time
while inference and judgment filled the first batch. Prime's model FLOPS
utilization figures use an A100 fallback peak for the Blackwell card and
should not be interpreted as measured full-pipeline utilization.

## Completed training and throughput, 2026-09-27 19:52 UTC

Four optimizer updates completed with nonzero gradient norms, a new adapter
weight hash at each step, and live-policy broadcasts. The last public adapter
is [step 4](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/8e3e1b28c65b10d59bdf46c297141b7ab3995171/gateway-overfit-four-20260928/step_4),
weight SHA-256
`484d6ecf9396fd38b4b1fc3b4c7927f57b6f23cb5266652dac2d477c96728582`.
The last online rollout batch had reward 0.5625 and 8/8 trainable episodes;
the four batch rewards were 0.5000, 0.6875, 0.6250, and 0.5625. These values
are stochastic and cannot rank the checkpoints by painting quality.

Across 185 equally spaced telemetry samples from 19:20:30–19:51:22 UTC,
inference GPU 0 averaged 98.2% utilization and was at least 50% busy on
98.4% of samples. Trainer GPU 1 averaged 20.2% and was at least 50% busy on
20.5%. These are sampled GPU engine activity fractions, not model FLOPS or
end-to-end utilization. The inference stage was saturated while the trainer
waited for batches; another inference replica would be more useful than
optimizing trainer kernels. At step 3 the orchestrator discarded 12 of 20
episodes as stale with `max_off_policy_steps=1` and about 22 in flight. This
is a concrete waste to fix before scaling: reduce the in-flight window and
benchmark fresh episodes per GPU-hour rather than merely keeping decoding
busy.

The rollout renderer produced receipts for 192 paint actions, of which 157
were valid; the mean measured paint time was 14.97 seconds. Invalid actions
were predominantly model JavaScript errors, not renderer service failures.
Standalone rendering capacity was roughly 0.41 paintings/second with eight
workers, higher than the observed end-to-end action rate. A renderer rewrite
would not relieve the measured generation bottleneck in this run and would
change the visual reward contract. The Gateway worker logged 95 requests for
92 unique pair IDs: 92 labeled responses and three timeouts that were later
retried. Its logged API cost was $0.06122. All four optimizer batches remained
8/8 trainable. These counts include speculative work later discarded by Prime.

The matched step-512 versus RL-step-4 evaluation uses four references, one
rollout per policy and reference, seed 918200, temperature 1, an 8,192-token
per-turn limit, four turns, and the same pinned renderer. Its gallery and
two-orientation Gateway labels are pending; this small development set is an
overfit check, not a held-out generalization test.
