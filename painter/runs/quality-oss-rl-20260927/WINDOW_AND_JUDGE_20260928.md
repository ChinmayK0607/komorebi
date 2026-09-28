# Four-scene recovery and in-flight test, 2026-09-28

Status: finite 8/12 in-flight GPU run launched at 2026-09-28 00:18 UTC;
subject-first API calibration is running separately. No visual result yet.

## Question and matched baseline

The previous four-update [Gateway overfit](GATEWAY_OVERFIT_20260928.md)
finished with nonzero gradients but split 2–2 with the frozen SFT step-512
baseline on four references. It sampled roughly 22 episodes concurrently and
discarded 12/20 at update three as stale. The step-two dog evaluation had no
valid canvas in four turns. Hypothesis: a bounded 8/12 in-flight window will
reduce stale work and cost per accepted update while preserving visual quality.
The matched throughput baseline is the prior 16/24 window on the same four
references, initial SFT adapter, 2×RTX PRO 6000 Blackwell layout, batch 8,
group 4, four turns, 8,192-token ceiling, four updates and `1e-6` LR. These
are very small, stochastic samples; report the changed interaction feedback
as a confound rather than claiming a clean concurrency effect.

The initializer is public
`CK0607/qwen3.8-27b-brush-painting/photo-curriculum-sft-20260922-balanced/step_512`
at revision `adf687e5b321514eda7be6eab050434fdc8e44ff`, weight SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Base Qwen3.8-27B revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
Prime revision is `26b3131d2716a4f8210b165df584b83b4bc54f61`.
The four-row data manifest SHA-256 is
`58a5d1f06910d60d5b3e5726a1e73692a08d8739fc582847bf23e5fd7619c62a`.
The launched source commit is
`65f679b01418b2f87dca55276a53e940edc2aea6`; the native TOML SHA-256 is
`8ec44fed5e4314e83c7fe9020e6aeb9a90ed6d78802b62032df03755f3be192b`.
Prime's pinned config validator accepted batch 8/group 4 and the exact
concurrency values `min_inflight=8`, `initial_inflight=8`, `max_inflight=12`.
Native task serialization and the node preflight passed before training.

## Concrete changes before spending

The matched evaluation gallery now retains every attempted `program.js` and
`painting.json`, even when no PNG exists; the public evidence publisher already
includes the review tree. This repairs a real evidence omission: the prior
step-two archive has its matched gallery and judge receipts but **not** the
four failed dog programs. That node was released, so those exact programs
cannot be recovered. Its training rollout archive contains 35 invalid paint
receipts, mostly malformed JavaScript and a few p5-name collisions. The live
feedback now gives a short specific correction for syntax errors, p5 global
redefinitions, and non-iterable inputs while retaining the actual renderer
error. This changes the interaction scaffold and must be disclosed in any
comparison with the previous run.

The visual judge remains the Vercel Gateway `xiaomi/mimo-v2.6-flash` with two
image-order orientations and image-bound receipts. A separate subject-first
prompt is prepared for an eight-call comparison on the four already-rendered
step-four pairs; it does not silently replace the live reward. The dog's
default preference was visually questionable because the winning RL canvas
has a faint, poorly formed animal. Compare prompt variants with image hashes
and any user preference before deciding whether to change the training
reward. Do not fabricate a human gold label from a model response.

## Operating bounds and decision

Use one finite 2-GPU node at no more than $2.38/hour if the previous 96 GB
card class is available; release it after public hash-verified evidence.
Previous observed GPU cost was at least $7.23. Keep combined campaign GPU
spend below the user's $25 ceiling. No recurring watcher or scheduled task.
Keep SFT baseline and final RL evaluation settings matched: same four
references, seed 918200, temperature 1, four turns, 8,192-token ceiling,
same pinned renderer. Record accepted, stale, cancelled and invalid counts,
wall time per update, GPU utilization over a stated window, and cost per
accepted judged episode. Preserve raw judge labels and failed programs.
Do not promote the new checkpoint solely from online reward; compare rendered
finals and retain the public SFT baseline if the visual result is not better.
