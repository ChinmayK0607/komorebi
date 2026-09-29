# Text-paint continuation: live failure audit

This is a **live diagnostic**, not a held-out result. The frozen 50-update
continuation and matched SFT/step-16 controls are in
[CONTINUATION_20260929.md](CONTINUATION_20260929.md). The full post-run audit
must replace the numbers below with final metrics, review the 47 same-seed
held-out paintings, and record GPU cost and public evidence hashes.

## What the current measurements support

At 27 admitted optimizer updates, the first-half mean relative reward was
0.2372 and the second-half mean 0.2381. The last five averaged 0.2625, but
that short window does not establish a trend. At least 26 recorded steps had
nonzero gradient norm; the public step-25 LoRA differs from its frozen
step-16 initializer. This confirms optimization activity, not better images.
The active judge compares each candidate only with a frozen SFT painting.
The almost-blank baker anchor `t600-412` lets a visibly weak RL patchwork
score 0.917; its rationale actually notices missing hands and dough. This is
a concrete *relative-anchor* failure. Hold visual promotion for a direct
matched comparison and stronger-anchor calibration.

The saved generated-trace stream at the same snapshot contained 270
trajectories, including speculative or stale work. Only 88/270 had a valid
first turn, 171/270 had a valid retained final canvas, and 102/270 recovered
from an invalid turn to a later valid turn. These are not the 27 admitted
optimizer-batch denominators. Across 810 turns, 283 were valid. Thus the
multi-turn scaffold salvages many episodes, but fresh-generation validity is
poor. A sampled error audit found hallucinated brush names (`cpen`),
unsupported methods (`brush.degree`), undefined variables, malformed or
unfinished JavaScript, and 180-second render limits. The full profile stores
the per-class counts and concrete turn receipts; do not explain all failures
as renderer latency or all losses as judge error.

The first 27 admitted batches averaged 283 seconds model generation,
91 seconds harness, and 20 seconds scoring per rollout; phase work overlaps,
so these are not additive wall-clock timings. Of 270 generated traces,
54 turns hit the renderer deadline; valid renders had a 19.1-second median
and 68.2-second 90th percentile. Over the sampled run window, inference GPU
0 averaged 97.5% utilization and trainer GPU 1 averaged 12.6%; the trainer
waited about 454 seconds per warning for a batch. The present bottleneck is
inference and its multi-turn/render work, not trainer arithmetic. Extra
trainer capacity alone would waste rental. [Renderer profiling](RENDER_PROFILE_20260929.md)
already found ~2× batch throughput with four software render workers but no
safe hardware-accelerated p5.brush canvas on the tested provider containers.

The two-order MiMo Flash judge disagreed by at least one-third on 32 of 171
saved valid judged canvases. This is a confidence/reliability warning, not
a human-validated error rate. Its receipts reported ~$0.114 for the 342
order views in this snapshot; rental and judge costs must remain separate.

## Changes to test only after the frozen run

1. Use a same-program offline replay set to measure syntax/API rejection
   before Chromium, pixel identity for successful renders, and the actual
   timeout/CPU savings. Report false rejection on valid programs. Never
   rewrite model code silently or change this run's renderer.
2. Compare 1 versus 2 inference workers and the existing four software
   render workers on identical saved tasks, reporting completed valid
   rollouts/hour, trainer busy fraction, and dollars per admitted update.
   Larger `max_inflight` alone cannot cure a saturated inference GPU.
3. Audit frozen SFT opponents on all 119 training briefs for near-empty or
   obviously wrong compositions, then calibrate relative comparisons against
   stronger candidates and direct human-labeled pairs. Avoid stacking
   independent detail/geometry bonuses that create new hacks.
4. Select new text/photo teacher episodes by actual visual fidelity and
   whether each subsequent turn improves the prior canvas. A saved valid
   canvas after an invalid last turn is not a successful repair. Keep weak
   episodes out of positive SFT, but retain them for recovery/negative data.
5. Read final held-out validity-inclusive and both-valid paired results by
   family before deciding whether to promote the new adapter or run more RL.
   Online reward alone cannot make this decision.

No new training run is authorized by this audit. The current finite run
continues unchanged while data generation on Codex Cloud CPU progresses.
