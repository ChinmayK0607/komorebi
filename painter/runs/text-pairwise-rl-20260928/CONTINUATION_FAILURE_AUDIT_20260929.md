# Text-paint continuation: live failure audit

This began as a live diagnostic and now records the **user-stopped 33-update**
continuation, before the held-out evaluation finishes. The frozen 50-update
plan and matched SFT/step-16 controls are in
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
The saved valid-trace mean relative score rises from 0.327 on 64 simple
canvases to 0.399 on 72 complex canvases even though the complex prompts
are harder; this is consistent with unequal opponent quality and task mix,
not evidence that the model became better over time. These are generated
trace subsets, not a matched per-prompt learning curve.

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

A later read-only snapshot of the native trace stream through 323 generated
three-turn trajectories (969 turns, including speculative work) found only
**7** calls with `finish_reason=length`; all seven used exactly 8,192 output
tokens and none yielded a valid action. The other **962** calls ended with
`finish_reason=stop`, yet 199 had a syntax-related error string and only 331
yielded a valid action. This rules out output-token truncation as the main
explanation for malformed programs at this snapshot. The three turn positions
had mean completion lengths of 2,249, 2,186, and 2,099 tokens. These are
generated-turn counts, not admitted-batch rates; verify the conclusion on the
final stream before changing the token policy.

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

An experimental [read-only syntax preflight](profile_js_preflight.py) used
Playwright's bundled Node parser (`--check`, no sketch execution) on a
deterministic sample of 16 saved programs while the production renderer was
untouched. It rejected all three programs whose saved Chromium receipts
reported syntax errors and rejected none of the valid renders in that small
sample. The gross renderer work potentially avoided was only **3.8 seconds**
over those 16 cases, before accounting for parallelism. This is a useful
fail-fast/feedback improvement, not a solution to model-generation time or
the 180-second render timeouts. A full saved-program profile and an
unchanged-pixel integration test remain necessary before adopting it.

The two-order MiMo Flash judge disagreed by at least one-third on 32 of 171
saved valid judged canvases. This is a confidence/reliability warning, not
a human-validated error rate. Its receipts reported ~$0.114 for the 342
order views in this snapshot; rental and judge costs must remain separate.

## User-stopped step-33 diagnosis (2026-09-29 22:02 UTC)

The launcher received `SIGINT` and exited with code 130; the trainer and its
inference workers stopped. The latest completed public, nonprivate,
hash-verified adapter is step 30. The final 33 admitted optimizer-batch means
average **0.23643**, with first 16 at **0.23763** and the following 17 at
**0.23529**; the last batch was **0.17708** and the last-five mean **0.22708**.
The frozen native-metrics SHA-256 is
`ec5c3301eb5badb6d6a63607383ff6a1416fe67d05b7b27bd26ebd6283499f95`.
There is no positive online trend, though changing prompt mix still prevents
reading this as a matched quality curve. Step-30 versus frozen-control
evaluation is underway and remains the decisive test.

A manually inspected four-pair probe is preserved under
`local-evidence/continued-live/live-pair-review/` (source tar SHA-256
`d641b258cf39e94c5602d1a8adf961dab83a8d4f560d7891564ef9255c06f8a6`).
These are **saved generated traces, including possible speculative work**,
not a random sample of admitted updates. Their score and visible result expose
how the relative reward can fail:

| Brief | Saved candidate score vs SFT | Manual visual finding |
| --- | ---: | --- |
| `t600-333`, red panda on bough | 1.000 | The SFT anchor is fragmented; the candidate has more organic washes but the head, masked face, folded paws, and ringed tail remain ambiguous. Both are poor references for a decisive “clearly better” margin. The two-order judge called the candidate a clear, anatomically complete panda, despite noting some blurred features. |
| `t600-412`, baker shaping dough | 0.917 | The SFT anchor is nearly blank. The candidate adds a torso/workbench but no clearly cupped hands, round loaf, basket, scraper, or baked baguettes. A large relative win does not mean the painting depicts the required action. |
| `t600-576`, two buses at snowy crossroads | 0.917 | The candidate improves dark street geometry over its faint SFT anchor, but the two separate turning buses and tiny pedestrians remain hard to identify; detail and contrast dominate the impression. |
| `t600-510`, acorn and oak leaf | 0.250 | The SFT anchor preserves an acorn and lobed leaf more clearly; the lower candidate score is directionally sensible here. |

These observations do not prove the judge is always wrong. They identify a
specific promotion failure: low-quality opponents plus overconfident
subject-recognition claims can grant large relative advantages to paintings
that still omit defining content. The next reward protocol needs an
**absolute visible-subject check inside the pairwise judgment** (or an
equivalent margin cap), calibrated on these failure pairs and genuinely good
counterexamples before any new RL. Avoid adding a pile of independent
detail, geometry, and style bonuses. Weak opponents should be replaced or
excluded, with their exact IDs and hashes frozen before training.

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
