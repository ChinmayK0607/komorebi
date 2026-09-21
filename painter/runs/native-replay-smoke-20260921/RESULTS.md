# Native replay smoke — completed 2026-09-21

The pinned Qwen3.8-27B trainer completed one real CUDA optimizer update from the exact contract-quality step-24 SFT adapter. All 992 initializer tensors matched before the first forward. The optimizer audit found finite gradients and values, 116,674,482 nonzero gradient elements, and 54,509,126 changed elements across 992 adapter tensors. Trainer exit status was zero.

This verifies the replay, multimodal packing, trainer handoff, backward pass, optimizer, and public checkpoint path. It does not establish painting-quality improvement, live asynchronous throughput, inference reload, or generalization. No new paintings or held-out evaluations were generated. The diagnostic adapter is not promoted over the matched SFT baseline.

## Inputs and scope

Immutable model, adapter, archive, and source identifiers are in [EXPERIMENT.json](EXPERIMENT.json). Implementation commit: `3462e84`; bootstrap/source baseline: `f73e2ec`. The deployed replay builder precedes one final same-sampling consistency assertion in that commit; all eight original records were independently checked to have identical sampling configs after collection. Actual deployed sources and the trainer config are identified by SHA-256 in the structured results. Launch/config code matches the reviewed source. [Verified scores](verified-scores.json) identify one originally complete eight-rollout bus group. Four deterministic program failures receive −2 and one parent-inspected relative improvement receives +1. Three rollouts are explicitly excluded: one scoring timeout, one uncertain label, and one reviewed tie omitted from this bounded diagnostic. The +1 painting is still mostly cropped and unacceptable as a finished painting; it is only better than the near-blank anchor.

The five selected rows produce mean-centered advantages [−0.6, −0.6, −0.6, −0.6, +2.4]. This deliberately selected group is not an unbiased training-quality sample. The prior scoring-error status is not used as a reward; only the explicit reviewed whitelist and program failures are admitted.

The pinned Qwen3.5 renderer was selected from the persisted runtime config for this Qwen3.8 model. Each original 2,341-token prompt matched exactly after processor reconstruction, with two image inputs per rollout. Generated actions were not retokenized. Saved log-probabilities retain their serialized rounding; they are not claimed to reproduce the original float precision. Native Prime conversion and packing produced two microbatches, 13,954 and 2,995 tokens, with no dummy batches. The retained 89,440,579-byte rank payload matched its pre-forward hash.

The generic handoff observer calls its producer `prime_rl.train_sink`; for this diagnostic the producer was the replay builder invoking native Prime conversion/packing. Its missing live score-sidecar observations are expected: scores are bound by the replay manifest and whitelist, not a live queue.

## Result and resource use

The native step took 112.56 seconds, including 110.37 seconds in forward/backward/optimizer work and 0.00034 seconds waiting for a batch. Reported loss was 0.01995, gradient norm 0.3164, and mean behavior-policy mismatch KL 0.00103. Peak allocated GPU memory was 74.38 GiB. These are mechanical metrics for one cold step, not evidence of converged learning or steady-state throughput.

GPU telemetry sampled one RTX PRO 6000 Blackwell GPU every ten seconds:

| Window (UTC) | Samples | Mean GPU utilization | Samples above 0% | Samples at least 90% |
|---|---:|---:|---:|---:|
| Measured lifecycle, 07:32:24–07:50:04 | 107 | 4.33% | 9.35% | 1.87% |
| Trainer process, 07:45:54–07:48:34 | 17 | 27.24% | 58.82% | 11.76% |
| Native step, 07:46:44–07:48:34 | 12 | 38.08% | 58.33% | 16.67% |

This was not saturated training. Setup, source debugging, model loading, and first-run kernel compilation dominate this one-step smoke. The first seven rental minutes and the post-preservation interval were not sampled, so the lifecycle figure is not a full-rental average. No inference GPU or live judge was waiting in this topology.

The pod was created at 07:25:07 UTC and confirmed absent by 07:52:10 UTC. At $1.29/hour this is approximately $0.58, an elapsed-time estimate rather than an invoice. Lium reported zero active pods after release. The watcher was deleted.

## Fixes and preserved evidence

The first dependency sync failed on a GitHub HTTP 500 while fetching the pinned vLLM wheel; a retry of the unchanged frozen lock succeeded. The reusable bootstrap now retries three times with 10/20-second backoff, unique attempt logs, and the original final exit code on exhaustion. Two mocked retry tests pass.

The first replay build exposed an incorrect tokenizer import, corrected to the pinned `renderers.base` API. Review also corrected the offline broadcast sink: it waits for Prime to create each sender-ready marker, then acknowledges both policy versions 0 and 1. Precreated markers would have been deleted by Prime. Seven offline replay/config tests, syntax checks, node config validation, native batch construction, and the actual CUDA step passed. No fake data or generated replacement tokens were used.

The [public diagnostic checkpoint](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/dbc3745c3a44b3fae1cdd8f68c565dfc2d0a6302/native-replay-smoke-20260921-diagnostic-only/step_1) is adapter-only, with all four uploaded file hashes independently checked against its immutable HF revision. Weight SHA-256: `09d253f9656779142d944e166870f2432062f36214448727a7fe8b8c1f79cc0b`. It does not contain optimizer state.

All 187 evidence members were verified after collection, including the replayable packed batch, initializer audit, exact update audit, metrics, source/config copies, and telemetry. Archive: [collected/evidence.tar.gz](collected/evidence.tar.gz), SHA-256 `84e86455a53a0cfffd01802f5e1c7ff03a5c6f8f88bc19de11e5023ee55e505e`. Generated evidence stays out of Git. [Structured results](result-manifest.json) retain the counts and immutable checkpoint link.

## Next decision

The trainer path has passed; do not repeat this smoke merely to obtain another update. The next learning experiment should start from the matched SFT checkpoint, use a small diverse set of reviewed relative-quality groups, and compare multiple updates against that same baseline. Keep the new geometry-first observations, preserve judge disagreements, and measure accepted complete-group throughput before sizing continuous live generation. The separate three-pair judge check still contains a false surface description, so order agreement alone cannot certify its factual grounding. Speed rewards remain off until quality learning is demonstrated.
