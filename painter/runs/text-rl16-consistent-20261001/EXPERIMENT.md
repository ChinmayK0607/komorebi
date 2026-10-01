# RL-16 continuation with repaired terminal credit

## Hypothesis and matched baseline

The user prefers the earlier `text-pairwise-scaled-20260929/step_16` paintings to the fresh SFT512-started run and authorizes training from that earlier checkpoint. Test whether conservative continuation of that policy under the current joint-image aesthetic judge, explicit valid-final-action contract and narrow p5.brush centroid repair improves fixed held-out paintings. This is a new optimizer run from adapter weights, not native optimizer resume. No merge with SFT or new SFT is performed.

Policy initializer: `CK0607/qwen3.8-27b-brush-painting@ac4d3eda8e5925e2d23992cfdc1a63620ebde573`, prefix `text-pairwise-scaled-20260929/step_16`, adapter SHA `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`, 233,590,600 bytes. Anonymous model metadata confirms public revision, byte size and LFS hash before any GPU rental.

Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; Prime `26b3131d2716a4f8210b165df584b83b4bc54f61`. Frozen reward opponent remains SFT512: `CK0607/qwen3.8-27b-brush-painting@adf687e5b321514eda7be6eab050434fdc8e44ff`, prefix `photo-curriculum-sft-20260922-balanced/step_512`, weight SHA `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`. Reward opponents are immutable existing PNGs; policy initializer and frozen opponent are separate.

## Scope and objective

Reuse the reviewed forty text briefs (12 simple, 10 intermediate, 18 complex) with semantic manifest SHA `13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`, and the fixed 47 held-out briefs (8 simple, 8 intermediate, 31 complex). Source archive SHA `ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3`. Report matched preferences versus frozen SFT512 and the earlier RL16 control where available; distinguish different-renderer replay comparisons from exact same-renderer controls. No train/validation leakage or new image-to-painting scope.

Judge: `xiaomi/mimo-v2.6-flash` through Vercel AI Gateway, unchanged `text-paint-sft-pairwise-consistent-v1`, side-keyed joint-sheet format source SHA `f94cfc9e880c8423f89a091486a0cf174cf1818c13d9acdf18fd21ace79aa069`. Provider weights are not immutably pinned. Order disagreements use balanced rechecks and unresolved judgments remain censored, never zero or tie. No additive detail or geometry rewards.

Enable `PAINTER_REQUIRE_VALID_FINAL_ACTION=1`: an invalid final recorded action receives zero reward even if a previous valid painting survives; preserve that painting's provenance. Valid finish/recovery/no-change remain judge eligible. This deliberately changes the terminal objective and prevents positive credit for a failed final action. It does not provide finer intermediate-turn credit. Service failures remain operational and missing evidence must not become model zeros.

Enable nonexecuting Node syntax preflight. Use explicit versioned p5.brush derivative with exactly `v[0]` to `t[0]` in the zero-area centroid fallback. Original asset SHA `9c6e271a78e5dcb23c9cfa9d085775317fe5fb2188f976465c83c1f7965d1ec8`; derivative SHA `db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b`; original renderer SHA `6181fb904a0c5dcd580f5c3cb01eaf341a14b7a15479e811d99ebb10b70bc841` remains unchanged. Patch provenance and license receipt are required. This repairs one vendor crash; long-tail rendering remains unresolved and no speedup is presumed.

## Finite limits and decision

LoRA LR1e-6, group8, batch16, three visually inspected turns, 8192 completion tokens/turn, context32768, maximum50 optimizer updates, public checkpoints every4. Select affordable separate trainer/inference GPU topology only after current pricing and fit checks. GPU rental hard ceiling **$20 total for this run**, including setup, final generation, preservation and release; Gateway charges separate. Register exact source/config/data/patch hashes, topology, rental start/price and stop/release deadlines before launch. Use existing finite process supervisors; no scheduled tasks or recurring automation.

Keep the registered admitted-batch checks at15/20/every5: latest-five mean must exceed prior-five mean to continue. These changing-prompt checks are a conservative operational stopping rule, not a matched visual progress test. Also report validity, valid-canvas preference, judgment censorship, per-family fixed held-out results, completed optimizer updates versus shipped batches, gradients, pipeline timing, interval GPU utilization and costs. A noisy online curve is not sufficient evidence for improvement; successful scaling requires better matched painting quality without validity collapse.

Preparation begins on 2026-10-01. No node rented or trainer started yet. Implementation source/config and Linux launch evidence are pending; the root research index will distinguish preparation from actual updates.

## Rental and launch preparation — 2026-10-01

Reviewed source `fb36ed92077eb48b96169085811eec1697f7ef3a` was handed off as a verified Git bundle. Relevant checks passed: six continuation contracts, 53 shared-run tests, ten evaluation tests, ten launcher tests, and seventeen credit/patch/group-admission tests (one Node fixture skipped on the Mac; the Linux syntax runtime is checked separately).

A two-card 96 GB node was rented at $2.38/hour. Conservative rental clock starts 2026-10-01T05:22:45.422071+00:00. Total GPU ceiling remains $20, including preparation; the absolute clock deadline is 2026-10-01T13:46:57.522911+00:00. A finite exact-node controller is active with a three-minute release margin. Training has not started: source transfer and pinned Linux preparation are in progress. No scheduled task was created.

Linux preparation completed and Prime launched the inference role at 2026-10-01 05:28:24 UTC (about 5m39s after the conservative rental start). Actual preparation verifies RL16 SHA `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c` and separate frozen SFT512 SHA `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`, with optimizer resume false. Config SHA `e6dc3fffa5211d32dab957917550ebc410470a75494f9b7c94cdce683a929a44`; concrete train-manifest SHA `72defda04d7c48bf7e357c05e618c2d19cf797cb8b615eabd441c3ded39b10a7` includes explicit Linux patched-runtime paths and differs from the path-independent semantic hash. Node18.19.1 runs the syntax preflight; all five Node centroid regression tests pass. The model download took about74s. The finite training supervisor is running; no completed optimizer update is yet observed.
