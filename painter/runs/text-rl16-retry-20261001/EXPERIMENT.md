# RL16 continuation retry: useful-group throughput

## Hypothesis and matched baseline

The first continuation attempt made zero optimizer updates: complete groups were operationally failed, judge-uncertain, or flat-zero. Test whether verified renderer access plus smaller independently judged groups produces useful updates without changing visual reward semantics. The user explicitly requests diagnosis and restart on2026-10-01. Start from the preferred earlier RL16 adapter, not the failed attempt or SFT512.

Initializer: CK0607/qwen3.8-27b-brush-painting@ac4d3eda8e5925e2d23992cfdc1a63620ebde573, text-pairwise-scaled-20260929/step_16, adapter SHA e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c. Fresh optimizer; no merge. Base Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; Prime26b3131d2716a4f8210b165df584b83b4bc54f61.

Frozen opponent: immutable SFT512 PNGs; model revision adf687e5b321514eda7be6eab050434fdc8e44ff, photo-curriculum-sft-20260922-balanced/step_512, weight SHA bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740. Same40 training briefs and47 fixed held-out briefs; semantic training SHA13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501. Matched preference against these frozen PNGs, plus retained RL16 evidence, is the evaluation baseline. The previous failed attempt cannot establish a learning result.

## Changes and unchanged objective

Reduce group size8 to4, keep optimizer batch16, increase inflight16 to32. This reduces the number of other traces discarded when one judgment remains uncertain and allows four briefs per batch. It changes group-relative estimation and may increase advantage variance; it does not invent labels or guarantee improvement. Retain strict whole-group censorship for operational errors/unresolved judgments and genuine zero rewards for invalid final actions. Flat-zero groups provide no relative learning signal.

Keep LR1e-6, three visually inspected turns,8192 completion tokens/turn, context32768, terminal valid-final-action contract, joint-sheet swapped-view MiMo Flash judge via Vercel, and no speed/detail/geometry bonus. Judge source SHA f94cfc9e880c8423f89a091486a0cf174cf1818c13d9acdf18fd21ace79aa069; provider weights are not immutable. Keep isolated centroid renderer runtime watercolour-p5-brush-2.2.1-centroid-fix-v1 with patched brush SHA db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b. Preparation verifies unprivileged renderer read/traverse access before model rollout.

Maximum50 updates if the registered admitted-batch checks at15/20/every5 permit continuation; public HF checkpoints and hash-verified evidence preservation before release. GPU limit for this retry is$18.55, keeping the first failed attempt plus retry below the user's$20 ceiling. Gateway charges separate. Missing-first-metric timeout remains a bounded operational stop, not an aesthetic criterion. No scheduled task or recurring automation.

## Pending launch

Source/config identity and current node fit/price will be recorded before launch. Actual optimizer updates, gradients, judgment availability, model-invalid outcomes, admission reasons, timing, interval GPU utilization, cost and matched results will distinguish working training from mere model generation. No new rental or training result is claimed at registration.
