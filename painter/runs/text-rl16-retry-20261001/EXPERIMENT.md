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


## Rental and source handoff

Reviewed source `7d950c75b9c074f82cf125901919499274caaf69` is bundled for Linux deployment. Two96 GB GPUs were rented at$2.38/hour; conservative rental clock begins2026-10-01T06:40:31.864181+00:00. The finite exact-node controller is running with$18.55 remaining cap and a180-second release margin. Model/runtime preparation and optimizer launch are pending at this receipt; no learning claim. The CLI required the node UUID because its listed human-readable alias did not resolve; both earlier failed rental requests created no pod and zero active rentals were checked before successful acquisition. No scheduler or recurring automation.


Linux preparation passed; Prime launched inference at06:45:12 UTC. Actual hardware: two NVIDIA RTX PRO6000 Blackwell Server Edition cards,97887MiB each, driver580.126.09. Actual config SHA `214b8a4dfa042c5018ff08c1fe90622ef9e90750f6892f3940091fcefe91d474`; concrete training manifest SHA `c9521a44f3c8df060a9c8957f447e0b08b63b05a08ef2736390a2ccc3381763c`. Config reports group4/batch16/min-inflight16/initial32/max32, LR1e-6, trainer loss `ipo` with eps0.1/adv_tau1.0/kl_tau0.001. Policy download independently verifies the pinned RL16 adapter hash; guard activation reports installed/group4. Versioned renderer receipt confirms Node18.19.1 and unchanged renderer-source SHA; all five Node centroid regression tests passed on Linux. Unprivileged renderer access passed before rollout launch. Adapter-server cold loading and trainer startup remain pending at this observation; no admitted reward or optimizer-update claim.

The public adapter metadata audit counts992 tensor keys, all in language-model layers, zero vision-tower tensors. Thus the serving startup warnings for unsupported vision LoRA wrappers do not indicate omitted trained vision weights for this initializer. Inference warmup completed at06:49:14 UTC: engine initialization160.42s including40.81s compilation. These are cold-start timings, not steady-state throughput or saturation.


Trainer/orchestrator started around06:50 UTC. The before-first-forward audit verifies992/992 exact RL16 tensor matches,496/496 nonzero B matrices, all finite,116,727,808 adapter parameters, fresh optimizer and native resume false. Training loop begins06:50:33; startup weight broadcast synchronizes inference before the orchestrator loop at06:50:35. Thus the active sampler uses the preferred trained adapter, not unadapted base or fresh SFT512. Rollouts reached32 inflight at06:51:15; at06:51:31 Prime's queue-overload controller reduced target concurrency to28 and canceled four youngest episodes. Preserve this pipeline-cancellation scope separately from genuine model errors. The32 setting is an initial/maximum bound, not sustained saturation. At this observation the first16-episode admitted batch is collecting; zero completed optimizer updates and no gradient/improvement claim. Group-admission JSONL is configured and forwarded for early decision visibility.


## Live observation: 2026-10-01 07:49 UTC

Four optimizer updates completed; fifth batch shipped and training pending at the snapshot. Effective admitted-batch mean rewards for batches1–5 are0.270833,0.166667,0.093750,0.166667,0.197917; all-completed means are0.154762,0.111111,0.085859,0.126984,0.129167. Neither series establishes upward learning; changing briefs, censorship and relative comparison make these different from matched held-out quality. Gradient norms for completed updates1–4 are0.010498,0.011230,0.007080,0.011292. No matched evaluation yet.

At07:46 UTC, group telemetry records30 admissions/32 observed groups, two exclusions for uncertainty/error, zero flat-group delegate rejections. This resolves the prior attempt's useful-group starvation at the operational level; it does not fix painting quality or validity automatically. Adaptive concurrency is approximately26–27 rather than sustained32. Last completed update took615.64s: approximately426s waiting for a batch and188s active training. Trainer reports986 tokens/s and66.5% active-step MFU; neither is whole-pipeline utilization.

GPU sampling covers06:41:53–07:48:55 UTC,805 five-second observations per card, including startup: inferenceGPU0 mean86.19%,86.96% observations above10%; trainerGPU1 mean14.76%,16.52% above10%. The two-GPU pipeline remains unsaturated and rollout-limited. This interval does not isolate model generation, rendering and external judging costs.

Step4 checkpoint is public and upload hash-verified: CK0607/qwen3.8-27b-brush-painting@91d7f7367a1636bdf9435d73702e19217f72e6ee, text-rl16-retry-20261001/step_4, adapter SHA e29db8256d2d8a8461b75c8990c8ee759fa0522432a0bdf03e118bdd905b876a,233590600 bytes. Rental clock estimate at07:48:59 UTC is$2.716, or$4.164 including the failed attempt; these are GPU clock estimates, not invoiced provider totals, and exclude Gateway. Continue the registered finite run to its first trend decision unless an operational failure or budget deadline intervenes; do not claim improvement or rent another node from this snapshot.


## Live observation: 2026-10-01 08:39 UTC

Nine optimizer updates completed; tenth batch collecting with28 inflight episodes. Effective admitted-batch rewards6–9 are0.177083,0.218750,0.135417,0.135417. Reward still fluctuates without a sustained upward trend. No matched visual evaluation has completed; do not infer a quality gain/regression from changing training batches. All nine completed gradient norms are finite/nonzero; latest0.014465. Group admission58/64 with six uncertainty/error exclusions and zero flat-group delegate rejections. Latest logged step9 duration is13m32s, comprising10m36s waiting and2m55s active training (rounding). Runtime remains rollout-limited.

Public, hash-verified step8 adapter: CK0607/qwen3.8-27b-brush-painting@3ab31b51d1da6c56bcb5eabcda90ef454d3f9b10, text-rl16-retry-20261001/step_8. Sampling06:41:53–08:38:56 UTC has1405 five-second observations per GPU including startup; inference mean91.68%,92.38% samples above10%, trainer mean16.45%,18.72% above10%. This is not two-GPU saturation. Clock-based rental estimate at08:39 UTC$4.70 retry/$6.15 including failed attempt, Gateway separate; below combined$20 cap. Continue to the registered first trend decision at15, preserve public checkpoints every4; no new rental/scheduler.


## Judging investigation: 2026-10-01 09:18 UTC

[Controlled frozen-pair audit](JUDGE_AUDIT.md) separates215 invalid final actions/364 completed traces from140 actual visual scores and9 unresolved judges.98 failed finals discard an earlier valid canvas.17/149 pairs initially disagree on winner across swapped views;59/132 agreeing winners disagree on magnitude. Six identical frozen-pair repeats confirm one opposite-winner change among two previously scored pairs, one magnitude-only change and three uncertain-to-scored transitions. Root inspected15 selected pairs and found both legitimate relative improvements and perceptual/spatial misjudgments, including an overrewarded filled horseshoe arch. No demonstrated ordering, arithmetic or consensus-aggregation bug; three offline audit tests pass. Do not treat selected root opinions as labels or changing-batch means as matched visual gains.177-file evidence is public and anonymously hash-verified; immutable receipt and cost are in the linked audit. Live run now has12 completed updates, recent effective rewards0.09375/0.0625/0.15625, GPU clock estimate$7.71 combined. Active reward, sampling, optimizer and stop rules unchanged; no new rental or schedule.
