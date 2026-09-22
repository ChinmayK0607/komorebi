# Photo curriculum RL

Status: retrying launch after setup failures; no RL optimizer update verified. See EXPERIMENT.json for actual outcomes and failures.

The user requested RL on the same data after inspecting the photo-curriculum SFT paintings. The hypothesis is that training complete paint → inspect → revise episodes with a terminal visual preference reward improves the midpoint SFT model's reference fidelity and painting quality. Speed optimization remains deferred.

## Matched baseline

Initialize a fresh RL optimizer from `CK0607/qwen3.8-27b-brush-painting`, namespace `photo-curriculum-sft-20260922-balanced/step_512`, revision `adf687e5b321514eda7be6eab050434fdc8e44ff`. The adapter SHA-256 is `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`. Base: `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Prime: `26b3131d2716a4f8210b165df584b83b4bc54f61`.

The previous one-turn evaluation is historical context, not the matched baseline for six-turn RL. Evaluate the frozen midpoint and RL checkpoints with the same prompts, reference images, turn allowance, decoding settings, and renderer. Keep all 28 photo development references and 240 procedural development references outside training. These are repeatedly used development sets, not a fresh untouched test set.

## Data and objective

Reconstruct the same 86 admitted photo references and 2,400 procedural training references with source IDs and content hashes. The 192 general SFT retention examples are not painting episodes and are excluded from this pure painting RL objective. Balance photo/procedural sampling and report actual source coverage rather than equating pool size with references trained on.

Use the existing holistic Luna-low pairwise rubric: beauty, coherent painted interpretation, and preservation of the reference's important identity and relationships. Require visible weaknesses for both candidates where supported. Two blinded orientations produce one terminal preference score; there are no additive aesthetic subrewards or token/turn savings bonuses. Model-caused failure to produce any valid canvas receives zero. Service failures and unavailable judgments are recorded as operational exclusions, not invented visual scores. Lazily generate and cache a frozen midpoint512 opponent using the same full-history six-turn loop; its adapter alias remains immutable across live-policy updates. A valid candidate against a model-invalid frozen canvas receives one, recorded separately from Luna judgments.

## Initial training allocation

Proposed: native Prime asynchronous RL, eight samples per reference, 32 episodes per update, learning rate 1e-6, unchanged rank-16/alpha-32 adapter coverage, up to six model responses, 8,192 output tokens per response within a 32,768-token context. Start with a finite 32-update allocation and publish checkpoints every eight updates. This allocation is an initial measurement of learning, not a claim that the whole reference pool receives coverage. Record clipping, context censoring, group reward variance, accepted training tokens, policy staleness, and actual updates.

Use distinct training and inference GPU roles and preserve behavior log probabilities across bounded asynchronous updates. Rent only after the preparation can use the node. Luna judging must have an automatic finite queue worker; interactive parent intervention must not be required for every training batch. Credentials remain outside source and evidence. No scheduled tasks or recurring watchers are authorized.

## Decision evidence

Compare rendered RL paintings directly against frozen-midpoint paintings on the same development references. Report wins/ties/losses, invalid outputs, all excluded cases, turn/token distributions, and representative regressions as well as improvements. Report separate trainer and inference utilization over measured intervals, judge waiting time, elapsed rental cost, and public hash-verified checkpoint revisions. Record exact data/config/source hashes at launch. Do not describe preparation, rollout collection, or an optimizer update as evidence of visual improvement.
