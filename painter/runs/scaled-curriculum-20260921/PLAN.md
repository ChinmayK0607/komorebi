# Scaled quality curriculum — 2026-09-21

## Decision

The next learning campaign is a broad quality curriculum, not another four-update diagnostic. The recent RL run admitted only two reference states despite twelve available families. A 32-task repair pool is a component, not the full next curriculum. Earlier larger SFT runs did exist; their regressions mean that increasing a nominal dataset count alone is not evidence of learning.

## Hypothesis and matched baseline

With the corrected shared paint → actual canvas → inspect/revise environment, exposure-balanced training across diverse references should improve reference-conditioned visual preference over the frozen contract-quality step-24 SFT initializer. Evaluation uses the identical environment and generation limits for initializer and candidate. Keep quality as the objective; speed pressure remains disabled.

Initializer: `CK0607/qwen3.8-27b-brush-painting`, revision `b7103dd9923f19449ccf814f9b67bb2dcfed0909`, prefix `contract-quality-20260918/step_24`.

Authoritative initializer weight SHA-256: `d10eeb3b7b53fc539aa9d5e71eb98bcfe1c59d6a4878c7a69c93b8b09b59bcfa`.
Base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Data and actual training exposure

Prepare 2,400 training and 240 development procedural source programs across twelve recipe families and varied layouts. These are preparation targets, not a claim of rendered image counts, distinct semantic families, or aesthetic quality. Audit existing real references and teachers separately and import only locally resolvable, provenance-preserved assets. Real-reference availability is not a reason to block the procedural stream. Keep teacher demonstration quality distinct from successful program execution.

Split before augmentation; group related references and variants together. Check image hashes after rendering. Historic evaluation images remain development-only; newly seeded procedural cases measure parameter/composition transfer, not unseen-family or photo generalization. Deduplicate reference identities before expanding opponents or repeated rollout draws.

Interleave families and difficulty, and cover distinct references before repeating them. Log eligible, sampled, rendered, judged, and optimizer-admitted references separately. Task, opponent, turn and rollout counts must never be described as unique reference counts.

Initial learning allocation: 128 optimizer updates, 32 admitted episodes per update, eight rollouts per reference/opponent group: 4,096 admitted episodes, or 512 complete group occurrences if all groups are full. Aim for 512 distinct training references in this first allocation. Missing/filtered groups can reduce actual coverage; publish measured admission counts rather than silently claiming the arithmetic target. Continue through the larger pool; checkpoint/evaluation intervals are observations, not requests for repeated human permission.

Starting settings: learning rate 1e-6, rank16/alpha32 inherited from initializer, six turns, 8,192 tokens per response, 32,768 total context, temperature 1.0, no speed/length/stroke reward. These are starting choices, not optimized hyperparameters. Pin native Prime and Verifiers to the tested revisions and use the active shared environment for training and evaluation. Track truncation explicitly.

## Judging and throughput

Retain Luna low's approved holistic reference-conditioned aesthetic A/B rubric, including visible weaknesses of both candidates. Do not add additive syntax, finish or length rewards. A missing judgment is not a loss. Preserve the current two-orientation receipt semantics during integration; quantify order disagreement and missing rewards.

Judge service is a real throughput constraint: the last four-update run waited about 1,039 seconds for batches, versus 329 seconds forward/backward; trainer mean utilization was 14.06%. At the previous aggregate rate, 128 updates with four times the episodes per update would be roughly 52 hours. This is a rough linear extrapolation, not a runtime promise: concurrent generation, caches, exclusions and judge throughput can change it substantially. Merely renting more GPUs cannot fix a slower judgment consumer.

Keep separate bounded generation and scoring work, prioritize completing groups, retain behavior log probabilities and policy versions, and measure judged/admitted tokens per second. Do not claim this decoupled scheduling is implemented until native integration verifies it. Scale useful judge throughput and generation together. Use distinct inference/trainer devices; additional devices depend on measured bottlenecks. Report utilization by role, over a stated window.

## Current execution scope

This preparation phase builds/audits curriculum source and integrates the shared environment. Rendering/compilation and model work run on Linux nodes only. No new optimizer updates have run. Source/data/config hashes and exact launch configuration are recorded before a training launch. No scheduled task, heartbeat, automatic restart or recurring automation is authorized.
