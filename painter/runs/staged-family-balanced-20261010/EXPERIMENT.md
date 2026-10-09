# Staged family-balanced curriculum — 2026-10-10

This is an offline preparation record for one curriculum intervention. No GPU, model, renderer, judge, provider, or external API was used.

## Hypothesis and matched baseline

The completed `diverse-flow-rl-20261009` run used a balanced mixed manifest, but asynchronous completion produced stage means of 0.2491, 0.2953, 0.1905, and 0.2622 and family exposure from 10 to 27 scored rollouts. Its spatial-pair stage was weakest. The matched baseline for the next experiment is the exact Flash12 policy, which produced 32/32 valid final canvases on the fixed 32-case panel. The new arm tests one change: deterministic sequential stage advancement with equal family quotas. It does not test a new model, renderer, rubric, data source, or SFT initializer.

The same-run four-step code/canvas SFT produced 32/32 valid final canvases both before and after supervision, so this arm starts directly from Flash12 with no new SFT warmup. That is a controlled decision recorded here, not an assertion that supervision is generally ineffective.

## Immutable identities

- Base model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Starting policy: `CK0607/qwen3.8-27b-brush-painting`, revision `1227a2427a8ae16014d2fdca4d1adb6855d3fa1c`, prefix `rl-judge-matched-20261004-flash/step_12`, SHA-256 `c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79`.
- Prior source tree commit: `113d40cdd446d1d406eb294bd50fa34f90cc0197`; Prime revision: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
- The exact prior curriculum generator is checked by SHA-256 `b86c1729a83a73ba1fc7bdf934146178c19abd21a0876c1e12ec404e4e5ffbe5`; its canonical proposal identity is `49262afcb3138fe4e21f2c98cff84a6bf3acb513c23d8762306c7ffb0a390f95`.
- The prior Flash medium judge, rubric, frozen SFT512 opponent, renderer, and p5 brush identities are copied into [PROTOCOL.json](PROTOCOL.json) and [config.json](config.json).

## Curriculum and controller

The generated [curriculum manifest](curriculum-manifest.json) retains the prior 96 training rows and 32 heldout rows by source-group, prompt hash, family, stage, and variant. Four stages run in order: single forms (4 sampling cycles), connected objects (4), spatial pairs (8), and scenes (4). Every sampling cycle schedules six Prime task groups, one prompt row from each of six training families. Each task group dispatches four rollouts, giving a 24-trace capacity per cycle; actual scored exposure is measured after admission because a group may contribute three traces when one rollout is judge-uncertain. Prime's `batch_size=24` counts trainable traces and may span multiple sampling cycles, so optimizer update receipts are recorded independently. Each family's four variants cycle deterministically. The 32 heldout rows are one fixed panel for the exact start and final policy.

The [controller](staged_curriculum.py) emits explicit pass/stop/advance receipts. A model-owned invalid code, canvas, current-canvas, or terminal action is recorded as a trainable reward-zero rollout and does not halt the schedule. Judge uncertainty or service failure is recorded as an excluded, unscored rollout with evidence; a group is admitted with three or four scored traces and censored below three. Missing or malformed ownership, status, evidence, validity, reward, row or family, duplicate group, wrong cardinality, or quota mismatch returns `status=stop`, `decision=stop`, and `admission=fail_closed`. A pass receipt advances only the sampling cycle. The `4/4/8/4` optimizer target receipts advance stages; this is schedule progression rather than a quality gate. Each stage has a finite two-times sampling-cycle refill cap and stops if its optimizer target cannot be filled. State does not advance after a censored or rejected cycle. The controller therefore records an actual staged protocol rather than labeling ordinary cyclic mixed sampling a curriculum.

## Configuration and pending decisions

The generated [config](config.json) preserves the prior Flash medium judge and renderer identities, sets learning rate `1e-6`, and reserves optimizer updates `4/4/8/4` for 20 total updates. Each stage starts with that many scheduled sampling cycles and may use up to twice as many refill cycles to fill partial Prime batches. It requires strict terminal action and current-canvas validity at admission. The schedule uses `batch_size=24`: six task groups, four rollouts per prompt, and a 24-trace cycle capacity. The preparation receipt includes an optional node-free validator for the pinned Prime revision `26b3131d2716a4f8210b165df584b83b4bc54f61`; an absent checkout is recorded as unavailable rather than passed. No hardware is allocated until that check and a repaired-renderer canary pass. The preparation receipt records these limitations and hashes.

A read-only probe found a local Prime checkout at the pinned revision, but its config import could not complete because the optional `openai` dependency is absent. The validator therefore reports `failed` with `validated=false`; this is a dependency limitation, not Prime compatibility evidence.

The next decision is to run those node-free compatibility checks and implement the runtime tail-purge hook. The hook must physically consume the exact recorded tail trace IDs before any stage transition; this preparation is therefore not launch-ready. If compatibility or the renderer canary fails, stop before allocation and revise the preparation. A later visual result must report every heldout case, invalid/censored case, stage receipt, utilization interval with denominator, and GPU rental-clock cost against the Flash12 baseline.

No optimizer, reward, visual, utilization, or cost result is claimed by this record.
