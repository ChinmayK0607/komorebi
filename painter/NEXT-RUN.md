# Current next work — 2026-09-21

The [scaled curriculum plan](runs/scaled-curriculum-20260921/PLAN.md) supersedes the earlier small-pilot proposal. The [research index](../RESEARCH_LOG.md) remains the entry point. Do not restart a historical watcher or use old run-local harnesses as the active implementation.

The completed [controlled comparison](runs/controlled-comparison-20260921/RESULTS.md) used the corrected shared native environment for both SFT and RL. It found no overall improvement from the four-update, two-reference RL checkpoint. Retain contract-quality step24 as the initializer; the 32-task repair pool is a small part of a much larger curriculum.

The 2,400 training/240 development procedural references are rendered and locally hash-verified; 128 real photos (100 train/28 dev) are also imported. The preparation node is released. See the scaled run RESULTS.md for exact scope. Generated source count, successfully rendered image count, semantic diversity and optimizer-admitted reference count are different quantities. Use explicit source groups and image hashes; historic development cases do not become novel holdouts through renaming. Preparation is not a quality result.

First larger RL allocation: 128 updates × 32 episodes, with 8 samples per reference group, targeting 512 distinct group sources. Interleave families/difficulty before repeating references or opponents. Record actual optimizer admission and exclusions. Continue data preparation independently of judging/inference integration. Use the shared `native_painting` loop, and the current native RL integration after review, rather than the broken historical three-turn prompt.

Luna low retains the approved holistic aesthetic A/B rubric and visible criticism of both candidates. Preserve missing/uncertain labels; do not invent scores to fill batches. The manual judge handoff has been a throughput bottleneck, so a larger data manifest alone does not make the trainer saturated. Record generation, judgment and trainer rates separately, and keep queues bounded by useful throughput and policy staleness.

No speed reward yet. Matched evaluation uses identical prompts, turn limits and token/context limits. Public hash-verified HF checkpoints remain the output. Record hashes, actual configuration and source revision before launch. No new recurring automation is authorized.
