# Offline reward and handoff repair

No GPU was rented and no optimizer step ran in this work. This is an offline replay of the stopped neutral pilot, plus infrastructure fixes and tests. It does not establish painting improvement.

## Reward result

The matched input is the same preserved archive, SHA-256 `b00964e4e486b4ec7220738e7ae52fa72f63ba2380810ab2b3d93d9f8a38739a`. The replay joins each recorded action and state to the reference/anchor/candidate byte hashes. It does not infer the action from the canvas pathname. Original receipts are preserved, including rejected labels.

There are 95 rendered attempts: 71 valid transitions and 24 program failures. The valid transitions map to 62 unique visual requests. Five reviewed requests are excluded: four contain false visual descriptions or comparisons, and one treats an ambiguous reference part as a required object. See [grounding review](GROUNDING_REVIEW.md) and [explicit exclusions](grounding-exclusions.json). This targeted review does not certify all remaining labels.

| Evidence scope | Visual transitions scored by v1 | Visual transitions scored by v2 |
| --- | ---: | ---: |
| 20 installed receipts | 12 | 12 |
| Installed + 4 receipts completed after shutdown | 13 | 16 |

These counts exclude the 24 deterministic invalid-program penalties. The supplemental receipts were never installed in the stopped run. Increased coverage is not increased judge accuracy. V2 separates relative correction preference from absolute finish acceptability and removes absolute-quality bonuses from paint/repair rewards. Known grounding errors remain excluded rather than rescued by order agreement. V2 remains an offline comparison; the live reward implementation has not been switched.

## Native evidence and remaining limitation

63 completed attempts have native trace token IDs, masks, sampled log-probabilities, and real group identifiers in the monitor stream. The other 32 rendered attempts lack saved native traces. A preliminary audit missed the existing fields because it checked alternate field names; this was corrected against the actual nested native schema and covered by a regression test.

The presence of native fields is not a trainer-ready batch. Packed multimodal data, native credit assignment and current policy eligibility still need verification through the pinned Prime path. The audit therefore reports `training_ready: false`. It finds six fully observed eight-member groups; separately, the installed evidence has two groups with at least four scored members and nonzero reward variance, including one group with only seven observed members. The supplemental v2 analysis has three such groups. These are diagnostic group-signal counts, not admitted training batches.

## Infrastructure changes

Repeated renders with identical image hashes now reuse the first queue request without resetting its timestamp. Concurrent arrivals use an atomic no-clobber install. Different rollout paths with identical bytes are accepted, while conflicting hashes and changed cached image bytes are rejected. Judge assignment and receipt files are preserved.

The trainer wrapper now observes the existing native Prime batch boundary and retains the exact `rank_N.bin` payload with its hash before forward execution. The original payload includes opaque multimodal data; a JSON summary is not substituted for it. Optional metadata or observation failures do not create a new optimizer admission gate. Prime continues to own grouping, score eligibility, stale-policy handling and native batch validation. The preservation script narrowly retains these transport files without adding model weights to the evidence archive.

Runtime state uses the current run's batch directory and parsed optimizer audit records, instead of treating setup completion as training progress. These changes have been reviewed against Prime revision `26b3131d2716a4f8210b165df584b83b4bc54f61`. The local suite passes 61 tests covering queue concurrency, nested native fields, packed/padded batch observations, missing optional metadata, wire retention and archive allowlisting. The actual archive was replayed separately for both evidence scopes. Final observer tests, Python compilation and shell syntax checks also pass. These are not Linux GPU integration tests, and the new observer has not yet run through a real CUDA update.

## Decision

Do not repeat the previous rental configuration merely with a weaker reconciliation rule. The reward change alone did not increase coverage on installed receipts, and an agreed comparison was itself wrong. Keep the original evidence, the explicit exclusions, and the revised formula separate.

The next node experiment should be a bounded native replay/update check using the completed traces and verified scores, with the exact packed batch retained and a parameter-change audit. That experiment has not run. Pending traces are not recoverable merely by retaining a later packed batch. The judge's remaining geometry errors also require a targeted validation before treating new visual rewards as dependable.

Detailed counts, report hashes, and local evidence paths are in [RESULTS.json](RESULTS.json). Source identifiers and validation results are recorded in [EXPERIMENT.json](EXPERIMENT.json). GPU utilization is not applicable for this offline work; incremental GPU rental cost is $0.
