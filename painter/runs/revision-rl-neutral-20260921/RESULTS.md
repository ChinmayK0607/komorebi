# Neutral-role pilot result

Stopped20 September2026 at21:15:44 UTC at the frugal deadline. **Zero optimizer updates; no new model checkpoint or quality-improvement result.** The trainer had8/16 rows ready; native files contained only the startup step0 broadcast. SIGTERM in the final trainer log is the intentional stop, not a discovered optimizer crash.

## What completed

The exact trusted SFT initializer was verified (992/992 tensors) before the native two-update job started. The32-case held-out baseline completed:32 valid final canvases,195/198 valid turns,13 explicit finishes,19 operational cutoffs,133 non-finish no-change turns. Mean generated tokens3910.7/case, mean6.19 turns. This is execution behavior, not a visual quality score. Task/model/config hashes and disjoint groups are recorded in EXPERIMENT.json, deployment-manifest.json and RUNBOOK.md.

Six batches received48 first-pass blinded views for24 requests. Twenty receipts were installed:10 usable and10 excluded. Batch006 finished after the stop and was not installed (1 usable,3 excluded). There were62 requests on the node when stopped. Raw labels, including factual failures, serialization-only corrections and an accidental duplicate diagnostic, remain preserved. No excluded judgment was converted to a tie for training.

## Findings and changes

- Reused judge context still produced a false missing-plane description; fresh-context judging correctly diagnosed that diagnostic pair. This suggests a context contribution but does not establish causality or general accuracy. Later views used fresh Luna-low subagents. Subjective proportion/acceptability disagreements remain.
- Queue export selected lexical hash order. A reviewed and tested change now selects oldest created_at first with stable ties; deployed queue SHA is in RUNBOOK.md. It preserves scoring semantics. It was deployed late in this pilot and did not produce an optimizer update before the deadline.
- Absolute acceptance was coupled to relative preference. Of the first16 requests,11 had confident order-consistent preferences;4 ofthose were discarded solely because absolute acceptance disagreed. That arithmetic does not prove their visual judgments correct. Standalone quality_signal_v2.py separates relative paint/repair rewards from absolute finish rewards. It is **not integrated or deployed**; the live run retained v1. Parent reviewed the code and passed50 tests.
- A delegated coordinator mistakenly used app tasks instead of collaboration subagents. Those three auxiliary tasks were stopped/archived; original outputs were preserved, and parent resumed direct one-view subagents. This added avoidable latency.

## Utilization and cost

Measured20:23:30–21:15:41 UTC at10-second intervals,314 samples per each of4 GPUs (1256 device observations). Mean utilization: rollout GPU20.83%, trainer0.089%, baseline GPU17.16%, unused GPU0%. Nonzero sample fractions:21.97%,0.318%,17.83%,0%. Full-node mean was9.52%; this includes warmup and baseline idle time. The pipeline was not saturated and external judging dominated waiting.

The exact owned node was released atapproximately21:18 UTC; provider inventory then showed no active pods. Estimated66-minute rental at$4.76/hour costabout$5.25, not a final invoice. Watcher deleted.

## Preserved evidence and next decision

Final archive41,664,395 bytes, SHA-256b00964e4e486b4ec7220738e7ae52fa72f63ba2380810ab2b3d93d9f8a38739a; all1674 file hashes verified locally, including baseline and RL inputs/rollouts. Separate baseline snapshot823 files is also verified. Raw/generated artifacts remain outside Git; small manifest/results are tracked. There is no trained checkpoint to upload; the pinned SFT initializer already exists publicly on Hugging Face.

Next: validate the standalone reward separation offline against saved, grounded examples and measure usable complete-group throughput before another rental. Prepare judging and a consumable training batch before holding a trainer idle. Existing receipt agreement alone is insufficient evidence of visual correctness; retain fresh views and independent spot-checks. Do not restart this run or rent a replacement automatically.

Offline signal-normalization audit across all24 labeled requests (including uninstalled006): v1 joint signal usable on11; v2 relative channel available on18, candidate absolute channel on18. Raw judgments were unchanged. This is coverage arithmetic, not independent visual validation or a transition-reward replay; shared-canvas visual requests can serve multiple actions. No extra GPU rental or training.
