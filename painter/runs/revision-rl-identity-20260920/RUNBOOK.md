# Fresh identity-aware judging pilot

Hypothesis: the explicit image-role mapping repairs false blank descriptions and increases the rate of grounded, reconciled rewards, allowing a short quality-only RL pilot from the contract-quality step24 initializer. This is a limited pilot, not promotion of the model or proof of broad judge reliability.

Matched model baseline, dataset groups, initializer revision/hash, optimizer and sampling settings are in EXPERIMENT.json. Initial setup used source baseline cf2bf9b plus reviewed restart changes bound by the bootstrap bundle hashes. Live queue integration will be copied and hash-verified before activation; source and overlay identifiers will be recorded before launch.

Native Prime runs a separate trainer and inference replica on two RTX PRO 6000 Blackwell Server Edition GPUs, each reported as 97,887 MiB / compute capability 12.0. Rental started 2026-09-20 14:57 UTC at $2.38/hour total, with automatic removal at 16:57 UTC. Connection and provider state are kept in temporary operational files outside Git. The setup-only deployment has a clean queue and no optimizer resume. No old rewards or the invalid one-update adapter are reused.

Initial configuration: 8 samples per state, 16 samples per optimizer update, maximum 32 in-flight episodes, at most 2 policy steps of staleness, 8 optimizer updates, learning rate 1e-6, completion budget 8192 tokens, sequence length 16384. Speed and length penalties remain zero. Initializer is the public pinned contract-quality step24 adapter. Checkpoint uploads use a new public Hugging Face run prefix; verify remote hashes before treating them as preserved.

The six-real-pair diagnostic had correct object/presence/cropping observations in both orientations, 6/6 acceptability agreement and 3/6 preference agreement. This is a small grounding check with parent image review, not artistic accuracy. The three preference disagreements remain masked. Relative ties for identical bytes are recorded separately; shared-image absolute acceptability still comes from the visual judge. The live controller must use --identity-mode. Start fresh Luna-low agents on the *-assets.json shards and assets.json; never send them private keys, policy identities, scores or prior labels. Save original labels before reconciling.

Before claiming progress: verify a finite native optimizer update and weight reload, inspect live reward grounding, report usable and excluded rewards, and evaluate against the identical SFT initializer with the same held-out references and budgets. Record utilization over a stated window for each device and judge-wait time; a busy inference GPU alone is not pipeline saturation. The supervisor stops after a sustained judge stall. Preserve evidence/checkpoints and release the exact owned pod on completion/failure or before lease expiry.

## Launch observation — 2026-09-20 15:07 UTC

The reviewed source overlay at commit `df02bf0` was copied and hash-verified on the node before activation. `deployment-manifest.json` binds the runtime modules and experiment override to the bootstrap bundle. Setup exited zero; the queue contained no old receipts. Native Prime confirmed the new run directory, inference on physical GPU 0 and trainer on physical GPU 1 at 15:07:40 UTC. The supervisor reports running. No optimizer update or quality improvement has yet been established. A five-minute watcher monitors progress, handles Luna-low judging and preservation, and releases the exact owned pod on completion/failure or before expiry.

The native initializer audit confirms 992/992 adapter tensors exactly match the pinned SFT weights, all finite, before the first forward pass. It records a fresh optimizer and no native resume. This confirms the intended starting checkpoint; it is not a training update.

## First live reward batch

`identity-live-001` exported six unique requests as twelve reversed views and twelve unique assets. The original three shards were redistributed across two available Luna-low agents after the agent thread limit prevented a third; `live/reward-batches/identity-live-001/assignment-override.json` records the split and confirms each request retains independent opposite-orientation judges. Original judgments were preserved; supplemental views were appended. The parent visually spot-checked the first cube triplet and confirmed visible objects and the large/small candidate distinction.

All twelve judgments passed the content/role validation. Six receipts were installed on the node: one usable preference/acceptability judgment and five masked for disagreement. This first live batch shows that preference noise is still a bottleneck, despite fixing observed false-blank descriptions on the grounding check. Do not report the judge as broadly reliable or the pilot as improved. Further batches and a native optimizer audit are pending.

Startup-inclusive telemetry from 15:07:18–15:13:38 UTC contains 39 samples per GPU: inference mean GPU utilization 15.4%, trainer 0.3%. This includes model loading/compilation and is not a steady-state throughput estimate. The 15:13:34 inference log reported approximately 318 generated tokens/second. The trainer was waiting for scored groups; the pipeline was not saturated. Raw telemetry remains on the node and a compact snapshot is in startup-telemetry.json.

Operational continuity: if a new fresh Luna-low agent cannot be spawned because of the thread limit, existing low agents `luna_identity_live_001_01` and `luna_identity_live_001_02` may process new batches. Build two supplemental assignments from the private request mapping so neither sees both orientations of a request. Keep private identities and other labels out of their prompts. Use only `--identity-mode` for all new pull/push operations. Batch 001 is complete and installed; do not repeat it.

## Pilot stopped — 2026-09-20 15:38 UTC

The watchdog stopped the native process group at 15:38:53 UTC after 900 seconds without receipt progress. There were 36 generated reward requests, six installed receipts, and no optimizer-update audit or trained checkpoint. Both GPUs were confirmed empty afterward. Batch002 export succeeded, but the SCP transfer stalled at 1,827,840 of 2,480,454 bytes; an authenticated SSH stream subsequently recovered the exact archive with SHA-256 `27fe8839d6773ded593cd098f18c5392292ad313e4a6afa5c408f7710a73ab2d`. This is a failed feedback-delivery pilot, not evidence of a painter quality regression.

The stopped-run evidence archive is 11,624,779 bytes with SHA-256 `6b64378174c89164c09829408febfbe860c8b398d8a312ffc3d10ee09489b950`; all 472 manifest members were hash-verified locally under `live/preservation/`. No new trained checkpoint exists to upload. Training reference/canvas files are also available from the original pinned bootstrap archive; these were recovered for offline diagnosis and checked against request hashes. No ML or rendering ran on the Mac.

The exact owned pod was released by approximately 15:44 UTC and Lium returned an empty active-pod list. The watcher was deleted. About 47 minutes at $2.38/hour is approximately $1.86, an elapsed-time estimate rather than an invoice.

Ten-second telemetry over 15:07:18–15:38:48 UTC, 190 samples/device, gives mean inference GPU utilization 16.75% and trainer 0.07%. Fractions of samples above zero were 17.37% and 0.53%, respectively. This startup-inclusive measurement is not steady-state efficiency; the pipeline was not saturated. See `utilization-summary.json`.

Post-stop batch002: 16 labels passed identity validation for eight requests; six reconciled, two masked. Four usable results are identical-image ties with both candidates unacceptable; two are non-identical preferences. The parent directly inspected the cup-reference/blank-candidate pair and confirmed the description. These labels were never installed into the stopped run. The remaining 22 saved requests are being judged independently in `live/preservation/extracted/reward-batches/identity-saved-004`. The incomplete local staging attempt003 is preserved; it lacked bootstrap reference files and produced no labels.

Next decision: fix bounded, resumable transfer first; assess all36 saved requests and distinguish identical-image controls from non-identical preferences. Do not silently relax agreement thresholds or replay these traces through an unsupported offline optimizer. The current native implementation waits on live receipts and does not support optimizer/data resume.

## Post-stop provenance failure found by parent image review

The saved004 batch is unusable as reward evidence. Parent inspection of view0041 found that reference asset0011 was an orange bus and candidate asset0003 a severely cropped bus fragment, while judge01 called them centered cubes. The agent explicitly admitted reusing descriptions from prior batches with the same ordinal asset IDs, and displaying only 11 assets rather than all required images before claiming completion. Judge02 correctly described this inspected bus pair and reports displaying its actual images; that does not validate the uninspected remainder. Raw labels remain unchanged. Do not report reconciliation counts for saved004 as reliable rewards, and do not train on them.

This establishes cross-batch image-identity confusion in one agent, not a general root cause for every earlier failed judgment. Follow-up implementation replaces batch-local ordinal asset IDs with full image SHA-256 IDs so different images cannot acquire the same identifier across batches. This fixes the demonstrated identifier collision but cannot itself prove actual inspection. Future audit uses small assignments and direct parent visual spot checks.

## Reviewed repairs

Luna xhigh implemented bounded SSH streaming with retry-safe reuse of exported batches, archive receipt/hash validation, temporary downloads and no-clobber installation. Push/status operations also have hard timeouts. Parent review caught and corrected a duplicate-key success-report bug. The queue now uses `asset-<full SHA256>` globally rather than per-batch ordinals; receipt validation rejects mismatched IDs. Old manifests and raw labels were not rewritten. The original runtime commit remains `df02bf0`; these repairs were made after shutdown and have not run in a new training pilot.

Parent execution of the queue/transport suite passed 17 tests, covering cross-batch identity collisions, same-byte stability, transport failure, hash mismatch, retries and conflicting existing artifacts. A one-pair, two-orientation regression probe (`live/reward-batches/sha-identity-probe`) correctly described the bus and rejected both severely cropped candidates. Both judges had already been informed of the earlier bus/cube error: this is a regression sanity check, not a fresh blinded accuracy test. The saved004 labels remain unusable.

The next run must keep individual judge assignments small and verify actual image descriptions. Permanent image IDs and valid JSON cannot prove inspection. Before another rental, use unseen known-answer pairs to test observation grounding and measure usable non-identical preferences; then launch a bounded native pilot from the same trusted SFT initializer. Group-completion scheduling and durable native replay remain unimplemented proposals, not available functionality.
