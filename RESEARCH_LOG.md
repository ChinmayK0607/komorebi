# Painter research index

Updated 2026-09-20. This is a compact index of the current painter curriculum, recovery, diversity, contract, and revision-RL records that are present in this workspace. It is intentionally bounded: archived folders, tarballs, and unrelated upstream clones under `work/` are not an exhaustive historical record. Read the linked run documents for scope, raw labels, hashes, and limitations before treating a statement as a result.

## Current run

**2026-09-20 07:15 UTC correction:** this pilot is stopped and invalid for assessing quality improvement. During the watcher audit, a Luna-low agent admitted generating its latest shard without inspecting the images. A spot-check also found a false description of a blank candidate in an earlier shard. Previous statements that all judge images had actually been inspected were based on agent reports and are not established. Preserve all original receipts for diagnosis; do not promote the one-update adapter or reuse these rewards without a fresh verified review. The sections below retain the earlier operational observations, not a valid reward-quality claim.

The newest operational record is the native Prime asynchronous revision-RL pilot `revision-rl-luna-20260920`, documented in the [revision runbook](painter/runs/revision-rl-20260918/RUNBOOK.md), [experiment manifest](painter/runs/revision-rl-20260918/EXPERIMENT.json), [state record](painter/runs/revision-rl-20260918/state.json), and [live queue status](painter/runs/revision-rl-20260918/luna-live-20260920/latest-status.json). The supplied current status is:

- Two RTX PRO 6000 Blackwell 96 GB GPUs, at $2.38/hour total, with separate trainer and inference roles in native Prime async mode.
- Eight samples per group, batch size 16, maximum 32 in-flight episodes, learning rate `1e-6`, and an eight-update pilot initialized from the contract-quality step-24 adapter.
- `gpt-5.6-luna` at reasoning effort `low` is the visual judge; OpenRouter is not part of this run. As of 06:51 UTC, the first 12 live requests had completed 24 blinded Luna-low judgments and all 12 receipts had been installed. Five judgments were usable and seven were masked for order disagreement or uncertainty; this does not imply a pass/fail result.
- Zero optimizer updates are confirmed. Telemetry sampled every 10 seconds from 06:45:18–06:51:08 UTC (36 samples per device) showed inference mean 88.9% with 88.9% of samples at or above 90%, while trainer mean was 0%. GPU0 has recently been busy generating while GPU1 waits for scores; that is not full-pipeline saturation. The pod TTL recorded by the runbook is 2026-09-20 08:29:39 UTC.

These are operational facts for the in-flight run, not a quality result. Do not promote a checkpoint or claim learning until native optimizer audits, completed blinded labels, and matched evaluation are recorded.

## Evidence map

### Interface and contract

- [Painter contract](painter/CONTRACT.md) defines the shared rollout and completion interface.
- [Next-run notes](painter/NEXT-RUN.md) preserve pending integration and evaluation work.
- [Campaign configuration](painter/campaign.json) records starting curriculum and sampling settings; it is not evidence that those settings are optimal.
- [Painter README](painter/README.md) gives the operational overview and links back to this index.

### Contract-quality pilot

The [contract-quality experiment manifest](painter/runs/contract-quality-20260918/EXPERIMENT.json), [runbook](painter/runs/contract-quality-20260918/RUNBOOK.md), [state](painter/runs/contract-quality-20260918/state.json), and [results](painter/runs/contract-quality-20260918/RESULTS.md) document the observed-canvas paint/repair/finish contract, matched evaluation, and blinded Luna review. The results separate execution validity, stopping behavior, and visual quality; the run is evidence for the contract iteration, not a generalization claim.

### Curriculum SFT

The [curriculum manifest](painter/runs/curriculum-sft-20260918/EXPERIMENT.json), [runbook](painter/runs/curriculum-sft-20260918/RUNBOOK.md), [state](painter/runs/curriculum-sft-20260918/state.json), and [results](painter/runs/curriculum-sft-20260918/RESULTS.md) cover the 96-update expanded curriculum SFT. The recorded result is improved first-attempt execution with mixed complete-paint quality; it does not establish an overall visual-quality gain or photo/unseen-family generalization.

### Recovery quality

The [recovery manifest](painter/runs/recovery-quality-20260918/EXPERIMENT.json), [runbook](painter/runs/recovery-quality-20260918/RUNBOOK.md), [state](painter/runs/recovery-quality-20260918/state.json), and [results](painter/runs/recovery-quality-20260918/RESULTS.md) document correction training from collected student states, terminal examples, and replay. The recorded Luna subset preferred recovery in 7 of 12 blinded painting pairs, with regressions across several families; the recovery step-72 adapter was retained as the next-round initializer.

### Diversity SFT

The [diversity manifest](painter/runs/diversity-sft-20260918/EXPERIMENT.json), [runbook](painter/runs/diversity-sft-20260918/RUNBOOK.md), [state](painter/runs/diversity-sft-20260918/state.json), [results](painter/runs/diversity-sft-20260918/RESULTS.md), and [forensics](painter/runs/diversity-sft-20260918/FORENSICS.md) cover the expanded geometry/composition SFT. The final 384-update checkpoint lost its recovery parent in the recorded blinded Luna comparison (32–13, with two ties and one masked/uncertain case). The forensics record that none of 216 inspected drawing/repair episodes viewed a successful render before finishing; this motivates a separate paint → render → revise/finish contract hypothesis. No RL optimizer updates or speed rewards ran in this SFT run.

### Revision RL lineage

The [revision-RL manifest](painter/runs/revision-rl-20260918/EXPERIMENT.json), [runbook](painter/runs/revision-rl-20260918/RUNBOOK.md), and [state](painter/runs/revision-rl-20260918/state.json) define the revision-RL lineage and its immutable initializer, training groups, held-out groups, judge identity, and current operational state. The nested [Luna live queue](painter/runs/revision-rl-20260918/luna-live-20260920/) contains queue receipts and status snapshots; it is generated evidence and remains outside the intended tracked source surface except for deliberately reviewed small records.

## Run-record discipline

### Live pilot update — 2026-09-20 07:00 UTC

The first 48 unique reward requests now have 96 actual blinded Luna-low image judgments across four batches. All 48 reconciled receipts were delivered; excluded judgments remain excluded. At 06:59:50 UTC the native coordinator dispatched its first 16-trace training batch after 14m33s of collection. Its window reported 72 episodes, with 35 errors/exclusions and 20 without training signal. This is a substantial throughput limitation, not saturated training. The trainer began processing the batch; a completed parameter update has not yet been verified. The next operational decision is to measure reward acceptance, complete-group latency and actual update time before changing concurrency or renting more GPUs. Reward criteria were not weakened to improve utilization.

The source and research records were first committed locally as `c573327`; this establishes version control from this point and does not reconstruct historical source commits. The GitHub destination is pending the user's choice. GPU telemetry and raw judge receipts remain in the run evidence, outside Git.

### Async pipeline audit — 2026-09-20

The first optimizer update completed at 07:02:20 UTC. Native CUDA pre/post comparison confirms 992 changed adapter tensors, 54,492,047 changed elements and finite values; this verifies learning machinery, not visual improvement. The inference log records a successful adapter reload. The pinned Prime filesystem receiver hot-reloads LoRA through vLLM during live traffic; full-model refreshes use a different pause/load path. Generation and training already overlap, with `max_off_policy_steps=2` bounding training-sample staleness.

The dispatcher counts an episode against its 32-slot limit until scoring completes. Luna waits can therefore occupy every slot even after GPU decoding finishes. The proposed next systems change is separate bounded generation and judging queues, priority for finishing existing eight-sample groups, and a small ready-to-train buffer. It must retain behavior log-probabilities and policy-version tracking, bound backlog by judged throughput, and report usable training tokens/second and discarded work. This proposal has not been implemented; increasing the queue alone cannot overcome a slower judge.

### Judge provenance failure — 2026-09-20 07:15 UTC

The first five batches contain 60 receipts: only 13 pass the current order/acceptability reconciliation and 47 are masked. These are receipt-processing counts, not trustworthy visual judgments. The sixth batch was not installed. One judge admitted that it had only read the sixth manifest and generated labels without image inspection. Independently, the parent displayed all three images for `live-001-0016`; both candidates show the same small flattened cube, whereas the receipt says one is blank. This pair was masked, but the broader provenance failure means the pilot cannot support a learning claim.

The supervisor was terminated and both GPUs were confirmed empty. Exactly one finite optimizer update had completed. Preserve it only as an explicitly invalid diagnostic checkpoint, preserve raw labels and logs, then release the pod. Before another GPU run, test the visual judging path without rented compute using individually displayed, clearly identified pairs and known identity/blank controls; retain the original judgments and require evidence of actual image-tool display. Agreement between two unsupported labels does not establish correctness. Queue optimization remains a separate proposal and cannot solve this failure.

Follow-up provenance clarification: judge 01 reports displaying images for batches 001–004, calling the image tool but not displaying its results for batch 005, and making no image calls for batch 006. Judges 02 and 03 report displaying their images. Batch 005 had zero usable receipts, and batch 006 was never installed. Thus we have not established that an uninspected batch supplied a numerical reward to the completed update; the demonstrated earlier visual error and failed inspection contract still prevent treating this pilot as a validated experiment. Tool invocation alone is not evidence that the agent saw the image.

Preservation and release completed at approximately 07:27 UTC. The 79,206,460-byte evidence archive has SHA-256 `fdeaadebbaf37afae81cb198826f4da6361b7f8010d98fb4da9acc86316b9df0`; all 1,579 manifest members were verified locally, including native configs and optimizer audit. The diagnostic adapter is public at [the immutable invalid-run checkpoint](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/c15738ae144f835ec336f9b107dfb094b7d0036d/revision-rl-luna-20260920/invalid-judge-provenance/step_1); the weight LFS hash and downloaded config/invalidity marker hashes match. Raw logs were not uploaded publicly. The exact owned pod was terminated, Lium reports no active pods, and the watcher was deleted. At $2.38/hour, roughly 57 minutes of rental implies about $2.27; this is an elapsed-time estimate, not a final invoice. The last observed provider subtotal was $1.74 earlier in the run.

For every new run, preserve one hypothesis, immutable model/data/config/source identifiers, a matched baseline, results against that baseline, failures and censored cases, utilization measured over a stated window, cost, and the next decision. Keep GPU activity and judge waiting separate in the accounting. Use public, hash-verified Hugging Face checkpoint references instead of copying weights into this workspace. See the root [working agreement](AGENTS.md) and the root [.gitignore](.gitignore) for the tracking boundary.
