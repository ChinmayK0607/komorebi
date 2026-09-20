# Painter research index

Updated 2026-09-20. This is a compact index of the current painter curriculum, recovery, diversity, contract, and revision-RL records that are present in this workspace. It is intentionally bounded: archived folders, tarballs, and unrelated upstream clones under `work/` are not an exhaustive historical record. Read the linked run documents for scope, raw labels, hashes, and limitations before treating a statement as a result.

## Current run

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

For every new run, preserve one hypothesis, immutable model/data/config/source identifiers, a matched baseline, results against that baseline, failures and censored cases, utilization measured over a stated window, cost, and the next decision. Keep GPU activity and judge waiting separate in the accounting. Use public, hash-verified Hugging Face checkpoint references instead of copying weights into this workspace. See the root [working agreement](AGENTS.md) and the root [.gitignore](.gitignore) for the tracking boundary.
