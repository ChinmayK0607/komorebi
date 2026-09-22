# Photo curriculum SFT: training complete

The fresh-base two-GPU training job completed all 1,024 updates successfully at 2026-09-22 01:15 UTC (06:45 IST). This records training completion; rendered visual evaluation is pending.

## What changed

Astra-high and Sol-high generated 100 reference-specific programs from the 100 training photographs. Linux rendering succeeded for 98. Luna-low inspected all 98; 86 were retained and 12 excluded for visual problems. Two rendering failures are separate from quality exclusions. The teacher gallery retains all candidates, including excluded examples. Parent review spot-checked actual reference/painting pairs and corrected stale/mistyped early label hashes before final admission; raw early review files are preserved locally.

Training includes 86 photo demonstrations, 2,400 distinct procedural reference/target pairs, and 192 general text/vision examples. These 2,678 unique examples supply 8,192 explicit sampled exposures over 1,024 updates at global batch 8. Repeated exposures do not add data diversity. All source examples occur at least once. Actual supervised-token mixture is 40.01% photo, 49.99% procedural, 10.00% general. The final labels and source are committed; source at launch is c5d713a.

## Implementation and launch

Pinned fresh Qwen3.8-27B, Prime 26b3131d2716a4f8210b165df584b83b4bc54f61; rank 16 / alpha 32 LoRA; BF16; vision frozen; learning rate 1e-5; two 96 GB GPUs with replication 2 / sharding 1; microbatch 1; global batch 8; exact assistant-target masking; 16,384-token sequence capacity with no target truncation. Real processor audit found maximum 6,039 tokens. Prime normalizes training gradients by supervised token count across ranks.

The initial 512-update startup was deliberately stopped with no completed optimizer update observed in its preserved logs: source coverage made its desired token proportions impossible. Preserved initial receipts show 21.62/72.97/5.41%. The 1,024-update relaunch reaches the intended 40/50/10 mixture. Its namespace is photo-curriculum-sft-20260922-balanced. Native checkpoints retain the last two states on the node for resumability; public adapters are exported and remotely hash-verified every 64 updates.

## Evidence and limitations

The run passed 64 updates and resumed training afterward. All observed losses and gradient norms were finite, with zero NaN loss counts. The step-64 public adapter is hash-verified; all 992 tensors changed from initialization and all 496 B tensors are nonzero. Procedural validation loss fell from 0.871 to 0.611; this is teacher-forced token prediction, not a visual score. Post-startup median step time was 8.82 seconds over steps 2–13. GPU means were 78.0% and 72.92% across 13 samples each during 22:15:54–22:17:54 UTC. Both were busy at every sample, but the pipeline is not fully saturated. Startup compilation is outside that selected utilization interval and its telemetry is preserved. Prime emitted impossible MFU estimates above 100%; those values are not evidence of efficiency.

Matched baseline is complete on all 28 held-out photographs: 4 valid first-paint canvases; 19 runtime/JavaScript failures; 5 responses lacking a complete JavaScript block, including 2 length cutoffs. All 28 remain in the denominator. This protocol is one response with 8,192 output tokens and 180 seconds rendering; it is distinct from the previous six-turn capability study.

Visual improvement is unknown until saved student checkpoints receive the same evaluation. The next decision compares real-reference fidelity and aesthetics against the immutable base, with failed paintings included. Neither teacher validity nor declining training loss is a visual-quality result. GPU rental rate is $2.38/hour; final utilization, elapsed cost and run results remain pending. No scheduled task was created.

Public trained checkpoint: [step64](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/b08effd090f8aa8ea5e05094f08588fb5237e950/photo-curriculum-sft-20260922-balanced/step_64).

## Completed training

Exit code 0. All 1,024 reported losses were finite with zero NaN losses. All 17 checkpoints, including initialization, are public and remotely hash-verified. Final procedural validation loss was 0.033264 versus initial 0.870856; this measures prediction of held-out procedural programs, not painting aesthetics or photo generalization. Median step time excluding the first was 8.02 seconds.

Whole-job utilization: GPU 0 averaged 72.45%, GPU 1 averaged 72.99%, with 1,098 samples each during 2026-09-21 22:11:52.983 to 2026-09-22 01:15:33.099 UTC. Busy fractions (at least 10% utilization) were 90.35% and 91.26%. This includes startup, validation and checkpoint work. It excludes the subsequent idle allocation period. At the 04:59 UTC status check, the node remained allocated and both GPUs were idle, approximately 3 hours 44 minutes after completion. The next finite job compares step 512 and step 1,024 with the preserved base first-paint evaluation on all 28 held-out photographs.
