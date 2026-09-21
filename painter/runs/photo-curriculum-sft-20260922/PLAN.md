# Photo curriculum SFT — 2026-09-22

## Hypothesis and comparison

Fresh-base Qwen3.8-27B trained on complete, rendered, reference-specific painting demonstrations will improve photo-to-painting structure and appearance over the unchanged base. The last COCO evaluation found frequent no-code plans and invalid programs in contract-quality step24; continuing that adapter is not the initializer for this experiment. This hypothesis does not assume that syntax validity establishes visual quality.

The matched baseline is `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Keep the canonical canvas-pixels-preview-v2 contract and frozen renderer assets. Compare base and new checkpoints with the same references, sampling, context, turn and output-token limits, and 180-second rendering deadline. Prior 120-second results are diagnostic context, not an exactly matched new baseline.

## Data scope

- Generate reference-specific demonstrations for the existing 100 COCO128 **training** photographs, divided evenly between Astra-high and Sol-high. Preserve reference hashes and teacher provenance in the two assignment manifests. Render and inspect initial examples early, while generation continues.
- Use the existing 2,400 distinct procedural training references as structural foundations, with their actual final programs. These targets are not aesthetic gold. Any repeated sampling is exposure, not additional reference diversity.
- Add only genuine coarse-canvas-to-complete-painting corrections where their input state and target are available. Do not fabricate successful inspection or train short plans as complete painting responses.
- Keep all 28 COCO development photographs and 240 procedural development references out of training. Twelve COCO dev images have already been inspected in the capability study; distinguish them from the remaining sixteen in evaluation records.
- Include retained general multimodal/text examples when their provenance and paths resolve. Record example counts and supervised token mass by source; do not let a large short-target pool silently dominate the photo demonstrations.

The planned 100 new teacher programs are candidates until rendering and review. Final accepted counts, data hashes, mixture, sequence limit and training hyperparameters will be recorded before launch. Teacher plans alone and reference-equals-canvas finish rows are excluded.

The proposed finite schedule is 1,024 updates × global batch 8 = 8,192 example exposures, covering every eligible source once before adding repeats and then deterministically interleaving the complete schedule. Requested supervised token proportions are 40% photo, 50% procedural and 10% general retention; actual fractions and exposure counts are authoritative. The planned optimizer uses rank-16/alpha-32 LoRA, BF16, a frozen vision encoder, 16,384-token full-sequence capacity and learning rate 1e-5. These settings remain subject to the actual full-target processor audit; no truncation is permitted.

The first matched visual comparison uses **all 28 COCO development photos, one painting response each**, 8,192 output tokens, temperature 1, seed 918200, thinking disabled, 16,384 context and a 180-second renderer deadline. This measures direct photo-to-program learning without confounding it with a weak revision policy. It is distinct from the preceding six-turn study. Base inference can run while teacher data is prepared; the same first-paint protocol will evaluate saved checkpoints. Multi-turn revision quality remains a separate subsequent measurement, not something inferred from first-paint scores.

## Execution

Prepare the finite SFT job on Linux, starting from the public base with a fresh optimizer and adapter. Preserve full assistant targets, EOS and correct image/template input. Use the existing Prime training implementation, reviewing distributed sampling, reduction and checkpoint export before choosing its supported launch mode. No new online judge dependency is placed inside the SFT optimizer loop.

A two-GPU node with 2 × RTX PRO 6000 Blackwell Server Edition (96 GB each) was rented at an advertised $2.38/hour. Renderer installation and data preparation precede training; GPU saturation is not claimed during setup. Record actual training utilization over an explicit sample window, elapsed rental cost, finite gradients, checkpoint identities and evaluation failures. Checkpoints must be public on Hugging Face and hash-verified. No recurring task or scheduled watcher is authorized or created.

Native optimizer/model/data checkpoints are retained temporarily on the node (last two, every 64 updates) for same-node resume, while portable adapters are uploaded publicly every 64 updates. A public adapter alone is not a full optimizer-state resume point. The frozen source, finalized dataset and exact resolved configuration will identify the launched run.

## Decision

Evaluate intermediate and final checkpoints against base on held-out reference-to-painting tasks, including failures in the denominator. Inspect whether successful revisions improve rather than merely consume more turns. Promote a checkpoint only on measured visual evidence; if an intermediate checkpoint is stronger, preserve and select it instead of treating the last update as inherently best.

## Launch admission adjustment

All 98 rendered photo candidates received Luna-low visual review. Admit 86 and exclude 12 weak targets; two additional candidates failed rendering. The initial 512-update launch was stopped during startup because coverage of every source made the requested token mix infeasible (21.62% photo, 72.97% foundation, 5.41% retention). Preserve that attempt separately. Increase the finite exposure budget to 1,024 updates and verify the actual token fractions before relaunch. No new optimizer result is inferred from initial adapter export.
