# Diversity SFT result: retain the recovery parent

The expanded-data SFT completed, but neither evaluated new checkpoint improved overall visual fidelity over its initializer. Retain public `recovery-quality-20260918/step_72`. Final 384 loses the blinded Luna comparison to that parent 32–13, with 2 ties and 1 uncertain/low-confidence pair. Midpoint 192 also loses overall.

[Comparison gallery](collected-final/evidence/comparison/index.html) · [Raw blinded labels and summary](luna-review/summary.json) · [Observed failure mechanisms](FORENSICS.md)

## Blinded visual result

Six isolated gpt-5.6-luna reviewers inspected all 144 assigned reference/A/B triplets. Model identities were hidden and A/B order balanced within comparison/reference strata. This covers 24 procedural references, two samples per reference, and all three checkpoint pairs. It is model judging, not human ground truth. Low-confidence and uncertain decisions are masked and remain in denominators. Repair/finish episodes have execution analysis but were not part of this final painting-pair review.

Wins below favor the candidate named first.

| Candidate vs anchor | References | Wins | Losses | Ties | Masked |
|---|---|---:|---:|---:|---:|
| Midpoint 192 vs parent | Previous | 7 | 16 | 0 | 1 |
| Midpoint 192 vs parent | New compositions | 9 | 13 | 0 | 2 |
| Midpoint 192 vs parent | All | 16 | 29 | 0 | 3 |
| Final 384 vs parent | Previous | 9 | 15 | 0 | 0 |
| Final 384 vs parent | New compositions | 4 | 17 | 2 | 1 |
| Final 384 vs parent | All | 13 | 32 | 2 | 1 |
| Final 384 vs midpoint192 | All | 24 | 20 | 1 | 3 |

Final 384's reference-averaged preference against the parent is 29.2%, with a reference-cluster bootstrap interval 17.7–39.6%. The two samples of one reference are treated as related observations. This interval describes this small development set and these reviewers; it does not estimate photograph generalization.

Each painting received an absolute-acceptability grade in two blinded comparisons. Requiring both grades to agree gives parent 32 acceptable / 7 unacceptable / 9 uncertain out of 48; midpoint 20 / 15 / 13; final 24 / 14 / 10. Disagreement stays uncertain. The parent itself has substantial remaining quality errors.

## What went wrong

The most concrete setup finding is that **zero of 216 drawing/repair episodes inspected a successful render before finishing**. All positive painting/repair demonstrations end with a program immediately followed by `FINISHED`. The rollout renders that program and honors the finish without another visual turn. Additional turns only repair failed execution. The available eight-turn loop therefore did not become visual self-correction.

Direct image/program inspections also found misplaced or shrunken objects and some object-shape substitutions. In one new bus composition, small local geometry is scaled again through an affine frame, explaining its tiny output. The new-house output also shrinks severely. Simplifying the coordinate representation is a plausible data ablation, not a proven cure. Detailed evidence and the distinction between observation and hypothesis are in FORENSICS.md.

The next demonstration contract should separate paint → actual rendered-canvas observation → revise or finish, and include genuine student visual mistakes. Keep the reference split fixed. Do not convert this development evaluation into training examples. Test coordinate simplification separately so its effect is identifiable. More updates on the same demonstration contract are not supported by this result.

## Executed training and behavior

- Base: pinned Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; initializer: recovery step 72, commit `057c193b1d5191fe3a06eb5e10744b383eba9f1f`.
- Rendered 6,912 programs for 2,304 procedural references across 12 known families. Luna visually reviewed 36 teacher targets; two uncertain targets were excluded, leaving 1,918 eligible new training references. This was stratified review, not exhaustive image grading.
- Completed 384 Prime SFT updates, batch 8, LR 3e-6, rank 16 / alpha 32, frozen vision encoder, fresh optimizer. Native initialization matched all 992 adapter tensors. Every update was finite and changed all 992 tensors.
- Exactly 3,072 draws: 1,536 new paint, 615 repair, 307 finish, 307 earlier teacher replay, 307 general replay. Measured assistant-token shares were 63.81%,25.47%,0.22%,8.83%,1.67%, respectively. Example shares and token shares differ.
- Full assistant targets plus EOS were supervised without truncation; 8,212 preprocessing rows accepted, zero rejected. Training/serving prefix, image-grid and pixel preprocessing parity passed for 0/1/2 images. Validation source groups are disjoint.
- All 288 matched evaluation episodes finished with valid final canvases. That does not imply visual success. Parent/midpoint/final had 10/15/14 invalid attempts, respectively. Midpoint had two 16,384-token cutoffs that subsequently recovered. Final had none.
- Held-out teacher-forced NLL fell from about 0.466 to 0.261 while visual preference regressed. Shared syntax/helper prediction contributes to this loss.
- No RL optimizer updates or speed rewards ran. Real photographs and unseen object families were not evaluated.

The first startup attempt made zero updates because intentional replay draws had duplicate row IDs. Unique draw IDs now retain original source IDs; exact messages and targets were unchanged, and the actual dataset loader was checked before the successful restart. A per-model evaluation producer lock was also added to prevent duplicate work. No duplicate final evaluator was launched.

## Preserved checkpoints and resources

All four new adapters are public and remotely hash-verified. They remain available for diagnosis; availability is not promotion.

- [Retained recovery parent](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/057c193b1d5191fe3a06eb5e10744b383eba9f1f/recovery-quality-20260918/step_72)
- [Step96](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/220cc8ae1659cf4d1bc6effef1cebe41d7a428fe/diversity-sft-20260918/step_96)
- [Step192](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/72f10dc27cdb1d935f7a6ddf7717ec875e07a2cc/diversity-sft-20260918/step_192)
- [Step288](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/8cfae0497dbddae2182295338ab0259f44e14162/diversity-sft-20260918/step_288)
- [Step384](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/25c6873ab1ccbb376b0cf8bbf7f14be93f372b4a/diversity-sft-20260918/step_384)

The final 499,513,124-byte archive contains 26,617 files, each verified by SHA-256. Archive SHA: `59f6aaaad54b7352c928f9e4fd46dad92c4d8e88e7daf50aa50a900a21d22301`. No weights were copied to the Mac. The H100 node was released at 10:43 UTC and provider lookup confirmed it absent. Last provider spend reading: $5.37. Blinded grading used preserved PNGs after release. No GPU remains rented for this run.
