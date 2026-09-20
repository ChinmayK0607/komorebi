# Diversity SFT: observed failure mechanisms

Follow-up: see [the regression/token audit](diagnostic/REPORT.md). The parent also finishes early, so that behavior alone does not explain the new regression. The follow-up traces confirmed coordinate-convention errors, distinguishes genuine wins from consistency, and compares per-turn token counts across all three runs.

This note distinguishes measured behavior from explanations that still need an ablation. It does not replace the complete blinded Luna comparison.

## The successful-render feedback loop was never used

`visual-feedback-use.json` audits all three checkpoints. Each has 72 drawing/repair episodes. Every one of their 72 successful painting actions declared `FINISHED` immediately. None continued to receive its successfully rendered canvas before deciding whether to revise. The other 24 tasks per checkpoint begin with an already complete painting and test stopping.

The extra turns seen in the gallery are execution repairs, not successful-render visual corrections. Parent/midpoint/final have 10/15/14 invalid attempts respectively; all eventually produce valid final canvases. Midpoint has two 16,384-token cutoffs, which subsequently recover. Final has no token cutoffs in this evaluation.

This behavior is consistent with our demonstrations: `recovery_data.materialize` appends `FINISHED` immediately after every positive painting/repair program. `rollout.run` honors that declaration after rendering, before presenting that render in another model turn. Our environment supports visual feedback, but the demonstrated policy bypasses it.

The next correction should make painting and finishing separate decisions: emit a complete program, show its actual canvas, then supervise either a specific visual correction or a finish decision. Include genuine student visual errors as starting canvases. Merely allocating eight turns does not teach or require this behavior. Whether this change improves fidelity still needs a matched evaluation; it is not established by this audit.

## Valid programs can encode the wrong geometry

In new-composition bus case `v3-bus-0029-paint/sample-00`, the reference contains a bus on the left and a sailboat on the right. The final checkpoint produces a very small bus and smaller boat with shifted placement. The program combines small local coordinates with another scale transform: the bus body spans local x49–124 and then applies scale0.56 plus translation. The reference program instead spans local x−170–170 and applies its transform once. This specific code explains the small body in that output; it does not prove the cause of every error.

The new-house sample likewise shrinks the subject markedly. The new-flower sample has an object/shape substitution, so the problem is broader than only a global canvas scale. These are direct inspections of preserved reference/parent/midpoint/final PNGs, not claims inferred from loss.

The expanded target representation introduces per-object affine frames and reusable local-coordinate geometry. Confusion between local and canvas coordinates is a plausible contributor. A useful isolated data ablation is to compile the same teacher geometry into a single explicit canvas-coordinate representation and verify identical node renders before training. Keep the reference split, initializer and evaluation scaffold fixed while testing that representation change.

## What the run verifies, and what it does not

All 384 updates are finite, all 992 adapter tensors change at each update, and the parent initialization matches exactly. All four checkpoints are public and hash-verified. Training and serving prefix/image preprocessing checks pass for zero, one and two images. All 288 evaluation episodes finish with a valid canvas. These checks establish execution and provenance, not visual fidelity.

Held-out teacher-forced NLL falls from about0.466 to0.261. Shared program syntax and helper code contribute to this number; it is not a reconstruction score. The blinded image comparison determines whether a checkpoint should replace the parent. Do not promote final384 solely because it has lower loss or more training.

The new references are procedural compositions within twelve known recipe families. This run does not establish photograph generalization or performance on unseen object families. There were no RL optimizer updates and no speed reward.
