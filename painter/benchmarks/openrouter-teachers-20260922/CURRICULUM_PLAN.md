# Teacher demonstration curriculum (pending teacher selection)

**Hypothesis.** A student trained on visually screened reference → paint → inspect → revise trajectories, ordered from short complete paintings to longer complex paintings, will improve reference fidelity and visual finish over the matched SFT initializer without losing its ability to execute the painting contract. The current matched baseline is photo-curriculum SFT step 512; RL step 24 is a separate diagnostic comparison, not a demonstrated visual winner.

The teacher is selected from the quality benchmark by blind, reference-conditioned pairwise comparisons of actual canvases, with render/transport failures reported separately. One attractive painting or syntax validity alone does not select a teacher. Keep model-specific latency and token cost separate from visual preference. Use reference images with known provenance, deduplicate by image identity, and keep evaluation references out of training.

## Progressive episode lengths

| Tier | Teacher turn cap | Reference mix | What a retained example teaches |
| --- | ---: | --- | --- |
| Short | 1–2 | Clear silhouettes, single objects, simple color groups and composition | A complete recognizable first sketch and valid stop after observing a canvas |
| Medium | 3–5 | Animals, vehicles, interiors and small multi-object scenes | Correcting proportion, placement and missing structure after visual inspection |
| Long | 6–12 | Crowds, occlusion, perspective, complex light and texture | Deliberate multi-turn refinement and aesthetic finish without losing the reference |

These are **maximums**, not quotas. A teacher may finish early after a valid render. Do not retain an incomplete short episode merely because it fits a token or turn budget. Raise the proportion of medium and long accepted trajectories as training proceeds, while continuing to sample earlier tiers so the model retains first-pass competence. Keep the source families and visual difficulty diverse within every stage.

For an accepted trajectory, preserve the reference, every assistant paint or finish response, the exact rendered canvas and feedback visible before the next response, program and render receipts, usage, and a human-readable reason for admission. Supervise actions conditioned on what the teacher actually saw. Keep invalid turns only when a subsequent valid repair is part of the accepted trajectory; do not label a failed render as a successful painting. Provider reasoning fields are evidence, not automatic SFT targets.

Build batches by accepted visual quality and coverage, not by raw generation count. The pairwise judge should see the reference and blinded canvases, record fidelity, aesthetics, ties/uncertainty, and the preferred image's weakness. Human spot checks should cover each subject family and the failure boundary. Mix prior general-purpose and validated painting-contract examples during SFT; measure matched held-out image-pair preference, final-canvas validity, explicit finish rate, and turn/token cost after each stage. Only after quality is reliable should a separate bounded-turn curriculum optimize speed.

**Not launched.** Teacher identity, source manifest, accepted-count targets, mixture ratios, spend, and training hyperparameters remain to be recorded before generation/training. The first pilot used 384,363 reported total tokens in 12 calls, so scale based on accepted demonstrations per dollar and per hour rather than multiplying that single episode by thousands without measuring yield.
