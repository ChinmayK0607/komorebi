# Curriculum repair candidate review — 2026-09-21

This is a small, manually reviewed candidate pool for the next structured painting stage. It responds to the completed [aesthetic RL run](../aesthetic-rl-20260921/RESULTS.md), whose admitted groups collapsed to `v2-box-000` and `v2-bus-000` even though the prepared state manifest named 24 states across 12 families. The candidate pool distributes two already-prepared complete references per family and reserves a third family-balanced set for development; these are reused synthetic assets, not 24 new references.

The machine-readable source is [candidate-manifest.json](candidate-manifest.json). It records the source PNG, teacher program, renderer receipt, and SHA-256 for every selected artifact. Every selected receipt is valid, has no renderer errors, reports a 600x600 canvas, and hashes the current program and PNG. Every selected program starts `function draw(){translate(-300,-300);`, which directly checks the coordinate failure seen in the prior RL traces. No rendering, model inference, training, node rental, or scheduling was performed for this review. The [repair gallery](repair-review.html) shows eight additional teacher-intermediate inputs beside their complete targets.

## Selected pool

The 24 train groups are the following two source IDs in each family:

| Family | Train groups | Reserved teacher-development row |
| --- | --- | --- |
| box | `v2-box-000`, `v2-box-001` | `v2-box-008` |
| bus | `v2-bus-000`, `v2-bus-001` | `v2-bus-008` |
| cup | `v2-cup-000`, `v2-cup-001` | `v2-cup-008` |
| cylinder | `v2-cylinder-000`, `v2-cylinder-001` | `v2-cylinder-008` |
| flower | `v2-flower-000`, `v2-flower-001` | `v2-flower-008` |
| house | `v2-house-000`, `v2-house-001` | `v2-house-008` |
| ribbons | `v2-ribbons-000`, `v2-ribbons-001` | `v2-ribbons-008` |
| sailboat | `v2-sailboat-000`, `v2-sailboat-001` | `v2-sailboat-008` |
| sphere | `v2-sphere-000`, `v2-sphere-001` | `v2-sphere-008` |
| still life | `v2-still_life-000`, `v2-still_life-001` | `v2-still_life-008` |
| tree | `v2-tree-000`, `v2-tree-001` | `v2-tree-008` |
| vase | `v2-vase-000`, `v2-vase-001` | `v2-vase-008` |

I opened all 36 `turn-3.png` files with `view_image`. The train and reserved images are complete, clean enough for foundation supervision, and visibly different in scale, color, and placement. The strongest paint-handling examples are ribbons; cups, boxes, cylinders, vases, and houses show clear connectivity; buses show a readable body, aligned windows, and attached wheels; flowers show petals meeting a central disk with a leaf and stem; sailboats show two sails attached to a mast over a complete hull; still lifes show a fruit beside an open vessel; trees show canopy masses connected through branches to a trunk; spheres show a stable silhouette, volume, and attached shadow.

These are foundation targets, not aspirational aesthetic gold. The recurring sawtooth brush edge, pale washes, and limited dimensionality remain visible. The manifest keeps those limitations explicit rather than converting the images into an unsupported quality claim.

## Roles and exclusions

The `turn-3.js` program and its rendered `turn-3.png` are a verified procedural teacher target. Because the target is generated from a known program, the target image may equal the reference image; that is valid for SFT. The manifest separately marks `reference_as_anchor_allowed: false` and `reference_as_opponent_allowed: false`. A future RL task must provide a distinct reviewed student or correction canvas as its opponent. Feeding the exact reference back as the anchor would create a trivial ceiling and would not test improvement.

The reserved `-008` rows are held out by this candidate manifest and were not admitted as train rows. They are pretraining/package-exposed assets, not novel or unseen generalization tests: all twelve `-008` groups and their turn assets appear in the `revision-rl-20260918/async-restart-20260920.tar.manifest.json` packaging manifest, and all twelve are explicitly marked `split=train` in `painter/data-v2/manifest.jsonl`. They are therefore reserved-next-run teacher-development rows only; do not report them as clean held-out checkpoint evaluation. The 000/001 rows were already present in the earlier RL state inventory, with only box-000 and bus-000 actually admitted in the completed aesthetic run. The manifest’s `exposure_audit` records this distinction. All prior evaluation groups `v2-*-007` and the listed `v3-*` groups are excluded.

Eight deliberate repair examples are now included. They use train groups only, never `-007`, `-008`, or `v3` groups, and pair an existing teacher intermediate with its existing complete turn-3 target. Four are structural completion cases (cup body/rim, cylinder rim, house fixtures, and still-life fruit); four are finishing-detail cases on an already legible object (bus lower contour/front window, flower seed texture/final center detail, ribbon contours, and sailboat hull/water lines). Each input and target has its own receipt and hash; no malformed student output was promoted to a teacher target.

## Coverage gap

The pool is entirely synthetic procedural painted targets. In the inspected Astra manual collection mirror at `outputs/brush-rl/runs/astra-manual-expansion-20260914/private-collection-backup/collection-mirror/`, there are 43 source-metadata JSON files and 90 API-cache JSON files but zero PNG, JPEG, or WebP image files. That mirror is therefore an inventory lead, not usable training data; other workspace locations were not exhaustively searched. Real-photo reference-to-painting work remains a later rung and should not be implied by this candidate pool.

The next run should use this pool to restore family-balanced structure and paint handling, keep the simple aesthetic reward, and use the eight inspected repairs as a small, explicit repair mix. The model’s prompt must preserve the coordinate and brush rules: the prior RL evidence shows that shortening that contract was a likely contributing factor to malformed placement and off-canvas output; this was not a controlled causal ablation.
