# What went wrong, and the next curriculum

2026-09-21 — offline review of `aesthetic-rl-20260921`. This is preparation, not a new training result. No GPU rental, rendering, training, or scheduled task was performed for this review.

## Assessment

The last pilot established that native RL updates the intended adapter. It did **not** establish broad aesthetic improvement. The parent inspected all four evaluation cases: the bus became more readable, the box changed paint handling, and the flower became executable, but placement, proportions, petal count and vase connectivity still fail. Four updates on two references cannot answer whether the method learns general painting.

There are concrete implementation and experiment-design defects behind the weak evidence. These should be repaired before increasing compute or changing the objective. Their individual effect sizes are unknown; no causal ablation has run.

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Native RL dropped the canonical painting instructions. | Saved native trace contains the short `_QueueContract.system_prompt()`, omitting the draw-time coordinate transform and brush-mode rules. | Training and the established painting interface differ. This is a scaffold regression. |
| Eight of 24 initial programs place `translate(-300,-300)` before `draw()`, including both trained bus states. | Initial programs and SHA-256 values in `parent-evidence.json`. The teacher checker only counted transforms, without checking placement. | A malformed starting canvas teaches repair of our coordinate mistake rather than normal reference painting. |
| Only box-000 and bus-000 reached the four updates. | All 32 score observations joined to native trace IDs. Each state’s anchors were enumerated before advancing to later source groups. | A 12-family inventory was mistaken for actual training coverage. |
| Exact repetitions did not receive the canonical no-change message. | Renderer emits `no_change`; historical native Env sends generic feedback instead. | The model is not explicitly told when its purported revision changed nothing. |
| Training and evaluation used different interaction histories and starting modes. | Native training preserves history and starts from an existing painting; the earlier evaluation starts fresh and reconstructs each turn. | Baseline/candidate evaluation is matched internally, but does not measure the practiced native interaction. |
| Pairwise labels disagree across image orderings for 8/32 admitted episodes. | Receipt join summarized in `parent-evidence.json`; reward audit retains original labels. | Averaging gives a number, but does not establish that the preference is visually grounded. |

[p5 documentation](https://p5js.org/reference/p5/translate/) explicitly states that transformations reset at the beginning of the draw loop. The setup/draw placement issue is therefore substantive. It does not prove that every bad canvas has this cause.

The [admitted-rollout audit](REWARD_AUDIT.md) finds 19/32 trajectories with any valid changed render and nine with every attempted render invalid. Because correction tasks retain the starting canvas, those nine can still receive a terminal-image score; that is not itself a reward implementation error. Likewise, caching an identical reference/opponent/candidate comparison is valid, although repeated receipts are not independent visual judgments.

There is a **confirmed judge error**, beyond subjective disagreement. The parent displayed the exact two hashed images for receipt `666be7f579afa532a7170940aea459a2ea2532388347c2265a5aa6b2cd5a1587`: candidate `d09fad…` is a lower-right bus fragment, opponent `c2c588…` is a full centered bus. The first orientation incorrectly says both are full buses and prefers the fragment; the reverse correctly prefers the full bus. Averaging produced 0.5 for two admitted trajectories. The solution is grounded image comparison, not a new validity/occupancy reward or a rule excluding every bad painting. A poor painting should remain available to learn from and receive the appropriate visual preference.

## Quality review of the proposed teaching pool

Luna-high inspected 36 existing procedural teacher images. The parent independently displayed one proposed training target in every family: flower-000, house-000, and the `-001` targets for box, bus, cup, cylinder, ribbons, sailboat, sphere, still life, tree and vase. These source files are under `recovery-quality-20260918/collected-final/evidence/data/`, with exact paths and hashes in the candidate manifest.

**Accept these for foundation supervision; do not label them aesthetic gold.** They have legible, connected objects and useful variation in placement, color and scale. The cup handle attaches, the bus windows/wheels belong to the body, the vase neck connects, and the flower petals meet its center. Ribbons offer useful curved paint handling. Weaknesses remain: repetitive jagged wash edges, pale low-contrast shapes, simplistic perspective, translucent bus wheels, and limited scene complexity. Photo-reference generalization is not covered.

The examples are curated existing assets, not newly generated demonstrations or 24 newly unseen references. A new manifest does not erase their exposure in previous SFT. Evaluation claims must distinguish references reserved from the next run from references never used to train the checkpoint.

## Next experiment direction

1. Use the canonical prompt and the same native paint → render → inspect interaction for training and evaluation. Keep student mistakes visible; do not silently repair their code.
2. Start with 24 fresh-start tasks across all 12 families and eight explicitly reviewed incomplete-painting repairs. Fresh-start means no current program/canvas in the prompt; it does not mean an unseen reference.
3. Order groups by family and reference so the first 12 fresh groups cover all 12 families. Measure the admitted groups, not merely the manifest. Rotate repair families across later cycles rather than permanently overweighting eight families.
4. Retain one holistic final-image preference: which painting is more beautiful and more faithful to the reference? Keep weaknesses for both sides in the explanation. No extra score for tokens, valid syntax, finishing, or speed. Retain raw order-specific judgments and audit visually unsupported claims separately from ordinary aesthetic disagreement.
5. Use a fixed baseline painting of the same reference as the fresh-task opponent. Never substitute the exact target image as an RL opponent. Repair-task opponents can be the inspected incomplete starting canvases. Generate missing baseline canvases on the Linux node using the same contract, interaction and limits as the candidate.
6. Compare the exact step-24 SFT initializer and step-4 pilot under this corrected interaction before claiming the pilot is a better initializer. Carry forward the policy supported by that comparison. A modest 24-reference coverage cycle is more informative than another four updates repeating two references.

The proposed next rollout cap is six model turns, applied identically to train and evaluation, with the generation/context budget recorded explicitly at launch. This is an operational cap, not a speed curriculum. Budget pressure comes after reliable painting quality.

## Prepared and reviewed

The [reviewed candidate manifest](candidate-manifest.json) now contains 24 training references and eight repairs. The parent inspected every repair input/target pair: four complete missing structure or objects (cup, cylinder, house, still life); four add finishing details (bus, flower, ribbons, sailboat). The flower already has a central disk in its intermediate canvas; its missing feature is seed detail. All 156 listed artifact hashes verify, and all 44 distinct teacher programs pass the coordinate and draw-placement authoring checks. The [repair gallery](repair-review.html) shows the exact existing images.

The [concrete task plan](curriculum-plan.json) records 24 fresh tasks, eight correction tasks, six turns, source-manifest SHA-256 `d27bd1cb1e1529c6cecad6bc4a08dbef1ef0240cc55aef3c147a640af4d12282`, and contract `canvas-pixels-preview-v2`, identity SHA-256 `dd99279a5f3f6e98126e82db8b026c1ba726500cc8efb3fde53633772b5c65fb`. The first 12 fresh tasks cover all 12 families. This plan orders the fresh block before the repair block; it does not claim interleaved online sampling. Twelve `-008` targets are reserved from this next run only: all already appear as training rows in the earlier data manifest and are unsuitable for claims of unseen-reference transfer.

Luna-xhigh implemented the next-run scaffold, and the parent reviewed it. The full shared prompt is restored; fresh starts, actual canvas feedback, no-change feedback and recoverable premature finish handling use one native Env. The loader consumes the prepared order and checks the imported prompt against the pinned contract. The canonical contract version was bumped for the new no-canvas finish feedback, and its foundation lock updated; historical run sources and labels remain unchanged.

Parent verification passed **11 scaffold/data tests and seven shared contract tests**. These include mocked native state transitions and actual-manifest preparation. They do not constitute GPU, renderer, native dependency, or learning validation. The [scaffold audit](SCAFFOLD_AUDIT.md) lists the Linux runtime dependencies.

The next operational work is to materialize the distinct baseline opponent canvases for the 24 fresh tasks on the Linux node and run the corrected interaction there. The eight repair opponents are already present. Fresh-task opponents are deliberately missing from this plan rather than filled with the exact reference image. Judge grounding remains an observed reliability problem; restoring the scaffold has not demonstrated a judge fix. Use the retained full-versus-clipped example as a concrete diagnostic during the next comparison work, while preserving the same holistic visual objective.

## Evidence and operational scope

The immutable model, adapter, optimizer and checkpoint identifiers for the audited pilot remain in [its experiment record](../aesthetic-rl-20260921/EXPERIMENT.json) and [results](../aesthetic-rl-20260921/RESULTS.md). This assessment does not overwrite its raw rewards or retrospective failures. `parent-evidence.json` records selected prompt, program and trace facts. The prior pilot’s trainer averaged 14.06% GPU utilization over 148 samples spanning approximately 24.5 minutes, versus 73.77% on inference; the pipeline was judge-limited, not saturated.

This offline preparation has zero additional GPU rental cost. New runtime validation and missing baseline rollouts still require a Linux node. No new checkpoint or quality improvement is claimed here.
