# Controlled comparison after the scaffold repair

2026-09-21. Hypothesis: under one corrected paint–render–inspect interaction, the four-update RL adapter produces better reference-conditioned paintings than its exact step-24 SFT initializer. The comparison may also show that the RL adapter is worse or indistinguishable. No new training is part of this experiment.

The primary comparison is 12 existing v2 evaluation references (`-007`), one per family, selected by ID rather than previous checkpoint performance. Both policies receive the same reference, full canonical prompt, complete interaction history, paired seed 918200, temperature 1.0, thinking disabled, six model turns and at most 8,192 generated tokens per turn. The served context is 32,768 tokens; this does not guarantee six maximum-length replies. Context and generation cutoffs, failed renders, final-canvas availability and explicit finishes are recorded rather than hidden.

Generation uses one shared active native environment. Evaluation bypasses reward waiting; images are assessed afterward. Training will import that same environment rather than a separate run-folder copy. Historical implementations remain frozen evidence.

Both checkpoints use the same unquantized base model and server configuration. Loading both LoRAs into one inference server avoids separate serving implementations. No optimizer or trainer is needed for this comparison. Select hardware by sufficient memory and total rental price; collect utilization over the actual generation window and release the node after evidence preservation.

The primary output is a reference/SFT/RL gallery containing all 12 cases and links to every turn. Parent visual assessment records reference fidelity and aesthetic preference, including weaknesses of the chosen painting. Luna's pairwise assessments, if collected, are separate measurements with original labels retained; an erroneous judge explanation does not become proof of a visual win. No extra validity, token-count, speed or finish reward is introduced.

This is a small paired diagnostic with one sample per reference and previously used evaluation families. It is not a broad generalization estimate or a controlled estimate of each individual scaffold fix. A change from old screenshots can reflect the new interaction/budget; only the SFT-versus-RL comparison within this run isolates the adapter choice under the new conditions.

Next decision: retain the adapter supported by the matched visual evidence and use it as the fixed baseline for the reviewed foundation curriculum. If neither is convincing, record the specific remaining behavior and continue from the SFT checkpoint; do not promote an RL checkpoint merely because optimizer steps completed.

No scheduled task, heartbeat, automatic restart or recurring watcher is authorized by this protocol.
