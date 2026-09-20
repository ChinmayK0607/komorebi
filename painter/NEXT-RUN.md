# Next work after the diversity comparison

Implementation update: the shared preview/coordinate contract and automatic snapshot check are now in [CONTRACT.md](CONTRACT.md). Paint/finish SFT targets and rollout behavior are updated in the root implementation. Historical run code remains unchanged. Thirty pure/stubbed tests passed; actual node compiler/render parity and matched-model evaluation are pending. Use this contract for the next run, rather than copying a historical run harness unchanged.

The diversity run is complete; read `runs/diversity-sft-20260918/RESULTS.md` and `FORENSICS.md`. Its GPU was released after all 26,617 final evidence files were hash-verified. Do not contact its endpoint, relaunch its pipeline, or duplicate completed Luna reviews. All four adapters are public. Neither evaluated new checkpoint should replace the recovery step72 parent recorded in `selected-checkpoint.json`.

The final checkpoint lost to the parent 32–13 in blinded Luna comparisons (2 ties, 1 uncertain). On new compositions it won only 4 of 24 pairs, losing 17, tying 2, with 1 uncertain. Lower NLL and fully valid final canvases did not track visual fidelity.

## Correct the demonstration contract

All 216 drawing/repair episodes finished immediately after their first successful painting. The additional turns repaired invalid execution; none inspected a successful render. Our positive demonstrations combine a complete program and FINISHED in one assistant response, and the rollout honors that declaration immediately.

Use separate paint → actual canvas observation → revise or finish decisions. Supervise corrections to concrete visual mistakes from student states, with the reference and current canvas both visible. A correct existing canvas should still finish without gratuitous repainting. A preview requirement changes the scaffold, so compare the retained parent and any trained candidate under that same updated contract. More allowed turns alone is not a fix.

## Isolate representation difficulty

The expanded corpus introduced per-object affine frames and local-coordinate geometry. Preserved examples show tiny objects caused by combining small local geometry with another scale transform; other cases also have identity/shape errors. Compile an alternative teacher representation into one explicit canvas-coordinate system, verify matching teacher renders on a node, and test it independently. This is an ablation to test, not a demonstrated solution.

Keep development references out of training. Use separate training references for new student-state corrections; do not consume the current development rollouts as SFT targets. Keep initializer, reference split and evaluation scaffold fixed when measuring a data change. Preserve complete targets and actual image/target binding checks.

## Subsequent learning and judging

Luna (`gpt-5.6-luna`) remains the requested visual judge. Preserve identities-hidden image reviews, uncertainty, absolute acceptability and category-specific errors. The 144-pair report covers painting reconstruction; repair quality still needs visual assessment. Agent review is not an already-integrated online RL reward endpoint.

Once the corrected interface demonstrates useful visual revisions, adapt the existing Prime rollout/advantage integration to the exact contract. Preserve sampled-token masks and inference log probabilities, use identical state/anchor groups, and verify actual optimizer updates. Keep speed rewards off while quality is unreliable. No additional training or GPU rental is currently queued.

## Avoid paying GPU rates for CPU rendering

Render and cache the next reference/corruption corpus on a CPU node before renting the training GPU. Keep all rendering, inference and training off the user's Mac. Transfer verified source/PNG hashes to the training node. Continue using public, remotely hash-verified HF checkpoints, unique IDs for repeated sample draws, and a per-model evaluation producer lock.
