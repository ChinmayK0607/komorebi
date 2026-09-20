# Contract-quality pilot — interim results

Training and candidate evaluation are still in progress. The selected recovery parent is unchanged.

## Complete Luna baseline review

All128 baseline episodes were judged from blinded reference/A/B images. Reviewed image hashes were validated. This compares each episode’s first valid (or supplied initial) canvas against its final canvas; it does not measure improvement from training.

| Task | Episodes | Final better | Tie | Final worse | Uncertain |
|---|---:|---:|---:|---:|---:|
| Painting |96|41|24|24|7|
| Repair |16|9|3|4|0|
| Finish/preservation |16|4|8|4|0|

Luna marked53/96 final paintings acceptable, compared with42/96 initial paintings. These are agent judgments, not human ground truth. Paint reference-clustered preference is0.592, bootstrap95% interval0.505–0.677. The benchmark covers known procedural families with held-out compositions; it does not establish photo generalization.

## Training and remaining review

The first24-update checkpoint is public and remotely hash-verified. Actual optimizer audits through update29 show finite updates to all992 adapter tensors. These are training integrity checks, not visual quality results.

Evaluate step24 and step72 under the same128-episode contract, then have Luna review every candidate revision pair plus all384 pairwise checkpoint comparisons. Painting preference remains the primary outcome; repair and finish probes are reported separately. Keep uncertainty visible. Review runs independently of training, and the watcher retains the full review scope.

Verified the actual training initializer:992/992 tensors exactly match the selected recovery checkpoint before the first forward pass, with a fresh optimizer. Training/serving prefixes, image grids and pixel tensors match for zero-, one- and two-image cases. Training and evaluation source groups have no overlap.

## Training complete

All72 contiguous optimizer updates completed successfully. Every update was finite and changed all992 adapter tensors; all per-tensor gradients were finite. Checkpoints24/48/72 are public and remotely hash-verified. See training-completion-verification.json for receipts. Matched candidate evaluation began at16:31:56UTC; checkpoint24 is running first. No candidate visual result is available yet.

## Evaluations complete; direct comparison pending

Parent, step24 and step72 each completed all128 episodes and produced128 canvases. Finish-within-limit counts are76,40,27 respectively. Candidate stopping behavior is worse; canvas validity alone does not establish quality.

Luna has now completed128/128 step24 initial-to-final reviews. Its96 painting episodes yielded39 improvements,38ties,16regressions and3uncertain/low-confidence cases;72/96 final canvases were marked acceptable. The parent revision review marked53/96 final paintings acceptable. These independently judged absolute scores are provisional; direct identity-blinded matched checkpoint comparisons remain the basis for choosing a model.

The final checkpoint’s128 revision pairs are collected with archive/member hash verification and Luna review is underway. No checkpoint has been promoted.

## Evidence preserved and GPU released

The complete archive contains15,125 files, all hash-verified locally, with no weights. The owned H200 pod was released and Lium confirmed it no longer exists. All384 blinded cross-checkpoint pairs are ready; Luna judging continues without GPU rental.

Painting-only behavior shows a stopping/repetition problem. Parent/step24/step72 median paint-turn lengths are668/644/651 tokens; mean episode tokens are4393/4790/5191. Finished painting episodes fall47/19/5 out of96. Unchanged successful paint submissions rise345/485/517 across those96 episodes. Output length is roughly stable per painting turn, while the candidates repeat more and spend more total tokens. Full results are in collected-final/evidence/behavior-summary.json. This observation does not replace the pending visual comparison.

The gallery is collected-final/evidence/comparison/index.html. Final selection remains pending complete Luna pairwise results.

## Current diagnosis

Visual inspection of sample0 for four new-composition references confirms real gains and remaining errors: both candidates reconstruct the bus-and-boat scene absent from the parent; final improves house structure but changes size/placement; midpoint still-life retains cylinder structure better than final; flower petal geometry remains inaccurate. This small inspection does not substitute for blinded full coverage.

A likely training-data mismatch explains stopping:300/300 finish-pool examples present the exact reference as the current canvas, whereas student-like states are taught a full teacher-program rewrite. Final passes all16 finish probes yet stops on only5/96 painting episodes. Finish examples are10.07%of sampled rows but0.285%of supervised target tokens; the token share alone is not a measurement of gradient contribution. Data should teach acceptance and targeted repair on realistic student canvases, with visual labels deciding which behavior is correct.
