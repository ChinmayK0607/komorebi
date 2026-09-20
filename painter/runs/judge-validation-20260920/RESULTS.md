# Luna low image-judging diagnostic — 2026-09-20

The tested judging path failed. It must not supply painting RL rewards. Displaying each image in a separate tool call did not resolve the pairwise failure. This result concerns this agent/tool workflow and these images; it does not establish the root cause or a general inability of the model to judge art.

| Test | Scope | Result |
|---|---|---|
| One triplet per call, three fresh Luna-low agents | 12 underlying pairs; 24 blinded views | All labels complete, file/hash binding passed; only 4/12 known-control views correct |
| Same first test, real-pair preference consistency | 6 real pairs in opposite orientations | 1/4 nonmasked comparisons agreed; 2 comparisons masked |
| Same first test, real-pair acceptability consistency | 6 real pairs | 4/5 nonmasked comparisons agreed; 1 masked |
| Isolated single-image descriptions, fresh Luna-low agent | 6 images: repeated cube/flower plus sailboat | All six subject/nonblank descriptions consistent with parent image review; no blank image in this probe |
| Separate calls for reference/A/B, two further fresh Luna-low agents | Same 6 controls in both orientations; 12 views | Only 5/12 correct; failed |

The six controls cover two identical-candidate pairs, three exact-reference versus wrong-subject pairs, and one exact-reference versus blank pair. Failure includes describing one of two byte-identical cube candidates as blank, choosing a side instead of tying identical flowers, and describing an exact reference match as blank. The copied cube files have SHA-256 `84747d4af33a6f3b84f3c003714cacc3a6c1c09c555651359deaaf6527272824`. Parent directly viewed the allegedly blank cube: it contains a peach/gray block and pale oval shadow.

The false blanks often involve repeated image bytes in a comparison. Image presentation, duplicate handling, image-role association and model behavior remain competing explanations. Tool calls and agents claiming every panel was visible do not distinguish those causes. The isolated-image result is encouraging but cannot rehabilitate the failed pairwise judgments. No results were overwritten and no uncertain judgments were converted into ties.

The comparison uses matched images but fresh agents and different ordering; it is an exploratory diagnostic, not a randomized estimate of the effect of changing presentation. The real-pair test is small, contains severe cropping failures, and has no human artistic preference gold labels. Agreement is not accuracy.

## Decision

Do not resume reward-driven training with this judging path. The next bounded repair to investigate is an explicit content-identity layer: represent each unique image once and bind reference/A/B roles to that image, treating exact identity deterministically rather than asking the vision model to rediscover it. Such deterministic cases must be reported separately from genuinely visual judgments. Then test distinct-image real pairs with factual observations and independent review. This repair is proposed, not implemented or validated here. The async queue optimization remains independent and cannot make unreliable rewards useful.

## Evidence and scope

- [Original protocol and implementation](PROTOCOL.md); [pre-run hypothesis and follow-up plan](EXPERIMENT.md).
- [Visual audit gallery](report.html), with original and follow-up labels; [original validation](validation.json).
- Original labels: `judge-01-labels.json`, `judge-02-labels.json`, `judge-03-labels.json`.
- Follow-ups: `single-image-observations.json`, `separate-calls-labels.json`, `separate-calls-reversed-labels.json`.
- Exact input copies, source hashes, view orientations and assignments: `artifacts/private.json`; original selection: `selection.json`.
- [Evidence hashes](EVIDENCE.md). Generated images/raw labels stay outside Git; reviewed source, protocol and research records are committed.

Judge identity is `gpt-5.6-luna` with reasoning effort `low`; an immutable model backend revision is not exposed by the subagent interface. Luna xhigh implemented the harness; the parent reviewed the code, verified actual control images and ran the focused tests. Source baseline was `9e5c4ab`; implementation and documentation are recorded in this result's Git commit.

No model training, inference rollout generation, image rendering, node rental or GPU work occurred. Incremental GPU rental cost: $0. GPU utilization: not applicable (no allocated devices and no measurement window). Judge-agent token cost and reliable per-item latency were not collected; do not equate zero GPU spend with zero total cost. Original failed pilot remains closed, and its checkpoint remains explicitly invalid for quality claims.
