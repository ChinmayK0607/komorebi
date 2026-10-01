# Live RL16 judging audit — 2026-10-01

## Question and scope

Does noisy or visually inaccurate judging explain the fluctuating training reward? Audit saved rollouts and exact frozen-pair repeats without changing the active training objective or renting hardware. Source/runtime/model/data identities remain in [EXPERIMENT.md](EXPERIMENT.md). This audit is exploratory diagnosis, not a matched checkpoint evaluation or a judge accuracy estimate.

Snapshot contains364 completed rollout traces,149 original judge receipts, and the metric rows available at collection. It includes speculative/excluded rollouts and arrival buckets; these are not the16 admitted examples of each optimizer update. The screened input archive SHA is `2017bea26d34c3dee12e43c79d6ddf33b5da1c6a93050fac440691ac69e8c8c4`. Trace arrival includes work through12 while fewer optimizer updates have completed. Avoid deriving an admitted-batch curve from arrival or dispatch labels.

## Observations

-215/364 completed traces have a model-invalid final action,140 a scored final canvas,9 an unresolved judge. Of215 invalid finals,98 retain an earlier valid canvas and117 never produce a valid canvas. Invalid-final categories:189 program failures,25 render timeouts,1 blank canvas. A closer classification finds82 syntax errors,27 undefined identifiers,12 unsupported functions,5 invalid array geometry,6 malformed geometry/types and57 other program failures. These genuine zero rewards are not aesthetic judge decisions.
-17/149 original pairs have conflicting initial winners after image-order reversal. Among132 initially agreeing winner pairs,59 disagree on preference strength. The scorer chooses the weakest agreeing magnitude; this is specified behavior, not an arithmetic sign reversal. Unresolved results are censored rather than replaced by zero/tie.
-The140 scored canvases average0.296429 against the frozen SFT opponent. Their reward bins are17 clear losses(0),52 moderate losses(1/6),47 slight losses(1/3),17 slight wins(2/3),7 moderate wins(5/6),zero ties. This conditional mean is selected by renderability/judge availability and is not an overall visual-quality result.
-Root visually reviewed15 deliberately selected pairs: six distinct moderate-win subjects, six original winner-conflict cases and three strong losses. Most moderate-win directions are plausible against weaker anchors; rugby-ball silhouette and brass-bell rim/clapper improvements are concrete. But the lantern lacks its requested carrying handle, and the horseshoe lacks readable nail holes, so the strength of those wins is less convincing than their reward alone suggests. Root prefers the smoother connected SFT feather despite the RL white band and detached barbs; this is a medium-confidence fidelity/aesthetic tradeoff. The failed horseshoe arch has incorrect opening/orientation and construction lines, yet some judge votes call it a coherent horseshoe. These are selected root judgments with roles/scores known, not blinded human labels.

## Controlled repeats

Repeat six selected frozen pairs with exactly the same MiMo Flash model ID, prompt and comparator implementation, preserving both image orders and conflict rechecks. All six original/repeat image-ID and SHA pairs match. Same prompt SHA `ee7c5feb68fb4d87fce2570d6d6c05844576dd0990651d322dbe94e11338723c`, side-judge source SHA `f94cfc9e880c8423f89a091486a0cf174cf1818c13d9acdf18fd21ace79aa069`, model `xiaomi/mimo-v2.6-flash` via Vercel, temperature0. Provider backend weights/version are not immutable.

| Pair | Original reward | Repeat reward | Finding |
|---|---:|---:|---|
| Feather2245f3 |0.667|0.333|Opposite winner on identical paintings|
| Lantern e1bafc |0.833|0.667|Same winner, magnitude changes|
| Metronome cc22f8 |uncertain|uncertain|Still unresolved|
| Duck ce34e3 |uncertain|0.167|Previously unresolved now favors SFT|
| Lamp767433 |uncertain|0.667|Previously unresolved now favors RL|
| Horseshoe bcefb2 |uncertain|0.667|Previously unresolved now favors the structurally suspect RL arch|

There is one opposite-winner flip among the two pairs originally scored, one magnitude-only change, three uncertain-to-scored transitions and one still-uncertain pair. Do not conflate availability transitions with opposite-winner flips or extrapolate rates from six selected hard cases. Six repeats used16 recorded provider calls for$0.0040422312 reported Gateway cost; failed/unreported billing is not independently reconciled. No new painting generation/rendering or GPU rental. Joint review sheets were assembled from existing PNGs on the Linux node.

## Interpretation and next action

Judge imperfections are real: same-image decisions drift, and visual reasoning sometimes rewards a named feature or bold mass despite weaker actual form. The implementation's image identities, score direction and received reward agreement show no demonstrated role/sign swap in this snapshot. Winner consistency across two orders reduces bias but does not prove visual correctness.

Small changing batches, correlated groups and the large invalid-final fraction independently explain substantial reward variation. Changing three slight losses to invalid finals changes a16-sample mean by0.0625; changing three slight losses to slight wins changes it by0.0625. A short reward sequence therefore cannot establish learning or judge failure by itself.

Keep the active run's reward unchanged, retain its registered finite stopping rule, and archive its evidence. Before another run, separate final-action validity, conditional aesthetic preference and a fixed matched validation panel in monitoring. Judge refinements should prioritize the coherent whole, connected parts, actual negative spaces and requested orientation before small detail matching, preserving beauty and warmth. Verify proposed changes on frozen images including unseen comparisons rather than fitting these examples or importing root opinions as training labels. Do not blindly switch to binary reward: with many invalid finals and few current wins it would make the group-relative signal substantially sparser. The fragility of revisions requires a separate sampler/termination/credit experiment, not merely another prompt rewrite.

## Evidence

[Side-by-side audit](local-evidence/judge-audit/audit-report.html), [machine summary](local-evidence/judge-audit/audit-report.json), [root visual observations](local-evidence/judge-audit/root-visual-review.json), original receipts in `local-evidence/judge-audit/rl-rollouts`, repeat receipts in `local-evidence/judge-audit/repeat`. Offline auditor `audit_judge_receipts.py` reads saved metadata only; it neither renders nor calls providers. Public preservation and reviewed source commit will be recorded after completion.


Public evidence: [177-file frozen-rollout judging audit](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/4dcea0eb56ce32b3993e6ae181f574cebcb8f701/runs/text-rl16-retry-20261001/judge-audit-v1.tar.gz),9,727,372 bytes, SHA `5d6460a519e03ec9f247e054ec88090ff4bb0abac6c284bc38b80f0348c63a7e`. Anonymous download hash verification passed; archived final report bytes match the reviewed local reports. The archive includes original149 receipts, six repeats, sanitized completed-trace metadata, metrics, root visual observations and15 existing-render joint sheets. Three focused offline-auditor tests pass, including incompatible consensus detection and repeat-denominator isolation. The snapshot has zero detected score-arithmetic, vote-consensus-selection, image-ID/hash-metadata or trace/receipt score mismatches. These checks cannot certify visual correctness.

At09:18 UTC the live run remains active with12 completed optimizer updates; effective rewards10–12 are0.09375,0.0625,0.15625. No consistent upward learning result or matched validation gain. Clock-based GPU spend$6.27 retry/$7.71 including prior failed attempt, Gateway separate. This audit does not alter the registered stopping check at15 or the$20 combined rental cap.
