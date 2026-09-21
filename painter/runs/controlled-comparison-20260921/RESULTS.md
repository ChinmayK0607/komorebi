# Controlled SFT versus RL comparison

**Completed 2026-09-21. Do not promote the four-update RL adapter as an overall improvement.** Under the repaired same native interaction, Luna-low preferred SFT on 7 of 11 comparable pairs and RL on 4. The remaining reference had no valid SFT canvas. Parent inspection finds clear wins and regressions and disagrees with several close preferences; see [direct review](PARENT_REVIEW.md). One sample on 12 existing procedural references does not establish broad generalization or a statistically reliable ranking.

[Open all twelve comparisons and every turn](review/index.html). [Exact protocol](PROTOCOL.md), [immutable experiment identifiers](EXPERIMENT.json), [raw mapped judgments](receipt-visual-review.json).

| Measurement | SFT step 24 | RL step 4 |
|---|---:|---:|
| Evaluated episodes |12|12|
| Valid final canvas |11|12|
| Explicit finish |5|4|
| Six-turn cutoff |7|8|
| Total model turns |61|65|
| Paint attempts |56|61|
| Unchanged valid paint attempts |16|25|
| Invalid program/action attempts |9|6|
| Completion tokens |40,336|49,159|
| Per-request token-length cutoffs |0|0|
| Renderer-service or native-trace errors |0|0|

RL used **21.9% more completion tokens**, with more repeated unchanged paintings and no consistent visual gain. The 8,192-token response ceiling was not reached; 15 episodes hit the six-turn operational cap. No speed reward or length reward was used. No optimizer ran in this comparison, and the native evaluator's default zero reward is not a visual score.

## What this resolves

Both policies used the same verified Qwen3.8-27B base, their exact public LoRA revisions, the complete canonical drawing prompt, full native history, actual rendered canvas feedback, explicit no-change feedback, a paired seed, and identical six-turn/8192-token/32768-context limits. Actual node dependency loading, task loading and renderer smoke were validated before generation. This separates a matched adapter comparison from the earlier inconsistent scaffolds; it does not causally isolate the effect of each repair.

A concrete behavior remains: RL's bus prose promises revisions while returning the same program five times. SFT can sometimes use feedback effectively (its bus becomes fully visible), but it can also repeat a simple undefined-variable error or falsely declare a geometrically wrong painting finished. The next learning objective remains the final visual outcome under this shared interaction, with coverage across all 12 families. No additive score for finishing, syntax, token count or stroke count is proposed.

## Runtime, cost and preservation

The 24 episodes ran from 14:55:30.954 to15:15:46.369 UTC, **20m15s**. One RTX PRO 6000 Blackwell 96 GB inference GPU averaged **96.59% utilization** across **121 ten-second samples**, 14:55:44.567–15:15:44.821 UTC; 96.69% of those samples were at least 90%. This is inference utilization, with no trainer present. Cold setup and artifact preservation are outside that utilization window.

All 310 evidence members are hash-verified locally. Server shutdown appended log lines after the first preservation manifest; a 25,981-byte finalization patch reconciles only that log and its manifest. Both archive hashes and the reconciliation are retained. No model weights were copied locally. The owned node was released, and Lium reports no active pods. Rental is estimated at **at most about $1.30** through the 15:24 UTC confirmation at the advertised $1.29/hour, including setup; this is an elapsed-time estimate, not an invoice.

Setup failures (wrong Python environment for a test, missing shipped tests, and missing hardware helper) are recorded in the manifest. The reusable bundle now contains its explicit dependencies, and setup invokes the shipped contract suite rather than arbitrary stale tests. The current source was reviewed and committed as `e237b87`; the report and final results are recorded in the subsequent results commit. No scheduled task or heartbeat was created.

## Decision

Keep **SFT step 24 as the fixed baseline and initializer** for the next family-balanced RL curriculum: 24 reference-painting tasks plus 8 reviewed repairs, with the corrected shared interaction. The four-update/two-reference RL adapter remains a diagnostic checkpoint. Materialize distinct fixed baseline opponents for the fresh tasks; retain one holistic reference-conditioned beauty preference with concrete critiques of both candidates. Further training has not been launched by this comparison.
