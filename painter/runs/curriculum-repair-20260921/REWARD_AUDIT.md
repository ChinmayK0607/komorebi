# Reward audit for the admitted trajectories

This is a bounded forensic audit of the 32 admitted training trajectories from [aesthetic-rl-20260921](../aesthetic-rl-20260921/RESULTS.md). It joins the four handoff `score_observations` files to the original receipt files and the preserved native trace stream. It does not relabel any receipt. Visual conclusions below are from selected representatives, not a complete 32-trajectory image review.

## Decision-relevant result

There was some real execution signal, but the admitted reward is not reliable evidence of useful terminal painting or self-correction. At the renderer level, 19/32 traces produced at least one valid changed render. The remaining 13/32 never produced a valid changed render; 9/32 were invalid on every observed turn and 6/32 contained at least one `no_change` turn. Only a small subset was visually inspected. The clearest positive example is step 1 trace `e13c8de40a7e4ffcaaf978ec1ae10fd2`: it turns the inverted box anchor into an upright box, makes a bad gray smear on turn 2, and recovers to a clean box on turn 3. This is evidence that self-correction occurred in at least one trajectory, not evidence that the reward trained that behavior reliably.

The main failure is grounding. The bus's admitted observed canvas is already a lower-right clipped fragment: [observed bus](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/observed/painting.png). The archived teacher is a centered complete bus: [archived teacher](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/archived-teacher/painting.png). In the first generated turn of the 16 admitted bus traces, 11 programs put `translate(-300,-300)` in `setup()`, one put `translate(0,0)` in `setup()`, four had no translate, and none moved the nonzero translate into `draw()`. This count comes from `traces[*].info.turns[0].program` in the preserved `stream/00000.jsonl.zst`, not from the task's shared initial state program. A renderer-valid turn therefore often meant a new off-canvas image, not a repaired canvas.

## Trace and receipt accounting

| admitted group | rows | used-receipt reward rows | valid changed traces | any `no_change` trace | all turns invalid | explicit finish |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| step 1, box | 8 | 1×4, 0.5×3, 0×1 | 5/8 | 1/8 | 2/8 | 3/8 |
| step 2, box | 8 | 1×3, 0.5×4, 0×1 | 6/8 | 0/8 | 2/8 | 2/8 |
| step 3, bus | 8 | 0.75×1, 0.5×6, 1×1 | 2/8 | 3/8 | 3/8 | 0/8 |
| step 4, bus | 8 | 0.5×2, 0×6 | 6/8 | 2/8 | 2/8 | 0/8 |
| **total** | **32** | — | **19/32** | **6/32** | **9/32** | **5/32** |

Across the traces there are 86 observed turns: 39 valid changed renders, 8 valid `no_change` turns, and 39 invalid turns. The handoffs contain 32 trace IDs but only 23 unique used request IDs. Reuse is material: request `fa6f2d8f0417de99f4040e998360d656b3bed6e83c79e9b53934f05775d32338` is attached to three step-1 traces; `bdc53bba04268b5b5ab8ce6392f885a239328bc4a4d8432e946aa47f610abd45` to two step-2 traces; `51852a8cb3c9a8536d0596e7fe7f10bb4dc222866d9b0d49e0bb87fb43174158` to six step-3 traces; and `666be7f579afa532a7170940aea459a2ea2532388347c2265a5aa6b2cd5a1587` to two step-4 traces. Each repeated request is an exact content-addressed cache hit with identical reference, anchor, and candidate hashes, so the reuse is valid caching. It does mean that receipt-level judgment counts and trajectory-level execution counts answer different questions. The used step-3 receipt rows have arithmetic mean 0.59375, while `RESULTS.md` reports the optimizer group reward as 0.75; the retained files do not expose enough of the native reward aggregation denominator to reconcile those statistics. Keep the native group statistic and report the row distribution separately; do not infer a training failure from this denominator difference.

## Order disagreement in used receipts

Reading the two orientation outcomes in the original used receipts gives 8/32 admitted occurrences with an orientation disagreement, counting seven strict win/loss disagreements and one tie-versus-win case. They are 3/8 in step 1, 2/8 in step 2, 1/8 in step 3, and 2/8 in step 4. At the unique-receipt level, four of 23 used receipts are strict disagreements and one more is tie-versus-win.

The exact used request IDs are:

- `fa6f2d8f0417de99f4040e998360d656b3bed6e83c79e9b53934f05775d32338` (three step-1 rows; receipt in `reward-batches/live-002`)
- `9b47ae1535631f6487722e028bef3bb88a544b3e377cf27accfdf5bdea6faef4` (step 2)
- `2c7cd2928cff8c8a058be007c1596ca1393e78486ec73b4623cdd6fc9cfbb35b` (step 2)
- `ff637df50b0c5fd0b3f94943021ad3d18bde0b39d92df9ed6a31c196668893a1` (step 3; tie versus win)
- `666be7f579afa532a7170940aea459a2ea2532388347c2265a5aa6b2cd5a1587` (two step-4 rows)

The last receipt is a particularly clear grounding failure. Both step-4 traces (`6458a7e5a91f4beb9794d0b9cc759385` and `d703e2ed3ab24e9cb11a7367f9177847`) have three invalid program turns, so their final judged candidate is the unchanged starting canvas by design. The candidate hash is `d09fad…`, the clipped observed bus; the anchor hash is `c2c588…`, the centered archived-teacher bus. Its first-orientation explanation incorrectly says both images are full buses and describes the candidate as the stronger full image; the reverse orientation correctly says the candidate is the clipped lower-right fragment. The `.5` is therefore a split pairwise judgment of the unchanged starting canvas, not evidence that either invalid trace produced a useful new render. The receipt is preserved at [666be7.../receipt.json](../aesthetic-rl-20260921/reward-batches/live-007/receipts/666be7f579afa532a7170940aea459a2ea2532388347c2265a5aa6b2cd5a1587/receipt.json).

## Selected visual examples

These examples were opened with the native image viewer. They are representative evidence only.

| trace / receipt | raw outcome | observed image evidence | interpretation |
| --- | --- | --- | --- |
| step 1 `e13c8de40a7e4ffcaaf978ec1ae10fd2` / `3f1af9ddd4e899d0a3fd8feb6ca1d3a76e30f9e457718a68e56180ea01aec003` | reward 1 | [turn 1](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-00/55552d15-c133-48fe-82fb-e5e88ffc9fc0/turn-1/painting.png), [turn 2](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-00/55552d15-c133-48fe-82fb-e5e88ffc9fc0/turn-2/painting.png), [turn 3](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-00/55552d15-c133-48fe-82fb-e5e88ffc9fc0/turn-3/painting.png) | Clear positive: upright box, then a large gray smear, then recovery to a clean box. This is the strongest direct evidence of actual revision in the selected sample. |
| step 1 `9dabbbfba1f942268b8813db5bb508db` / `383f1ba7a7afae34af82c9227a0ff224b3a54e641a5208262ba0085d5bbc1d3e` | reward 0 | [turn 1](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-00/1303c949-8df7-41e6-a83c-607033bdfb2c/turn-1/painting.png) | The candidate is a valid upright warm/cool box and is more recognizable than the inverted [starting anchor](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-00/final/painting.png), while the receipt says the anchor wins both orientations. Preserve this as a label-grounding concern, not as a relabel. |
| step 2 `7db90fa12094433fbb56a0ae6c6dc2fc` / `d245e6714c663c8a298ed7cc9830f382f0a0cbb22218be711b215ef984d9fd71` | reward 1 | [turn 1](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-00/bdca90cb-2301-4133-8774-b3c8d3900b2b/turn-1/painting.png), [turn 3](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-00/bdca90cb-2301-4133-8774-b3c8d3900b2b/turn-3/painting.png) | Valid changed renders produce a stable centered box over turns. This is useful execution, though it is still one selected example. |
| step 3 `00392a6f0d3e4734831aa9e8cb7a543e` / `ff637df50b0c5fd0b3f94943021ad3d18bde0b39d92df9ed6a31c196668893a1` | reward 0.75 | [turn 1](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-02/e1003af4-5693-470c-baeb-3c08f77014bf/turn-1/painting.png), [turn 2](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-02/e1003af4-5693-470c-baeb-3c08f77014bf/turn-2/painting.png), [turn 3](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-02/e1003af4-5693-470c-baeb-3c08f77014bf/turn-3/painting.png) | All three renders are valid but remain a tiny lower-right bus fragment. The high step-3 reward is therefore not evidence of repairing the starting canvas. |
| step 3 `6a81ec0dd91146f2b815e8ad687421e6` and five peers / `51852a8c3c9a8536d0596e7fe7f10bb4dc222866d9b0d49e0bb87fb43174158` | reward 0.5 each | [shared final canvas](../aesthetic-rl-20260921/collected-final/rl-states/state-02/canvas.png) and [observed starting canvas](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/observed/painting.png) | The six traces reuse a tie/tie receipt. Three are all-invalid; three contain no-change turns followed by invalid turns. The image path is shared fallback state, so it is evidence of the state/receipt linkage, not a claim that one PNG uniquely represents all six traces. |
| step 4 `6458a7e5a91f4beb9794d0b9cc759385` and `d703e2ed3ab24e9cb11a7367f9177847` / `666be7f579afa532a7170940aea459a2ea2532388347c2265a5aa6b2cd5a1587` | reward 0.5 each | [observed clipped canvas](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/observed/painting.png), [archived teacher](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/archived-teacher/painting.png); both traces show three `failure_kind=program` turns in the native trace | The unchanged clipped starting canvas receives a split `.5` pairwise judgment. The first orientation misdescribes its candidate as full; the reverse orientation correctly identifies the clipped candidate. This is a judge-grounding error, while scoring the unchanged canvas after an invalid action is expected behavior. |
| step 4 `c53f7836ba3449b1a02a5bb7e41323eb` / `b74e49ec04c73242fa68bdc7e12f9a02ca58641632f298cc72bacaa1c6573cae` | reward 0 | [turn 1](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-02/849835e9-27d9-43eb-a635-d11c44393a5d/turn-1/painting.png), [turn 3](../aesthetic-rl-20260921/collected-final/rl-rollouts/state-02/849835e9-27d9-43eb-a635-d11c44393a5d/turn-3/painting.png) | Valid changed renders remain clipped at the lower-right; turn 2 is `no_change`. This is a useful negative control: renderer validity did not fix grounding. |

## Translate placement and repair

The first-generated-turn program count is:

| admitted task (first generated turn) | `translate(-300,-300)` in setup | `translate(-300,-300)` in draw | `translate(0,0)` in setup | no translate |
| --- | ---: | ---: | ---: | ---: |
| box, steps 1–2 | 0/16 | 12/16 | 0/16 | 4/16 |
| bus, steps 3–4 | 11/16 | 0/16 | 1/16 | 4/16 |

No first-generated bus program moves the observed off-canvas translate into `draw()`; 11 retain it in `setup()`, while four are no-translate variants and one uses `translate(0,0)`. The task's state-02 initial program is shared across the 16 bus traces; this table is about the first generated action program. The valid off-canvas examples above show why renderer validity cannot stand in for useful terminal painting. The bus error is visible in the observed program itself: [observed `program.js`](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/observed/program.js) translates in `setup()`, whereas the complete teacher uses [teacher `program.js`](../aesthetic-rl-20260921/collected-final/aesthetic-anchor-pool/state-02/archived-teacher/program.js) with the translate in `draw()`.

## Next curriculum decision

1. Keep the simple pairwise beauty reward. Record each trace-to-request cache key, candidate/anchor hashes, both orientation outcomes, and terminal canvas so cached receipt-level judgments can be separated from trajectory-level action evidence.
2. Repair the coordinate contract in the prompt or starting-program curriculum before another judged run. The next bus examples should demonstrate the teacher's `draw()` translation and compare complete versus clipped canvases directly; judge poor paintings as ordinary losses under the same beauty preference.
3. Use deliberate repair tasks next: inverted box → centered box and clipped bus → centered bus, with a matched same-reference opponent. Inspect a small random sample before importing labels, without adding validity, finish, or occupancy reward terms.
4. Keep order labels and grounding failures as raw evidence. Use the disagreement rate to assess judge reliability and prompt grounding; do not silently relabel the source judgments.
5. Treat this run as evidence that a few trajectories can execute and revise, not evidence that the four update rewards measure useful aesthetic improvement. The visual audit here is selected and bounded.
