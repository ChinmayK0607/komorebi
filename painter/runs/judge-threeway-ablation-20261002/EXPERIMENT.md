# Three-judge frozen-image ablation — 2026-10-02

## Hypothesis and matched comparison (registered before calls)

The user enjoys selected old scaled RL16 and grounded RL24 paintings, but finds the latest Pro continuation broadly worse, with an exceptional observatory. Judge choice may change aesthetic supervision even with an identical rubric. Compare MiMo Flash, MiMo Pro, and GPT-6 Luna on the exact same saved image pairs and text briefs. This measures frozen-image judgments, not a causal training effect or a new checkpoint improvement.

Freeze the existing whole-form/beauty/warmth side-keyed v2 rubric: prompt SHA-256 `d21e4644e7a3874a0694b2ad406420e733b7638222eb7717e7f5f91cbdf1613c`. No prompt edits, code exposure, checkpoint labels or desired human answers in judge inputs. Exact model IDs: `xiaomi/mimo-v2.6-flash`, `xiaomi/mimo-v2.6-pro`, `openai/gpt-6-luna`. Gateway Chat Completions, explicit requested `reasoning.effort=medium`, equal 4,096 maximum output tokens, identical image sheet/strict JSON schema, omitted sampling controls on all models. Provider internal effort is not equal: Xiaomi documents non-none effort values as the same enabled thinking mode, while OpenAI differentiates effort. Record returned model identity and measured reasoning/output tokens. This assay changes reasoning configuration from old provider-default calls; it is not a reproduction of those defaults.

Select approximately 24 diverse frozen pairs, including user-highlighted successes and regressions, from the historical 47-brief held-out panel. Both image orders per model per pair, no adaptive model-specific rechecks in the primary assay. A fixed six-pair repeat subset measures repeat consistency separately. Strict parser or returned-model failure and order disagreement remain visible and censored; do not turn them into ties or rank a judge by how much it favors RL. All call receipts retained immediately; resume only matching manifest/config identities. Source, image, brief and request hashes are bound in the generated manifest before paid execution. Selected examples are diagnostic, not a random generalization sample; human alignment can only be reported for explicitly supplied pair labels, never inferred from broad praise.

## Checkpoints and evidence

- SFT512 opponent: `CK0607/qwen3.8-27b-brush-painting@adf687e5b321514eda7be6eab050434fdc8e44ff`, prefix `photo-curriculum-sft-20260922-balanced/step_512`, adapter SHA-256 `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
- Old scaled RL16: same model repository at `ac4d3eda8e5925e2d23992cfdc1a63620ebde573`, prefix `text-pairwise-scaled-20260929/step_16`, adapter SHA-256 `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`.
- Old grounded RL24: same model repository at `7d201165315595f2c611b355e6ba615f1067c75f`, prefix `text-grounded-pilot-20260930/step_24`, adapter SHA-256 `1397e30d2f2a802640c4aebec110eb657650b430d8591863e4831aa2a65aa5d4`.
- New Pro RL12: same model repository at `bff1ac729def86567d6d8f8573afde133041a7f6`, prefix `text-rl16-pro-20261002/step_12`. Its incomplete gallery displays last valid saved turns, not universally valid final actions; selected pairs must retain that caveat.

Old scaled RL16 used an admitted 68-brief anchored curriculum and original Flash margin-v1 rubric. Its expanded source pool remains public at `CK0607/komorebi-painter-teachers@ff7f9877379cb538557911ffe7496830ca9391c3`, `pools/text-paint-expanded-v3-20260929.tar.gz`, archive SHA-256 `482aa0d700ba42d5ae6b4dd8f09c4836863d38ccd24c4a287bbbb753c0c05c92`, manifest SHA-256 `712349be327ca2f5cc800aa42eb5529d0d9b9d54a3508a3aa3ed8c8c9c26b12d`. Grounded RL24 used 40 interleaved training briefs with 47 held-out briefs and started from RL16. These are different curricula and reward protocols; a same-curriculum training ablation would need one frozen recipe and fresh matched runs, not comparisons of their old aggregate reward curves.

## Historical token census

Recovered all 185 Pro continuation training comparison receipts from anonymously re-hash-verified public archive `CK0607/komorebi-painter-experiments@ae734053c5d994e4e9cfcc5629ff53fa9d055c8d`, `runs/text-rl16-pro-20261002/evidence.tar.gz`, SHA-256 `3a2c022ce7ffa627db5d6135e1dc9dd4ae70691eb5bdefdf245345258850db64`. Saved under ignored `../text-rl16-pro-20261002/local-evidence/judge-receipts/`; model weights and other raw archive contents were not copied.

169 scored /16 uncertain receipts, 466 HTTP calls, all reported MiMo Pro. Per call: mean completion 1,088.406, median 964.5, p90 1,722; mean prompt 1,470.682; mean reasoning 660.431. All 466 report usage and cost. Completion includes reasoning; visible output is not the billed total. Mean output per full receipt including rechecks is 2,741.605; reported cost totals $0.6810719448. Generated/speculative receipt scope, not exactly admitted trainer batches or the entire provider bill.

Prior same-eight-pair, two-repeat offline provider-default calibration: Flash49 HTTP calls mean completion1,488.7 and reasoning1,159.2; Pro49 calls mean completion1,385.4 and reasoning950.6. These are separate samples, not a controlled effort comparison with this new assay.

## Execution and decision

No GPU rental, training, rendering, automation or scheduled task. Use existing saved PNGs and finite paid API requests only. Keep generated images/receipts/report outside Git. Public HF evidence will be screened for credentials and hash verified after execution. The next decision is which judge settings warrant one controlled low-cost RL continuation from the preferred old checkpoint. Do not claim that a cheap judge or more RL wins alone proves better supervision. GPU ceiling for any future run remains $20 and the user's 15/20-update trend stop applies; no training launched here.

## Completed supporting assay

Reviewed source commit `f1c1a4d`; manifest content SHA-256 `2f5e19f828dd814d6f1371f4887d890ac6fb682f529792b42057087aa2ff15d2`, runner SHA-256 `153c9f53de89bfafaff0485a59ca71fa4e84ae6e2650472397a23e2cda18936a`. Completed180 calls, exactly60 per model, with24 selected pairs and six fixed repeated pairs. No GPU, rendering or training ran. All usage/cost fields were present; preserved at `local-evidence/results/summary.json` and individual call/round receipts. Fixed-side gallery at `local-evidence/report/index.html`.

| Model | Valid calls | Mean completion/call (includes reasoning) | Mean reasoning/call | Mean seconds/call | Reported cost |
|---|---:|---:|---:|---:|---:|
| GPT-6 Luna |60/60|795.85|495.27|9.78|$0.03444676|
| MiMo Flash |59/60|1229.98|865.50|26.73|$0.0304362688|
| MiMo Pro |54/60|1250.00|782.77|23.46|$0.0949957932|

Luna had25 agreeing/5 conflicting valid reversal rounds; Flash24 agreeing/5 conflicting/1 invalid; Pro13 agreeing/11 conflicting/6 invalid. Invalid outcomes remain censored, not ties. Flash's invalid call exhausted its4096-token output limit; Pro's invalid calls had stopped outputs with schema deviations (extra/missing/misspelled fields). Complete judgments were not silently repaired or repaid. Repeats are included and the selected24 pairs are not a random sample. Agreement and latency do not establish human alignment or causal training quality. Total reported API cost$0.159878822, distinct from GPU spend and the provider invoice.

The user's clarified next step is [three actual matched RL runs](../rl-judge-ablation-20261002/EXPERIMENT.md), not choosing a winner from this offline assay. No actual ablation arm has started yet.
