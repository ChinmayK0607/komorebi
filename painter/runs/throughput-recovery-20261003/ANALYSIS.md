# Offline diagnosis and throughput repair — 2026-10-03

The October 2 three-judge campaign did not produce a usable quality ablation. Flash, Pro and Luna each completed two optimizer updates; none reached the interval-four adapter checkpoint or the registered step-12 evaluation. These are operational observations, not evidence that one judge trains a better painter. The GPU node was released October 2 at 22:25:32 IST. This recovery work rents no GPU and makes no paid API calls.

## Hypothesis and matched baseline

The limiting factor was useful episode throughput and an incompatible completion/checkpoint budget, rather than a demonstrated gradient failure or judge expense. The baseline is attempt 6, source `68d5db6618a0264f26ec52248da0a1662727adff`, on eight A100 80-GiB cards at $9.20/hour, one trainer and seven inference replicas. All arms used 40 training briefs, 47 held-out briefs, batch 16, group 4, 96 in-flight episodes, maximum policy age one, three drawing turns, the same grounded beauty/warmth rubric, and interval-four adapter saving. CPU activation offload was enabled.

Immutable identifiers:

- Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- RL16 initialization: `CK0607/qwen3.8-27b-brush-painting@ac4d3eda8e5925e2d23992cfdc1a63620ebde573`, `text-pairwise-scaled-20260929/step_16`; adapter SHA-256 `e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`.
- Frozen SFT512 opponent: same model repository at `adf687e5b321514eda7be6eab050434fdc8e44ff`, `photo-curriculum-sft-20260922-balanced/step_512`; adapter SHA-256 `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
- Prime: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
- Training semantic manifest SHA-256: `13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`.
- Judge rubric SHA-256: `d21e4644e7a3874a0694b2ad406420e733b7638222eb7717e7f5f91cbdf1613c`.

## What consumed the time

| Saved observation | Flash | Pro | Luna |
| --- | ---: | ---: | ---: |
| Actual optimizer updates | 2 | 2 | 2 |
| Valid render receipts / all receipts | 225/432 | 230/470 | 264/509 |
| Render timeouts | 37 | 31 | 31 |
| Valid-render median, seconds | 24.4 | 25.0 | 24.6 |
| Active episodes canceled at the recorded policy update | 84 | 86 | 80 |
| Queued stale traces dropped | 28 | 16 | 24 |
| Admitted groups / observed groups | 20/24 | 17/20 | 23/24 |
| Mean judge call latency, seconds | 36.19 | 38.38 | 10.11 |
| Recorded judge calls | 86 | 113 | 130 |
| Recorded judge API cost, USD | 0.042405 | 0.163022 | 0.073471 |

The 96-episode pool was much larger than the two batch windows permitted by the policy-age bound. When the first weight update arrived, most active episodes were canceled and queued old traces were dropped. Weight broadcasting did resume fresh-policy generation; this is waste from cancellation and staleness, not evidence of a broadcast deadlock. Group admission was mostly successful and was not the main bottleneck.

Completed episodes spent a mean 309/295/302 seconds in model generation and 738/646/593 seconds in the environment harness, respectively. Harness time includes waiting for a render slot as well as rendering; it is not a CPU compute measurement. These summaries exclude canceled unfinished episodes. The faster Luna calls still yielded only two updates, so buying a faster judge alone would not solve this run.

The two trainer gradient norms were nonzero in every arm: Flash 0.0045/0.0047, Pro 0.0131/0.0094, Luna 0.0051/0.0081. Peak trainer memory was approximately 73.6/65.8/65.7 GiB. The second observed update took 318/370/252 seconds. Two updates do not establish steady-state throughput or effective visual learning.

## Reward was often validity feedback rather than beauty feedback

The strict final-action setting rejected a failed last correction even when the environment retained an earlier valid painting. Of 106/119/145 completed traces, 71/75/85 received `model_invalid_final_action`; 36/40/48 of those retained a valid canvas. Thus 124 completed episodes with an existing painting received zero before any aesthetic comparison. Other final-action failures had no valid retained canvas.

This is a deliberate existing reward contract, not a parsing error. It makes terminal execution reliability a hard prerequisite for beauty credit. It can help prevent a failed last edit from earning credit, but it also collapses a good earlier canvas and a wholly failed episode to the same reward. These counts do not establish which policy-gradient updates were dominated by those zeros. The offline repair leaves the reward contract unchanged; scoring retained valid canvases would be a separate research comparison, not an infrastructure fix.

About half of render receipts were invalid. Frequent model-code failures included unset brushes/colors, array arguments passed where scalar coordinates were required, helpers colliding with p5 global names, duplicate lifecycle definitions, and unsupported brush APIs. The existing compact API reference does not specify every signature. An opt-in versioned reference can clarify these conventions without silently changing historical prompts. Its effect on model validity must still be measured.

## Measured utilization and cost

GPU CSVs sampled approximately every ten seconds. Inference means were 70.47%, 71.31%, and 72.45% across 2,268/2,254/2,261 device samples. Trainer means were 12.32%, 12.21%, and 13.71% across 324/322/323 samples; nonzero trainer busy fractions were 19.75%, 18.32%, and 17.34%. Windows were approximately 54 minutes per arm: Flash 14:04:47–14:58:46 UTC, Pro 15:02:17–15:55:58 UTC, Luna 15:59:13–16:53:02 UTC. These include warm-up but exclude preparation/archive work outside those windows. Busy inference hardware did not imply a saturated training pipeline.

Attempt 6 request-to-release GPU-clock estimate was $27.31594. Prior failed starts cost an estimated $18.42724, giving $45.74318 for the campaign; these are elapsed-clock estimates, not provider invoices. Saved judge receipts report $0.278899 combined, a lower bound excluding unfinished/unrecorded calls. GPU time dominated the observed spend.

Each arm's $13.70 allocation allowed about 89.35 minutes. The 32-minute evaluation/archive/release reserve left only 57.35 minutes for startup and training. Interval-four saving therefore depended on completing four updates inside a window that produced only two. Ordinary final saving was not guaranteed under signal-driven deadline termination. No trained adapter survived this attempt.

## Repairs and next decision

1. Preserve a render slot through cancellation until its blocking worker has stopped. Terminate owned nested worker/browser process groups cooperatively, without altering painting programs or historical frozen assets. Use a distinct opt-in renderer derivative and runtime manifest.
2. Provide an opt-in recovery configuration with 32 in-flight episodes, unchanged batch 16/group 4/policy-age one, and an adapter checkpoint at every completed update. Public upload/hash verification remains part of the existing export path; publishing overhead must be measured. This is a proposed occupancy setting, not a proven speedup.
3. Produce a reusable offline audit distinguishing dispatched/admitted work from actual updates, stale drops, cancellations, validity, judge calls, and sampled utilization. Preserve raw evidence separately; never publish private connection state.
4. Clarify drawing API signatures in a separate opt-in prompt version. Keep the registered ablation's prompt and reward hashes intact.

Do not repeat all three paid arms with the old recipe. First measure the corrected renderer on Linux CPU and the new occupancy/checkpoint recipe in a short single-arm run, within the user's $20 GPU-rental ceiling. Report useful completed episodes and published updates per dollar, render-slot waiting, policy-age cancellations, and held-out visuals. No throughput improvement is claimed from offline unit tests. A retained-canvas reward comparison, if chosen, must be separately labeled and matched.

Parent review passed 83 tests in the combined offline suite and all 18 historical ablation contract tests. One existing pure centroid-function fixture was skipped because Node.js was unavailable; the copied derivative's `--help` executed successfully. Lifecycle tests use dummy POSIX processes, including a separate-session worker and grandchild; they do not exercise Chromium. [Validation and tested source hashes](validation.json) and [application instructions](RECOVERY.md) preserve the exact scope. Linux rendering, pixel parity, interval-one public publication and useful throughput remain unmeasured.

## Evidence provenance

Public dataset: [CK0607/komorebi-painter-experiments](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/tree/b7aa7cfcc1353fdbd5c403e37dea4f83eb6624c6/runs). All three archives were anonymously downloaded and SHA-256 verified before metadata extraction:

- Flash `runs/rl-judge-ablation-flash-20261002/evidence.tar.gz`, 136,318,302 bytes; `5928a75e9766cc8f96667c4249db59801ba6c4516007f88d642e7e7afadbe3cc`.
- Pro `runs/rl-judge-ablation-pro-20261002/evidence.tar.gz`, 149,830,658 bytes; `315c1977ecdded2125c6da19c78982a6d5cdcdf01c79872dedbcdd19c811e5f8`.
- Luna `runs/rl-judge-ablation-luna-20261002/evidence.tar.gz`, 159,651,920 bytes; `da1a103a990afc041a2240ab03480de068f1f666e161ccfdbaf1b4312a058059`.

Local metadata and compact trace extracts remain under the ignored `painter/runs/rl-judge-ablation-20261002/private/offline-analysis/` directory. Full programs, responses, images and weights are excluded from the tracked audit. The prior terminal record is [allocation-attempt6-result.json](../rl-judge-ablation-20261002/allocation-attempt6-result.json). The root research index is intentionally bounded rather than exhaustive.
