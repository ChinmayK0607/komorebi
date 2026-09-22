# Photo curriculum SFT results

The larger curriculum improved first-attempt execution and produced some clearer reference paintings. It did not produce consistently high-quality painterly outputs. Retain midpoint step 512 as the next experiment initializer; retain final step 1024 as a comparison, not an automatic replacement.

## Matched result

| Policy | Valid first-paint canvases / 28 | Completion tokens | Length cutoffs |
| --- | ---: | ---: | ---: |
| Immutable base | 4 | 130,166 | 2 |
| Step512 | 19 | 86,085 | 1 |
| Step1024 | 18 | 81,947 | 0 |

All 28 held-out development photographs remain in each denominator. These are one-response results at 8192 output tokens, seed 918200, temperature 1, thinking disabled, 16384 context and 180-second rendering. They must not be compared as if equivalent to the earlier six-turn capability evaluation. Actual system prompts match; prompt-token totals match at 22,033 per policy. The student checkpoints ran concurrently with four cases per GPU, versus eight concurrent baseline cases. This declared batching difference preserves combined renderer concurrency but does not guarantee identical floating-point sampling.

Luna-low inspected every blinded midpoint/final pair, with displayed-image hashes verified: midpoint 15, final 11, neither 2. Among 11 pairs with two valid paintings, midpoint 7, final 4. Eight cases rendered only at midpoint, seven only at final, two in neither. These are descriptive preferences on a small development set, not a decisive statistical quality win or an RL reward result. The absolute acceptability bar was clarified after the first nine labels; do not pool their acceptability flags into an overall rate. Raw labels remain unchanged.

Parent inspection finds useful improvements over base in café, kitchen and ski scenes, but major anatomy/action failures, simplified lighting and flat geometric surfaces remain. The clock-tower output trades solid color for lost side-face perspective. One valid tennis rendering is a disconnected arrangement of shapes. Parent disagrees with Luna on the café preference and several absolute acceptability/rationale judgments; see PARENT_VISUAL_NOTES.md. Runtime success is not equivalent to visual quality.

## Training and remaining failures

Training completed 1,024 CUDA updates, exit 0, zero NaN losses, with 86 admitted real-photo teacher demonstrations, 2,400 procedural references and 192 general examples. The 8,192 sampled exposures delivered 40.01/49.99/10.00% supervised tokens. Procedural teacher-forced validation loss fell 0.870856→0.033264; midpoint was 0.036954, so most token-loss improvement preceded the second half. This validation loss is not an aesthetic score.

All 17 adapters, including initialization, are public and remotely hash-verified in CK0607/qwen3.8-27b-brush-painting under photo-curriculum-sft-20260922-balanced. Midpoint revision adf687e5b321514eda7be6eab050434fdc8e44ff; final revision 94b97165474a5c0b56bd1eb3e3fc81b710741cca. Native optimizer states are node-local and excluded from preservation; future training can initialize from public adapters but cannot exactly resume the released optimizer state.

Actual failure messages include wrong iterable/array argument shapes, undefined symbols, invalid array lengths, unsupported constructor use and malformed/incomplete JavaScript. The earlier generic brush-API classification was incorrect and was corrected using structured turn errors. No sampled renderer deadline failures were implicated. vLLM applies the trained language adapters, including all four GDN projection targets; ignored wrapper warnings concern frozen vision modules.

The stale config-generator startup and concurrent compiler-cache race were fixed before any student evaluation samples completed. Both failed startup attempts are preserved. No scheduled task was created. One local placeholder-image fixture violated the node-only compositing agreement; it is excluded from Linux validation. Actual training, inference, painting rendering and final review composition ran on Linux.

## Decision

Use step 512 for the next reference-conditioned, multi-turn learning experiment. Focus added demonstrations/corrections on executable argument shapes, declared helpers, object geometry and articulated subjects. Keep actual canvas/error feedback between turns and score reference fidelity plus aesthetics through pairwise visual comparison. The next experiment must evaluate against the same step 512 initializer at the same multi-turn budget. Do not introduce speed pressure yet or treat the 26/28 union of two different checkpoints as a single-policy score. No further optimizer run was launched in this status turn.

The complete visual evidence is review-matched/index.html. The blind review is review-blind/index.html; labels are review-labels/student-blind-*.json. Final preservation, evaluation utilization and rental accounting are recorded separately once cleanup completes.

## Utilization, cost and preservation

Whole training job GPU averages were72.45/72.99% across1098 samples each,22:11:52–01:15:33UTC. Evaluation averages were84.73/83.21% across131 samples each,05:24:00–05:45:41UTC, excluding server startup. The subsequent preservation/approval wait has idle GPUs and is not included in either interval. The earlier post-training idle allocation was about3h44. Elapsed rental estimate at06:01UTC is$21.30 and keeps accruing at$2.38/hour; this is not a provider invoice.

All public adapters are verified. The1.65GB private evidence archive exists on the node, with remote SHA b986c2e67399232ea88198aa519190c282fe7210bf96800ec5e19537e67fb26b; local download and verification remain pending explicit authorization after automatic approval review rejected that transfer. The node has not been released. Final galleries and labels are already local.
