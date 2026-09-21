# Qwen real-photo capability check

Compare unadapted Qwen3.8-27B with contract-quality step24 on twelve COCO development photos chosen before generation, covering varied subjects and scene complexity. Use the exact shared native paint → rendered canvas → inspect/revise loop for both policies: one sample, paired seed 918200, temperature 1, six turns, 8192 tokens per response, 98304 context, thinking disabled. Generation does not wait for a judge and no training runs.

Inspect every reference, first valid painting, final painting and raw turn record. Distinguish recognizable subject/layout from beauty, and distinguish rendering validity from either. Preserve failures, repeated unchanged paintings, context/turn cutoffs and missing outputs. Compare base and SFT within this matched setup; first-to-final changes show whether revision helps. If both are weak, identify the specific demonstration gaps instead of inferring that the general model cannot paint. A different planning mode, if tried, must be reported as a separate condition.

The decision is whether current Qwen rollouts are useful starting points for real-photo RL, or whether teacher demonstration/correction data should be expanded first. Parent visual review is primary, with the existing Luna-low rubric available for a separate measurement.

Run on a Linux GPU node. Preserve hash-verified evidence before release. No scheduled tasks or recurring watchers.

Operational amendment: the first 32k-context attempt exhausted context before the fourth request on two base episodes that had already painted valid canvases. It was interrupted and preserved as `attempt-1-context32k`, including completed no-code failures and partial episodes. The primary rerun keeps the frozen photo cases, six-turn/8192-per-response budgets, prompts, model pins and seed; it increases context to 98304, server concurrency from four to eight, and wall-clock rollout timeout from 1200 to 3600 seconds. Prefill batching remains 32768. These changes remove observed scheduling/context limits; no old censored attempt is presented as a completed six-turn result.
