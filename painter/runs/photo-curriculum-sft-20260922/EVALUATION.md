# Matched photo evaluation after SFT

Hypothesis: the larger, visually screened curriculum improves valid first-paint programs and reference fidelity over the immutable base. Compare the midpoint and final adapter to distinguish continued learning from degradation during longer training.

Policies: unchanged base Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (preserved baseline), step 512 (public commit adf687e5b321514eda7be6eab050434fdc8e44ff), and step 1024 (public commit 94b97165474a5c0b56bd1eb3e3fc81b710741cca). Adapters use the photo-curriculum-sft-20260922-balanced namespace of CK0607/qwen3.8-27b-brush-painting. Source checkpoint bytes must match their public upload receipts before serving.

Use all 28 held-out COCO development photographs in the frozen manifest, one response each, temperature 1, seed 918200, thinking disabled, 8192 output tokens, 16384 model context, and the same canonical painting prompt and 180-second renderer deadline. Preserve failures and length cutoffs in the denominator. These are development references, not a sealed test set. No additional training or model selection has used their new generated outputs.

Run midpoint and final inference on the two now-free GPUs. Limit each evaluation to four concurrent episodes (eight total) to maintain the baseline's combined renderer concurrency of eight. The baseline used eight episodes within one policy. This is a declared throughput change; request batching need not be numerically identical. Rendering and model inference stay on the Linux node.

Report program/render validity separately from visual quality. Show all references and all recorded outputs, with failures explicit. Luna-low pairwise review should judge reference fidelity, composition, structure and aesthetics together, explain the preference, and name defects in the preferred painting. Preserve disagreements with parent inspection. A lower procedural validation loss is not evidence of improved photo painting.

This is a finite evaluation job with owned inference-server cleanup, not a scheduled task or recurring watcher. Preserve the exact configs, checkpoint receipts, outputs and metrics before releasing the node.
