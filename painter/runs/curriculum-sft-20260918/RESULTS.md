# Expanded curriculum SFT — completed experiment

The 27B model learned valid brush code more reliably on its first attempt, but the complete painting result is mixed. This run does not establish an overall visual-quality gain.

[Open all 96 matched comparisons](collected-final/evidence/comparison/index.html). Each card shows reference, base and trained output. The sample selector exposes all eight attempts per reference; failed and unfinished episodes remain visible.

| Observed measure | Base | SFT step 96 |
| --- | ---: | ---: |
| Evaluated episodes | 96 | 96 |
| First attempt rendered successfully | 41/96 | 60/96 |
| Episode produced any valid canvas | 96/96 | 95/96 |
| Model declared FINISHED | 96/96 | 78/96 |
| Mean turns | 2.13 | 4.39 |
| Mean generated tokens per episode | 3,708 | 2,454 |

FINISHED measures stopping behavior, not painting quality. A valid canvas means the program rendered, not that it obeyed every brush constraint or matched the reference. There is no quality pass@k score yet. All attempts use the same references, prompts, sample seeds and operational limits (eight turns, 16,384 response tokens, 1,800 seconds). Base is the unchanged original model without an adapter, even for baseline samples collected after SFT.

In a fixed sample-0 inspection of four families, the flower gains separate petals and the house better matches the broad structure. The bus loses distinct wheels, and the still life gains unwanted striped/radial marks. This is an unblinded qualitative inspection, not an aggregate win rate. Several trained episodes repeat valid revisions until the eight-turn limit. Actual renderer errors are fed back; these failures were preserved rather than silently retried or removed.

The next quality iteration should target correct rounded shapes, preservation of already-correct parts, recovery from the student's own erroneous canvases, and recognizing a completed painting. Existing demonstrations cover canonical three-stage construction; they do not directly teach recovery from every student error or stopping on already-complete states. This distribution gap is a plausible contributor to the loops, not yet a proven causal diagnosis. Use the collected student states for teacher corrections and pairwise judging before scaling another training run. Keep speed rewards off.

Training completed 96/96 optimizer updates; all 96 audits observed finite gradients and changed adapter weights. Development assistant-token NLL fell from 0.4461 to 0.2119. Dataset: 384 procedural references in 12 families, 1,152 rendered stages; 288 train and 96 development references. Training pool: 864 painting turns plus 192 existing text/chart/document examples. Measured mixture: 70% current objects, 20% primitive review, 10% general replay by assistant tokens. The 96-update run uses 768 training example slots, not the entire unique pool. Evaluation uses twelve development references, one per family, and does not establish photograph or unseen-family generalization.

All checkpoints at 0/24/48/72/96 are public and hash-verified. [Final adapter](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/0778179fb79cfa4d7885788b743e101beae0dc38/curriculum-sft-20260918/step_96). Step 0 is initialization. Saved artifacts are adapter-only, without optimizer state.

Evidence: 7,229 files verified against the remote archive and per-file SHA-256 manifest; archive hash `386fe6cebfc9f718e26b0a5d29ab08368bb46e60f3e728a3cb9105915d173a1f`. No weights were downloaded to the Mac. All painting rendering, inference and training happened on the H100. The node was released; last provider cost was $2.19. The watcher is paused.
