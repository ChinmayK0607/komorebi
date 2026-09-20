# Recovery quality results

The recovery adapter learned useful stopping behavior and improved several object constructions. Improvement remains uneven: Luna preferred it in 7/12 blinded painting pairs, while five families regressed. The user prefers the new paintings, so it is the candidate initializer for the next round; preserve the previous adapter for regression comparisons.

Training completed 72 Prime SFT updates at learning rate 5e-6, global batch 8, LoRA rank 16/alpha 32, warm-starting the immutable prior step96 adapter. Every update was finite and changed all 992 adapter tensors. This run did not perform policy-gradient RL.

Evaluation used twelve fresh procedural references, three tasks and two attempts per task: 72 episodes for each adapter, with the same scaffold. The previously evaluated references used for training corrections were retired into training. These results do not establish photographic or unseen-family generalization.

| Behavior | Previous | Recovery |
|---|---:|---:|
| Blank painting: first attempt executable | 9/24 | 21/24 |
| Blank painting: mean turns | 2.38 | 1.17 |
| Blank painting: mean output tokens | 1,516.5 | 743.7 |
| Partial-canvas repair: mean turns | 1.50 | 1.04 |
| Completed-canvas tasks: kept unchanged without painting | 0/24 | 24/24 |
| Completed-canvas tasks: mean output tokens | 681.5 | 17.0 |

Both adapters ultimately produced valid canvases and signaled completion for all 72 tasks. For repair/finish tasks, validity can reflect the supplied starting canvas and is not an independent painting-quality result.

## Luna visual review

A gpt-5.6-luna subagent inspected all 36 reference/A/B PNGs for the twelve paint sample00 pairs, with model identities hidden until completion. Recovery: **7 wins, 5 losses, no ties**. Independently acceptable: **6/12 recovery, 5/12 previous**. These are small-sample model judgments, not human ground truth or a statistically established broad win rate.

Gains: sailboat, tree, box, sphere, house, still life and bus. Regressions: flower, ribbons, cylinder, vase and cup. Sphere was relatively better but still unacceptable. A factual petal-count audit corrected the reference count without changing the flower preference; original and amended judgments are preserved.

See [the reviewed preview](visual-preview/index.html), [decoded Luna results](luna-review/results.md), and [all evaluation cases](collected-final/evidence/comparison/index.html).

## Separate Qwen diagnostic pass

Qwen3-VL-32B judged 72 paired outcomes in both image orders: 2 stable recovery wins, 25 ties, 3 losses, and 42 masked for uncertainty/order disagreement. Only 16/38 recorded transitions have unmasked rewards. The task mixture includes finish/repair as well as blank painting, unlike the Luna subset. This automated pass is not a dependable sole reward signal yet. Preserve its raw judgments for diagnosis; masked values are not zeros or ties and development rewards must not become training batches.

## Preservation

Public checkpoint steps 24/48/72 are hash-verified. The final checkpoint is [step 72](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/057c193b1d5191fe3a06eb5e10744b383eba9f1f/recovery-quality-20260918/step_72).

The weight-free archive contains 8,838 files, 132,518,093 bytes, SHA256 `a660330d245f64ff9113bc72e0ec2e49a1d73a8c14a8aadc7f5ce9ef8a3b1cd6`; every archived file was verified. The node was released and provider lookup confirmed not found. Last provider-reported cost was $1.83; this is not a later billing reconciliation. The watcher is paused.

## Next data

`painter/data/diversity-v3-20260918` contains 2,304 new reference specifications (1,920 train/384 dev), with 6,912 syntax-checked programs for complete references and intentionally flawed input canvases. They still require node rendering and Luna review. Geometry, placement, object-count and overlap diversity increase; the underlying twelve recipe families remain procedural. Retain previous successful training examples and general replay, and keep speed penalties off.
