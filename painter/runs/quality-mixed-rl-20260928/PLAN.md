# Mixed-reference fixed-anchor RL, 2026-09-28

Status: inputs public and independently hash verified; node launch pending.
No quality result exists for this run yet.

## Hypothesis and matched baseline

The last four-scene run was rewarded against one static SFT sample per scene.
The frozen sphere and flower samples were weaker than the matched SFT
evaluation samples, and the dog anchor barely depicted the dog. Thus an online
win against that anchor did not establish progress against ordinary SFT output.
Its 8/12 in-flight window also dropped six queued traces and canceled eleven
stale episodes. Hypothesis: a wider set of explicit, hash-bound anchors and an
8/8 in-flight window will yield more usable visual comparisons per GPU-hour
and less stale work. This is not a claim that static anchors solve visual
reward, and the two changes are confounded in this run.

The matched quality baseline is the public photo SFT step-512 adapter at
revision `adf687e5b321514eda7be6eab050434fdc8e44ff`, weight SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Evaluation runs that adapter and the final RL adapter on identical references,
seed, temperature, four turns, 8,192 tokens per turn, and renderer. Seven photo
references are held out from this RL run; three procedural training references
are also evaluated so overfit and generalization are not mixed.

## Data, reward, and operating bounds

The [offline builder](../quality-oss-rl-20260927/build_fixed_anchor_curriculum.py)
assembled 12 COCO photos and three procedural paintings for training; seven
additional photos are held out. Each image is bound to a source SHA-256. The
training manifest SHA-256 is
`83adfcee25a83badae322c0c6d39905bfd0f2de42a9c88c4d82e4b370e76bcf6`;
the held-out manifest SHA-256 is
`26833a8161b276d24a9dd07ad04fb88b06701e481b8c5a06b57f7234c14c4f4e`.
The [public dataset bundle](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/320337fd3bcf60d128b50787766e6e5be7ec9897/runs/quality-oss-rl-20260927/mixed-fixed-anchors-v1.tar.gz)
is 6,454,514 bytes, SHA-256
`1d4c098f6a40f82091cccf870611cc7c82b4e0a951762fd0d6c2172494a10fcd`;
an anonymous re-download matched. The local ignored
[anchor review](../quality-oss-rl-20260927/collected-mixed-anchors-v1/anchor-review.jpg)
is a visual inventory, not a quality result.

The fixed photo anchors are **one-turn** SFT step-512 paintings from an earlier
matched gallery; the three procedural anchors are final canvases under a
four-turn cap. Some photo anchors omit details. A 4-turn RL painting beating
a one-turn anchor is a training signal, not evidence of a better 4-turn
policy. The held-out matched comparison is decisive; online reward and
training-scene wins cannot by themselves promote the checkpoint. The live
two-orientation Vercel MiMo Flash judge prompt remains unchanged because the
recent 19-case human category audit did not favor the subject-first variant.
There is no additive speed reward. Valid rendering only makes a canvas
eligible for visual scoring.

The proposed finite run initializes from SFT step 512 on Qwen3.8-27B at
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, uses Prime RL at
`26b3131d2716a4f8210b165df584b83b4bc54f61`, one trainer and one inference
GPU, batch eight/group four, 12 optimizer updates, four turns, 8,192-token
ceiling, LoRA learning rate `1e-6`, exact eight in-flight, and checkpoints at
steps four, eight, and twelve. Record the actual generated TOML hash and
source commit before launch. Each checkpoint must be public and hash verified.

The combined campaign GPU spending observed before this run was at least
about $10.70; this run plus evaluation and node setup must stay below the
user's $25 GPU-only ceiling. The currently available two-card 96 GB node is
$2.38/hour. Available four-card 140 GB nodes are $13.60/hour and the only
larger 96 GB class is eight cards at $10/hour, so a larger whole node is not
cost-effective within the remaining ceiling. This pricing is a 2026-09-28
inventory snapshot, not a fixed quote. No recurring watcher or scheduled
task is authorized.

## What to record afterward

For every checkpoint: accepted updates, gradient norms, checkpoint revision
and local/remote SHA-256, judged score distribution and invalid/censored
rollouts. For throughput: inference and trainer utilization over the same
stated time window, accepted episodes/hour and per GPU-dollar, dropped and
stale episode counts, renderer attempt duration, API judge latency and cost.
For quality: held-out and training splits separately, paired wins/ties,
invalid counts, image-order conflicts, full turn galleries, and manual visual
inspection of subject structure and beauty. Publish raw receipts/rollouts
before deleting the node. Retain step-512 SFT unless held-out evidence supports
a better painting policy.
