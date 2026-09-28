# Mixed-reference fixed-anchor RL, 2026-09-28

Status: four-update diagnostic completed and evaluated; public evidence was
independently hash verified; the GPU node was released. No RL checkpoint was
promoted. The
staged source commit is `a549955a9917a6c86f16b1f70cb036e501528143`;
the transferred Git bundle SHA-256 is
`da0ff6c52f626e5fb58ebfcf237c1e8a5191b6fb9a238c764f046fca2831adde`.

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
steps four, eight, and twelve. Record the actual generated TOML hash before
launch. Each checkpoint must be public and hash verified.

The node-generated native TOML SHA-256 is
`5050f0f62e3d3262deebad5cd8b0ee5fe1e969c273bbe3d718e1f65b1e43ae78`.
Prime's pinned validator accepted this config, and the native task preflight
serialized a fixed-anchor photo row successfully. The inspected TOML has
`min_inflight=initial_inflight=max_inflight=8`, batch eight/group four,
12 steps, and 15 expected tasks. No training had started at this check.

The first launch reached inference warmup and then stopped before any rollout
or optimizer step: `run_from_sft.py` assumed the first task always used a lazy
opponent and rejected this fixed-anchor task. The runner now verifies the
fixed anchor's SHA-256 and skips lazy generation; the focused runner tests
pass. This was a launch-scaffold failure, not an RL result. The failed attempt
used source `a549955a`; record the patched source commit on restart.

The immediate retry also made no rollout: Prime found the failed launch's
nonempty output directory and correctly refused to treat it as a fresh run.
That directory, containing only configuration and startup logs, was renamed
and retained. The active restart uses patched source commit
`407836865085d00d537da0508d54ea9c15ff4082`; a direct node-side CPU check
returned `fixed_verified` for the first anchor before model warmup. The
setup and launch scripts now exercise that check and detect an existing run
directory before loading the model on future attempts.

The combined campaign GPU spending observed before this run was at least
about $10.70; this run plus evaluation and node setup must stay below the
user's $25 GPU-only ceiling. The currently available two-card 96 GB node is
$2.38/hour. Available four-card 140 GB nodes are $13.60/hour and the only
larger 96 GB class is eight cards at $10/hour, so a larger whole node is not
cost-effective within the remaining ceiling. This pricing is a 2026-09-28
inventory snapshot, not a fixed quote. No recurring watcher or scheduled
task is authorized.

## Live rollout diagnosis, 2026-09-28 06:44 UTC

The active run's source commit is `407836865085d00d537da0508d54ea9c15ff4082`.
After model warmup at 06:15 UTC, its first completed group produced no admitted
gradient; the next group admitted four episodes and was waiting for the rest
of the first batch. Eight scored rewards included seven zeros and one tie at
0.5. This is an early sample, not a training or quality result. The first 86
rendered turn receipts comprised 18 valid and 68 invalid canvases. Common
errors included malformed JavaScript, collisions with p5.brush global names,
and unsupported brush calls. One syntactically valid no-drawing program
produced an all-black canvas; it earned zero reward. A tested source fix
(`5bc2680`) now marks near-uniform canvases invalid for **future** launches;
it was not installed in the running process and is not counted as this run's
behavior.

The one inference GPU was decoding continuously at roughly 170 aggregate
output tokens/second with eight active requests at this observation. Through
the earlier 06:16–06:36 UTC telemetry window, its mean sampled utilization
was 96.6% over 122 ten-second samples and it was at least 50% busy in 96.7%
of those samples; the trainer GPU averaged 0% before its first update. This
is not a saturated training pipeline. More inference workers on the existing
card are unlikely to improve throughput appreciably. The live available
four-card 140 GB node was $13.60/hour and the larger 96 GB class eight cards
at $10/hour, versus $2.38/hour for this two-card node, so migration would
consume the remaining GPU budget before a matched evaluation. An early-stop
path will publish and compare a public checkpoint if 12 updates cannot fit
the ceiling; online wins against weak one-turn photo anchors cannot promote it.

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

## Result and decision, 2026-09-28

The active run stopped after its first planned public checkpoint at step four
because invalid actions and fixed-anchor zero-contrast groups made further
sampling inefficient. This was a deliberate early stop, not completion of the
configured twelve updates. All four updates had nonzero gradient norms:
0.0165, 0.0352, 0.0415, and 0.0486. Their online mean rewards were 0.125,
0.375, 0.3125, and 0.500; these values compare against static anchors and do
not establish improved painting. End-to-end step durations were 33m35s,
9m45s, 18m31s, and 18m42s. Step one discarded 12/20 completed episodes for
no signal; step three discarded 12/20 (eight stale, four no signal). The
65 completed traces had 49 zero, six tie (0.5), and ten win (1.0) terminal
rewards. Their 261 rendered turn attempts produced 110 valid and 151 invalid
canvases. A locally tested blank-canvas eligibility fix was committed after
the running source commit and did not affect this run.

The [public step-four adapter](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/ab5f4162f7626a1838a8b7b1bf603e22909987c1/quality-mixed-anchors-20260928/step_4)
has local and upload-receipt weight SHA-256
`dc75aedf0dddef52d19e916c6daaa6668ecc4c4a1534511d25d87ab2d4ba9023`;
the receipt states public and verified. The frozen SFT step-512 and RL
step-four policies were evaluated on identical ten references, seed 918200,
temperature 1, 8,192-token per-turn ceiling, four-turn cap, renderer and
server profile. The evaluation completed all ten cases per policy. Its
two-orientation MiMo Flash judge scored six SFT wins, two RL wins, and two
ties, with no censored judgment. On seven held-out photos: five SFT wins,
one RL win, one tie; RL had three invalid final canvases, SFT none. On three
training shapes: one win each and one tie. Manual review of the [local
gallery](collected-step4/review/index.html) agrees with the broad held-out
conclusion: SFT preserves recognizable people and scene structure on the
frisbee and tennis photos where RL becomes blocky; RL is missing entirely on
three held-out cases. The small procedural paintings are closer, and RL's bus
is a plausible local gain. The ten-case set is a diagnostic, not a population
quality estimate. Retain SFT step 512; do not continue this reward recipe.

The public [evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/dc8044d996d52c46e2fc0a7b901ebd2a28c390b4/runs/quality-mixed-rl-20260928/step-4-evidence.tar.gz)
has 1,020 screened files, 34,774,022 bytes and SHA-256
`154ad715234899e23a05d7e32479510eeb8445c42eb78043792b06977825c24e`.
It was anonymously re-downloaded at its commit and hash checked. The local
gallery transfer archive also passed a source/destination SHA-256 check.
The archive contains renderer programs, canvases, reward files, judge receipts,
and matched evaluation turns. It does **not** contain Prime's packed trace
JSONL or trainer/orchestrator logs; those were not selected by the evidence
packager and were lost when the node was removed. Thus the recorded gradient
and discarded-episode counts are transcribed from the live logs, while the
exact native sampled token/log-probability arrays are no longer recoverable.
The visual result, checkpoint, reward receipts, and telemetry remain
independently inspectable. The archive selector must include screened native
traces and run logs in future experiments before node deletion.

During 06:16:36–07:40:38 UTC, 505 ten-second samples per GPU showed mean
sampled utilization 99.1% on inference GPU 0 (99.2% of samples at least 50%)
and 12.5% on trainer GPU 1 (12.9% at least 50%). This is not full-pipeline
saturation. During the matched evaluation's 07:44:08–08:12:58 UTC window,
174 ten-second samples per GPU averaged 72.4% on SFT GPU 0 and 87.8% on RL
GPU 1. Model decoding dominated elapsed time, but rendering was still
nontrivial: valid evaluation turns averaged 23.2 seconds for SFT and 9.4
seconds for RL, with one hard-render deadline on each side. The last observed
node charge was about $5.84 at $2.38/hour. Together with at least ~$10.70
observed earlier in the campaign, observed GPU spend remained below $25.
The node was removed and a fresh `lium ps` returned no pods. No scheduled
task or watcher was created.

Next method decision: before another GPU launch, fix invalid p5.brush
generation with a small matched validity evaluation, and replace the sparse
anchor-only 0/0.5/1 signal with calibrated **pairwise candidate ranking within
each rollout group** when every candidate loses to a strong anchor. Keep a
single visual-quality ranking criterion tied to the reference; retain the
anchor as a check rather than adding unrelated metric bonuses. Reorder the
curriculum from simple shapes to photos and compare it with the same frozen
four-turn SFT baseline. The current Prime config preserved manifest order,
which put all twelve photos before the three procedural shapes. Test the
within-group ranking and renderer speed on CPU/Cloud evidence before renting
another GPU. More inference capacity is useful only once the extra rollouts
can enter useful gradients; this run's 24 GB three/four-card options could
not host the pinned 27B trainer/inference split as configured, while the
four-H200 whole node cost $13.60/hour at the inventory check.
