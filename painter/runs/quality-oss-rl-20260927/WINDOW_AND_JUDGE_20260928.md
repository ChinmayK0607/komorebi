# Four-scene recovery and in-flight test, 2026-09-28

Status: finite 8/12 in-flight GPU run and matched step-four evaluation completed;
public evidence is hash-verified and the GPU node was released. Do not scale
this checkpoint on the current reward and rollout configuration.

## Question and matched baseline

The previous four-update [Gateway overfit](GATEWAY_OVERFIT_20260928.md)
finished with nonzero gradients but split 2–2 with the frozen SFT step-512
baseline on four references. It sampled roughly 22 episodes concurrently and
discarded 12/20 at update three as stale. The step-two dog evaluation had no
valid canvas in four turns. Hypothesis: a bounded 8/12 in-flight window will
reduce stale work and cost per accepted update while preserving visual quality.
The matched throughput baseline is the prior 16/24 window on the same four
references, initial SFT adapter, 2×RTX PRO 6000 Blackwell layout, batch 8,
group 4, four turns, 8,192-token ceiling, four updates and `1e-6` LR. These
are very small, stochastic samples; report the changed interaction feedback
as a confound rather than claiming a clean concurrency effect.

The initializer is public
`CK0607/qwen3.8-27b-brush-painting/photo-curriculum-sft-20260922-balanced/step_512`
at revision `adf687e5b321514eda7be6eab050434fdc8e44ff`, weight SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Base Qwen3.8-27B revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
Prime revision is `26b3131d2716a4f8210b165df584b83b4bc54f61`.
The four-row data manifest SHA-256 is
`58a5d1f06910d60d5b3e5726a1e73692a08d8739fc582847bf23e5fd7619c62a`.
The launched source commit is
`65f679b01418b2f87dca55276a53e940edc2aea6`; the native TOML SHA-256 is
`8ec44fed5e4314e83c7fe9020e6aeb9a90ed6d78802b62032df03755f3be192b`.
Prime's pinned config validator accepted batch 8/group 4 and the exact
concurrency values `min_inflight=8`, `initial_inflight=8`, `max_inflight=12`.
Native task serialization and the node preflight passed before training.

## Concrete changes before spending

The matched evaluation gallery now retains every attempted `program.js` and
`painting.json`, even when no PNG exists; the public evidence publisher already
includes the review tree. This repairs a real evidence omission: the prior
step-two archive has its matched gallery and judge receipts but **not** the
four failed dog programs. That node was released, so those exact programs
cannot be recovered. Its training rollout archive contains 35 invalid paint
receipts, mostly malformed JavaScript and a few p5-name collisions. The live
feedback now gives a short specific correction for syntax errors, p5 global
redefinitions, and non-iterable inputs while retaining the actual renderer
error. This changes the interaction scaffold and must be disclosed in any
comparison with the previous run.

The visual judge remains the Vercel Gateway `xiaomi/mimo-v2.6-flash` with two
image-order orientations and image-bound receipts. A separate subject-first
prompt is prepared for an eight-call comparison on the four already-rendered
step-four pairs; it does not silently replace the live reward. The dog's
default preference was visually questionable because the winning RL canvas
has a faint, poorly formed animal. Compare prompt variants with image hashes
and any user preference before deciding whether to change the training
reward. Do not fabricate a human gold label from a model response.

The four-pair subject-first calibration finished with all eight image-order
calls scored. In SFT-score order sphere/bus/flower/dog, the default prompt
gave `1/0/1/0`, while subject-first gave `0/0/1/0.5`. Thus wording flipped
the sphere and made dog orientation-dependent; it did not establish a better
judge. The two prompts agree on bus and flower. The eight calls' provider
usage totals approximately $0.00770. Raw image-bound labels and prompt hashes
are saved under the ignored `collected-eval/judge-calibration/` evidence tree
and will be backed up publicly before releasing the node. Keep the default
live reward for the finite window run; seek human subject-fidelity labels
before changing the training reward.

At 00:26 UTC the launch finished a cold-start prewarm of one frozen SFT
opponent; the first receipt was created around eight minutes after launch.
Additional opponent cache entries are generated on demand. This is a
one-time cost separate from steady-state rollout throughput. The public
evidence packager now retains hash-bound opponent receipts and canvases,
and a next-node setup option restores them only from an archive with a
supplied exact SHA-256. Those restore changes were prepared after the
training source commit and do not affect this run.

## Operating bounds and decision

Use one finite 2-GPU node at no more than $2.38/hour if the previous 96 GB
card class is available; release it after public hash-verified evidence.
Previous observed GPU cost was at least $7.23. Keep combined campaign GPU
spend below the user's $25 ceiling. No recurring watcher or scheduled task.
Keep SFT baseline and final RL evaluation settings matched: same four
references, seed 918200, temperature 1, four turns, 8,192-token ceiling,
same pinned renderer. Record accepted, stale, cancelled and invalid counts,
wall time per update, GPU utilization over a stated window, and cost per
accepted judged episode. Preserve raw judge labels and failed programs.
Do not promote the new checkpoint solely from online reward; compare rendered
finals and retain the public SFT baseline if the visual result is not better.

## Result and decision, 2026-09-28

The finite run exited zero. All four accepted batches contained 8/8 trainable
episodes; their online rewards were 0.50, 0.625, 0.50 and 0.50. Trainer gradient
norms were 0.0962, 0.1006, 0.1436 and 0.0767. The four adapter files changed
from the initializer and each has a public, hash-verified HF receipt. Step four
is [public at revision `61f06efa`](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/61f06efa09a3419fc8b902503a344dce9ecad7eb/gateway-overfit-window-20260928/step_4),
with weight SHA-256
`1fae84a66095115a79aa81aa4665966ec0b93f8cbc4057cdde936fd17dbf1fdb`.
The online reward sequence does not establish visual improvement.

The matched evaluation used the four pinned training references, seed 918200,
temperature 1, four turns, the same 8,192-token per-turn ceiling and renderer.
It completed both policies successfully. The [local gallery](collected-window-eval/review/index.html)
shows three MiMo pairwise wins for the frozen SFT step-512 policy (sphere/vase,
bus/mug/tree, flower/cylinder). RL's only win was the dog/skateboard photo:
SFT produced no valid final canvas, so the scorer correctly assigned a
deterministic RL win without an API call. The RL dog remains visibly crude.
Visual inspection agrees that the simple-scene RL variants lose important
structure: its mug handle floats free, and its sphere/vase and flower/vessel
are less faithful. This one-seed four-scene result is an overfit diagnostic,
not a generalization estimate or evidence that concurrency alone caused the
regression; the error-feedback scaffold also changed.

Training collection ran from 00:27:34 to 01:08:44 UTC. Across 245 ten-second
telemetry samples per GPU, inference GPU 0 averaged 98.0% engine activity
(98.4% of samples at or above 50%), while trainer GPU 1 averaged 13.5%
(13.9% at or above 50%). These are sampled busy fractions over that interval,
not FLOPS. Accepted collection steps took 13m56s, 8m44s, 10m29s and 5m25s.
Prime logged six dropped queued traces and eleven canceled stale in-flight
episodes across the run; these are different units and must not be summed as
completed renderings. The smaller 8/12 window therefore did not remove stale
work. The live renderer recorded 185 attempts, 139 valid and 46 invalid; 21
invalid attempts contained JavaScript syntax errors, nine undefined names,
and eight unsupported API calls. Mean receipt wall time was 17.0 seconds.
Previously measured parallel renderer capacity was about 0.41 paintings/s,
above this run's end-to-end action rate, while inference was busy almost
continuously. A renderer rewrite is not justified by this profile.

The separate subject-first prompt calibration used the *previous* window's
four rendered pairs. Its eight orientation calls changed the sphere winner
and made the dog orientation-dependent; it did not establish better agreement
with a human gold set. The current live reward kept the original prompt.
The prior calibration receipts and image hashes are included under
`matched-eval-step-4/review/prior-step4-judge-calibration/` in the public
archive, explicitly scoped to that prior checkpoint. Do not present them as
labels for this window's step four.

The [public evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/b377353e87fb56003b3ed1509d34f97c8fad0606/runs/quality-oss-rl-20260927/window-overfit-20260928-evidence.tar.gz)
contains 847 screened files, including the actual window config, raw failed
programs, renderer receipts, matched images, judge labels, and frozen opponent
receipts/images. Its SHA-256 is
`05c3667ae14a9850968225ee8e22026e10166c546bc8b89a47673d2a17fe3581`;
an unauthenticated re-download matched. The evidence packager fix is commit
`f0e035b`, made *after* training; training itself used `65f679b`. The
additional targeted feedback for `brush.ellipse()` and undefined names is
commit `b608142`, also post-run and untested by training. Focused feedback
tests pass.

The last observed Lium spend for this node was $3.47 at about 1.5 hours;
combined with the prior observed at-least $7.23, campaign GPU spend was at
least approximately $10.70, below the $25 GPU-only ceiling. Vercel API costs
are separate. The exact owned node was removed and `lium ps --format json`
returned `[]`. No scheduled task was created.

**Next decision:** retain SFT step 512. Do not launch a larger version of this
RL recipe. Keep renderability as eligibility, not visual reward. The default
MiMo prompt remains in place pending actual beauty labels. Expand the
category and aesthetic calibration beyond the small development set, focusing
on equally poor/tie cases and wrong-subject but attractive-background pairs.
Then test a group-aware or tighter in-flight policy on a broader
mixed-reference curriculum, measuring quality-evaluable updates per GPU-hour
and stale work; only scale if a matched held-out gallery shows a visual gain.

## Human category-label audit, 2026-09-28

The earlier `golden-categories-v1` export contains 24 unique reviewed cases,
but only 19 distinct usable A/B/tie preferences for this comparison. All
reference and candidate pixels were checked against the frozen image hashes;
the panel manifest identity is
`f70b8d592f101f351ff175224ffc33bbcfc4ecb1f3b78d65577e8b7e3086df78`.
The reviewed human-label JSON SHA-256 is
`423c2916ee90d238f05c6fd3128abad962301cb90ab1c20d501b388d85bc62e3`.
The finite [calibration script](calibrate_human_categories.py) called the
Vercel MiMo Flash judge on each case with both the live and subject-first
prompts and both candidate orders: 76 image-bound receipts, zero failures,
total provider-reported cost approximately $0.0509. It performed no local
model inference, rendering, or GPU rental. The raw receipts and
[summary](collected-window-eval/human-category-calibration/summary.json) are
local ignored evidence; the summary SHA-256 is
`5ac0507ad262311094aecf221939d5cb408fd71f10726af48c69e658bd4fddfb`.

The live/default prompt exactly matched 16/19 human preferences and had one
image-order split. The subject-first prompt matched 15/19 and had three
splits. Default category counts were completeness 3/3, identity 1/1,
layout 5/5, preservation 5/6, shape 0/1, and values/color 2/3. These tiny
per-category denominators are descriptive only. The two human ties (house
values and leaf shape) were missed by both prompts, which chose one candidate
despite both paintings being comparably incomplete. The preservation
headphone case split with A/B order under both prompts. Inspection of the
[disagreement sheet](collected-window-eval/human-category-calibration/disagreements.jpg)
supports the tie/ordering concern. The current default prompt is preferable
to the subject-first variant on this development set, but 16/19 agreement on
category preferences is **not** a validated beauty reward or a held-out
generalization result. Do not retune on these same 19 and claim success.
