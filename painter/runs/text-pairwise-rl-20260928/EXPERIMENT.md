# Text painting: fixed-SFT pairwise RL

Status: GPU baseline rendering in progress; no training result claimed.

Rental update (2026-09-28 15:04 UTC): a two-96-GB-GPU node was rented at
**$2.38/hour total** with a ten-hour automatic rental expiry (maximum quoted
GPU rental $23.80). This is a finite rental limit, not a recurring watcher.
Setup completed after removing an unnecessary `pip --version` probe from the
uv-managed Prime runtime; baseline sampling is now running on two GPUs. No
baseline or training result has been observed yet.
The two-orientation Gateway smoke judged an obvious red-circle-versus-blue-circle
pair correctly (score 1.0 for red under a red-circle brief). This confirms
endpoint/schema/image delivery, not judge reliability on real paintings.

The first baseline attempt was stopped after its earliest completed traces
showed repeated full HTML documents and invalid attempts to construct
`p5.brush`. One inspected final canvas was a weak but recognizable towel roll;
the failures were chiefly format/API errors, not a scored RL result. The
attempt is retained separately. The text-only system prompt and missing-code
feedback now state the renderer's JavaScript-only/global-brush contract; all
baseline prompts must be regenerated under this changed protocol before RL.

## Hypothesis and matched baseline

On text painting briefs, Qwen3.8-27B initialized from the photo-curriculum
SFT step-512 LoRA may learn to paint better than that *same* frozen LoRA when
each three-turn rollout is compared against a pre-rendered step-512 painting
for the same brief. This is an experiment, not a guaranteed improvement.

Base model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Initializer: public `CK0607/qwen3.8-27b-brush-painting` revision
`adf687e5b321514eda7be6eab050434fdc8e44ff`,
`photo-curriculum-sft-20260922-balanced/step_512`, weight SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Prime revision: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
Matched baseline: that initializer sampled with exactly the RL evaluation
prompt, renderer, three-turn limit, temperature, and per-turn token ceiling.

## Data and reward

The frozen prompt pool has 72 authored, unique text briefs from the 600-scene
teacher visual audit: 40 train (24 simple, 16 intermediate), 16 validation
(8+8), and 16 same-tier reserve. The source audit SHA-256 is
`ea018fce4336c6c3de2dc0ee496b141f45162dc5f3e7428294ea6398b8c2f8f8`;
the selected pool SHA-256 is
`c91ca18440e054d360ac45b071066c7c5c94c62cf5526dfb3791d97ae07fc7fb`.
Teacher paintings/ratings are not RL rewards or held-out student results.

Before training, render step-512 on every brief and hash the fixed canvases.
If the pinned single-sample baseline produces too few valid anchors, rerun
**only invalid briefs** with the same policy, prompt, renderer, and decoding
settings but a predeclared different seed. Freeze the *first valid* canvas in
attempt order, never the prettiest judged canvas. Keep the single-sample
validity rate and retry count separate in results; the comparator then is a
validity-conditioned SFT baseline rather than a matched one-sample baseline.
Training rows with invalid or blank anchors are replaced only by predeclared
same-tier reserve rows, if possible. Report all rejected rows. Validation
membership is never selected using RL outcomes. The frozen SFT painting for
each training row is the only visual comparator. The MiMo-v2.6-Flash judge
sees the brief and a two-image contact sheet in both A/B orders. Each view
votes win/tie/loss; their mean is the terminal reward. Invalid model output is
zero. A judge or renderer outage is censored, not scored as bad art. There is
no detail bonus, within-policy rank, or speed penalty in this run.

## Run and decision

Candidate configuration: 1 training GPU + 1 inference GPU, 4 rollouts per
brief, batch 8, up to 20 updates, LoRA learning rate `1e-6`, checkpoint every
4 updates, maximum 3 painting turns, maximum 8192 completion tokens per turn.
This is a finite run with a **$25 GPU rental cap for this run**, inclusive of
setup, SFT pre-rendering, training, evaluation, and public checkpoint upload.
The node hourly quote and timestamp must be recorded before rental; stop
before the cap if throughput cannot complete the plan. Vercel judge spend is
recorded separately; the user specified $25 as GPU spend only.

Evaluate the fixed validation set at step 0 and available checkpoints using
the matched generation settings. Report validity, paired judge win/tie/loss,
reward by update, unique training briefs reached, censored episodes, GPU busy
fractions by role and time window, wall time, and cost. Promote only a
checkpoint with held-out improvement over the frozen step-512 baseline and no
material validity regression. If no checkpoint qualifies, keep the SFT
baseline and use the failure evidence to choose the next change. Do not
describe rising training reward alone as generalization.
