# Relative-quality RL: failure-mode audit

2026-09-28. This is a post-run audit, not a new training result. The frozen
model, curriculum, configuration, source IDs, checkpoint hashes, rental meter,
and [matched gallery](local-evidence/review/index.html) are recorded in
[PLAN.md](PLAN.md). The full raw [public evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/98b4cfbdecee8699a78520c23c2784a21d90cebb/runs/quality-relative-rl-20260928/step-4-evidence.tar.gz)
was anonymously downloaded again for this audit; its 126,313,885-byte payload
matched SHA-256 `249eb93f32bf34f320ee1c60952949034616cff7343fd60d22f3c81f1088ed82`.
The archive's trace, group-credit, trainer, judge, telemetry, and evaluation
records were examined without relying on the deleted node.

## Outcome and what the optimizer actually learned from

The matched ten-case final evaluation favors the frozen SFT step-512 adapter:
SFT 4 wins, RL 1 win, four ties, one case with neither canvas valid. On seven
held-out photos: SFT 3, RL 1, two ties, one censored. RL has an additional
missing final kitchen canvas (photo 089). This small developmental set is
enough to reject promotion, not to estimate a population win rate. RL's lizard
and sink (photo 208) visibly improves, so the update is not universally bad.
However, the coffee scene (photo 127) goes blank and the owl clock (photo 078)
loses visual quality. The prior mixed-reference four-update run also lost to
SFT, 6–2 with two ties, so this is a repeated directional failure across two
reward designs, not a single unlucky comparison.

The training signal was real but narrow. The four accepted batches each had
eight trainable traces and zero truncation, with sampled-token advantages on
every trace; 22 of the 32 final canvases were render-valid. Their references
were photos 049, 572, 431, and 634: **32
accepted episodes from only four distinct photos**, despite a nine-row
curriculum containing six photos and three shapes. Photo 322's eight-way group
was scored but not used; sphere, bus, and photo 562 appeared only as partial
or canceled groups. A rising reward mean across those groups cannot be read
as improvement because each mean belongs to a different image. The four
nonzero gradient norms (0.0228, 0.0378, 0.0131, 0.0277) and public adapters
show optimization occurred, but four updates cannot establish a scaling
trend.

## Algorithm and credit assignment

The fixed opponent was correctly upgraded to a saved four-turn SFT final;
the previous run's weaker one-turn photo-anchor mismatch was removed. This
run mixed that fixed-opponent score 50/50 with a relative rank among valid
paintings in the same eight-rollout group. In photo 431, **all five valid
paintings lost to SFT**, yet four obtained positive adjusted rewards from
their internal ranks. A mean-centered group update can therefore reinforce
the best painting in a group that is still worse than the frozen baseline.
This is mathematically consistent with the configured objective, but poorly
aligned with the stated goal of beating an already strong SFT painter. The
score does not contain an absolute visual-quality threshold or a direct
teacher imitation term. It assigns the same final outcome advantage across
all sampled tokens in all four turns; it does not pinpoint the revision that
improved or damaged the canvas.

This is **not** evidence of a missing loss mask. In the archived Prime native
trace stream, all 40 traces covered by five group-credit audits had sampled
mask counts exactly equal to the audit's sampled-token counts: 422,910
sampled tokens across those groups, versus 702,782 total tokens including
context/tool echoes. The audited trace nodes show unsampled system, user, and
tool/context tokens masked off. Every audited trace had nonzero mean-centered
credit; this includes negative credit for invalid outcomes. This checks
recorded masking and group credit, not the internal correctness of every
optimizer kernel or whether the reward is useful.

The eight-way judge is a material source of rank noise. The two reversed
presentation views chose different top paintings in **all four accepted
groups**. In photo 431, five of ten candidate pairs reversed order between
the views (Kendall tau 0); the other accepted groups had two of ten, two of
15, and four of 15 pair reversals. Averaging the two views reduces but does
not eliminate this ambiguity. No held-out human agreement study establishes
that the eight-way ordering matches the user's aesthetic preference. The
final matched pairwise judge was more interpretable: it identified the blank
coffee canvas and the improved lizard, and its two orientation scores were
stored per case. It is still an automated developmental label, not a human
ground truth.

The model's final-action contract is also leaky. On photo 127, the last
program calls `brush.fill` but uses p5 `beginShape` after `noStroke`; its
render receipt says valid, yet the saved final image is blank white. Thus
"JavaScript ran" is not the same as "a painting exists." The RL run's
matched evaluation had 19 valid render attempts among 26, versus 21 among
27 for SFT. RL photo 089 had zero valid attempts in four turns; photo 127
first tried an undefined variable and then the nonexistent
`brush.strokeTexture` method before producing the blank final. SFT also had
syntax/API failures, so we should not ascribe all action fragility to RL.
The previous run showed the same broader pattern: invalid actions and
subject loss despite nonzero gradients. A final-image visual-emptiness check
and precise p5/p5.brush action feedback would address a demonstrated failure,
without turning validity into a substitute for beauty.

## Infrastructure and throughput

The pipeline was inference- and feedback-bound, not trainer-compute-bound.
Across 445 ten-second GPU samples from 10:10:37–11:24:38 UTC, inference GPU 0
averaged 94.7% utilization (94.6% of samples above 80%); trainer GPU 1
averaged 12.7% (12.4% above 80%). The Prime training-step MFU percentages
are unsuitable as a hardware comparison because Prime fell back to an A100
peak-FLOPS constant for the RTX PRO 6000 Blackwell. In the matched eval's
valid turns, render time had a 19.2-second median for RL and 20.3 seconds for
SFT, with observed SFT valid renders as slow as 146 seconds; each policy also
had a 180-second render deadline failure. The renderer is not the only
bottleneck, but it is not yet reliably cheap for multi-turn RL.

The two reversed eight-way judge calls for an accepted group took up to 152,
188, 166, and 25 seconds respectively, with the group waiting for the slower
view. These latency tails hold the trainer idle. Prime was allowed eight
in-flight episodes but only one reference per eight-member update, so more
inference workers can amplify stale-work loss without widening accepted
curriculum coverage. At `max_off_policy_steps=1`, Prime explicitly dropped
eight queued traces after step two and canceled two in-flight episodes; step
three then reported 16/24 discarded as stale and 63.6% canceled episodes.
These counts refer to different log windows and should not be summed into a
unique-episode total. Eight more in-flight episodes were canceled after the
final batch. The accepted update remained 8/8 trainable, but paid generation
was not converted into gradients.

Run control caused avoidable spend. The first launch was stopped before any
optimizer update because a zeroed `/proc/environ` read and a too-early audit
check were misdiagnosed as a missing reward hook. The archived group audit
proves the hook was active. On retry, a stale judge receipt made the worker
exit and required state isolation plus manual restart; the source launcher
now rejects stale state and supervises the worker, but that fix was not the
cause of reward correctness. After the fourth optimizer update and checkpoint
upload, the Prime orchestrator stayed in a zero-inflight drain loop; it was
stopped only after training completion was verified. These are lifecycle
faults, separate from the algorithm's failure to improve paintings.

## Credits, evidence, and decision

The two-GPU node was rented at $2.38/hour and cost **$6.39 total** from
creation through deletion. The 74-minute measured training window alone is
approximately $2.94 at that rate; the remaining approximately $3.45 covers
setup, the aborted attempt, evaluation, archival/upload, and other time
outside that window, not an itemized allocation. The prior campaign's
recorded GPU-spend lower bound was about $16.54, making the combined known
lower bound about **$22.93**. Since earlier records are incomplete, this is
not proof of remaining headroom under the previously stated $25 GPU ceiling.

Logged Vercel Gateway receipts total $0.06203 for both training attempts and
$0.006293 for the resumable 16-view matched judge, so **$0.068323 is a
provider-cost lower bound**, not a bill. The first gallery scorer timed out
after paid calls without per-view receipts; that unmeasured usage is excluded.
The replacement scorer saves hash-bound receipts after each orientation and
resumes without rebuying completed views. Checkpoints at steps two and four
and the full evidence archive are public and hash-verified. The node was
deleted; the latest `lium ps --format json` inventory was empty. No watcher
or scheduled task was started.

Retain SFT step 512. For the next finite run, first fix the demonstrated
empty-canvas/action-semantic error and make group scheduling guarantee actual
reference diversity, including shapes, among accepted batches. Use the
frozen SFT painting as a meaningful quality floor and calibrate a simpler,
faster pairwise judge on a small set of human-labeled comparisons before
using it to update weights. Limit queue depth or adjust policy staleness only
with measured discard and trainer-busy fractions; adding GPUs or making the
group larger under the current contract would mostly produce more costly
stale samples. Keep the sampled-token mask audit and matched four-turn
evaluation, and reject any next checkpoint that wins only by staying valid
or by looking better than weaker siblings while losing to the baseline.
