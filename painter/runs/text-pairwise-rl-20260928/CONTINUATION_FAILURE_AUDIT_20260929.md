# Text-paint continuation: live failure audit

This began as a live diagnostic and now records the **user-stopped 33-update**
continuation, before the held-out evaluation finishes. The frozen 50-update
plan and matched SFT/step-16 controls are in
[CONTINUATION_20260929.md](CONTINUATION_20260929.md). The full post-run audit
must replace the numbers below with final metrics, review the 47 same-seed
held-out paintings, and record GPU cost and public evidence hashes.

## What the current measurements support

At 27 admitted optimizer updates, the first-half mean relative reward was
0.2372 and the second-half mean 0.2381. The last five averaged 0.2625, but
that short window does not establish a trend. At least 26 recorded steps had
nonzero gradient norm; the public step-25 LoRA differs from its frozen
step-16 initializer. This confirms optimization activity, not better images.
The active judge compares each candidate only with a frozen SFT painting.
The almost-blank baker anchor `t600-412` lets a visibly weak RL patchwork
score 0.917; its rationale actually notices missing hands and dough. This is
a concrete *relative-anchor* failure. Hold visual promotion for a direct
matched comparison and stronger-anchor calibration.
The saved valid-trace mean relative score rises from 0.327 on 64 simple
canvases to 0.399 on 72 complex canvases even though the complex prompts
are harder; this is consistent with unequal opponent quality and task mix,
not evidence that the model became better over time. These are generated
trace subsets, not a matched per-prompt learning curve.

The saved generated-trace stream at the same snapshot contained 270
trajectories, including speculative or stale work. Only 88/270 had a valid
first turn, 171/270 had a valid retained final canvas, and 102/270 recovered
from an invalid turn to a later valid turn. These are not the 27 admitted
optimizer-batch denominators. Across 810 turns, 283 were valid. Thus the
multi-turn scaffold salvages many episodes, but fresh-generation validity is
poor. A sampled error audit found hallucinated brush names (`cpen`),
unsupported methods (`brush.degree`), undefined variables, malformed or
unfinished JavaScript, and 180-second render limits. The full profile stores
the per-class counts and concrete turn receipts; do not explain all failures
as renderer latency or all losses as judge error.

A later read-only snapshot of the native trace stream through 323 generated
three-turn trajectories (969 turns, including speculative work) found only
**7** calls with `finish_reason=length`; all seven used exactly 8,192 output
tokens and none yielded a valid action. The other **962** calls ended with
`finish_reason=stop`, yet 199 had a syntax-related error string and only 331
yielded a valid action. This rules out output-token truncation as the main
explanation for malformed programs at this snapshot. The three turn positions
had mean completion lengths of 2,249, 2,186, and 2,099 tokens. These are
generated-turn counts, not admitted-batch rates; verify the conclusion on the
final stream before changing the token policy.

The first 27 admitted batches averaged 283 seconds model generation,
91 seconds harness, and 20 seconds scoring per rollout; phase work overlaps,
so these are not additive wall-clock timings. Of 270 generated traces,
54 turns hit the renderer deadline; valid renders had a 19.1-second median
and 68.2-second 90th percentile. Over the sampled run window, inference GPU
0 averaged 97.5% utilization and trainer GPU 1 averaged 12.6%; the trainer
waited about 454 seconds per warning for a batch. The present bottleneck is
inference and its multi-turn/render work, not trainer arithmetic. Extra
trainer capacity alone would waste rental. [Renderer profiling](RENDER_PROFILE_20260929.md)
already found ~2× batch throughput with four software render workers but no
safe hardware-accelerated p5.brush canvas on the tested provider containers.

An experimental [read-only syntax preflight](profile_js_preflight.py) used
Playwright's bundled Node parser (`--check`, no sketch execution) on a
deterministic sample of 16 saved programs while the production renderer was
untouched. It rejected all three programs whose saved Chromium receipts
reported syntax errors and rejected none of the valid renders in that small
sample. The gross renderer work potentially avoided was only **3.8 seconds**
over those 16 cases, before accounting for parallelism. This is a useful
fail-fast/feedback improvement, not a solution to model-generation time or
the 180-second render timeouts. A full saved-program profile and an
unchanged-pixel integration test remain necessary before adopting it.

The two-order MiMo Flash judge disagreed by at least one-third on 32 of 171
saved valid judged canvases. This is a confidence/reliability warning, not
a human-validated error rate. Its receipts reported ~$0.114 for the 342
order views in this snapshot; rental and judge costs must remain separate.

## User-stopped step-33 diagnosis (2026-09-29 22:02 UTC)

The launcher received `SIGINT` and exited with code 130; the trainer and its
inference workers stopped. The latest completed public, nonprivate,
hash-verified adapter is step 30. The final 33 admitted optimizer-batch means
average **0.23643**, with first 16 at **0.23763** and the following 17 at
**0.23529**; the last batch was **0.17708** and the last-five mean **0.22708**.
The frozen native-metrics SHA-256 is
`ec5c3301eb5badb6d6a63607383ff6a1416fe67d05b7b27bd26ebd6283499f95`.
There is no positive online trend, though changing prompt mix still prevents
reading this as a matched quality curve. Step-30 versus frozen-control
evaluation is underway and remains the decisive test.

A manually inspected four-pair probe is preserved under
`local-evidence/continued-live/live-pair-review/` (source tar SHA-256
`d641b258cf39e94c5602d1a8adf961dab83a8d4f560d7891564ef9255c06f8a6`).
These are **saved generated traces, including possible speculative work**,
not a random sample of admitted updates. Their score and visible result expose
how the relative reward can fail:

| Brief | Saved candidate score vs SFT | Manual visual finding |
| --- | ---: | --- |
| `t600-333`, red panda on bough | 1.000 | The SFT anchor is fragmented; the candidate has more organic washes but the head, masked face, folded paws, and ringed tail remain ambiguous. Both are poor references for a decisive “clearly better” margin. The two-order judge called the candidate a clear, anatomically complete panda, despite noting some blurred features. |
| `t600-412`, baker shaping dough | 0.917 | The SFT anchor is nearly blank. The candidate adds a torso/workbench but no clearly cupped hands, round loaf, basket, scraper, or baked baguettes. A large relative win does not mean the painting depicts the required action. |
| `t600-576`, two buses at snowy crossroads | 0.917 | The candidate improves dark street geometry over its faint SFT anchor, but the two separate turning buses and tiny pedestrians remain hard to identify; detail and contrast dominate the impression. |
| `t600-510`, acorn and oak leaf | 0.250 | The SFT anchor preserves an acorn and lobed leaf more clearly; the lower candidate score is directionally sensible here. |

These observations do not prove the judge is always wrong. They identify a
specific promotion failure: low-quality opponents plus overconfident
subject-recognition claims can grant large relative advantages to paintings
that still omit defining content. The next reward protocol needs an
**absolute visible-subject check inside the pairwise judgment** (or an
equivalent margin cap), calibrated on these failure pairs and genuinely good
counterexamples before any new RL. Avoid adding a pile of independent
detail, geometry, and style bonuses. Weak opponents should be replaced or
excluded, with their exact IDs and hashes frozen before training.

## Changes to test only after the frozen run

1. Use a same-program offline replay set to measure syntax/API rejection
   before Chromium, pixel identity for successful renders, and the actual
   timeout/CPU savings. Report false rejection on valid programs. Never
   rewrite model code silently or change this run's renderer.
2. Compare 1 versus 2 inference workers and the existing four software
   render workers on identical saved tasks, reporting completed valid
   rollouts/hour, trainer busy fraction, and dollars per admitted update.
   Larger `max_inflight` alone cannot cure a saturated inference GPU.
3. Audit frozen SFT opponents on all 119 training briefs for near-empty or
   obviously wrong compositions, then calibrate relative comparisons against
   stronger candidates and direct human-labeled pairs. Avoid stacking
   independent detail/geometry bonuses that create new hacks.
4. Select new text/photo teacher episodes by actual visual fidelity and
   whether each subsequent turn improves the prior canvas. A saved valid
   canvas after an invalid last turn is not a successful repair. Keep weak
   episodes out of positive SFT, but retain them for recovery/negative data.
5. Read final held-out validity-inclusive and both-valid paired results by
   family before deciding whether to promote the new adapter or run more RL.
   Online reward alone cannot make this decision.

At the time of this live audit, no separate training run was authorized;
the later user request stopped this run and authorized a corrective run.

## Final frozen training trace and root-cause assessment

The user stopped training after 33 admitted optimizer updates on 2026-09-29
22:02 UTC. The statements in the preceding pre-stop section are historical;
the new request authorizes a separate, finite corrective run after evaluation
and pod release. The full frozen profile is
`evidence/profile-continuation.json`; the generated-completion audit is
`evidence/completion-failure-profile.json`. Both are preserved for the public
evidence archive. The step-30 checkpoint is the latest completed public
adapter; steps 31–33 changed the live optimizer but have no promoted adapter.

All 33 admitted batches had nonzero gradient norms (mean 0.01482) at fixed
LoRA LR 1e-6; mean mismatch KL was 0.000276 and 16 off-policy rollouts were
dropped. Mean admitted-batch reward was 0.23643; the first 16 averaged
0.23763 and the next 17 averaged 0.23529. The last five averaged 0.22708.
This supplies no upward online trend. The model is being updated, so the
failure is not a missing-gradient bug. The sequential curriculum changes
from simple to complex, so these means are not a matched generalization test.

The native saved stream includes 339 generated trajectories and 1,017 turns,
including speculative/stale work. Only 107/339 first turns and 212/339 final
canvases were valid. Invalid-turn error categories are **not mutually
exclusive**: 209 syntax, 97 undefined symbol, 87 unknown brush name, 70
render timeout, 53 type, 43 unsupported brush API, 14 invalid color, 7 blank
or no canvas, and 3 premature finishes were counted. Only 7/1,017 generation
calls stopped at the 8,192-token ceiling; the other 1,010 stopped normally.
Thus simply raising max output tokens would address very few failures.
Multi-turn revision recovers some invalid first attempts, but it spends
inference time and rendering on a high rate of malformed programs.

The 212 saved two-order judge receipts, again including speculative work,
have a mean score of 0.35653 and mean order gap of 0.125; 39/212 have an
order gap of at least one-third. Their provider-reported cost is $0.13965
across 424 views, mean view latency 24.23 seconds. This is the recorded
judge cost for saved receipts, not an independent billing statement and not
the full cost of all provider attempts. The manual four-pair probe above
shows that high scores can be awarded for weak paintings when their SFT
opponent is still worse. The direction of the acorn/leaf low-score example
was sensible; this is a calibration and anchor-selection failure, not proof
that every judgment is inverted.

The full 10-second GPU telemetry window covers 1,948 samples per card from
2026-09-29 16:38:11 to 22:02:47 UTC (19,476 seconds). Inference GPU 0
averaged **98.04%** utilization, with 97.95% of samples at or above 90%; the
trainer GPU 1 averaged **12.37%**, with only 11.55% of samples at or above
90%. Thirty-one trainer wait warnings averaged 482.6 seconds. Admitted-batch
phase means were 300.2 seconds model generation, 96.2 seconds harness, and
19.6 seconds scoring; phases overlap and should not be summed into wall time.
The optimizer step mean was 565.4 seconds. The pipeline is inference-bound,
not saturated as a two-GPU training system. Larger trainer GPUs or just more
in-flight requests on that saturated inference card would not fix this.

The low changed-pixel-coverage screen found seven suspicious frozen training
anchors below 5% of their background color. All seven were visually checked:
`t600-412` (baker) is nearly blank and `t600-580` (tidepool) lacks the
requested coherent subject; both are unfit opponents. `t600-529` (padlock and
key), `t600-442` (mantis), `t600-429` (deck chair), `t600-540`
(shuttlecock), and `t600-541` (pencil/eraser) remain at least identifiable
despite sparse or faint marks. Automatic coverage exclusion would therefore
reject some usable simple paintings. The red-panda and snowy-crossroads
anchors in the four-pair probe are further weak opponents that require
manual curation or replacement before reuse.

The leading failure modes are (1) unreliable JavaScript/brush API generation,
causing expensive unproductive turns; (2) relative reward against uneven
frozen SFT opponents, compounded by subject hallucination and judge order
variance; and (3) inference/render latency leaving the trainer idle. The
fixed 47-case held-out comparison is still needed to establish whether the
step-30 paintings improved despite these flaws. The follow-up should freeze
a manually curated opponent set, make visible subject/action fidelity an
explicit condition for a *large* pairwise margin, and test a short stable
curriculum against the same held-out cases at multiple checkpoints.

### Grounded-margin calibration and curated opponent pool

The proposed [grounded pairwise protocol](../../text_pairwise_grounded_judge.py)
was tested on the **same four frozen saved pairs**, before using it for any
training. It still makes two image-order calls to MiMo-v2.6-Flash. Each call
must describe visible evidence and mark the candidate's defining-content
fidelity as `clear`, `partial`, or `missing`. A partial candidate's single
pairwise score is capped at 2/3; a missing candidate is capped at 1/2. This
is a cap on preference strength, not another additive quality metric. The
full model responses and image/prompt hashes are retained in ignored
`local-evidence/continued-live/grounded-calibration.json` for publication
with the corrective-run evidence.

| Saved pair | Old score | Grounded raw | Grounded after cap | Manual interpretation |
| --- | ---: | ---: | ---: | --- |
| red panda `t600-333` | 1.000 | 0.750 | 0.667 | Both views called the candidate partial; high margin was reduced. |
| baker `t600-412` | 0.917 | 0.833 | 0.667 | Both views called the candidate partial; high margin was reduced. |
| snowy crossroads `t600-576` | 0.917 | 0.833 | 0.833 | **Still overconfident:** both views called it clear despite ambiguous buses/pedestrians. |
| acorn/leaf `t600-510` | 0.250 | 0.167 | 0.167 | Lower candidate score agrees with manual preference for SFT. |

This is a small, selected calibration set, **not** a measured judge accuracy
rate. It demonstrates that the cap catches two failures and misses one; the
remaining ambiguity must not be hidden. The next pilot excludes the known
weak `t600-333`, `t600-412`, `t600-576`, and `t600-580` opponents instead of
assuming the new prompt alone cures them. A positive-control swap on
`t600-510`, putting the recognizable SFT acorn/leaf
painting in the candidate position, scored **0.833** with both order views
calling it `clear`; the cap therefore does not suppress this good example.
The full receipt is in ignored
`local-evidence/continued-live/grounded-positive-control.json`. The manually
reviewed pilot pool has 12 simple, 10 intermediate and 18 complex briefs interleaved
in a fixed order. The [source generator](curate_grounded_pool.py) verifies
every brief and anchor hash; its path-independent content SHA-256 is
`13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`.
On the source node its path-dependent manifest SHA-256 was
`d005b1d86277c1db92107f555c6ff143710501c72bea5adaf8668f7f4b704693`
and the 119-case parent SHA-256 was
`f32ac68bb524321d8f96271c28ef8fc227630276d56c49d594bf62417ab73923`.
These are visual-review selections, not gold-standard paintings. A pilot
must still report individual failures and same-seed held-out quality.

The source 509 MB gzip evidence archive also exposed a setup bottleneck:
the old rehydrator repeatedly sought into a compressed tar while verifying
182 canvases, consuming over three minutes of one CPU and remaining
unfinished. The [single-pass rehydrator](rehydrate_scaled_anchors.py) streams
the pinned archive once, still checks the archive digest and every extracted
brief/canvas digest, and completed a fresh Linux-node rehydration. Its output
contained the expected 182 valid anchors, 119 train, 47 validation and 16
reserve cases; the curated 40 content hash matched the value above. This
changes setup time, not the old run's data or reward.

The corrective-run Git bundle SHA-256
`50b2a3d74355fac916b4ed5f5621a839fed9c0d9c70c36f544558aa6a73c2e1c`
was transferred and independently verified on the old Linux node. A
**non-training** preflight built the grounded config with pinned Prime,
verified two GPU roles, and round-tripped the first RL task through the
installed Verifiers serializer (`t600-510`, 31 wire fields). The tested
initializer was public RL step 16; the actual pilot initializer is selected
after held-out comparison. The preflight used the already installed renderer
runtime and started no training, telemetry or GPU server. The live package
install was restored to the old repository afterward; the active held-out
evaluation uses its own pinned process and remains unchanged.

## Partial matched visual inspection (not a final result)

While the frozen step-30 evaluation was still rendering, 28 of the 47
predeclared cases had completed. On those *same* case IDs, step 30 produced
25 valid canvases, versus 22 for SFT step 512 and 23 for RL step 16. These
are partial validity counts, not a preference or quality comparison; the
remaining 19 cases and the blinded scoring may change the conclusion.

The saved `t600-332` cuttlefish is more coherent at step 30 than at SFT or
step 16, but still reads as a spotted jellyfish because its fin, eye, and
overall anatomy are missing. On `t600-309` red combine harvester, step 16
has a legible machine and cleaner composition, whereas step 30 is diffuse
and muddy. SFT is coarser but also recognizable. These two inspected cases
show why higher program validity cannot be promoted as visual improvement;
the full 47-case comparison must govern checkpoint selection. The local
render evidence is under the ignored `local-evidence/continued-live/partial-step30-spotcheck/`
directory; the public archive will carry the underlying matched renders.

## Final matched evaluation, publication, and release (2026-09-29 23:40 UTC)

All **47 predeclared text-to-paint held-out briefs** completed under the
same three-turn, 8,192-output-token, seed-918210, 600×600 p5.brush
contract. There were **zero censored cases**. Frozen SFT step 512 and RL
step 16 each rendered 41/47 valid canvases; step 30 rendered 43/47. In the
original SFT-anchored two-order MiMo Flash protocol, step 30 scored a mean
0.461 (18 wins, 20 losses, 9 ties against the SFT anchor). The paired
step-30 minus step-16 anchor-score delta was +0.0496 across all 47, with
a bootstrap 95% interval of **[-0.0443, +0.1436]**. On the **38 both-valid**
step-16/step-30 cases, the delta was **-0.0022**, with interval
[-0.0877, +0.0811]. Thus the anchor-score increase is principally a
validity/recovery gain, not demonstrated quality gain among paintings that
both policies could render. Compared with SFT anchor scores, the all-case
step-30 delta was +0.0656, interval [-0.0337, +0.1596].

Fresh *direct* candidate-vs-control paintings were also judged in both
image orders, with invalid outputs counted instead of dropped. Step 30
beat SFT in 28 cases, lost 16, tied 3: mean 0.5798, bootstrap interval
[0.4929, 0.6649], 15/47 material order disagreements. Against RL step 16,
it won 25, lost 15, tied 7: mean 0.5496, interval [0.4663, 0.6294],
21/47 material order disagreements. Both intervals include an effective
tie at 0.5. The direct results are suggestive, **not a conclusive visual
promotion**, especially given the previously demonstrated judge hallucination
and order sensitivity.

Manual inspection of the saved 43 valid triplets found real gains and
regressions. The step-30 octopus (`t600-377`) has more legible tentacles and
underwater setting than SFT, though its limbs are segmented and mechanical.
The twin-bell clock (`t600-439`) is much more recognizable than SFT. By
contrast, the red combine harvester (`t600-309`) loses the step-16 machine's
silhouette in a muddy wash. The table lamp (`t600-422`) becomes a broad,
less coherent block than the SFT shade and base; the hand fan (`t600-516`)
spills spokes and decorative triangles outside its intended arc; the
sailboat (`t600-560`) weakens the SFT's clear hull and sail relationship.
These are selected visible examples, not a new numerical human score. The
[local full gallery](local-evidence/continued-live/continued-gallery/index.html)
has all 47 cases, including invalid-output placeholders; its source tar was
copied locally with SHA-256
`f965f697a6e7f45ed75596bb257e5a7c29e0a23135826d265735dc5be9f40c9b`.

The [public, anonymously hash-verified evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/1cc58226c67fee23542a427d4ec937b3df1012c2/runs/text-pairwise-rl-20260928/continued-step-30-all-20260929.tar.gz)
is 651,541,297 bytes; its LFS SHA-256 is
`4bd5365d3123f53d02862b61b899ead9bcea157df77b93c22cfe4e59d359f801`.
It contains the frozen native metrics, raw rollout and judge receipts,
completion/GPU profiles, matched scores and renders, and gallery. The
[step-30 public adapter](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/103663c9d0fa61cf1476435116966a2c9e68dcb4/text-pairwise-continue-20260929/step_30)
has adapter-model LFS SHA-256
`4e89d6165227f189f196134d3ce4286ea9ec60b24e6fd0ca28e47357eecb12a9`.
The old run's preparation source commit was
`ed234af61fd475c5bbef849fe90bd068fa7b820e`; its immutable config and
curriculum files are in the evidence archive. The last Lium spend read was
**$17.31** at $2.38/hour, below the $25 GPU ceiling; the final invoice is
not available. Saved judge receipts report $0.13965 for 424 training-order
views, not total provider billing. The exact old pod was deleted after
public verification, and Lium returned an empty active-pod list.

**Next decision:** use public RL step 16 as the corrective pilot initializer,
because step 30 improves validity without a demonstrated both-valid quality
gain and visibly regresses on several structural briefs. Freeze step 16,
step 30, and SFT as matched controls. Test the reviewed 40-opponent pool and
grounded margin protocol in a separate finite run; do not claim that the
protocol is reliable until intermediate and final same-seed held-out
paintings improve without new boundary or muddling failures.
