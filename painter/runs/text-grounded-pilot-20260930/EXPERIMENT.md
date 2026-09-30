# Grounded text-paint RL corrective pilot

## Hypothesis and controls

The 33-update continuation changed weights but had no upward online reward
trend. A manually inspected failure set found large relative wins against
unrecognizable SFT opponents. This pilot tests whether a fixed, mixed pool of
visually recognizable frozen SFT anchors and a visible-subject pairwise margin
cap produce a rising **same-seed held-out** preference/validity trajectory
without sacrificing the best existing paintings. The 40 training briefs have
12 simple, 10 intermediate, and 18 complex tasks, interleaved to avoid a
simple-to-complex reward shift. Frozen validation is the previous 47-case
split, with a prespecified balanced 16-case subset for intermediate review.

The matched controls are public SFT step 512, public RL step 16, and public
RL step 30, evaluated under the same three-turn, 8,192-token, seed-918210,
600×600 p5.brush contract. The initializer is selected only after the old
step-30 matched result is read; its exact public revision, prefix and LFS
SHA-256 will be written to the launch receipt. Do not treat online reward as
validation. The old run's 47-case output and the new pilot's 16/47-case
outputs must be compared validity-inclusively and visually, with censored
renders reported, not dropped.

The prespecified file named `balanced16-validation-manifest.json` contains
8 intermediate and 8 complex briefs, **no simple briefs**. Its name is
misleading: use it as a fixed *harder-case* midpoint panel, not as a balanced
sample of all three training families. The full 47-case validation contains
8 simple, 8 intermediate, and 31 complex briefs; report final results by
family before making any generalization claim. The fixed subset hash and
contents must remain unchanged during this pilot.

The completed 47-case comparison selected **RL step 16** as initializer:
`CK0607/qwen3.8-27b-brush-painting@ac4d3eda8e5925e2d23992cfdc1a63620ebde573`,
prefix `text-pairwise-scaled-20260929/step_16`, adapter LFS SHA-256
`e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c`.
Step 30 improved valid-canvas count from 41/47 to 43/47 but did not improve
the 38 both-valid paintings' SFT-anchored scores (paired delta −0.0022);
direct step-30 versus step-16 preference had a 95% bootstrap interval
crossing 0.5 and conspicuous muddy/structural regressions. The [final old-run
audit](../text-pairwise-rl-20260928/CONTINUATION_FAILURE_AUDIT_20260929.md)
records the full metrics, public hashes and visual examples.

## Immutable preparation

- Base `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Prime `26b3131d2716a4f8210b165df584b83b4bc54f61`.
- Source archive `scaled-final-20260929.tar.gz` public SHA-256
  `ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3`.
- Curated 40-row manifest **content** SHA-256 (ordered IDs, brief hashes,
  opponent hashes, and families; independent of node paths)
  `13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`.
  The first node's path-dependent manifest SHA-256 was
  `d005b1d86277c1db92107f555c6ff143710501c72bea5adaf8668f7f4b704693`;
  the new node will record its own path-dependent hash after rehydration.
- Reward protocol `text-paint-sft-pairwise-grounded-v1`, two-order
  `xiaomi/mimo-v2.6-flash` via Vercel AI Gateway. The selected four-pair
  calibration and its unresolved snowy-crossroads error are in the
  [failure audit](../text-pairwise-rl-20260928/CONTINUATION_FAILURE_AUDIT_20260929.md).

## Finite execution and decision

Use one inference and one trainer GPU on a two-card node, 24 optimizer
updates maximum, batch 8, group 4, three turns, 8,192 tokens, LoRA LR
`1e-6`, fixed 8/8 in-flight window, public checkpoints every four updates.
The actual node price and removal deadline must keep rental below the prior
**$25 per-run GPU ceiling**, leaving three hours for matched evaluation,
hash-verified public evidence and pod removal. Vercel charges are separate.
No watcher or recurring task is created.

Inspect the frozen balanced-16 subset at the midpoint and the same subset
plus the full 47 at the final checkpoint. Report online reward and validity
curves alongside matched held-out preference and visual examples. An upward
trend is supported only if the same held-out tasks improve at later
checkpoints, validity does not collapse, and manual visual inspection does
not reveal new reward hacks. Otherwise do not scale this protocol.

Record the final config hash, source commit, model revision, adapter hash,
curriculum hash, phase timings, GPU sampling window and denominators,
provider-reported judge cost, actual GPU rental, failed/censored cases,
public archive/checkpoint hashes, and next decision after the run.

## Live execution record (not a held-out result)

The old pod was removed after the old-run archive was anonymously hash-verified.
The new two-card RTX PRO 6000 Blackwell rental began 2026-09-29 23:45 UTC at
$2.38/hour and has a finite removal deadline of 2026-09-30 09:45:03 UTC.
The actual training source is `a965832148f6cbbdbd32588a6bbe054b28da64d5`;
three pre-optimizer setup defects and their source fixes are documented in the
[research index](../../../RESEARCH_LOG.md). The node's 40-row path-dependent
manifest SHA-256 is
`caa4ef000d9fdd514f31b9a1d7f2e741896904a0a6e657d3ece61fe64744dc45`.
The final Prime config SHA-256 is
`50ddb71b2947c8941ddbe54f3aea9a80ecdab5173eac62e5562e55c1036902ec`;
the frozen 16-case subset SHA-256 is
`0a16138115a44dfa81fd01bfd72e61e6ac5691f6ebecc4aea713c17e0b0bcdb7`.

The first four optimizer updates completed by 00:47:54 UTC with nonzero
gradient norms 0.0090, 0.0126, 0.0129, 0.0120. Their **effective-batch**
reward mean was 0.17969; first two batches averaged 0.26563 and the
next two 0.09375. This is neither an upward trend nor a matched visual
result. Trainer logs reported batch waits of 6m03s, 7m12s and 10m54s on
steps 2–4; the inference card is the throughput bottleneck. An allocator
mapping warning occurred before step 3 but its update succeeded; a real
OOM/crash has not been observed. The public step-4 checkpoint was verified at
HF revision `7bbe79e1945377ccd6ad1cfe38817d21112a6158`, adapter SHA-256
`1cd51bb44a84d96a3562050a356d854bc71544401d3be0259ea4a84f8401d112`.
These are interim facts. Preserve the final raw metrics and interval telemetry
before making any utilization or quality claim.

The frozen prior-control archive SHA-256
`4bd5365d3123f53d02862b61b899ead9bcea157df77b93c22cfe4e59d359f801`
was anonymously verified on the new node. It yielded both control summaries
and 82 valid SFT/step-16 paintings for direct two-order comparison. A finite
post-training [finalizer](finalize_pilot.sh) is prepared from source commit
`48606cb6488b7fd909aacae4f46019d83d4b9892` plus verifier-source archive
fix `71508fbaacd3845a567b810ed1b14effdab80e2d`. At this preparation
snapshot it had not yet run; its completed result and packaging repairs are
recorded below.

At the 01:14 UTC live snapshot, step 7 had completed with nonzero gradient
norm 0.0073 despite one CUDA allocator *mapping warning* just before it. The
warning is not a fatal OOM: step 7 succeeded, and the process and rollout
coordinator continued. Its batch took 12m07s, including 9m32s waiting for
rollouts and 2m34s of active update. Seven unpaired **all-completed** rewards so far are
0.19444, 0.29167, 0.11111, 0.14063, 0.54167, 0.20139, 0.12500. This is
high variance and not a general upward trend.

There were 57 saved two-order visual-judge receipts in this speculative
snapshot, which is **not** the admitted optimizer-batch denominator. The
candidate subject was graded clear in both orders on 8, partial in both on
38, and missing in both on 3; the other 8 had mixed category views. The
curated anchor was clear in both orders on 25, partial in both on 20, and
mixed on 12. Candidate margin caps were 1.0 on 8 receipts, 2/3 on 44, and
1/2 on 5. Mean capped receipt score was 0.3304 and 10/57 order gaps were at
least one third. The cap does visibly restrict many incomplete candidates,
but manual contact sheets still show some weak anchors (e.g. `t600-429` chair
frame and `t600-444` T-handle corkscrew), and three early `t600-429` candidate
canvases do not consistently preserve chair geometry. The 47 held-out cases
and direct controls remain required before any quality decision.

At the separate step-10 profile, the **effective optimizer batches** had
10/10 nonzero gradients (mean norm 0.01330), learning rate fixed at `1e-6`,
mean output 5,807 tokens per three-turn trajectory, p90 7,873, and zero
generation-truncated fraction. Mean reported phase times per batch were
257s model inference, 58s harness, and 29s scoring; these phases can overlap,
so they are not a wall-clock decomposition. This points to model generation
and invalid-code retries ahead of judging as the main throughput problem.
The separate **125 saved generated traces** include speculative and stale
work: 80 had valid final canvases, only 47 had a valid first turn, and 42
recovered after an invalid turn. Another 41 traces had a valid turn followed
by an invalid turn. Because the current paint action rewrites the complete
sketch each time, a revision can lose validity even after the model has made
a renderable painting; this is an observed failure mode, not proof that
incremental patches would improve visual quality. Across 375 generated turns,
the primary invalid-turn classes were syntax (89), undefined symbol (32),
unknown brush (27), type error (23), unsupported brush API (18), and render
timeout (14).
Those counts are turn-level and should not be called admitted-batch failures.
The valid renders had median elapsed 16.17s and p90 59.89s. This establishes
that an 8,192-token cap was not the dominant failure cause in this window;
program validity and inference latency were. The final-run profile must
repeat these statistics with its complete denominators.

The node's pinned model `config.json` reports `model_type=qwen3_5` and
`Qwen3_5ForConditionalGeneration`; Prime's resolved orchestrator renderer is
`qwen3.5`. Saved trace nodes contain the intended p5.brush system prompt and
the text brief. This rules out an obvious model-family renderer mismatch in
this run, though it does not prove every chat-template or token-mask detail.
The observed invalid programs include hallucinated brush names such as
`cwash`, consistent with code/API errors rather than a missing brief.

An independent CPU read of the published step-12 adapter against its
hash-verified RL-step-16 initializer found matching shapes and keys for
992/992 tensors, with nonzero changes in 992/992. The Frobenius norm of the
delta was 0.022998 versus initializer norm 51.732831 (relative 0.0004446).
This rules out a no-op checkpoint, while the small relative movement does
not itself establish a visually meaningful update. Repeat at the final
checkpoint and pair with held-out images before judging the learning rate.

The step-16 speculative receipt audit found 9/68 generated judged canvases
graded `clear` in both image orders for dispatch steps 1–8, versus 8/62
for dispatch steps 9–16. The other 59/68 and 54/62 had at least one
`partial` or `missing` view, respectively; large order gaps occurred on
13/68 and 12/62. This is **not** an admitted-batch or held-out comparison:
the generated work includes speculation, repeated briefs, and changing task
mix. It is nevertheless an early warning that the new visible-subject cap
has not yet increased clear subject depictions among saved judged rollouts.

## Training completion and fixed evaluation

The finite trainer completed all 24 planned optimizer updates normally at
2026-09-30 03:38:46 UTC. The step-24 adapter is public and hash-verified at
`CK0607/qwen3.8-27b-brush-painting@7d201165315595f2c611b355e6ba615f1067c75f`,
prefix `text-grounded-pilot-20260930/step_24`, with LFS SHA-256
`1397e30d2f2a802640c4aebec110eb657650b430d8591863e4831aa2a65aa5d4`.
The public step-12 midpoint is revision
`0c3dd598cf7ae87c0dbfcac2e953883c00f37a61`. The one-shot finalizer
started at approximately 03:40 UTC and is producing the prespecified
same-seed held-out 16- and 47-case comparisons. The successive eight-update
blocks of effective optimizer-batch reward averaged 0.22005, 0.23958, and
0.22135. The middle bump did not persist; these unpaired training batches
have different prompt samples and cannot establish improvement. Final matched
results, complete-run failure denominators,
utilization, and release receipts belong below after the evidence archive is
publicly verified. The rental snapshot at about 03:48 UTC was $9.62; this is
not the final billed amount.

The public step-24 adapter differs from its hash-verified RL-step-16
initializer in 992/992 tensors. The relative Frobenius delta is
0.00057235, measured by `verify_adapter_delta.py` with finite-value and
shape checks; the full JSON receipt is `evidence/adapter-delta.json` in the
run archive. This is evidence of nonzero weight updates, not visual progress.

The prespecified harder-case midpoint panel has completed: step 12 produced
15/16 valid final canvases, zero censored cases, and scored 5 wins and 11
losses against frozen SFT opponents under the grounded two-order protocol
(mean candidate score 0.359375). This panel has eight intermediate and eight
complex briefs, no simple briefs. The result is an early warning, not a
step-24 comparison or a general-population estimate. The same 16 prompts
must be scored at step 24 before judging a training trajectory.
Of the 15 actual midpoint image-pair receipts, 13 capped the candidate for
`partial` or `missing` defining content, and one had an image-order score
gap of at least one third. The sixteenth scene was model-invalid and scored
zero without an image-pair call. These are the held-out midpoint denominators,
distinct from the speculative training receipt counts above.

A direct ID comparison of the frozen manifests confirmed 40 distinct
training briefs, 47 distinct held-out briefs, and zero overlapping IDs.

I visually inspected five selected midpoint pairs against the exact anchor
images used by this judge, including three losses and two wins. The losses
are substantive: `t600-423` loses the chameleon's curled tail and connected
head/crest; `t600-425` breaks the seahorse's upright S silhouette into
separate pieces and omits the curled tail; `t600-426` enlarges the stapler but
muddies its jaw and collapses the two offset paper sheets into ragged white
shapes. The ocarina win `t600-445` is also concrete: the new painting joins
the open mouthpiece to the body and shows the edge hole that the anchor lacks.
The library win `t600-339` adds a more legible building and cantilevered
slabs, while retaining rather mechanical horizontal edges. This selected
spot-check supports some midpoint judge directions but is not a measured
judge accuracy rate or a substitute for reviewing step 24. The five source
canvases and two-order receipts are retained in the node evidence archive;
local spot-check copies are ignored by Git.

The complete training-only GPU telemetry file ends at 03:39:11 UTC, before
held-out evaluation. It contains 1,325 ten-second samples **per card** from
2026-09-29 23:58:27 to 2026-09-30 03:39:11 UTC: inference GPU 0 averaged
97.01% utilization and was at least 90% busy on 96.83% of samples; trainer
GPU 1 averaged 12.57% and was at least 90% busy on 11.40%. Restricting to
the active 00:04:07–03:38:41 UTC window (1,288 samples per card) yields
99.60% / 12.92% mean utilization and 99.46% / 11.72% of samples at or above
90%, inference/trainer respectively. These are device utilization samples,
not delivered training throughput; the inference card was saturated while
the trainer waited for admissible rollouts. The evaluator is a separate
later period and is excluded from both windows.

During the final 47-case evaluation, a single `nvidia-smi` snapshot showed
GPU 0 at 100% and GPU 1 at 0% (the trainer had exited). This is **only a
snapshot**, not an evaluation-window utilization estimate. The current
finalizer evaluates midpoint and final checkpoints serially on GPU 0, leaving
the second rented card unused for most evaluation work. A future run should
test parallel, nonoverlapping evaluation shards or concurrent checkpoint
servers on separate cards, with unchanged seeds/render contract and a
measured cost-per-comparable-painting benefit before adopting the change.

## Final held-out result, failures, cost, and decision

The final fixed 47-case text-only evaluation completed at step 24. The
grounded SFT-anchor reward had 46 comparable cases, one censored judge
failure (`t600-309`), 40 valid candidate canvases among those 46, and 20
wins, 2 ties, 24 losses: mean 0.42391. This is not an improvement over the
anchor. By family, its 30 comparable complex cases averaged 0.42222 (13
wins, 1 tie, 16 losses; one further complex case censored); eight
intermediate averaged 0.39583 (3/1/4); eight simple averaged 0.45833
(4/0/4). Five complex and one simple case were model-invalid and scored
zero. The canvas collector found 41/47 valid final canvases because the
censored case had a canvas but no admissible anchor-judge score. Do not merge
these denominators.

The separate, two-order, validity-inclusive **direct visual** comparison
of step 24 against frozen SFT yielded 23 wins, 6 ties, 17 losses over 46
comparable cases (one censored), mean 0.50543 and 95% bootstrap interval
[0.42029, 0.59239]. Against its own RL-step-16 initializer it yielded 22
wins, 7 ties, 18 losses over all 47, mean 0.49113 and interval
[0.40248, 0.57801]. The prespecified harder 16-case midpoint comparison
of step 24 versus step 12 gave 7 wins, 3 ties, 6 losses, mean 0.52083 and
interval [0.40104, 0.64063]. Every interval spans 0.5; no comparison
establishes a reliable gain. The step-24 versus step-16 direct means were
0.5242 on 31 complex briefs, 0.3542 on eight intermediate, and 0.5000 on
eight simple; the intermediate regression is an explicit warning. Family
counts are small and not independent evidence of generalization.

The effective optimizer-batch reward averaged 0.22005, 0.23958, and
0.22135 across consecutive eight-update blocks; its last five averaged
0.19167. All 24 updates had nonzero gradients (mean norm 0.01231), and the
final adapter changed all 992 LoRA tensors versus the initializer (relative
Frobenius delta 0.00057235). This excludes a no-op update but cannot turn
the nonrising online curve into held-out visual progress. The complete
saved-trace audit has 257 generated trajectories, including speculative or
stale work: 182/257 valid final canvases (70.8%), only 104/257 valid first
turns (40.5%), 94 recovered after an invalid turn, and 88 had a valid turn
followed by an invalid one. Across 770 turns, error-class flags included
149 syntax, 72 undefined-symbol, 50 unknown-brush, 45 type, 36 unsupported
brush-API, and 32 render-timeout cases. These overlapping turn flags must
not be added as distinct failed trajectories. Valid render receipt time was
median 15.65 seconds and p90 65.91 seconds (426 receipts). Mean model
inference phase was 265 seconds versus 71 harness and 34 scoring seconds
per admitted batch; phases overlap. The 23 logged trainer waits averaged
432 seconds. This identifies generation, program validity, and slow renders
as throughput bottlenecks ahead of the judge API.

The saved 180 two-order training judge receipts (speculative-inclusive)
had 35 image-order gaps of at least one third. Their provider-reported
judge cost was $0.1744 for 360 views; this is **only** the training-receipt
subset, not the total Vercel bill. The serial held-out judges and parallel
teacher generation also used Vercel; actual billed totals were not exposed
by these receipts and are not invented here. The two-GPU Lium rental ran
from approximately 2026-09-29 23:45 UTC through node deletion on
2026-09-30, at $2.38/hour for the whole node. Its final displayed subtotal
was **$14.62**, under the $25 GPU ceiling. About $9.3 accrued before the
03:39 UTC training exit; about $5.3 more accrued during serial midpoint,
47-case inference, API comparisons, and archive work. Those phase amounts
are approximations from clock time, not separately metered charges. The
training-window telemetry above shows a saturated inference card and a
lightly used trainer; evaluation then occupied only one rented GPU and
later performed CPU/API-only judging while both cards were still billed.
The prior, separately closed 33-update rental was $17.31, so the two recent
GPU runs alone cost about $31.93, excluding Vercel and earlier runs.

Manual review of selected final cases confirms that the mixed statistics
hide meaningful regressions and wins. In `t600-342` beach cabin, step 24
loses the readable cabin and lets the turquoise sea cut across its walls;
the SFT and RL-step-16 versions preserve more of the building. In
`t600-423` chameleon, step 24 breaks the head/body/tail into fragments
and loses the coiled tail, despite a valid canvas. In `t600-466` mountain
train, step 24 adds a coherent red train, repeated viaduct piers, trees,
and river where earlier versions were sparse; it still looks fairly
mechanical. In `t600-323` polar bears, step 24 is preferred to a very weak
anchor because its aurora and two bear outlines are present, yet the animals
remain thin wireframes. Thus a relative win against a weak anchor is not
an absolute-quality pass. These four examples were selected for diagnosis,
not sampled to estimate an error rate. The complete local
[47-case gallery](local-evidence/gallery/index.html) and public archive
allow further review.

Finalization preserved completed scores and **did not repeat paid calls**
after two packaging faults: missing `zstandard` in the pinned Python
environment and inclusion of evaluator `cache/` files in the screened
archive. The dependency was installed on the node; `archive_pilot.py` now
excludes generated caches, and all Python heredocs in the finalizer compile.
The 3,421-file, 454,805,809-byte evidence archive was anonymously
hash-verified at public HF dataset revision
`ffdcc865cf46495d72ca4215c15866ddb077bebf`, path
`runs/text-grounded-pilot-20260930/grounded-pilot-step-24-20260930.tar.gz`,
SHA-256 `e34870e580a74fcf621268e10f5f45c54b0a91060c9b7c74bf4fab8cb23282c4`.
The step-12 and step-24 adapters remain public at the revisions above.
The gallery, summary, profile, and curve were also copied to ignored local
`local-evidence/`. The owned pod was removed and `lium ps` returned an
empty inventory. No scheduled watcher was created.

**Decision:** do not scale this reward/training recipe yet. Keep the frozen
SFT and RL-step-16 checkpoints as controls. For the next *proposed* run,
first make the policy produce valid p5.brush programs more often, avoid
replacing a valid canvas with an invalid full rewrite, and introduce an
absolute subject/structure floor so wins over very weak anchors do not
masquerade as good paintings. Then test a fixed, family-balanced held-out
trajectory with direct pairwise judgments and manual spot checks before
buying a longer GPU run. To reduce spend, batch or shard inference
evaluation across available cards and move post-render API judging and
archiving to a CPU-only environment; measure cost per comparable valid
painting, rather than assuming high utilization of one card means an
efficient two-card rental. This is a research direction, not a launched run.
