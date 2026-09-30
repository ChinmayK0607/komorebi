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
fix `71508fbaacd3845a567b810ed1b14effdab80e2d`. It has **not yet** run.

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
the primary
invalid-turn classes were syntax (89), undefined symbol (32), unknown brush
(27), type error (23), unsupported brush API (18), and render timeout (14).
Those counts are turn-level and should not be called admitted-batch failures.
The valid renders had median elapsed 16.17s and p90 59.89s. This establishes
that an 8,192-token cap was not the dominant failure cause in this window;
program validity and inference latency were. The final-run profile must
repeat these statistics with its complete denominators.

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
