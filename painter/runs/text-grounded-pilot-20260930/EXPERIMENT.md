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
gradient norms 0.0090, 0.0126, 0.0129, 0.0120. The corresponding admitted
batch reward mean was 0.17969; first two batches averaged 0.26563 and the
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
