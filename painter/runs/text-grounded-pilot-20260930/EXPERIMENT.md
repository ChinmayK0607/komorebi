# Grounded text-paint RL corrective pilot (prelaunch)

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
