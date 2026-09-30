# SFT-512 initialization with consistent joint-view RL

## Hypothesis and matched comparison

Initialize directly from the publicly hash-verified SFT step-512 adapter.
Test whether conservative RL under the revised aesthetic/readability prompt
and identity-keyed, balanced pairwise supervision improves held-out painting
quality relative to that same SFT checkpoint. This avoids inheriting the
previous RL-step-16 judge's optimization history. Keep the step-512 canvases
as immutable per-prompt reward anchors. A rising admitted-batch reward alone
is not a held-out result or a guarantee of better aesthetics.

Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Initializer: `CK0607/qwen3.8-27b-brush-painting@adf687e5b321514eda7be6eab050434fdc8e44ff`,
prefix `photo-curriculum-sft-20260922-balanced/step_512`, adapter SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Prime: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
Judge: `text-paint-sft-pairwise-consistent-v1`, source `1b9ac1f`,
`xiaomi/mimo-v2.6-flash` through Vercel AI Gateway. The provider exposes no
immutable weight revision. Both paintings are always presented together.

The existing interleaved 40-brief training split (12 simple, 10 intermediate,
18 complex) and 47 held-out briefs remain fixed. Training semantic manifest
SHA-256: `13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501`.
The source public archive SHA-256 is
`ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3`.
No train brief is drawn from the selected frozen judge-calibration panel.
The 47 held-out cases contain 8 simple, 8 intermediate and 31 complex briefs;
report by family and include failed/censored outcomes. Original SFT paintings
are reused with their bytes/hashes, not regenerated as a moving opponent.

## Finite execution limits

Maximum 50 optimizer updates, LoRA LR `1e-6`, three visual-feedback turns,
8,192 completion tokens per trajectory, group eight and batch sixteen
(two briefs per admitted optimizer update). GPU topology is selected after
availability, memory and whole-node price checks; do not assume inexpensive
24/32/48-GB cards fit independent BF16 trainer and inference roles. Public
checkpoints are planned every four updates, with finite hash-verification
and retained raw rollouts, renders, metrics, gradients and judge receipts.

The user's hard ceiling is **$20 total GPU rental per run**, including setup,
evaluation and evidence preservation; Vercel cost is separate and logged.
Register actual rental start, hourly price and training/removal deadlines
before launch. Use a finite process supervisor, not a scheduled Codex task,
recurring automation or heartbeat. No node has been rented at registration.

At updates 15, 20 and then every five updates through 50, compare the latest
five effective optimizer-batch reward means with the preceding five under
the fixed sampler. If the latest mean does not exceed the prior mean, stop
training, preserve the latest public checkpoint, evaluate it and release the
pod. This is an operational early-stop rule, not a statistical significance
test. Do not reset the rule or cherry-pick a better starting window after
observing rewards. Missing metrics, stalled admission, crashes or the cost
deadline must terminate finitely rather than silently continue spending.

Unresolved judge choices return no reward. Genuine invalid model programs
retain a separately labelled zero; operational renderer/judge failures are
not visual failures. Before optimizer start, verify how the pinned Prime
version filters uncertainty across rollout groups and records it. Never
coerce `None` to zero or 0.5. Log group admission/censoring and zero-output
fractions; group-eight admission could be inefficient if uncertainty is high.

## Pending launch evidence

Record reviewed source commit, frozen config/manifest bytes, initializer and
renderer hashes, actual GPU model/topology/price, rental start/deadlines,
credential-free launch receipt, and native serializer/scorer/gradient checks.
Afterward report matched held-out preferences and visual examples, admitted
and all-completed reward curves, group losses, turn validity, parser and
position conflicts, timing/utilization windows and denominators, public
checkpoint/archive hashes, actual GPU/API cost, and the next decision.
