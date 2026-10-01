# RL16 continuation with MiMo Pro

## Hypothesis and matched baseline

Authorized2026-10-02: a stronger MiMo Pro comparator with the reviewed whole-form v2 rubric, explicit supported brush API instructions and an inspect/keep-or-revise choice may yield more useful, visually aligned gradients than the previous Flash continuation. These interventions are combined in this pilot; no isolated attribution or guaranteed improvement is claimed. Start from the user's preferred old RL16 adapter with a fresh optimizer. Do not initialize from the failed recent continuation or merge adapters.

Policy: CK0607/qwen3.8-27b-brush-painting@ac4d3eda8e5925e2d23992cfdc1a63620ebde573, text-pairwise-scaled-20260929/step_16, adapter SHA256 e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c. Base Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. Prime26b3131d2716a4f8210b165df584b83b4bc54f61. Native IPO loss with GRPO group-normalized advantages, terminal trajectory credit; no claim of turn-level credit assignment.

Frozen opponent: SFT512 immutable paintings from CK0607/qwen3.8-27b-brush-painting@adf687e5b321514eda7be6eab050434fdc8e44ff, photo-curriculum-sft-20260922-balanced/step_512, adapter SHA256 bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740. Reuse the40 reviewed text training briefs and47 held-out briefs. Semantic training manifest SHA256 13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501; public source archive SHA256 ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3. The final fixed-prompt comparison against SFT512 is distinct from online changing-batch reward. A direct parent comparison remains desirable; previous incomplete first-turn recovery is not a matched three-turn parent result.

## Registered method and bounds

MiMo xiaomi/mimo-v2.6-pro via Vercel AI Gateway; provider weights are not revision-pinned. Prompt SHA256 d21e4644e7a3874a0694b2ad406420e733b7638222eb7717e7f5f91cbdf1613c. Joint side-by-side image, strict side JSON parsing, both image orders, conservative rechecks/consensus. Pro showed improved recognition on the saved horseshoe example but not established general judge reliability; unresolved comparisons remain censored, never synthesized as rewards. Invalid final model actions retain genuine zero rewards. No additive geometry, detail, warmth or speed bonus.

Group4, optimizer batch16, LR1e-6, maximum50 actual updates if the finite trend policy permits continuation. Three inspected turns,8192 completion tokens per turn,32768 context. Supported API primer, explicit finish choice, syntax preflight and valid-final contract enabled. Patched centroid renderer watercolour-p5-brush-2.2.1-centroid-fix-v1: renderer SHA256 6181fb904a0c5dcd580f5c3cb01eaf341a14b7a15479e811d99ebb10b70bc841 and brush SHA256 db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b. Retain exact lexical virtualenv interpreter. Limit simultaneous rendering according to measured node CPU quota; the offline profile did not establish a new renderer speedup.

GPU rental hard cap$20 including setup, evaluation, preservation and release; Gateway charges separately. Completion reserve and exact-node release controller bound the finite command. Stop at registered15/20/every-five trend checks if the latest five admitted-batch rewards fail to exceed the prior five; explicitly record optimizer completion separately from batch admission and avoid treating a batch dispatch as an update. Public hash-verified checkpoints every4 updates, final raw evidence and saved turn renders before release. No scheduled task, heartbeat or recurring automation.

## Launch status

Preparation only at registration. Reviewed source/config hashes, actual rental price/hardware/CPU quota, start time, adapter tensor audit, first completed optimizer update and interval GPU telemetry will be recorded after verification. No learning or saturation result is claimed.
