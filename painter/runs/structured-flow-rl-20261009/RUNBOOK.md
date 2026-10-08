# Operator runbook

The local plan is complete when the curriculum proposal metadata, code-supervision receipt (when supplied locally), source commit, and both public adapter identities are available. The 32/16 pilot materialization receipt is produced on Linux after the SFT512 endpoint and renderer are verified; it must say `anchors_verified: true` and still state that 208 train and 44 heldout rows are pending.

Node preparation stages the pinned base model, Prime runtime, renderer and two public adapters. It generates the 48 frozen opponents and matched Flash12 panel, renders and reviews the demonstrations, and performs the four-update SFT warmup before writing `preparation-receipt.json`. The warmup adapter initializes RL with a fresh optimizer. A preflight checks the launcher before the finite supervisor starts RL.

Checkpoint four is an observation boundary. Checkpoint eight is the final training boundary. The supervisor records truthful four/eight-update reward observations after the process exits; raw optimizer metrics are retained throughout. This short pilot does not stop on a noisy four-update trend or promote from changing reward windows. Promotion requires the matched 16-case pre/post evidence. A process crash, missing optimizer metric, or finite deadline is a failed or stopped run and must retain its evidence.

Compare the fresh Flash12 pretraining panel with the selected post-training public checkpoint using the same order, seed, renderer, policy instructions and generic Flash medium judge protocol. SFT512 is the frozen reward opponent, not the matched improvement baseline. Keep invalid final actions, missing terminals, renderer failures, and censored cases separate from scored preferences. Do not substitute a retained earlier image for a missing final action.

Record measured GPU utilization as a run-level busy fraction or interval with the measurement window and denominator. If the trainer waits on external judge calls, record the pipeline as unsaturated. Record actual provider usage and rental cost separately. The plan and receipts contain no credentials or provider connection details.
