# Aesthetic multi-turn RL — 2026-09-21

Status: four optimizer updates completed, matched evaluation collected, checkpoints and evidence hash-verified, node released.

## Question and scope

Does terminal reference-conditioned aesthetic preference improve the exact contract-quality SFT step-24 initializer through actual paint → render → inspect/revise trajectories? This bounded run uses native Prime RL, eight rollouts per group, one group per update, fresh AdamW at 1e-5, rank-16 LoRA, three generated turns, 8192 completion tokens per turn and a 16384-token total context. Context and turn limits are operational limits, not speed rewards. Reward is the mean of two displayed A/B orientations from Luna low; uncertain or unavailable judgments are missing. No additive validity, finish, length or stroke reward is used.

The available curriculum has 24 references across 12 families and 41 reference/opponent tasks. Default source-order sampling admitted only `v2-box-000` and `v2-bus-000`: two opponent groups of eight episodes per reference. This is two-reference training, not 12-family coverage. Training starts from an existing painting state; the held-out evaluation starts from the reference alone. Baseline and candidate evaluation share the same historical rollout scaffold, which resets user context between turns; the native training environment preserves interaction history. These differences limit conclusions about transfer.

## Training evidence

| Update | Adapter values changed | Finite | Group reward | Mean turns |
| --- | ---: | --- | ---: | ---: |
| 1 | 77,814,138 | yes | 0.6875 | 2.625 |
| 2 | 69,210,553 | yes | 0.6250 | 2.750 |
| 3 | 65,398,122 | yes | 0.7500 | 3.000 |
| 4 | 62,637,197 | yes | 0.1250 | 3.000 |

Each update changed all 992 adapter tensors. This is 32 admitted training episodes; the reward rows involve different reference/opponent tasks and are not a learning curve. Native observed maximum off-policy distance was one update. The step-3 collection window reports a 46.7% cancelled fraction (7 of 15 records in that window), not a failure rate among the eight admitted episodes. The preserved native trace stream has 39 records: 32 admitted episodes and seven `user_closed` traces for `v2-box-001`. Final drained work is also recorded in native logs. No quality improvement is inferred from finite updates or training reward.

Attempt 1 failed before forward/backward or any optimizer update because an observation wrapper assumed a filesystem receiver field on Prime's ZMQ receiver. Attempt 2 used the corrected, tested non-gating observer and finished with exit 0 at 11:02:27 UTC. Its exact source commit is `7bb93511155f46b3f1deb2172d9b86d935ee7df6`. Model, initializer, data, prompt, configuration and archive identifiers are recorded in [EXPERIMENT.json](EXPERIMENT.json).

## Checkpoints

Both portable adapters are public and hash-verified by the uploader:

- [Step 2](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/2e26fbc4f05430d892d08a44fc3ea3cd60885fc6/aesthetic-rl-20260921/step_2), weight SHA-256 `7c29a65caee5fee56d880d6a73474f3ea33647e9266ca4ac2c36006738d875b9`.
- [Step 4](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/2fab61ebb52a00f7e1c9c3829bad1668719d07b3/aesthetic-rl-20260921/step_4), weight SHA-256 `fcbfb3c234b8a10ae3d948e90c778319359f1072f45823c7f90dd50bb69adba3`.

These are experimental checkpoints, not a claim of promotion.

## Judging and operational failures

Two workers did not complete image inspection for their initial larger live-007 assignments. Uninspected labels were removed before any live-007 import. Eight original files remain in `reward-batches/live-007/uninspected-labels/`; the other worker reported removing its unverified labels. All 16 live-007 views were then explicitly re-inspected in two-view assignments before import. Earlier batches retain original labels and agent inspection reports; parent spot checks are not an exhaustive audit of judge accuracy. Two inspected live-008 comparisons were uncertain and excluded; a parent spot check confirmed that both compared buses were almost wholly off-canvas. Missing views remain unscored, never false losses.

The retry package unintentionally included generated reward batches and collected baseline evidence. It was sent only to the authorized node; explicit runtime input paths remained unchanged. The packaging allowlist was subsequently corrected and its two tests passed. The fix does not retrospectively alter the launched bundle.

## Resources

Two 96 GB RTX PRO 6000 Blackwell Server Edition GPUs at $2.38/hour total, with separate inference and trainer roles. Across 148 ten-second samples per GPU from 10:37:47.490 to 11:02:17.906 UTC, mean utilization was 73.77% inference and 14.06% trainer; fractions above 10% utilization were 74.32% and 17.57%. The pipeline was not saturated. Batch waits totalled about 1,039 seconds across four trainer steps; forward/backward totalled about 329 seconds. Prime's displayed MFU used an A100 fallback peak and is not a valid hardware-specific efficiency estimate.

No scheduled task, heartbeat, recurring job or scheduled termination was created. Lium removal succeeded and `lium ps` subsequently reported no active pods, verified by 11:14:17 UTC. The conservative elapsed-time rental estimate is about $3.02 through that verification time, excluding judge quota; it is not a final provider invoice.


## Matched held-out result

[Visual comparison](review/index.html), [raw held-out judgments](heldout-judging/results.json).

| Reference | SFT valid canvas | RL valid canvas | Two-order Luna result |
| --- | --- | --- | --- |
| Box | yes | yes | RL preferred in both orders |
| Bus | yes | yes | order disagreement; reverse explanation misdescribes image sizes |
| Flower | no | yes | no visual baseline to compare |
| Vase | yes | yes | order disagreement |

Both policies hit the three-turn operational cutoff on all four cases; neither explicitly finished. Tokens were box 1883→1639, bus 2138→2614, flower 2169→2112, vase 2101→2181. There was no speed objective. The flower's successful render is an execution improvement on this one reference, but it has four petals against the reference's six. Parent inspection also finds residual geometric/placement problems and a disconnected vase rim/neck. Four samples and noisy pairwise judgments do not establish overall aesthetic improvement.

The reverse-bus judge describes A as the larger, stronger bus and B as small/washed out. Direct inspection shows the reverse: A is the small SFT bus and B the larger RL bus. Its raw label is preserved as a judging failure, not converted into a confident win or silently relabeled. This is evidence that order averaging alone does not solve image grounding. These held-out labels were generated after training and never supplied training rewards.

## Preservation and next decision

The frozen evidence archive is 123,703,837 bytes, SHA-256 `654b50976c3d9cffdb7a3a1ff48c1b4853dded62cfa4466fbdfcc62f207b5a33`. All 5,502 archive member hashes verified locally; no model weights were copied into the workspace. The original live telemetry file advanced during collection, so the final manifest was computed from frozen archive-member bytes before re-downloading and verifying. Native rollout traces are retained as Zstandard-compressed JSONL; ZMQ packing evidence is the decoded batch observer, not a claimed filesystem `.bin` transport copy. Local raw judge labels, quarantined labels and held-out comparison artifacts are retained separately.

Next use this as a completed small learning run, not another infrastructure smoke or a promoted model. Keep step 2 and step 4 as candidates. A broader continuation should distribute admitted groups across reference families and use stronger same-reference opponents; the available manifest alone does not achieve that. Keep the simple reward. Service judgments with small inspected assignments and retain order/grounding failures, rather than adding more reward terms or treating runtime validity as beauty. No further training or scheduled task is running.
