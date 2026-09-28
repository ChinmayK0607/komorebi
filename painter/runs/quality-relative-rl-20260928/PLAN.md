# Eight-candidate relative-quality RL diagnostic

Status: prepared, not launched. No GPU is rented for this run as of this record.

## Hypothesis and matched baseline

The previous mixed-reference RL run used one-turn SFT photo anchors against
four-turn student paintings. Its step-four checkpoint lost to the four-turn
SFT step-512 baseline on the matched ten-case evaluation (SFT 6, RL 2, two
ties), and 49/65 completed training traces earned zero reward. Hypothesis:
four-turn SFT anchors plus a within-reference visual ranking will give valid
but imperfect paintings useful relative credit even when all lose to the SFT
anchor. This is a diagnostic of trainability, not evidence of quality gain.

The immutable baseline is Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with the public SFT
step-512 LoRA at revision `adf687e5b321514eda7be6eab050434fdc8e44ff`,
weight SHA-256 `bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
The prior failure analysis is [FORENSICS.md](../quality-mixed-rl-20260928/FORENSICS.md).

## Inputs, comparison, and scope

The nine-row training curriculum contains six photos and three procedural
shapes. The photo opponents are exact saved final canvases from the matched
four-turn SFT evaluation. Photo 077 was excluded because its SFT canvas was a
weak anchor. Source report SHA-256:
`82c2de5f2fe7561509f0aa743e2a12d64e2b95d6cd2b399af2694fe23dd8933d`.
Manifest SHA-256:
`0d4cf5c68527a089dc118a02b05d767aafa81a2467d3a0c506983f41b6223ddf`.
The public [curriculum archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/0068bc95df0b2bc0d06c42f066761c939d9baffd/runs/quality-relative-rl-20260928/four-turn-relative-curriculum-20260928.tar.gz)
has SHA-256 `105620c5998a4d4a5b49126588e7722bb1c9be80eb88caa92069fe9f97ed76ec`;
anonymous re-download and all manifest-bound image hashes verified.

The terminal MiMo Flash comparison against the frozen SFT canvas remains the
quality anchor. At Prime's group-finalization point, a second MiMo Flash
judgment ranks the valid final canvases in the same eight-rollout group. It
sees the reference and all valid candidates, twice with reversed presentation
order, and must name every candidate's visible weakness. The adjusted reward
is half the original fixed-anchor score and half the within-group rank. An
invalid canvas receives zero. No speed or validity bonus is added. This
changes the reward recipe and is not directly comparable to the prior online
reward series. Raw judge responses, input hashes, costs/usage, original and
adjusted rewards, and post-GRPO sampled-token advantages must be retained.

An offline five-candidate smoke test on saved tennis paintings ranked
recognizable SFT canvases above the fragmented RL outputs in both image
orders. The two responses cost approximately $0.00079 combined. A later
seven-candidate smoke exposed response truncation at 3,072 tokens; after
raising the cap to 12,288, both reversed views returned complete rankings
with the same bottom five and swapped top two. They took 168.4 and 92.8
seconds, costing $0.001947 and $0.001374. The request timeout is now 360
seconds; judge latency is a measured throughput concern. These selected
checks test format and obvious-case ordering, not eight-way judgment accuracy
or human agreement. The local all-anchor-loss unit test confirms two
different valid paintings still get different token credit. Any eight-way
truncation or omission stops the run rather than silently changing the reward.

## Finite run and decision

Pinned Prime RL source is `26b3131d2716a4f8210b165df584b83b4bc54f61`
with verifiers submodule `828488fffe31aa3332b9d1bd4bd9ee320e375cf1`.
The Prime TrainSink pre-hook source SHA-256 is
`8618ae01796d616f973bee57966474da184fff50e324f6ac4a93b52becce3ca8`;
the patched source is expected to hash
`a07a615094fa7a57b44cb8fed5ee463911cb50143b12343a16cd94e430a69405`.
Record the committed painter source ID and generated TOML hash before launch.

Use one trainer GPU and one inference GPU, eight rollouts per reference,
optimizer batch eight, exactly eight in flight, four optimizer updates,
four painting turns, up to 8,192 completion tokens per turn, LoRA learning
rate `1e-6`, and public hash-verified checkpoints at steps two and four.
Batch eight means only one reference per update; the nine-row pool is not
equivalent to nine exposed training references. Record actual reference
coverage and reward/gradient data by step. No scheduled watcher is authorized.

Evaluate step four and SFT step 512 with matched four-turn settings on seven
photos excluded from this curriculum; report three training shapes separately.
Publish the full token/credit and judge evidence with anonymous hash
verification before releasing the node. Promote only if the held-out
paintings improve visibly without a material validity regression. Otherwise
retain SFT step 512. Track GPU rental against the user’s cumulative $25
GPU-only ceiling; previous campaign spending was at least about $16.54, so
remaining headroom is limited. Vercel API usage is itemized separately.
