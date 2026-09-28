# Eight-candidate relative-quality RL diagnostic

Status: eight-candidate clean retry running; first optimizer update completed.
The matched visual comparison and final cost are pending.

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
seconds; judge latency is a measured throughput concern. An exact
eight-candidate format smoke, using seven individual saved paintings and one
triptych distractor of the same reference, returned complete reversed-order
rankings in 33.0 and 36.6 seconds at $0.000597 and $0.000507. The best two
swapped order; weaker middle candidates were less stable. These selected
checks test format and obvious-case ordering, not human agreement. The local
all-anchor-loss unit test confirms two
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

## Launch record and corrected diagnosis

The first launch used painter source `1008a5e9686edb0fa1698b64cf14de8ed0747227`
and generated TOML SHA-256
`d2cfff1f93830d59fd28868344f8fa646e3a8adb992711dd04452a93309dd7e6`.
It started at approximately 09:41 UTC on a two-card 96 GB Edition node quoted
at $2.38/hour. Prime loaded the SFT initializer and ran rollouts, but it
was interrupted before its first optimizer update. The first complete
eight-member group had seven trainable traces (one renderer service failure),
four valid final canvases, and a real two-orientation relative-rank audit.
All seven traces had nonzero sampled-token advantages. The two rankings
agreed exactly on four valid horse paintings. One subsequent group was only
partially complete. The 251 MB output, raw rollouts, judge receipts, telemetry,
and group audit remain separately preserved on the node as an interrupted
attempt; they are not a trained checkpoint.

The interruption was based on a mistaken diagnostic: reading `/proc/environ`
after Prime changed its process title showed zeroed bytes, and the group audit
was checked before it appeared. Those observations did not prove the hook was
missing. The scorer was in fact active on the first launch. Source
`e472d6b4bb45be138499464f8ec255af14c38293` additionally forwards the
group-hook module, gateway key-file path, and Python import path explicitly
through Prime's child-environment wrapper; twelve focused local tests pass.
The clean retry started at approximately 10:10 UTC with the same model,
curriculum, group size, and generated TOML. This defensive change must not be
presented as the reason the reward began working. A stale terminal-judge
receipt from the interrupted attempt made the retry's worker exit; its
reward-batch directory was isolated, and a fresh worker was started without
resetting the GPU run. The future launcher now refuses any nonempty prior
rollout, judge, reward, or group-credit state and supervises the terminal
worker, stopping the run if that worker exits first.

The retry's first group had eight trainable traces, five valid final canvases,
four positive adjusted rewards, and nonzero sampled-token credit for all
eight traces. The two reversed-order MiMo rankings returned complete results;
this is evidence of a usable reward path, not visual improvement. The trainer
confirmed optimizer step one at 10:46:06 UTC: loss 0.0018, gradient norm
0.0228, learning rate 1e-6, 600 training tokens/s, reported MFU 40.9%, and
92.2 GiB peak memory. The inference GPU generated the next group while the
trainer updated. The node's whole-life rental meter read $3.03 at 10:48 UTC;
that includes setup and the interrupted attempt. Four updates, public
checkpoints, matched evaluation, and final spend remain pending. The
interrupted attempt's raw evidence will be included in the public audit,
excluding a duplicate step-zero weight file.

Optimizer step two completed at 11:03:20 UTC with gradient norm 0.0378 and
the same 1e-6 learning rate. Its 233,590,600-byte adapter is [public on Hugging
Face](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/a2e78de540035fcd39039fac6710e13dd8e1528e/quality-relative-eight-20260928/step_2),
commit `a2e78de540035fcd39039fac6710e13dd8e1528e`, weight SHA-256
`4ca0b90b26d3d818fd96a21eda0f101fbfc4717bc42a839a211be851767a0d9a`.
Anonymous repository and LFS metadata agree with the upload receipt's byte
count and SHA-256. The second group had six valid canvases and five positive
adjusted rewards, with nonzero token credit on all eight traces. This is not a
held-out visual result. A separate copy of the small step-two upload receipt
is retained on the node in case Prime prunes the local step-two directory.
