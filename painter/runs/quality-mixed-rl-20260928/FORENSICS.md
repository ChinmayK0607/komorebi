# Why step-four RL did not beat SFT step 512

2026-09-28. This is an audit of the completed diagnostic, not a new run. The
frozen comparison and immutable model/data IDs are in [PLAN.md](PLAN.md). Raw
turns and paired judgments are in [the gallery](collected-step4/review/index.html)
and the [public evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/dc8044d996d52c46e2fc0a7b901ebd2a28c390b4/runs/quality-mixed-rl-20260928/step-4-evidence.tar.gz).

## Finding

The optimizer ran, but its reward was a weak proxy for improving the already
strong SFT policy. The twelve photo opponents were frozen **one-turn** SFT
paintings; the RL policy had four turns. A win against one of those anchors did
not mean a win against a fresh four-turn SFT painting. The three procedural
opponents were four-turn finals, so the comparison strength also varied by
family. The online reward rose across four updates, while the matched
four-turn evaluation went the other way: SFT won 6/10, RL won 2/10, and two
tied. On seven held-out photos SFT won 5, RL won 1, and one tied. Three RL
photos had no valid final canvas; SFT had none missing.

The signal was sparse. The live run counted 65 completed traces: 49 zeros,
six ties and ten wins. The surviving archive contains exactly 38 judged reward
files (22 losses, six ties, ten wins); invalid outputs return zero in the task
without writing a judged reward file. The 261 archived training render
attempts include 110 valid and 151 invalid. Among the invalid attempts, 66
reported JavaScript syntax errors and 66 reported undefined/incompatible API
or runtime errors; the remaining 19 had other failures. These are render
attempts, not independent episodes. The archive has 77 episode directories,
including 12 without a completed reward at the stop, so directory count must
not be treated as the number of completed traces.

The model also failed to use feedback reliably. In held-out photo 572, RL
repeated the same invalid program four times (`v is not defined`). In photo
634 it never submitted a renderable program. In photo 431 it reverted on its
last turn to the first-turn program, yielding disconnected rectangles instead
of the recognizable tennis player, racket and court in SFT. Photo 322 similarly
reverted to its first-turn program, reducing the player to a silhouette. The
held-out [montage](collected-step4/heldout-montage.jpg) shows these examples.
The paired judge did recognize an RL gain on the skatepark photo 077, where
the SFT output was abstract, and the small train-set bus is a plausible local
gain. Stronger block-in on a few images is compatible with a net regression:
the policy often preserved coarse color/layout while losing subject identity
and brush detail.

The common painting helper still calls p5.brush, so this is not evidence that
the evaluation silently switched to a p5-only renderer. The sampled RL
programs could be much simpler: on photo 431 the RL final program was 1,744
bytes versus SFT's 4,338-byte final and omitted the recognizable player and
racket. This is one illustrative case, not a corpus-wide complexity
estimate. The renderer rejected many actions on both sides, but 17/24 RL
evaluation paint attempts were valid versus 24/27 SFT attempts.

Nonzero recorded gradient norms (0.0165, 0.0352, 0.0415, 0.0486) and the
verified step-four adapter show that an update happened; they do **not** prove
that token masking or credit assignment was ideal. The public evidence archive
omitted Prime's native packed token traces and trainer logs, which were lost
with the released node. We therefore cannot determine whether a training-mask
bug contributed. Nor can ten paired cases estimate population quality
precisely. The strong direction of the held-out result, invalid finals and
visible subject loss are enough to reject this checkpoint for promotion.

## Decision for the next experiment

Retain SFT step 512. First make the comparison target match the task: use a
frozen four-turn SFT opponent on photos as well as shapes, and verify the
candidate and opponent each receive the same turn/context budget. Preserve
within-group rankings of renderable candidates so that a group in which every
sample loses to a strong opponent still contains useful relative quality
information; keep one reference-conditioned visual-quality criterion rather
than adding unrelated rewards. Treat syntax/API failures as explicit model
outcomes and test whether targeted feedback yields a changed, valid revision.
Before another paid run, confirm the corrected signal on a small fixed set
with the SFT policy as the matched baseline, and package native token traces,
trainer logs and judge receipts before releasing the node. More GPUs would
accelerate the current weak signal, not resolve it.
