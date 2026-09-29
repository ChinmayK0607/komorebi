# Scaled text-paint RL: post-run forensic review

2026-09-29. The [frozen protocol and results](NEXT_SCALED_RUN.md) define the
model, manifests, seeds, curriculum, action contract, and matched SFT control.
The [public evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/e39e6e58fc830eb7be0a77a8bdde7d906943d1d3/runs/text-pairwise-rl-20260928/scaled-final-20260929.tar.gz)
has SHA-256 `ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3`.
Its 2,601 screened files include the frozen manifests, full rollout stream,
judgment receipts, telemetry, traces, comparisons and gallery. The
[local side-by-side gallery](local-evidence/scaled-step16-public/evidence/gallery/index.html)
was streamed from that public archive and independently hash-checked.

## Decision

Keep the step-512 SFT adapter as the default. The public step-16 RL adapter
is an interesting research checkpoint, particularly for small mechanical
objects, but has **no demonstrated overall visual advantage**. Do not
continue this RL configuration automatically. The first useful next work is
offline judge calibration against independently chosen beautiful paintings
and a bounded, valid p5.brush action path. Only after those two repairs should
another short RL run test whether reward and held-out visual quality rise
together.

## Matched evidence

The 47-case held-out set was never used for checkpoint selection; an
outcome-blind training-reward audit stopped the planned 24-step run at its
already scheduled step 16. The initial and trained policy were sampled under
the same `native-painting-eval-v3` contract, three turns, seed 918210 and
8,192 output-token ceiling per turn. Both made 41/47 valid final canvases;
there were no provider-censored held-out cases.

| Comparison | Step-zero SFT | Step-16 RL | Paired result |
| --- | ---: | ---: | --- |
| Preference against the same frozen SFT anchors, all 47 | 0.3954 | 0.4113 | +0.0160, bootstrap 95% interval [-0.0833, +0.1135] |
| Same anchor comparison, 37 both-valid | 0.4347 | 0.4842 | +0.0495, interval [-0.0450, +0.1486] |
| Direct fresh-SFT-versus-RL, all 47 | 0.500 is a tie | 0.5319 | 26 RL wins, 4 ties, 17 losses; interval [0.4433, 0.6188] |

The direct judge's family means were 0.5208 for eight simple prompts,
0.8021 for eight intermediate prompts, and 0.4651 for 31 complex prompts.
The complex mean's bootstrap interval was approximately [0.3575, 0.5726],
so its negative direction is concerning but not proven in isolation. The
fixed eight-case train probe did not show a lift: seven comparable pairs
gave RL -0.0476 versus SFT, interval [-0.2024, +0.1071]. One of the probe's
eight IDs had no saved training rollout, leaving six comparable exposed
pairs near tied (+0.0139). Generated trace exposure does not prove gradient
admission. Online rewards fluctuated across 16 steps rather than rising;
every optimizer step nevertheless had a nonzero gradient norm. This was
learning from a weak, noisy signal, not a frozen adapter.

The blind, predeclared sixteen-case human read is in the protocol record.
It found better RL mittens, hand-crank drill, matchbox and a valid seahorse
where fresh SFT was invalid. It also found a much worse concrete library,
a weaker combine composition and stapler, and three invalid complex cases.
The older RL step-20 pocket **compass** is a convincing isolated win, but
that prompt is not evidence that the new step-16 policy generalized. In some
disagreements, the judge's detailed rationale identifies brief elements the
human read missed; in others, it rewards object detail despite a mechanical
look. The two judging orders produced different strength values on 18 of
37 both-valid held-out pairs, including six large gaps. These labels are
useful diagnostics, not yet a calibrated beauty metric.

## Failure mechanisms

**Action and renderer:** Of 186 saved generated training traces, 110 ended
with a valid canvas and 76 did not. The 186 includes possible speculative or
stale traces; it must not be presented as the admitted gradient batch. Only
47/186 first turns rendered valid. Across all turns, the audit counted 125
JavaScript syntax, 91 brush-API, 87 JavaScript runtime and 34 render-timeout
errors; these counts are turn failures and can overlap within an episode.
Complex traces were valid in 58/112 cases. A held-out snowplow reset its
`for` loop index inside the loop on turn one, then corrected that assignment
but still exhausted the 180-second renderer limit on turns two and three.
It never produced a canvas and held up final evaluation for about nine
minutes. A more generous token allowance alone does not resolve bad code
or unbounded brush work.

**Reward and curriculum:** The pairwise prompt explicitly checks subject
count, silhouette and spatial relations before coherence, beauty, color and
natural brushwork. That is reasonable for fidelity, but it can prefer a
detailed diagram over a more enjoyable painting. The frozen opponents were
the *first valid* SFT canvases, not an artist-screened quality bar. For the
frosted spiderweb training case, MiMo assigned the structurally clearer RL
web a 0.667 score while also calling it diagrammatic and noting its missing
frost beads. Beating a weak opponent does not prove attractive output. The
curriculum exposed 50 distinct IDs in saved generated traces, versus 68
available train anchors, and the evaluation showed benefits concentrated in
intermediate objects rather than complex spatial scenes. Scaling this exact
reward would amplify the wrong objective on the difficult cases.

**Throughput and credit:** In the measured 04:35:52–07:26:05 UTC training
window, 1,022 ten-second samples per GPU show inference GPU mean utilization
96.76% (96.58% of samples at least 90%) and trainer mean 10.92% (9.78% at
least 90%). The pipeline was inference-bound; extra trainer GPUs would have
little use. One otherwise idle GPU was used for the fixed eight-case probe
after training stopped. The two-GPU Lium rental cost $20.43 at removal,
below the per-run $25 cap; `lium ps` then returned no pods. Public archived
MiMo Gateway receipts total $0.164521 across 244 judgments: $0.073575
training, $0.031455 SFT control, $0.005201 probe, $0.029289 held-out anchor,
$0.024617 direct held-out and $0.000384 canary. These are recorded receipts,
not a claim about unlogged provider attempts. GPU time dominates this run's
measured spend.

**Experiment control:** The parallel probe initially hit an eight-case
manifest selector bug; it failed before inference, and commit `d6d1851`
fixed the explicit `all` scope. After all 47 held-out canvases rendered,
editing the active Bash wrapper in place caused it to exit with an unexpected
EOF before scoring. The saved renders were scored directly without rerunning
model inference. The first evidence package attempted to include vLLM
compilation caches, which its own safety check rejected; commit `668fde4`
excluded them. The final 533,129,801-byte archive was uploaded publicly,
anonymously downloaded and SHA-verified before the node was deleted. These
recovery steps consumed paid time and are reliability defects to remove.

## Next experiment, only after offline repair

1. Build a small blind set of genuinely appealing SFT, RL and teacher
   paintings across simple, intermediate and complex prompts. Ask human
   raters for both overall beauty and reference/brief fidelity. Score every
   pair in both orders with MiMo Flash and report agreement by family; revise
   the prompt or model until it reliably rewards *better-looking* art. A
   relative judge should be compared with the actual human choices before
   spending on more gradients.
2. Bound the painting action space: syntax and allowed-brush-method checks
   before browser rendering, a per-program work budget or loop watchdog,
   shorter failure detection, and a safe fallback to the last valid canvas.
   Verify that a failed turn can recover rather than consuming three full
   renderer timeouts. Use the same checks at training and inference.
3. Replace first-valid SFT opponents with curated high-quality same-brief
   paintings, while keeping a direct fresh-SFT control. Test on a small
   balanced curriculum before scaling. Promote a checkpoint only if direct
   held-out human-calibrated preference, validity, and complex-scene quality
   improve together. If the canary passes, buy inference capacity ahead of
   trainer capacity and record run-level utilization and cost again.
