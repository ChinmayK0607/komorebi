# Frozen painting judge alignment

## Question and fixed controls

Can a general prompt revision make the same MiMo Flash judge prefer readable,
coherent, attractive paintings in line with the user's feedback, without
rejecting pale watercolor or good paintings for minor accessory errors?

This is a local API-only comparison of already rendered PNGs. No painting
generation, rendering, model training, GPU rental, or scheduled task is part
of this experiment. Use `xiaomi/mimo-v2.6-flash` through Vercel AI Gateway,
temperature zero, the existing image-sheet construction, two image orders,
and the existing pairwise strength conversion and visible-subject cap.
Compare the frozen grounded-v1 prompt with a separately versioned grounded-v2
prompt. Keep raw labels, reasons, image hashes, prompt hashes, order gaps,
provider-reported usage/cost and latency; errors are censored, never visual
losses. Provider-reported cost is not a verified billing total.

The source gallery is the public-hash-verified grounded-pilot archive at
HF dataset `CK0607/komorebi-painter-experiments`, revision
`ffdcc865cf46495d72ca4215c15866ddb077bebf`, archive SHA-256
`e34870e580a74fcf621268e10f5f45c54b0a91060c9b7c74bf4fab8cb23282c4`.
Local images are under the preceding run's ignored `local-evidence/gallery/`.
The manifest freezes exact briefs, image byte hashes, pair roles, old score
metadata, and human feedback before new paid judgments. Judge requests omit
case IDs, checkpoint names, old scores and human preferences.

Historical gallery scores are **relative preferences against a fixed SFT
anchor**, not absolute painting quality. Worse, prior SFT/RL-step-16 scores
used `text-paint-sft-pairwise-margin-v1`, while pilot-step-24 scores used
`text-paint-sft-pairwise-grounded-v1`. They cannot be used as a consistent
cross-checkpoint ranking. This test recomputes both prompt versions on the
same frozen pairs, including direct step-16 versus step-24 comparisons.

## Feedback and generalization

The user identified nine scenes: ferret `t600-370`, pantry `t600-375`,
beekeeper `t600-413`, mountain train `t600-466`, farm worker `t600-503`,
fan `t600-516`, wool ball `t600-542`, sand yacht `t600-560`, and kingfisher
`t600-591`. Explicit preferences are step 16 over step 24 for the beekeeper
and kingfisher, and the frozen SFT anchor over the other fan paintings.
Other comments describe qualities or defects and are not converted into
invented forced pairwise labels. Preserve them as qualitative feedback.

The new prompt emphasizes a clearly readable main subject, connected parts,
believable contact and containment, and an appealing finished painting.
It distinguishes a subtle but coherent wash from an illegible wireframe or
fragmented main subject. Contrast, darkness, saturation, and more detail do
not independently earn preference. Precision and warmth can coexist;
proportionate local faults (such as an extra accessory) do not erase an
otherwise strong subject. A judge must not infer anatomy or contact that is
not visible, and its reason must agree with its stated weaknesses.

Add eight deterministic, previously unlabelled control scenes from the
remaining gallery, spanning available families. Report the feedback panel
and controls separately. This is calibration against a small, selected set,
not evidence of general judge accuracy or RL improvement. Avoid repeated
prompt tuning until this panel is memorized: the result can reject a prompt
change as well as support it. Preserve disagreement between image orders
rather than hiding it in an average.

## Standing GPU-run policy from this discussion

Future GPU runs have a **$20 whole-rental cap per run**, including setup,
training, GPU evaluation, and preservation before release. API charges are
separate. Set the finite rental deadline from actual node price and elapsed
spend, with a preservation reserve inside the cap; never use the old $25
default for a new run. CPU-only API judging and packaging should run after
GPU release where possible.

At optimizer updates **15 and 20**, check the effective-batch reward mean
over the most recent five completed updates against the preceding five.
If it has not increased, stop training and preserve the completed run.
Freeze the sampling mixture so this signal is not deliberately raised by
changing task difficulty. This is a frugal compute-stop rule, not proof of
visual improvement; checkpoint promotion still requires matched held-out
painting comparisons. Implement the finite in-run guard before the next
launch; no recurring watcher or scheduled task is authorized. Canvas
validity changes remain a later experiment after the judge calibration.

## Decision to fill after execution

Record human-preference agreement with exact denominators, image-order
disagreement, changes on the unlabelled controls, failures/censoring,
reported usage/cost, and manual review of reasons. Keep grounded-v1 intact
for reproduction. Do not announce a successful judge or launch RL merely
because the selected calibration pairs agree.
