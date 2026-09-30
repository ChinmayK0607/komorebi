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

The API pass started on 2026-09-30 at 06:29:44 UTC from source commit
`8440db9` (the committed bytes match those used by the process), with four
concurrent pairs and one observation per prompt/order. The 36-pair manifest
content SHA-256 is
`5f998bc55c7e20bf0160eea5a4484b02e7ae1a1627799465850aca20136783e0`.
The frozen old prompt SHA-256 is
`214702df5bdf4e2465236ae0758903439d2cadb3e9680095a4cb3ee71a5a9452`;
the new prompt SHA-256 is
`7f0379436d72610561687948fd41b859a5b400017b84859d639e06fb84471c08`.
The input manifest and raw receipts are in ignored
`local-evidence/pass1/`; that pass requests 144 views before any parse retries.

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

The user clarified that these preferences concern **which image should get
the higher reward**, rather than an absolute grade. In addition to direct
pair preferences, inspect step-16 and step-24 rewards against the same frozen
SFT opponent under each fresh protocol. A reversal between that reward
ordering and a direct comparison is itself useful evidence of unreliability.
Keep this secondary diagnostic separate from the precomputed five-label
direct-agreement denominator; it is not five additional independent labels.

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

## Presentation follow-up registered before its calls

Pass 1 completed with only 2/5 explicit direct preferences aligned under v2,
versus 1/5 under v1. V2 still reversed its winner between image orders on
11/36 pairs. That is insufficient support for deploying the revised judge.

The next finite API pass keeps the exact v2 prompt, model, generation
settings, cap, pair list and two-order scoring, but supplies two original
600-pixel PNGs separately with adjacent `Painting A (first image)` and
`Painting B (second image)` text. Reuse pass-1 v2 receipts as the baseline;
do not purchase them again. Request 72 new views before retries, at eight
concurrent pairs. This tests the presentation package (separate images,
original resolution and explicit adjacent labels), not the isolated effect
of any one component. The local contact sheet uses the fallback small font
when DejaVu is absent; no claim is made that this caused the past failures.
All 36 pairs remain in the test, including the eight control scenes.

Judge requests still omit human preferences and checkpoint identities.
Do not rewrite the prompt to force the fan or beekeeper result. Report
explicit agreement and order consistency along with inspection of readable
subjects, faint workers, extra wool-ball needles and the failed sand yacht.
The split path remains separate from both frozen v2 and the training judge.
