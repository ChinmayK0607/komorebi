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

The presentation pass started at 06:40:51 UTC from source commit `a44eb3e`.
Its manifest SHA-256 is
`cfd1234885dff6ef0a491562665f256e1c509467364ff4b5628df15dec8e7a2e`.
Nine focused alignment/presentation tests and five existing margin/grounding
tests passed before launch. The reviewer corrected an initially permissive
winner parser to use the margin parser's strict winner/strength validation,
matched fidelity-validation retries, and exact baseline receipt identities;
the faulty draft made no API calls. No existing training code was modified.

## Results and decision — 2026-09-30

Both finite passes finished. Their frozen reports are
[prompt-only](local-evidence/pass1/report.html) and
[presentation](local-evidence/split-pass1/report.html), with raw view receipts,
image hashes, normalized order votes and provider usage beside them.
The generated [analysis](local-evidence/alignment-analysis.json) preserves
the direct preference constraints, rewards against the common frozen SFT
opponent, and separate control-scene decisions.

| Metric | Grounded v1, sheet | Grounded v2, sheet | Same v2, separate PNGs |
| --- | ---: | ---: | ---: |
| Scored pair receipts | 35/36 | 36/36 | 35/36 |
| Explicit direct preferences matched by averaged reward | 1/5 | 2/5 | 3/5 |
| Explicit preferences matched by **both** order votes | 1/5 | 2/5 | 2/5 |
| Disagreeing order winners, all scored pairs | 9/35 | 11/36 | 11/35 |
| Order margin gaps at least 1/3 | 9/35 | 12/36 | 11/35 |
| Disagreeing order winners, feedback panel | 7/27 | 10/28 | 8/27 |
| Disagreeing order winners, eight controls | 2/8 | 1/8 | 3/8 |
| Mean / median successful view latency, seconds | 35.28 / 31.08 | 43.97 / 38.83 | 53.72 / 47.02 |
| Recorded provider cost, USD | 0.03635 | 0.04427 | 0.05014 |

The five preferences are a selected calibration set, including three related
comparisons of the **same fan**. They are not independent random examples;
these counts are not estimates of general judge accuracy. No confidence
claim or trained-model improvement is inferred from them. Each condition
has one two-order observation, not repeated stochastic evaluation. The
second pass was sequential in time, so provider load is another limitation
of its latency comparison. Both passes had at most 16 concurrent view calls.

### What improved and what remains wrong

The new prompt recognizes the wireframe ferret as unfinished rather than
awarding it a large win for apparent anatomical detail. Its reward against
the identical frozen SFT opponent falls from 0.667 under fresh v1 to 0.333
under v2; split presentation gives 0.500 with conflicting votes. These are
relative margins, not absolute grades. The clean pantry stays at 0.667;
the mountain train stays a clear win (0.917 → 1.000); and the excellent
step-24 wool ball goes from 0.750 to 0.833, with both orders agreeing under
v2. The extra needles do not erase its coherent round form and appealing
wash. The lost step-24 sand yacht is demoted from 0.250 to 0.083 under v2
(0.167 with split inputs). The two sand-yacht parser failures are censored,
not scored as visual losses.

The direct kingfisher comparison changes from preferring step 24 under v1
to preferring step 16 under both v2 presentations, matching the user. The
beekeeper remains a failure: v2 sheet votes confidently favor opposite
images (order gap 0.667), and split votes also oppose one another (gap 0.500)
with the averaged reward incorrectly favoring step 24. Reasons still claim
grip or connected sleeves not supported by clearly visible marks. Therefore
explicit anti-hallucination wording did not solve the perception problem.

Reward ordering against the common SFT opponent also matters. Sheet v2
gives beekeeper step 16 a higher reward than step 24 (0.667 vs 0.500), but
split presentation reverses that order (0.333 vs 0.500). For the kingfisher,
sheet v2 ties the anchored rewards (0.500 each) despite a correct direct
preference, while split gives 0.583 vs 0.250, matching the user but with
conflicting votes on the step-16 anchor comparison. These differences show
why an average alone cannot be treated as reliable supervision.

The fan is not fixed. Step 24 gets 0.667 against the user-preferred frozen
SFT under fresh v1, 0.750 under v2, and 0.583 under split presentation.
The latter reduces the overreward but still chooses the wrong winner and
has a 0.500 order gap. Split presentation's third nominally correct human
preference is frozen SFT over step 16 (0.417); its two votes disagree, so it
does not add a third reliable preference. Both v2 presentations correctly
prefer the frozen fan over the nearly unpainted matched-SFT fan.

For the faint farm worker, sheet v2 raises the step-24 reward from 0.417 to
0.583, but that average masks a huge 0.833 order gap. Split presentation
instead gives 0.250. The subject visibility/readability distinction is in
the rubric, but the judge has not applied it consistently; this is not a
reason to hard-code a saturation or darkness reward. Pale coherent painting
remains acceptable. On additional scenes, v2 consistently prefers the more
coherent chameleon and preserves the mitten shape. It also changes the
harvester and stapler ranking; those have no user labels and must not be
counted as additional successful calibration examples. Split inputs increase
control order disagreement from 1/8 to 3/8, so they are not a general repair.

### Failures, scope, spend and next decision

One v1 direct sand-yacht pair and one split step-16 sand-yacht/anchor pair
failed validation after bounded retries (`ValueError`). Credential-safe
receipts retain the exception class but not the provider body, so the exact
validation subcause is unavailable. No failed judgment became zero reward.
There are 212 saved successful view receipts across both passes. Their
reported costs total **$0.130756192**; failed/retried calls with discarded
responses are outside that total, so it is not a verified complete invoice.
Saved receipts cover 339,202 completion tokens and 270,308 prompt tokens.
Model IDs are requested/reported API identifiers; an immutable provider
weight revision is unavailable. GPU spend and GPU utilization are both
inapplicable: **no GPU was rented or used**, and no painter generation,
renderer, optimizer, rollout or training data changed.

**Decision:** retain the general v2 rubric as an evaluated candidate, but do
not silently replace the training judge with either tested path. The prompt
helps several clear visual failures; Flash still invents spatial contact and
can prefer the mechanical fan against the user's taste. Separate-image
presentation alone does not remove that problem. Before new GPU work, the
next useful judge experiment is a single stronger-judge comparison on these
frozen pairs plus independent human-labelled pairs, or grounded visual
preference examples spanning different subjects. Do not keep tailoring
instructions to these five preferences or introduce extra additive rewards
to hide the mismatch. Preserve both the positive RL examples and the
failure cases. The $20 GPU cap and finite trend-stop policy remain in force
for the next actual run; their launcher integration is still pending.

### Public preservation

The [169-file calibration archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/69e9abf6d94741da472c31f93260fb3531695e92/runs/judge-alignment-20260930/frozen-judge-calibration-20260930.tar.gz)
contains both passes' manifests, raw receipts and reports, the tested PNGs,
analysis and reviewed experiment source. It is 13,852,946 bytes with SHA-256
`6945eb52b69800087e6cd0c4054ec8f6de86b3fee57e295a912195bb04c75b26`.
Anonymous streaming download at revision
`69e9abf6d94741da472c31f93260fb3531695e92` reproduced both the byte count
and hash. Public images are candidates/evaluation evidence, not newly
approved training examples. The first upload attempt stopped on a local
Hugging Face Xet cache-write restriction; HTTP upload with a task-local
cache succeeded. No GPU, additional judging, or broader filesystem grant
was needed to repair publication. The artifact was frozen before adding
this preservation receipt; its copies of this document omit this paragraph.
