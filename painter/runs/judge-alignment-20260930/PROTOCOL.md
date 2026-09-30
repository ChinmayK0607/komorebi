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


## Joint-view identity repair — registered 2026-09-30

The user chose joint presentation for future judging. The separate-PNG path
remains historical evidence, not the default. In the saved joint-v2 pass,
11/36 pairs have conflicting normalized winners: six prefer B in both
positions and five prefer A in both positions. Both opposite-role patterns
map correctly; this is not an arithmetic reversal defect. The beekeeper
reasons change their claims about readable torso/contact along with order.
This observation alone cannot separate position bias from ordinary response
instability; the previous experiment has no same-order repetition control.

**Hypothesis:** persistent content-based neutral image labels, legible joint
600-pixel panels, and visible assessments before the verdict reduce identity
confusion and conflicting preferences while retaining the user's intended
quality comparison. Both canvases are always visible together. Stable IDs
follow image bytes when their positions swap. Canonical request enumeration
prevents caller anchor/candidate role from changing which request runs first;
it does not by itself remove perceptual position bias.

Test all 36 frozen pairs against reused grounded-v2 joint receipts from the
pass-1 content manifest `5f998bc55c7e20bf0160eea5a4484b02e7ae1a1627799465850aca20136783e0`.
Use the same `xiaomi/mimo-v2.6-flash` Gateway model at temperature zero.
No case IDs, human notes/preferences, checkpoint names, or prior verdicts
enter requests. Keep the same general quality/readability rubric rather than
adding subject-specific desired answers. This is a protocol package test: IDs,
layout, evidence-first contract and disagreement handling change together.
The opaque provider model alias has no supplied immutable backend revision;
record the exact response model identifiers without inventing a pinned version.

Two balanced orders are judged independently. An initial disagreement gets
one fresh joint-view adjudication without previous reasons or winners. A
two-of-three identity majority uses the weakest supporting preference
magnitude; no majority stays uncertain with no numerical reward. Malformed
and failed requests remain censored, never zero or an invented tie. Recovered
majorities are reported separately from initial order agreement, so retrying
cannot hide position conflicts. Agreement is not correctness. Unlike v2,
the new identity contract does not apply the candidate-only fidelity cap;
compare winner agreement and conflict rates, not old versus new reward levels.
Fidelity remains recorded in visible assessments rather than a new additive
reward. No training scorer is replaced during this experiment.

The finite test permits 72 initial successful judgments plus at most 36
adjudications, with bounded parse retries. Preserve all successful retry
usage and raw responses, request image/brief/prompt/format hashes, failures,
latencies, and source revision. Reuse existing renders: no GPU rental, model
generation, image rendering, or scheduled watcher. Report raw order conflicts,
resolved versus unresolved conflicts, explicit direct labels (five selected
comparisons, three of one fan), and controls separately. Run-level paid usage
is provider-reported and not a verified invoice. Freeze source and configuration
hashes before the first call. Do not claim the swap issue fixed merely because
conflicts are hidden, converted to ties, or filtered out of the denominator.

The strategy is informed by [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926), which studies evidence-before-rating and balanced positions for text judges. Applying it to paintings is an experiment, not an established visual-judging guarantee. The saved two-order calibration already detects winner instability; the proposed bounded recheck is a reliability policy rather than proof of a better visual evaluator.


### Identity pass launch

Reviewed source commit `92a775617a29d74fbbf6fb18f2fcf664d888645c` passed
19 focused tests and compilation. The API-only pass started at
2026-09-30 08:16:18 UTC with six concurrent pairs (up to 12 initial
view calls); every image request contains both paintings together.
Manifest content SHA-256:
`56a9051c2eda3f23b64f17a4146c1108aba54637d2072c1907abf1f35c121f61`.
Prompt SHA-256:
`02a0d9783fbb31b7a115a7be0e79d6eee9424c3b40d13224192fe246dc62ed8c`.
Judge module and runner hashes, source commit and timestamp are frozen in
`local-evidence/identity-pass1/launch-record.json`. Evidence receipts preserve
initial conflicts even if a malformed fresh vote is censored. No painting
generation, GPU rental or training launch. The fresh vote repeats the
canonical order, so two-of-three support can retain position bias; recovered
choices are provisional, never evidence that perceptual order bias vanished.


### Lossless response replay amendment — registered during the API pass

Saved replies reveal format-only rejections: known keys with leading spaces,
`_preference` instead of `preference`, and extra irrelevant top-level fields.
The strict pass is retained unchanged, including all paid retries. A separate
offline replay will trim only known key names, accept the explicit preference
alias, and ignore extra top-level fields after validating every required
field. Reject alias collisions, changed/unknown image IDs, invalid enums,
contradictory winner/magnitude, and incomplete evidence or weakness lists.
No prose, winner, preference, image ID or human label is invented or altered.

Replay selects the first semantically valid saved response for each view,
not the answer closest to the desired outcome. Raw calls, costs and the
original strict receipts remain intact; replay metadata lists transformations.
Two newly recovered initial views may disagree where the strict runtime never
requested a third call. Such pairs remain pending/uncertain; replay does not
invent a fresh vote or claim them resolved. A fully observed three-way split
stays uncertain without a fourth vote. Source is
`replay_identity_receipts.py` with its focused tests, separate from frozen
source commit `92a7756`. Report replay results separately from strict runtime
results, and preserve failures in the full 36-pair denominator.


## Schema and symmetric recheck repair — registered before its API calls

The 36-pair identity pass finished with 123 attempt receipts; 122 include
provider usage/cost totaling $0.0496900824, and one request failed without
known usage. Mean/median successful recorded call latency is 34.95/32.26
seconds. The exact-fields parser censored 20 pairs. Lossless offline replay
recovers 30 initial pairs, with eight initial winner conflicts, six still
incomplete, 22 initial agreements, four saved two-of-three choices and four
conflicts lacking a saved fresh view. These do not demonstrate that stable
labels removed position bias. The same-order retry replies also show
response variability, so not every flip can be attributed solely to position.

Next, retain the same Flash model, prompt, frozen image bytes and joint layout,
but request an explicit dynamic JSON Schema with enum image IDs and typed
required assessments. [Gateway structured-output documentation](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/structured-outputs) describes this request format; support by the actual
Flash backend must be observed rather than assumed. Start with two existing
pairs for support, retaining and reusing their replies if the full pass runs.
Do not silently fall back to JSON mode if schema requests are rejected.

The scorer's conflict recheck must also be symmetric: repeat **both** image
orders in parallel, then require three of four votes for the same image
identity. A stable preference for a screen position produces two versus two
and therefore no reward. A failed call or no identity consensus stays
uncertain with `score=null`, never a fake tie or visual zero. Use the weakest
supporting magnitude for an accepted preference; genuine agreed ties and
byte-identical canvases retain 0.5. This prevents contradictory observations
from producing confident gradient supervision; it does not guarantee that
agreed judges are aesthetically correct. No new additive reward or human
preference is added to requests.

If the API support checks pass, test all 36 matched pairs with six concurrent
pairs and cached support receipts. Initial checks are 72 views plus at most
72 balanced recheck views, each with at most one malformed-response retry.
Freeze source/config/manifest before calls. Report initial conflict, final
accepted/uncertain counts, parser and provider failures, selected human
preferences and separate controls, latency and all reported API cost. The
new `text_pairwise_consistent_judge.py` is isolated from the frozen identity
pass and is not installed into training merely because its local tests pass.
No GPU rental or scheduled task is authorized by this repair.


## Live format tolerance amendment — before the next full-panel calls

Both two-pair schema support canaries returned HTTP 200 but some replies
violated the requested schema (a leading space on `preference`, irrelevant
extra fields, and one incomplete reply). Thus schema enforcement is not
confirmed; no full-panel strict-schema pass was launched. The seven original
API attempt receipts remain unchanged in `local-evidence/consistent-pass1`.
Offline lossless normalization recovers both initial views of both pairs,
with agreement for each, using the first semantically valid saved response.

Use the previously registered conservative normalization in the live scorer:
trim whitespace on known keys, accept the explicit `_preference` alias, and
ignore irrelevant extra top-level fields only when all required fields are
valid. Reject alias collisions, arbitrary spelling repairs, unknown IDs,
incomplete evidence, and contradictory winner/preference. Preserve raw
responses and record transformations. This is tolerant parsing of observed
responses, not proof of provider schema enforcement or a change of verdict.
The same schema request, Flash model, prompt, joint sheet, and balanced
three-of-four conflict rule remain.

A new immutable manifest in `local-evidence/consistent-normalized-pass1`
binds the reviewed amended source and the same 36 frozen pairs. Reuse the
two saved support pairs after conservative replay, retain all seven paid
attempts and their original provenance, and call only the remaining 34 pairs
with six concurrent pairs. No human labels enter requests. Separate
recovered format problems, remaining parser failures, initial disagreements,
accepted balanced rechecks, and unresolved uncertainty. No GPU or training
launch is part of this experiment.


## Joint-view repair result — completed 2026-09-30

Reviewed scorer source: `1b9ac1f953767417e6be429d9f8e5059b1f37470`.
Report source: `6a95a1e`. Frozen full-panel manifest content SHA-256:
`aa82509cbe6a2843caedd412f01d09baa4e352271d8b72970e516b707d469e21`.
The model alias was `xiaomi/mimo-v2.6-flash`; the provider did not disclose
an immutable weight revision. The same 36 comparisons and original image
bytes were retained; original
pass-1 and separate-input receipts/manifests/summaries were hash-checked
unchanged. Two support pairs were conservatively replayed without new calls;
34 remaining pairs received fresh independent calls.

Results in `local-evidence/consistent-normalized-pass1/report.html`:

| Observation | Count |
| --- | ---: |
| Both initial replies readable | 33/36 |
| Initial image-identity disagreements | 9/33 |
| Accepted with both initial orders agreeing | 24/36 |
| Accepted after balanced three-of-four recheck | 3/36 |
| Uncertain, score null | 9/36 |

Uncertainty consists of four fully observed splits, three incomplete initial
pairs and two incomplete rechecks. The eight unlabelled control comparisons
had seven readable initial pairs, one initial conflict and six accepted
final decisions. On the user's five explicit selected preferences the new
final decisions agree twice, disagree once and remain uncertain twice. Three
preferences concern the same fan, so this is not a general accuracy estimate.
The frozen-SFT versus RL-step-16 fan is still wrongly preferred by both
orders; the beekeeper is now uncertain rather than receiving a confident
contradictory score. This repair improves safe aggregation, **not established
visual accuracy**. Previous joint-v2 had 11/36 initial disagreements; this
small, selected and prompt/format-changed panel does not establish that
position bias disappeared. Do not promote the judge on consistency alone.

The new full pass records 112 API attempts with 112 usage receipts, including
all seven reused support attempts once. Provider-reported cost totals
$0.0349368488; `cost`, `gateway_cost` and `market_cost` are alternative fields,
not three expenses to add. Mean/median recorded call latency was 35.54/32.45
seconds over all 112 attempts. There were 25 unsuccessful parsing attempts
(including bounded retries) and no recorded HTTP/provider errors; 36
normalization transformations were retained. The earlier strict identity
pass cost $0.0496900824 with one unknown-usage failure. Thus reported usage
for these two repair passes totals $0.0846269312, excluding that unknown
failure and the separately archived prior calibration. **GPU cost: $0.**
No node, training, recurring automation, or background watcher was launched.

The report uses fixed LEFT/RIGHT positions for every pair, two columns at
all widths, friendly checkpoint names, and display-only translation of
opaque image IDs in explanatory prose. Technical data and raw prose are
preserved. HTML checks found all 36 pair sections and 72 existing image
links, with no swap script. Browser visual verification was not performed.
All 36 focused tests passed. Reviewed implementation commits are `1b9ac1f`
and `6a95a1e`; generated data remain outside Git.

Next decision: retain the joint-sheet identity scorer and balanced conflict
guard as an isolated candidate; unresolved pairs must be censored rather
than treated as ties, visual failures, or zero rewards. Integration into a
future training launcher must explicitly handle null rewards and preserve
full comparison groups. Do not claim the currently closed training run
used this candidate. The remaining aesthetic mismatch calls for a broader
blind human-preference set or judge-capacity comparison, without fitting
the prompt further to only these five selected choices.


Public preservation: [402-file evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/b35a7ec1d8921e145134b4ca16ee38a7f166f48d/runs/judge-alignment-20260930/joint-judge-repair-20260930.tar.gz) at immutable HF
revision `b35a7ec1d8921e145134b4ca16ee38a7f166f48d`, 55,731,287 bytes, SHA-256
`b997e1d664479d2a8dbf352a6181a98e896d770ea65e7d2f0d19b9328055ebf9`. Anonymous streamed download matched the
complete archive hash and byte count. The bundle includes the frozen original
paintings, original joint calibration receipts, new raw calls, normalized
replays, report, immutable manifests and reviewed scorer/report sources.
It contains no credential files. The verification receipt remains in
`local-evidence/consistent-normalized-pass1/public-upload-verification.json`;
the transient local archive was deleted after remote verification.
