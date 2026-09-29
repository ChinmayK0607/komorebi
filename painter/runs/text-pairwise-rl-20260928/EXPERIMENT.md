# Text painting: fixed-SFT pairwise RL

Status: the finite 20-update graded-margin run and all predeclared held-out
checkpoint evaluations are complete. Step 20 had a small positive result on
the original validation set, but an independent reserve check tied SFT. No
confirmed visual-quality improvement is claimed or promoted.

Rental update (2026-09-28 15:04 UTC): a two-96-GB-GPU node was rented at
**$2.38/hour total** with a ten-hour automatic rental expiry (maximum quoted
GPU rental $23.80). This is a finite rental limit, not a recurring watcher.
Setup completed after removing an unnecessary `pip --version` probe from the
uv-managed Prime runtime. The corrected SFT baseline and fixed anchors are
complete; no RL training result has been observed yet.
The two-orientation Gateway smoke judged an obvious red-circle-versus-blue-circle
pair correctly (score 1.0 for red under a red-circle brief). This confirms
endpoint/schema/image delivery, not judge reliability on real paintings.

The first baseline attempt was stopped after its earliest completed traces
showed repeated full HTML documents and invalid attempts to construct
`p5.brush`. One inspected final canvas was a weak but recognizable towel roll;
the failures were chiefly format/API errors, not a scored RL result. The
attempt is retained separately. The text-only system prompt and missing-code
feedback now state the renderer's JavaScript-only/global-brush contract; all
baseline prompts must be regenerated under this changed protocol before RL.
During the corrected baseline's 15:46:08–15:50:58 UTC sample window, both
inference GPUs read 100.0% mean utilization (30 ten-second samples per GPU,
all 30 at least 90%). No trainer was running in that window. This is a
five-minute baseline measurement, not whole-run utilization.
The corrected first pass completed all 72 briefs with 55 trace-level valid
canvases and no trace-read errors. The stricter anchor admission accepted 54;
its frozen manifest reached 40 train examples using predeclared same-tier
reserves but only 12/16 held-out anchors. The 18 unusable cases (11 original
train, four held-out, three reserve) were sampled once more under seed 918201.
The retry produced 16 valid trace-level canvases from 18 attempts. All 16
held-out anchors are now frozen; the 40-row training manifest SHA-256 is
`72953465cec6888a20d50aea326162fe08da4fc523ba566fb8f750e930b11a9a`
and the 16-row validation manifest SHA-256 is
`c59aa89b22ceb6dccf13367cb93fa29a1c35592f13c63306b7e1ddb4b18a9075`.
First valid means first valid in fixed attempt order, not the better looking
attempt. Visual spot checks of the first-pass acorn/leaf and chameleon show
recognizable but simplified subjects; there is no RL comparison yet.
The independent step-0 sample used the same pinned step-512 adapter and the
held-out 16 briefs with a fresh fixed seed, 918210. It scored 0.423 on 13
comparable cases: four wins, three ties, six losses against the frozen SFT
renders. Two renderer service failures and one malformed judge response were
censored. The raw result predates the one-retry fix for malformed judge
responses and is retained unchanged. Training began at about 16:55 UTC; the
first two admitted batches had 8/8 trainable traces and effective reward 0.50
and 0.3125. The first gradient norm was 0.0466, showing a real optimizer
update, but not validation improvement. From 17:02–17:22 UTC, inference GPU
utilization averaged 99.9% and trainer GPU utilization 6.2% across 122
ten-second samples per GPU. The pipeline is inference-bound.

At step 4, the pairwise train reward fell to 0.0625 after 0.50, 0.3125,
and 0.375 in steps 1–3. The trainer confirmed four nonzero-gradient updates,
but spent 14m55s waiting for its fourth batch versus 58.3s of active work.
The step-4 LoRA was uploaded publicly and hash-verified at revision
`6066f125e0e15ec69b52b7799219afb29d2fa062`. The first attempt was
stopped after that checkpoint for a full held-out comparison. On all 16
validation briefs, step 4 had 13 comparable cases (3 wins, 3 ties, 7 losses),
10 valid candidates, 3 censored, and mean reward 0.346. The independently
sampled SFT step-0 control had 13 comparable cases (4 wins, 3 ties, 6
losses), 12 valid candidates, 3 censored, and mean reward 0.423. On the 11
cases comparable in both runs, both means were 0.409. The checkpoint is not
promoted. Among 82 training-attempt traces, 38 were visually scored, 32 had
no valid canvas, and 12 failed due to renderer deadlines. The invalid
programs include missing code blocks and JavaScript syntax errors; this is
not explained by the 8192-token ceiling, as only one invalid model call ended
at length. The first attempt's [public evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/676d60b189df2cff02459138fdb9682efb85ba5f/runs/text-pairwise-rl-20260928/first-step4-evidence.tar.gz)
contains 768 screened files (48.4 MB), including frozen prompts/anchors,
rollouts, traces, judge receipts, validation summaries, and telemetry. Its
anonymously verified SHA-256 is
`9359749a69aad9724b4548bcc884d9e3de87798b8d707ac20dbaa90498cf1eef`.

A second attempt starts from the same step-512 adapter, frozen 40-row train
manifest, and visual comparator. Its only reward-path repair classifies a
completed renderer receipt with a hard program timeout as an invalid model
action with feedback; absent workers and other service failures remain
censored. It has a distinct Prime run name and output root so the first
attempt is immutable. The node preflight passed source/task serialization
and the 1-trainer/1-inference GPU roles. Spend was $8.14 at restart, still
under the $25 GPU rental cap. The default eight-case pilot contains three
censored cases in the step-0 control and is too small to decide promotion.
Instead, check the corrected attempt on the full frozen 16 after step 8,
then only repeat at a later checkpoint if the first comparison and remaining
budget justify it. Report every checked checkpoint and selection explicitly;
neither rising training reward nor a subset alone promotes the adapter.
The second attempt was stopped before an optimizer update after a curriculum
ordering audit: the task loader's default family interleave alphabetically
put intermediate tasks ahead of simple ones. Its earliest completed traces
all lost to the strong SFT anchor, so continuing would not test the intended
easy-to-hard hypothesis. The third attempt preserves the frozen manifest
order (24 simple, then 16 intermediate) and retains the render-timeout
classification fix. It again starts from step 512 with a distinct run name;
preflight must show a simple brief first. Validate on the full frozen 16
after an informative checkpoint and report all attempts, including this
stopped one. No corrected checkpoint has been promoted.

The third attempt was also stopped before an optimizer update: after eight
finalized rollout units Prime reported no admitted training payload, because
the pairwise win/tie/loss reward was uniformly zero within early groups.
This is a reward-resolution failure, not a compute or optimizer failure.
The [margin protocol](../../text_pairwise_margin_judge.py) retains the same
two-order comparison to the frozen SFT painting and the same visual rubric,
but asks for one seven-level overall preference strength rather than only
win/tie/loss. It is not a sum of aesthetic or structure subrewards. In a
12-pair calibration, the two visually inspected improvements scored 0.667
and 0.833, an inspected stapler regression scored 0.250, and eight candidates
that the old judge had scored zero took four distinct values from 0.083 to
0.500. The maximum difference between reversed views was 0.500; this remains
a noise limitation. Identity scored 0.500 and an all-white blank scored 0.000
with both orientations in agreement. The same saved step-0 SFT validation
canvases were rescored under the margin protocol: mean 0.393 on 14 comparable
cases, 13 valid candidates, 2 censored. A fourth attempt, named
`text-pairwise-margin-20260928`, passed preflight from the original step-512
adapter with simple briefs first and started when GPU rental spend was $9.88.
Its reward and checkpoint evaluations must use the margin protocol; old
binary-judge validation scores are not comparable to it.
The fourth attempt was stopped before an optimizer update after its training
worker revealed a Python import-path error in the new judge module. Four
visual scoring attempts failed as service errors; none were assigned visual
losses. The fallback import was fixed at source commit `7777de8` and verified
inside an isolated Prime-worker-style Python path, then by a real two-order
MiMo judgment of an inspected seahorse pair (score 0.667, no order gap).
The fifth attempt, `text-pairwise-margin-workerfix-20260928`, preserves the
same frozen manifest, step-512 initializer, easy-to-hard order, graded
pairwise protocol, and render-timeout classification. It passed preflight
with the simple first brief. Its first two eight-rollout batches completed
with train rewards 0.0938 and 0.1562, respectively. The trainer logged
nonzero gradient norms 0.0064 and 0.0190, mismatch KL 0.0003 and 0.0004,
and no rollout truncation. These batches covered different training briefs;
their reward difference is not a within-task learning curve. One judge
response with contradictory winner and strength survived a retry and was
censored as a service failure. No held-out improvement is claimed until a
checkpoint is evaluated against the frozen 16-case margin baseline.
Steps 3 and 4 then produced train rewards 0.0417 and 0.3958 on further
different briefs, with nonzero gradient norms 0.0099 and 0.0354. The large
step-4 batch reward is not a held-out gain. The step-4 adapter was uploaded
publicly and hash-verified at Hugging Face revision
`8a990737e719e86d8f91125c951d15066200d71c`, path
`text-pairwise-margin-workerfix-20260928/step_4`, with adapter weight SHA-256
`1279982e1f6d1dbc53c2bad82ced7fee0e2b447989ef1ec228fce6a700eb4c2a`.
Step 8 also completed and uploaded publicly with hash verification at revision
`90e49aadce95d09fd402dbc1d6a63fc6477a26fb`, path
`text-pairwise-margin-workerfix-20260928/step_8`, adapter weight SHA-256
`496ccb13a01575e3c0ddc82236dc4c41d809f6e32001ffe9de82dc76ff3a32f6`.
Its batch reward was 0.1354 after step 7's 0.2083, again across different
briefs; step 8's trainer gradient norm was 0.0092. The node's reported total
rental spend was $13.77 at approximately 20:51 UTC, still below the $25 cap.
Step 12 completed on the progressing curriculum with batch reward 0.2396,
trainer gradient norm 0.0242, and a public hash-verified adapter at revision
`bab8a97dc9871ef436980e5c5b7eee9c4ad86665`, path
`text-pairwise-margin-workerfix-20260928/step_12`, weight SHA-256
`12e959e5e28e469b7926e15cc5aabf5e4f9a1668b8a3c1c95ab69ed86a7c7fd3`.
This remains a training signal, not a held-out quality result.
For this attempt, compare the public step-4, step-12, and final step-20
adapters on the same full 16-case held-out set under the margin protocol,
subject to the rental deadline. This spacing was chosen before seeing any
checkpoint validation result. If the finite run ends early, use its latest
public checkpoint in place of step 20. Keep all checked results, including
regressions, and reserve time for evidence upload and pod termination.
During the active fifth attempt from 19:34–19:57 UTC, GPU 0 (inference)
averaged 99.46% utilization with 99.3% of 143 ten-second samples at or above
90%; GPU 1 (trainer) averaged 10.04%, with 9.8% of samples at or above 90%.
This measures a 23-minute interval, not whole-run saturation. It confirms
that inference rollout production, not judge latency or gradient compute,
currently bounds throughput on this two-GPU node.
The three stopped attempts' partial rollouts, configs, raw traces, logs, and
GPU telemetry are in a [public, anonymously hash-verified archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/5d3e58d94c02b989de1df767d9ccc382736597bb/runs/text-pairwise-rl-20260928/stopped-attempts-evidence.tar.gz)
of 299 screened files, SHA-256
`4491bc4ee89c68d3d7a6566513f0baa3cf97ef3c8597c185aa93a2d9f89e3cd9`.
Gateway usage reported $0.007977 for the 24 calibration image-order calls,
$0.000351 for the two blank-control calls, and $0.008493 for 26 margin
step-0 rescoring calls. These are receipt totals only; any failed/retried
requests without usage receipts are not included. Vercel spend is separate
from the $25 GPU rental cap.
The 12-pair calibration, blank control, image inputs, and raw judge receipts
are in a [public, anonymously hash-verified evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/f24a696d9a6cbe060b88faffc0f3cc034d876201/runs/text-pairwise-rl-20260928/margin-calibration.tar.gz)
with SHA-256 `b0fbd04e54fb69222ec273ee432e2db7b24081b0551694b71eb42ce3610deda9`.

Data scaling has been staged independently of this live 40-brief run. A
deterministic expansion of the same audited source contains all 290 distinct
text briefs: it preserves all 72 original assignments and prompt hashes,
adds 186 complex briefs to train and 32 complex briefs to blind validation.
The expanded pool is `prepared/expanded/pool.json`, SHA-256
`181d7fc907681b5d9729edf1e8be4091b775ecd1a33086e6b96c1a64d2b06406`.
Its [public, anonymously hash-verified archive](https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/blob/a9ea9bc36ab9bbed6e8abe4313f88367aee69810/pools/text-paint-expanded-20260928.tar.gz)
has archive SHA-256 `5c249648cd5f1c041967070e89362b6750cb9833d3d52f6ea8d75746e002c2ee`.
These 218 additional briefs have **not** been SFT-rendered or trained. Their
inclusion in a later RL run requires separately frozen, hash-verified SFT
opponents and a same-protocol step-0 control; the 32 new validation prompts
must remain blind to training. This staging makes the scaling option concrete
without changing the current experiment's baseline or splits mid-run.

## Independent reserve confirmation (specified before its results)

The step-20 checkpoint has a small positive paired signal on the original
16-case validation set. Check it on all **14 original reserve briefs unused by
training or that validation set**, without selecting cases by SFT or RL
quality. The fixed reserve pool SHA-256 is
`032dc136011d357414b9004b5240da183543b17239c5a6e644094ec401d0d791`.
Sample the same pinned step-512 SFT adapter and the public step-20 RL adapter
once each, on the same brief and three-turn/8192-token settings, with seed
`918220`. Record invalid canvases and operational failures separately.

For each case with two valid canvases, judge the fresh pair with the same
two-order MiMo margin protocol. Count RL-valid/SFT-invalid as 1.0,
RL-invalid/SFT-valid as 0.0, and both invalid as a 0.5 pairwise tie while
reporting both validity rates. Service or judge failures are censored.
Report the all-comparable mean, the visual-only mean, and a deterministic
case-bootstrap 95% interval; do not call the checkpoint a confirmed win if
the interval includes 0.5 or visual review contradicts it. This is a
confirmatory comparison, not another training or checkpoint-selection set.

### Observed outcome

The fifth attempt completed all 20 optimizer updates from the pinned
step-512 SFT adapter. Each update admitted eight trainable rollouts, and
trainer gradient norms were nonzero, including 0.0337 at step 20. The
public, hash-verified [step-20 adapter](https://huggingface.co/CK0607/qwen3.8-27b-brush-painting/tree/7da0d70bc595625375f6912ea64c7fda90ee6f67/text-pairwise-margin-workerfix-20260928/step_20)
has weight SHA-256
`37fcfc2cae2c226b9e757ad3191148cdbc3a7dc279f4fda9ff2ddb2b3ccfe4ad`.
The training batch rewards cannot serve as a learning curve: their briefs
change through the curriculum, and speculative prefetch sometimes cancels
stale rollouts.

Under the **same margin judge and on the 14 cases comparable with the
independent step-0 SFT sample**, SFT averaged 0.393. RL step 4 averaged
0.345 (paired difference -0.048; 12/16 valid), step 12 averaged 0.357
(-0.036; 14/16 valid), and step 20 averaged 0.429 (+0.036; 15/16 valid).
The step-20 paired case-bootstrap 95% interval for its difference from SFT
was [-0.167, +0.226]; two SFT service/judge failures were censored, not
visual losses. All three RL evaluations covered the full 16 prompts and had
zero censored cases. Step 4 and step 12 are regressions, and step 20 is only
a weak positive signal. The complete visual comparison is in
`validation/checkpoint-gallery/index.html`.

The separate 14-reserve comparison was sampled once per policy at seed
918220, after excluding reserve IDs used as train replacements. The SFT
first-sample anchor manifest SHA-256 is
`b35c19a9f4e3a6f02cbdc0ea934c2e7f9350b8fe21299eef21fb205c537243ee`.
Both SFT and RL yielded 13/14 usable canvases; 12 cases had two valid
canvases and their mean visual score was exactly 0.500. Including the two
one-sided invalid cases as predeclared, the all-14 pairwise mean remained
0.500 (six RL wins, two ties, six losses; bootstrap 95% interval
[0.363, 0.643]). There were no censored reserve cases. Summary SHA-256 is
`00cd3ce165eb8d475a26961be9545389ef5f414cebc14340d706296269191747`.
The reserve visual comparison is in
`validation/independent-reserve-14/gallery/index.html`; its 13 cards cover
the SFT-valid subset, and `anchors/audit.json` retains the one SFT-invalid
case. This independent tie does **not** confirm the step-20 gain.

Visual inspection supports the mixed result: the RL chalk-sticks painting
is more legible than its fresh SFT counterpart, while its pinecone loses
the overlapping-scale structure. On the original validation set, some RL
paintings improved when valid, but step-4 plan-only responses ended normally
after about 60 tokens with no JavaScript block despite an 8192-token cap.
Another invalid response repeated the same `scale is not a function` error
on all three turns. Action validity and effective correction remain live
failure modes, not token-budget truncation.

From 19:34 to 22:20:18 UTC during this fifth training attempt, 998
ten-second samples **per GPU** gave inference GPU 0 mean utilization
99.82% (99.8% of samples at least 90%) and trainer GPU 1 mean utilization
11.65% (10.7% at least 90%). This is a near-full training-window sample,
not an instantaneous reading: the pipeline was strongly inference-bound.
MiMo Gateway receipts total $0.068924 for 110 training judgments,
$0.028694 for 41 original-validation judgments, and $0.008098 for 12
reserve judgments ($0.105716 combined); failed calls without usage receipts
are excluded. GPU rental was **$21.53 at the final pre-removal snapshot**,
below the $25 per-run cap and $23.80 automatic whole-node maximum. The
node was removed afterward; `lium ps` returned no active pods.

The complete [public evidence archive](https://huggingface.co/datasets/CK0607/komorebi-painter-experiments/blob/180ea85f472f39dd7506adff4c764a7e5fc6053f/runs/text-pairwise-rl-20260928/margin-final-20260929.tar.gz)
contains 2,083 screened files (147,628,262 bytes): training traces and
metrics, rollouts, GPU telemetry, exact scoring receipts, all three held-out
evaluations, reserve samples, audit and both visual galleries. An anonymous
download at immutable revision
`180ea85f472f39dd7506adff4c764a7e5fc6053f` matched SHA-256
`de7a74c8a3c08096f7dec6b8942cde9c14c2c34b4ad70e88d248692e7746aee3`.
The post-run Prime stream was decompressed from `.jsonl.zst` and screened
as JSONL before packaging; model weights remain in the separately verified
public checkpoint repository.

Decision: preserve step 20 as a research checkpoint, but do not promote it
as a demonstrated improvement. For the next bounded run, first repair the
action/correction scaffold and establish code-output validity on a small
fixed set, then freeze SFT opponents for a substantially broader training
curriculum from the already published 290-brief pool. Keep its 32 new blind
complex briefs out of training, match the SFT initializer and validation
settings, and respect the $25 GPU cap per run. More inference workers should
take priority over more trainer GPUs given the measured imbalance.

## Hypothesis and matched baseline

On text painting briefs, Qwen3.8-27B initialized from the photo-curriculum
SFT step-512 LoRA may learn to paint better than that *same* frozen LoRA when
each three-turn rollout is compared against a pre-rendered step-512 painting
for the same brief. This is an experiment, not a guaranteed improvement.

Base model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Initializer: public `CK0607/qwen3.8-27b-brush-painting` revision
`adf687e5b321514eda7be6eab050434fdc8e44ff`,
`photo-curriculum-sft-20260922-balanced/step_512`, weight SHA-256
`bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740`.
Prime revision: `26b3131d2716a4f8210b165df584b83b4bc54f61`.
Matched baseline: that initializer sampled with exactly the RL evaluation
prompt, renderer, three-turn limit, temperature, and per-turn token ceiling.

## Data and reward

The frozen prompt pool has 72 authored, unique text briefs from the 600-scene
teacher visual audit: 40 train (24 simple, 16 intermediate), 16 validation
(8+8), and 16 same-tier reserve. The source audit SHA-256 is
`ea018fce4336c6c3de2dc0ee496b141f45162dc5f3e7428294ea6398b8c2f8f8`;
the selected pool SHA-256 is
`c91ca18440e054d360ac45b071066c7c5c94c62cf5526dfb3791d97ae07fc7fb`.
Teacher paintings/ratings are not RL rewards or held-out student results.

Before training, render step-512 on every brief and hash the fixed canvases.
If the pinned single-sample baseline produces too few valid anchors, rerun
**only invalid briefs** with the same policy, prompt, renderer, and decoding
settings but a predeclared different seed. Freeze the *first valid* canvas in
attempt order, never the prettiest judged canvas. Keep the single-sample
validity rate and retry count separate in results; the comparator then is a
validity-conditioned SFT baseline rather than a matched one-sample baseline.
The retry command is `prepare_retry_pool.py --pool prepared/pool.json --source
baseline --output prepared/retry-pool-1.json`, followed by
`PAINTER_POOL_PATH=$PWD/prepared/retry-pool-1.json
PAINTER_EVAL_OUT=$PWD/baseline-retry-1 PAINTER_EVAL_SEED=918201
PAINTER_BUILD_ANCHORS=0 bash render_baseline.sh`. After retries,
`build_anchors.py --pool prepared/pool.json --eval-root baseline --retry-root
baseline-retry-1 --output anchors-frozen` freezes the first valid result.
Training rows with invalid or blank anchors are replaced only by predeclared
same-tier reserve rows, if possible. Report all rejected rows. Validation
membership is never selected using RL outcomes. The frozen SFT painting for
each training row is the only visual comparator. The MiMo-v2.6-Flash judge
sees the brief and a two-image contact sheet in both A/B orders. Each view
votes win/tie/loss; their mean is the terminal reward. Invalid model output is
zero. A judge or renderer outage is censored, not scored as bad art. There is
no detail bonus, within-policy rank, or speed penalty in this run.

## Run and decision

Candidate configuration: 1 training GPU + 1 inference GPU, 4 rollouts per
brief, batch 8, up to 20 updates, LoRA learning rate `1e-6`, checkpoint every
4 updates, maximum 3 painting turns, maximum 8192 completion tokens per turn.
This is a finite run with a **$25 GPU rental cap for this run**, inclusive of
setup, SFT pre-rendering, training, evaluation, and public checkpoint upload.
The node hourly quote and timestamp must be recorded before rental; stop
before the cap if throughput cannot complete the plan. Vercel judge spend is
recorded separately; the user specified $25 as GPU spend only.

Evaluate the fixed validation set at step 0 and available checkpoints using
the matched generation settings. Report validity, paired judge win/tie/loss,
reward by update, unique training briefs reached, censored episodes, GPU busy
fractions by role and time window, wall time, and cost. Promote only a
checkpoint with held-out improvement over the frozen step-512 baseline and no
material validity regression. If no checkpoint qualifies, keep the SFT
baseline and use the failure evidence to choose the next change. Do not
describe rising training reward alone as generalization.
