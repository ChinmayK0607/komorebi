# Spatial readability teacher candidates

Hypothesis: compact demonstrations with visible subjects, clear silhouette and
occlusion, restrained layering and selective brush edges will address current
RL failures better than adding mechanical detail. This is a finite eight-scene
candidate batch; it does not alter the live run's data or reward.

Teachers: four gpt-6-astra/high and four gpt-5.6-sol/high authored text-to-paint
scenes. Runtime contract: pinned p5.js 2.3.2/p5.brush 2.2.1, 600x600 WEBGL,
one top-left translation, complete sketch per turn. All painting uses actual
brush APIs. Preserve initial code, actual Linux renders, teacher critiques and
revisions. Do not call a prewritten alternate sketch a correction trajectory.
These new prompts are training candidates, never additions to frozen validation.

Matched comparison: each initial painting versus its inspected revision, same
prompt, renderer assets, seed and canvas. Acceptance requires hash-verified
renderability and an actual visual review. Good unchanged first canvases may
finish after inspection; no forced meaningless revision. Keep rendering serial
with low CPU priority on the already GPU-required training node to avoid
competing with the live renderer; no new rental or model inference GPU.
Generated source, images and raw plans remain outside Git in
`painter/data/spatial-readability-20260930`. Only small reviewed receipts and
this experiment record may be tracked. Candidate status is not SFT admission.

Before generation source: root experiment repository commit 33e1957. The
ongoing RL remains bound to source 0772bb6, SFT step512 and its original judge.
Source/program/prompt hashes, rendered scope, failures, render timings,
review outcome and next decision will be recorded after collection. Additional
GPU rental cost: zero new machines; concurrent node time cannot be attributed
as free or split into a separate causal cost without measurement. Agent usage
is separate and no API charge is assumed.


## Actual collection and teacher/root review — 2026-09-30 18:30 UTC

Eight text-only candidates now have eight selected valid final canvases.
There are twelve hash-verified valid renders across the collected turns and
five retained model/render failures: Astra01 first-turn timeout at the 60s
candidate cap; three Astra turn2 global `tint` helper collisions; and Sol04
turn2 with the same helper collision. The timeout does not establish invalidity
under the live 180s cap. A separate Sol04 browser-environment launch error is
operational and retained separately, not model failure. No rendered failure
is exported as a positive SFT target. Invalid first programs and actual error
feedback remain evidence for potential future repair trajectories.

Every final selected image was actually displayed and inspected by its teacher
and root. Three good first-turn images finished after inspection. Five scenes
received render-driven revisions; three needed runtime repair before a valid
revised image. First and final views use the same renderer seed per scene:
Astra93001–93004 and Sol0. Source, PNG, renderer-asset and receipt hashes are
retained; source-defined random seeds remain part of the immutable programs.
Hybrid opaque p5 underpainting with p5.brush texture/accents is explicitly
identified, rather than described as pure brush painting. Rounded readable
forms and cleaner overlap improved; some canvases remain flatter/illustrative,
and the small boat's bow is compact. No held-out training benefit is claimed.

The complete eight-task review manifest is
`painter/collected/spatial-readability-20260930/review/manifest.json`, with
side-by-side initial/final gallery at `review/index.html`. Both are generated
outside Git. Final source/PNG identities and render time follow:

| Candidate | Selected turn | Render seconds | Program SHA-256 | PNG SHA-256 |
| --- | ---: | ---: | --- | --- |
| spatial-readability-20260930-astra-01 | 3 | 2.274 | `0ce2218ee5400cc833c91bc5b897fdf95aa20f1b038f65852d5f21e3a948cf43` | `c3a05c5daaaf34672ad96d90095d845a45992bdf1aa952282991c0e38a4d80de` |
| spatial-readability-20260930-astra-02 | 3 | 2.624 | `8d1300ac5a5541760e5a3a4ef80edb61be4f1305bdd9e29076f2a8c3ae79082f` | `2d7ca5d51ecd836f7cfd1f2bd8ea1de1ca6ae91dc736e9e293ae4d245604f8a1` |
| spatial-readability-20260930-astra-03 | 3 | 2.423 | `44dd098e36b58e117ecd841658f36fffe8875b9a721eb2d2a7e38afb89069ac0` | `fce1cad27844b4021301e579ab4edd186d6954108f86308f02f6ae195b7f9a74` |
| spatial-readability-20260930-astra-04 | 1 | 43.955 | `d223ba6ff33691512f827c2ad274f02c0f1ac265980ac42996c864e86ec593e2` | `cb8a1cfc7a772d7dc76f8334cc3ea2aea192e366c3e8d297bd17d1c1648e7065` |
| spatial-readability-20260930-sol-01 | 2 | 14.242 | `d8e5f52c47911f843cf0662824b3d9c6e07f342432dbd8459b458c1960ed86b2` | `fa9c87a5b733c679ae9f5386cbd91b3a84ca5a088304b505c673603dc99a6c30` |
| spatial-readability-20260930-sol-02 | 1 | 37.835 | `93303b124962aa28401e05ce3e1c622cc7db127bb2422c82210cb33e5a0aa7c4` | `f3540f320af641a62f6e5970bacdb3c3c8ece99155ab8ec023926396d71c0aff` |
| spatial-readability-20260930-sol-03 | 1 | 44.365 | `6da8f0905bd25b45b3d3fcd14b4b49158187d5d63da444c934d132922e7a784f` | `dd1a6719a1702df9792798c48acd7192e258c2c51805f6cbc534060c39773631` |
| spatial-readability-20260930-sol-04 | 3 | 2.727 | `82cec21ea510ceec0edcd1eb7149f6406892cbaa57dea5b87cdd8ff24206df03` | `6998f4622c78fa8ca723b69facedcd4f32bce0f5a2f8fc4a08fc21d5cbc9f59b` |

Review status means teacher/root inspected and retained, with human labels
absent; it is not a human acceptance or training-admission claim. No SFT/RL
training was launched on these samples. Next decision: use these as concrete
candidate repair demonstrations after preserving the actual multi-turn
messages, and measure student validity/visual transfer before claiming the
spatial gap has closed. Finalized prompts/programs/critiques were copied into
the owned node's screened teacher-candidate subtree before RL preservation.
Public upload verification is recorded separately when it completes.


Public preservation verified: the eight selected program/PNG pairs and their
source trajectories are included in the stopped RL run's public HF archive
`CK0607/komorebi-painter-experiments@c4f05445e70442dbab465b667c1854e4f9d206bf`,
path `runs/text-sft512-consistent-20260930/interpreter-retry/evidence.tar.gz`,
SHA-256 `4ca075b3794e2434b8b6eaec0bc2a2884eeaead20bc92d6001466d3ac9cdfbcf`.
The entire anonymous archive and all16 selected program/PNG member hashes
were checked by the parent. They remain explicitly candidate data. The node
was released; no further GPU rendering is pending for these eight scenes.


Complete finalized teacher bundle: public dataset
`CK0607/komorebi-painter-teachers@13c5ea1fc9221eaee7eb06d755b218660f26f4e0`,
path `spatial-readability-20260930/teacher-eight.tar.gz`,139 screened files,
3,049,378 bytes, SHA-256
`f5628221255f51860613555a79a6e80c584f49f47c2766687987c4a2f5011401`.
Anonymous downloaded bytes match. This archive includes the final teacher/root
inspection metadata that arrived after the earlier RL archive snapshot, plus
original/revised code, renderer receipts, valid canvases and gallery. All
teacher candidates remain distinguished from human labels and training gains.
No GPU rental was required for the supplemental publication.
