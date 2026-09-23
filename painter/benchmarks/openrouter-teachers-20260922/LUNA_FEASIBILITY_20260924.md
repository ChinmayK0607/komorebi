# GPT-6 Luna high teacher feasibility, 2026-09-24

## Question and baseline

Can GPT-6 Luna at high reasoning produce reference-faithful, attractive
brush-program paintings that justify using it instead of MiMo V2.6 Pro for
the higher-quality data tier? This is a **screen**, not a replacement decision.
The matched references are COCO128 IDs `110` (restaurant people and pizza),
`247` (airplane), and `520` (harbor and birds). These were in the existing
training-reference pool, so this is not an unseen-generalization test.

The MiMo Pro quality-track baseline for `110` is the complete final canvas in
public `full-quality-03-20260923`. For `247` and `520`, the latest candidate
was interrupted by renderer timeouts; public, provider-free replays in
`renderer-replay-full-quality-09-20260923` and
`renderer-replay-full-quality-03-20260923` recovered the saved programs.
Those offline canvases were never seen by MiMo for a subsequent revision.
MiMo had up to 12 benchmark turns; the Luna Codex-agent screen below begins
with one initial sketch, so the two are not a matched policy comparison.

## Source and scope

Three initial Luna high programs were created by a Codex `gpt-6-luna` agent
after visually inspecting the original JPGs. They are public at dataset
`CK0607/komorebi-painter-teachers`, immutable commit
`12f9cdb1d02d56264949f37664808db658af5e94`, under
`inputs/luna-high-smoke-20260924/`. The manifest SHA-256 is
`5cdc0011d6536f2444e923a6e0238c4611a1ee9f1664216e5b6221f4a2c147e8`;
the three source files were independently fetched anonymously and hash
checked. Generation used Codex agent scaffolding, so its API token usage and
dollar cost are unknown. It does not establish that the plain Gateway model
behaves the same way. No GPU was rented for this screen.

Linux rendering uses the pinned Watercolour renderer and source commit
`be61cb5` on `codex/painter-teacher-cloud`, a 600-second deadline and three
workers. The first provider-free render task at `b0e4e55` stopped before
rendering because its validator incorrectly required a 64-character SHA-256
for a 40-character Hugging Face Git commit. The fixed validator at `be61cb5`
was tested against the public manifest before resubmission. The first failed
task made zero model calls and produced no painting or result receipt.

A separate **paid** Gateway episode on reference `110` uses
`openai/gpt-6-luna`, explicit `high` reasoning, the same quality-track
paint/render/revise contract, and up to 12 turns. Its source is `c091a95`; the
task-local config override and effective config hash must be preserved in the
episode receipt. This compares an actual API teacher with MiMo on one matched
reference. The model ID is an alias; no immutable provider weight revision is
available from this setup. The known public token list price was
`$0.10` input and `$0.50` output per million on 2026-09-24, but actual run
cost is not inferred if provider fields are absent.

## Decision rule

Check valid render count, reference structure, beauty, and pairwise visual
preference on the same image. Keep invalid, timed-out, and interrupted cases
visible. Compare actual paid calls, tokens, elapsed time and cost per accepted
demonstration when available. A good first-pass Codex-agent painting alone is
insufficient to replace MiMo Pro; the API episode and subsequent independent
references must support that decision. No speed curriculum is part of this
screen. The verified results and next decision follow.

## Verified rendering and API results

The three independent Codex-agent programs were rendered concurrently in
finite cloud task `task_e_6ab428322e30832bb9bfa874024fb015` at source
`be61cb5`. All three timed out at the 600-second per-program deadline. The
complete ten-file archive was anonymously fetched and verified against public
receipt `runs/luna-high-initial-20260924/receipt.json`, bundle SHA-256
`66996c3bde97a3557cf601dd9dddf4c5a22772823601df5089c88fe464918ee5`,
dataset commit `1c15871f39b0adadedf9e38aeb9a1836f7429054`. There are no
canvases from that batch. The same exact programs were rerendered on one
software-rendered Chromium worker, with a 900-second deadline. All 3/3 were
valid, taking 318.019, 236.593 and 228.534 seconds for references 110, 247
and 520. Its public run `luna-high-sequential-20260924` was anonymously
reconstructed and hash-verified: bundle SHA-256
`6bdba599ef0da7304970b1a0876fa18a748b61f3fb8ba57774899040411e37c5`,
dataset commit `b79278b4405cbc2d6e2a32e70208a4c7ef1774ab`, source
`cb1c196`. This isolates contention between three concurrent software
renderers as the likely cause of the first failure; it does not imply that
sequential rendering is optimal across independent VMs.

The Codex Luna painter inspected those actual canvases and revised each
program once. The revised source manifest is public at dataset commit
`1c9abfcfbf4482a18384673354d54f66f63eaad3`, SHA-256
`d6999f58ab7516f87ece53e2ea0cf3a3fc74ceb4e66426b4e6557ed3d329d114`.
Its provider-free, one-worker render at source `94d4baa` produced 3/3 valid
canvases in 95.536, 72.397 and 70.743 seconds. The public run
`luna-high-revision-20260924` was anonymously reconstructed and hash-verified:
bundle SHA-256
`abe70173f4abf96a20973cb3e3eb3547d649aacce2dc2b71aec3a0cf26abb37e`,
dataset commit `62705bca316d8c5e979076609ac6fd48acd987b5`.
The Codex-agent generation cost remains unknown. These are one-step visual
revisions, not the same Gateway API scaffold used below.

Three **paid** Gateway episodes used the identical quality-track prompt SHA-256
`dc030b846c01bc6a68be4d5aa6371b42d0814306a6c5e0eb23336c0e09d8eacb`,
effective config SHA-256 `591a53f3d83b830a72075a77050d0288627141bbbf7b836a066513dae099563a`,
explicit `high` reasoning, 12-turn cap and 600-second render override. Source
was `c091a95` for 110 and `cb1c196` for 247/520; the latter only added
documentation. The model ID was the mutable `openai/gpt-6-luna` alias, with
no immutable provider weight revision exposed. Each run made exactly 12
fresh provider calls and hit the turn limit, rather than finishing early.

| Reference | Valid renders | Prompt / completion tokens | Active seconds | Listed-rate cost estimate | Public bundle SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Restaurant 110 | 9/12 | 471,933 / 97,657 | 1,051.118 | ~$0.096 | `0c340822b247b7ba71ce36193f4a8144a3adb05cb0af5f003ab701146257bcb4` |
| Airplane 247 | 11/12 | 425,183 / 90,987 | 1,423.130 | ~$0.088 | `b1a81b24b8df8a7881887766591decbba57b4c68dd151b792a45851e6b5670c8` |
| Harbor 520 | 6/12 | 301,873 / 70,104 | 613.800 | ~$0.065 | `a1944514212dc2f203d3efd2ac0fe8e208d11949e9ce52fa3f79b72de2425983` |

The corresponding public run IDs are `luna-high-api-coco110-20260924`,
`luna-high-api-coco247-20260924` and `luna-high-api-coco520-20260924`.
All three bundles were anonymously reconstructed and hash-verified. Their
dataset commits are, in row order,
`53e3e5efd8f16f5e216c99643260b05e7e2f080d`,
`d70d962c2bdea803258153c90c0e5f9771fcc92e`, and
`477edc7dc2738bdee79876fc5a2a7de15fc48789`. Provider cost fields were
missing on every turn, so the table's dollar amounts are **not measured
charges**. They apply the publicly listed [Luna](https://vercel.com/ai-gateway/models/gpt-6-luna)
$0.10/$0.50 input/output and [MiMo Pro](https://vercel.com/ai-gateway/models/mimo-v2.6-pro)
approximately $0.44/$0.87 per million input/output tokens, checked
2026-09-24. Billing details or discounts can change the actual amounts.

The matched MiMo Pro restaurant episode completed ten turns, nine with valid
canvases, using 691,442 tokens and 4,074.278 active seconds. At the same
listed-rate calculation it is about $0.36, versus Luna's ~$0.096. MiMo Pro's
airplane and harbor episodes were interrupted after four and five turns by
renderer errors; their saved latest programs were replayed offline. Their
partial token costs must not be compared with Luna's complete 12-turn costs
as if both had the same training opportunity.

## Visual judgment and decision

The [local comparison gallery](results-ai-gateway-20260923/luna-high-comparison/index.html)
shows each reference, the best available MiMo Pro canvas, Luna's Codex-agent
first pass and revision, and Luna's paid API final canvas. Parent inspection
is unblinded and qualitative:

- **Restaurant:** The paid Luna episode corrects a confused first canvas into
  recognizable people and pizza, but its final image is flat. The Codex-agent
  brush version has attractive wash texture but a conspicuous dark gap and
  weak people/hand anatomy after one revision. MiMo Pro is the better
  finished painting to this reviewer.
- **Airplane:** Luna's paid final canvas reproduces the frontal aircraft,
  propeller, landing gear and apron structure well. It is graphically stiff;
  the Codex-agent revision is more painterly but hazy. The MiMo Pro canvas
  has richer brush texture, while Luna's paid version is arguably closer in
  geometry. The result is mixed, not a decisive overall win.
- **Harbor:** Luna's paid final canvas contains the bridge, birds and water
  but reads like crisp vector art. Its Codex-agent brush version has more
  atmosphere, though the pier is too dark and the scene is foggy. MiMo Pro
  better balances structure and painterly finish here.

The paid Luna finals relied on ordinary p5 filled shapes for large color
fields, despite the shared prompt's preference for `brush.fill` polygons;
the MiMo Pro restaurant final used brush fills. This likely explains some
of the aesthetic gap, but is an inference from these programs, not a proven
model limitation. Luna can be a useful lower-listed-price **structural**
teacher, especially for hard objects. The present three-case evidence does
not justify replacing MiMo Pro for the higher-quality aesthetic tier. The
next bounded experiment should compare a brush-fill-specific Luna prompt or
distill the Codex-agent visual revision scaffold, then apply a blinded
pairwise reference-and-beauty filter to select demonstrations. Keep MiMo Pro
as a candidate rather than paying both on every image. No GPU rental,
scheduled task, or recurring watcher was used in this feasibility screen.
