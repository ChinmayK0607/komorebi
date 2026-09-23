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
screen. Results and the next decision will be appended after public receipts
are verified.

## First observed results

The three independent Codex-agent programs were rendered concurrently in
finite cloud task `task_e_6ab428322e30832bb9bfa874024fb015` at source
`be61cb5`. All three timed out at the 600-second per-program deadline. The
complete ten-file archive was anonymously fetched and verified against public
receipt `runs/luna-high-initial-20260924/receipt.json`, bundle SHA-256
`66996c3bde97a3557cf601dd9dddf4c5a22772823601df5089c88fe464918ee5`,
dataset commit `1c15871f39b0adadedf9e38aeb9a1836f7429054`. There are no
canvases from that batch. A finite, sequential one-worker rerender with a
900-second deadline is running as `luna-high-sequential-20260924`; this tests
whether three simultaneous software-rendered Chromium instances caused the
timeouts. It makes no model calls. The initial timeout is censored evidence,
not a visual-quality score.

The paid Gateway episode on reference `110` completed at source `c091a95`
with task-local high reasoning, quality-track 12 turns and a 600-second
renderer override. Its public archive was anonymously fetched and SHA-256
verified: run `luna-high-api-coco110-20260924`, bundle
`0c340822b247b7ba71ce36193f4a8144a3adb05cb0af5f003ab701146257bcb4`,
dataset commit `53e3e5efd8f16f5e216c99643260b05e7e2f080d`. It made exactly
12 fresh provider calls and produced nine valid canvases. Turn 2 supplied no
complete program; turns 6 and 10 failed on undefined functions. The last
valid canvas is turn 12, with a recognizable mother, child, restaurant and
pizza. The first valid canvas was visually confused; later turns corrected
the structure, but the result remains flat and graphic compared with MiMo
Pro's softer, more painterly matched-reference final canvas. This is the
parent's unblinded assessment, not a calibrated pairwise judge result.

The Luna run used 471,933 prompt and 97,657 completion tokens (569,590
total) and 1,051.118 active seconds; the matched MiMo Pro quality episode
used 559,073 prompt and 132,369 completion tokens (691,442 total) and
4,074.278 active seconds. The provider supplied no actual dollar cost for
either run. Applying the publicly listed per-token prices *only as an
estimate* gives about $0.096 for Luna and $0.36 for MiMo Pro on this
reference, with possible billing differences unmeasured. The list prices
were [Luna](https://vercel.com/ai-gateway/models/gpt-6-luna) $0.10/$0.50
and [MiMo Pro](https://vercel.com/ai-gateway/models/mimo-v2.6-pro) about
$0.44/$0.87 per million input/output tokens when checked on 2026-09-24.
This one example supports testing Luna as an inexpensive structural teacher,
but does not support replacing MiMo Pro for the aesthetic tier. Paid
high-reasoning episodes on the matched airplane and harbor references are
underway before a broader decision.
