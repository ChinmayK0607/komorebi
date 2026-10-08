# Smudging diagnosis and proposed next curriculum

Recorded 2026-10-09. Offline source/image inspection only; no new rendering, API judgments, optimizer updates, GPU allocation, or scheduling. The user prefers Flash after viewing the matched ablation. Treat that as the research choice, not a measured population-level win rate.

## Exact example and evidence

The attached image is pixel-identical in RGB to Flash step12, case `t600-349`, turn3: the passenger jet and luggage-loading scene. The archived terminal trace records a valid paint action, with the episode ending at its three-turn cap.

- Program SHA256: `07a8c80567e9670444c69f83a2ea665db4287469b9ca946210005e066395d73d`.
- Canvas SHA256: `ad9ff4e11d8a8e2bbfcb7f0f651f6315e0001fd0504e73fa69d85744fadc5c12`.
- Program and canvas: `local-evidence/completed/flash/validation/step-12-balanced16/eval/evaluation-rollouts/consistent-step-12/t600-349/c768bf44-1183-4d55-9b81-5797410396ce/turn-3/`.
- Authoritative trace: under `local-evidence/completed/flash/validation/step-12-balanced16/eval/`, in its native `traces.jsonl`.
- Public archive and immutable dataset revision: [completed evidence receipt](completed-public-evidence.json).
- Source/model/data/rubric identifiers and matched scope: [protocol](PROTOCOL.json), [results](RESULTS.json).

The final program contains 30 calls to its filled-polygon helper, including overlapping broad sky/tarmac fills and aircraft shading. It never calls `fillTexture` or `fillBleed`. The inspected, pinned vendored export-basis asset (`9c6e271a78e5dcb23c9cfa9d085775317fe5fb2188f976465c83c1f7965d1ec8`) initializes fill texture to 0.8, border strength to 0.5, outward bleed to 0.07, and scatter to true. `fill()` changes color/opacity/activation, without resetting those controls. The registered centroid patch changes only the degenerate-polygon fallback, not these fill settings.

The small foreground elements are also mainly thin pencil contours, while the aircraft is a weakly separated angular polygon. Excessive material effects and weak geometry/value separation are both plausible causes. The current upstream [p5.brush reference](https://github.com/acamposuribe/p5.brush#fill-operations) explains the independent bleed/texture/scatter controls; the pinned local asset is the authority for this run's defaults.

Turn2 to turn3 only relocates the crew block after the safety cones. It does not alter the aircraft geometry, colors, texture controls, or background fills. Valid recorded render times are 36.77, 35.83, and 31.64 seconds for turns1/2/3. These are one-case render timings, not run-level utilization. Each sketch resets the background and redraws a complete version: this is not simply accumulated paint from successive turns.

Conclusion: a concrete material-control omission and ineffective revision are established. Their individual causal contributions to the smudging are not yet measured. Do not claim that a renderer change alone repairs the geometry, or that this proves Flash assigned the example a high training reward: it is heldout evaluation without a recorded quality judgment here.

## Cheap causal check before changing the painting interface

Replay the exact program on Linux CPU using the same renderer, canvas dimensions and seeds. Compare original against bleed-only control and texture-only control, then combined control. For the interventions, test `fillBleed(0.01, 'out')` and `fillTexture(0.2, 0.1, false)` without changing geometry/color/opacity; these are candidate settings, not validated defaults. Preserve original code, intervention diffs, hashes, timings and images. Inspect subject separation, retained brush character and shape boundaries. Repeat on a handful of other weak and good examples so this is not tuned solely to the airport image. No GPU is needed for this replay.

If the controls help, demonstrate explicit background-versus-subject fill settings to the policy. Keep painterly variation and useful soft edges; do not globally remove all texture or silently rewrite old programs/renders. Material control cannot substitute for composition or connected forms.

## Curriculum proposal for discussion

Use `xiaomi/mimo-v2.6-flash` as the judge through the existing Gateway path. Continue text-to-painting first. A proposed next pool is 240 distinct training briefs plus 60 heldout briefs across four levels; these counts are targets, not a claim that a new dataset is built. Split by source/subject family and near-duplicate template, before generating variants. Include multiple palettes, orientations and materials at each level.

| Level | Skills | Example tasks |
| --- | --- | --- |
| 1 | Bounded shapes, readable values, controlled pigment, contact shadow | sphere, mug, fruit, folded cloth |
| 2 | Connected parts, roundness, proportions, negative space | lamp, fan, simple toy, one animal |
| 3 | Relative scale, overlap, support, perspective | two-object still life, simple vehicle, small cabin |
| 4 | Focal hierarchy and spatial organization in scenes | person using a tool, animal pair, street or loading scene |

Start sampling predominantly levels1/2. Increase levels3/4 based on a fixed per-level heldout comparison, preserving roughly a quarter of sampling for earlier skills. Treat that replay fraction as an initial choice, not an optimum. Use opportunities with useful preference differences, including states near the current policy's competence; endless already-solved tasks and uniformly impossible scenes provide little signal. Keep fixed heldout prompts/anchors and separate curves by level. A changing training mixture makes its raw average reward unsuitable as the sole trend measure.

Do not impose a speed/stroke budget yet. First learn finished, readable, attractive paintings. Later add the budget curriculum with matched quality evaluations. Keep reference-to-painting as a subsequent, separately reported extension.

## Judging and credit assignment

Keep a single holistic pairwise decision: a clearly readable subject, coherent connected shapes and spatial relationships, composition, controlled values/color, and attractive brushwork. Judge muddling when pigment/noise destroys separation or identity; a pale readable painting and deliberate soft edges are not defects. Neither additional detail, darkness, warmth alone, nor mechanical precision is an automatic win. Require a concrete visible reason and weaknesses for both images, including the winner. Preserve strict schema, opaque image identities and balanced order checks. Known controls should include readable pale work versus faint wireframe and controlled brushwork versus muddy texture; preserve raw disagreements rather than inventing successful labels.

The current native reward function produces one final-canvas score for an episode. It contains no separate reward for each revision, so that signal alone cannot identify which edit helped. Avoid immediately adding independent bonuses for texture, detail, warmth, turn count or tool use.

Proposed first credit experiment: sample alternative one-action repairs from the exact same brief/history/current canvas, with a keep-as-is choice. Compare each repaired canvas against that unchanged incoming canvas. Train only newly sampled assistant action tokens; frozen prefix, user/tool/image observations get no loss. Do not mix different incoming canvases into a common GRPO group. Include complete first-paint tasks so repair competence does not replace composition from scratch.

A crucial limitation: standard group centering can give the least bad harmful repair a positive advantage when every sampled repair is worse than keeping the canvas. Subtracting a fixed 0.5 from all rewards does not fix this because centering cancels the constant. A signed preference against the unchanged canvas can serve as an action baseline without group recentering, but that is a distinct estimator experiment, not an already-validated GRPO drop-in. Compare it with terminal-only GRPO under matched conditions and verify the actual action-token advantages before scaling. A weak initial block-in that helps later can still be unfairly penalized by local visual comparisons, so local repair credit is for finished-canvas repair states initially, not every early construction turn.

For later free multi-turn RL, use an explicit return/advantage design that preserves downstream credit. Adding adjacent pairwise wins does not guarantee a coherent final objective: pairwise margins are ordinal and can be non-transitive. Keep final quality versus a fixed anchor as the outcome measure, and report revision improvements/regressions separately.

## Next decision

Flash is selected by user preference. First confirm the material-control intervention using preserved programs, then assemble the staged curriculum and compare a narrowly scoped repair-credit change against the current terminal-only method. No new expensive judge sweep is needed. The GPU node remains released; this document does not launch training or change historical results. The per-run GPU ceiling must be made explicit in the next prelaunch record; the prior matched-ablation ceiling was specific to that completed campaign.
