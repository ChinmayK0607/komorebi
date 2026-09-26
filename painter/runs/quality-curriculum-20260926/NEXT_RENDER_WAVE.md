# Finite diverse first-paint render wave, 2026-09-26

**Hypothesis.** The already-authored simple CC0 references and selected text prompts may contain enough legible first paintings to seed more actual-canvas inspection and correction. Renderer validity alone is not a positive label. We will render and visually triage every output, then correct a diverse subset against the exact observed canvas. The 11-image screened-CC0 pilot is an operating baseline for render validity/throughput, not a matched visual-quality comparator because the inputs differ. For every future correction, the matched baseline will be its own first-turn canvas on the same reference or prompt.

The candidate wave has 28 disjoint simple CC0 image references across objects, animals, landscapes, food and architecture, plus 20 text prompts across architecture, transport, food, weather, activity and animals. All source archives were publicly uploaded and anonymously SHA-256 verified before launch. The frozen 28-photo student evaluation references are excluded from these authoring inputs.

| Finite run ID | Public batch | Count | Source archive SHA-256 | Immutable dataset revision |
| --- | --- | ---: | --- | --- |
| `teacher500-sol-simple-cc0-v1-20260926` | `sol-simple-cc0-select-v1` | 16 | `afa3db414c41327d37b0522b8fb00b432ae1b2a73ede548bb2c5a6d21cb75fe2` | `0df6bf6ec4ce2ca67944a5166622b96711257793` |
| `teacher500-sol-simple-cc0-v2-20260926` | `sol-simple-cc0-select-v2` | 12 | `d1df2ace6c2be02485cd5e851d3654d1c95791fdc3552309aa5941259b52dee6` | `0aa45a5b9b9577fb034ebf524227b530d346ee40` |
| `teacher500-astra-text-curated-20260926` | `astra-text-curated-v1` | 20 | `5fa73041b5c481d0717c584cc39b7c6e86991682457d335fd389933c815bc278` | `56d15da7d1d3c2acfefb41b6b04d056a3779fa2d` |

Launch source is Git commit `4da195c800c106a8638a33d9944c8414e4caa3f2` on `codex/painter-multiturn-sft`. Launcher SHA-256 is `e7a94f13d71254013711e17de968f370c0d3702ce751ea2f7ebec41cbffb6836`; cloud renderer driver SHA-256 is `6c0cc2de76de4f369c0271f315546bb7a9a0f0e1cc809d1d82ef5cae520b8520`. The pinned renderer itself is SHA-256 `ad440b1aa8612e293b52fa5e68c2252926722852155eb8cd145d369a862a8a63`. Each finite Codex Cloud Linux CPU task uses two bounded render workers, a 600-second per-program timeout, the public source receipt, and public hash-verified result publication. No teacher API calls or GPU rental are needed for this render phase.

**Decision after collection.** Display the actual reference-or-prompt and canvas for each program; record missing/invalid/censored cases, category-specific strengths and flaws, and choose a small diverse subset for render-conditioned teacher corrections. Report per-run wall time and worker count. Cloud CPU utilization and billed cost are not instrumented, so do not infer saturation or dollar cost from worker count. No scheduled watcher is authorized or used.
