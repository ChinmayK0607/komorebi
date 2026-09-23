# Preparation validation — 2026-09-22

Status: preparation checks passed; a one-reference paid quality pilot subsequently ran. This file records the preparation checks, while the root research log records the pilot outcome.

- Parent ran the complete offline suite: **16 tests passed**. The suite covers content extraction, context exhaustion, key redaction/retry usage, cached-response reuse after renderer failure, full two-turn image conversation reconstruction, speed deadline classification, two-track counts, API reasoning payload schema, package boundaries, archive traversal rejection, offline dry runs, custom-model selection and real CLI compatibility.
- Full input dry run: **40 references × 5 models × 2 tracks = 400 episodes**, at most **3,000 assistant requests before service retries**.
- All selected reference hashes match; parent displayed and inspected all final 40 image files. A collage was replaced with a complex single-scene kitchen image before model generation.
- Shell syntax and Python compilation pass; Git whitespace check passes.
- Real Linux renderer smoke: fixed polygon-and-line sketch, no model/API calls. Ran the same benchmark `render_program` entry point under the unprivileged painter user, using the existing pinned renderer. Canvas600×600, sandbox enabled, network disabled. Source and PNG hashes verified independently. Elapsed3.512 seconds, painting1.803 seconds; these measure the test sketch only, not teacher benchmark latency.
- The wrapper's isolated benchmark environment bootstrap succeeded on Linux, including importing tqdm. Shared training/renderer environments were not modified.
- No benchmark episode, automatic judging, SFT admission, GPU rental, training restart or scheduled task was started by this preparation.

## Render evidence

- `source_sha256`: `321ca5ac6c99de64096856329368f9bc12b5a863b20f12800a54eba5cb92a381`
- `png_sha256`: `ad8dc56b803ddbc4f153867c85ec09731b1eaf5cbcda03cd409c5821aff53a1f`
- `renderer_sha256`: `ad7ba24d632f0f16989741560c3af0fe42e426275c1df01954b60ddd69b63395`
- `chromium_version`: `145.0.7632.6`
- `p5.brush.js` SHA-256: `9c6e271a78e5dcb23c9cfa9d085775317fe5fb2188f976465c83c1f7965d1ec8`
- `p5.min.js` SHA-256: `87adc350e8ec0e9bced22d4f03c181bdd208dc997d3956ab3ec2e90537643c9a`

Full local smoke receipt: `renderer-smoke/node-smoke-receipt.json` (generated evidence, ignored by Git). Provider connection details and credentials are not part of this record.

## Limits

The separate AI SDK holiday example was smoke-tested against the Gateway before the benchmark. Offline tests use mocked transport and renderer replies, except for the explicit real-node renderer smoke. These checks validate preparation and key runtime plumbing; the later one-reference pilot is recorded separately in the root research log. Neither establishes model quality, realized price, or successful completion of all 400 episodes. Gateway model versions are not immutable weight revisions.
