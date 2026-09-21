# Scaled curriculum inventory — 2026-09-21

This is a metadata and SHA-256 audit only. No rendering, model inference, or ML was run locally. The reusable inventory code is [`painter/data_inventory.py`](../../data_inventory.py); its tests pass from the `painter/` directory.

## Headline counts

The canonical painter manifests contain **2,748 task/reference rows and source groups**: 2,736 procedural rows and 12 real-photo rows. Local image roots contain **444 unique image SHA-256 values**. The v3 manifest adds **2,304 declared target/program SHA-256 values** with `render_status=pending_node_render`; those are not counted as rendered images. The canonical declared train/dev split is 2,256/492 with no source-group overlap.

The fresh COCO128 reference pool in [`real-pool/reference-manifest.json`](real-pool/reference-manifest.json) adds **128 locally present, hash-matched real images** (100 train, 28 dev), bringing the combined explicit curriculum scan to **2,876 rows, 572 unique local image hashes, 140 real-photo rows, and 2,304 unrendered v3 target hashes**. The COCO pool has `sft_teacher_status=no_sft_teacher_claim` for all rows and `visual_review_status=pending`.

The ASTRA high-structure reference manifest contains **66 distinct declared photo hashes** (48 train, 6 validation, 12 sealed test; 33 iNaturalist and 33 Commons). It has **zero local image files**. Hashing embedded task payloads found 32 of those reference hashes in JSONL payloads; the other 34 are metadata-only locally. The associated archive has 196 bounded-brush-v2 painting task rows over 26 source groups. These are reusable task data only after export and contract conversion; there are **0 canonical p5_js teacher programs** in that archive.

## Canonical datasets

| Manifest | rows / split | families | source type | local/hash status | programs |
| --- | ---: | ---: | --- | ---: | ---: |
| [`painter/data/manifest.jsonl`](../../data/manifest.jsonl) | 48 (36 train, 12 dev) | 6 | procedural known program | 48 local image hashes | 144 p5 JS files |
| [`painter/data-v2/manifest.jsonl`](../../data-v2/manifest.jsonl) | 384 (288 train, 96 dev) | 12 | procedural known program | 384 local image hashes | 1,152 p5 JS files |
| [`painter/data/diversity-v3-20260918/programs.jsonl`](../../data/diversity-v3-20260918/programs.jsonl) | 2,304 (1,920 train, 384 dev) | 12 | procedural known program | 2,304 declared target hashes; 0 local renders | 6,912 inline programs |
| [`work/.../evidence/reference-manifest.json`](../../../../work/structured-painting-campaign-20260917/correction-pilot/preserved-corrections-xz/evidence/reference-manifest.json) | 12 train | 6 | real photo | 12 local image hashes; all declaration matches | 0 p5 JS |
| [`real-pool/reference-manifest.json`](real-pool/reference-manifest.json) | 128 (100 train, 28 dev) | COCO object classes | real photo | 128 local image hashes; all declaration matches | 0 teacher claims |

The v1/v2 program paths all exist and contain the required `createCanvas(600,600,WEBGL)` and `translate(-300,-300)` markers (1,296/1,296 program files). The v3 inline programs contain those markers (6,912/6,912 values), but its render and visual-review status remain pending. Marker presence is a static contract check, not a render validation.

## Real references and teachers

The normalized 12-reference preserved Commons pool is [`real-reference-pool.jsonl`](real-reference-pool.jsonl). It carries repo-relative reference paths, source groups, family, source URL, license, raw/source hashes, declared split, exposure, historical-eval exclusion, and hash verification. All 12 are previously exposed historical teacher/eval sources; keep them train-only and do not call them clean heldouts. Ten distinct references have valid first-round bounded-brush-v2 trajectories; `commons_30557288` has one additional valid round-two artifact; two (`commons_1566947`, `commons_61735555`) have invalid initial teacher outputs. Thus there are **11 complete native trajectory artifacts over 10 distinct refs, 2 invalid refs, and 0 canonical p5_js trajectories**. The normalized manifest records the per-row distinction.

The 66 ASTRA declarations are normalized in [`astra-real-reference-candidates.jsonl`](astra-real-reference-candidates.jsonl). Their declared source splits are preserved, exposure is `previously_exposed` only where a matching task payload was found and `unknown` otherwise, and every row is marked historical-eval-excluded for clean-heldout accounting. The path is a declared logical path; it does not exist as a local file. `teacher_compatibility` is bounded-brush-v2 where task rows were observed, and never canonical p5_js.

The ASTRA manual mirror was also scanned at `outputs/brush-rl/runs/astra-manual-expansion-20260914/private-collection-backup/collection-mirror/`: 43 source-metadata JSON files and 90 API-cache JSON files represent 3,570 unique Commons pages, but there are **0 PNG/JPEG/WebP files and 0 data-image payloads** outside caches. These are metadata leads with provider SHA-1 values and URLs, not ready local reference assets. The machine-readable summary is [`real-expansion-leads.json`](real-expansion-leads.json).

## Exposure and leakage

Canonical v1/v2/v3 source groups have no declared train/dev overlap. Existing run/evaluation manifests reuse declared v2 `-007` dev groups and selected v3 dev groups; that is historical exposure, not fresh heldout evidence. The 12 preserved Commons references are historical teacher/eval sources and have no clean local heldout split. The COCO128 pool is marked fresh by its own manifest (`mode=fresh`) after checking 24 historical hashes/source groups, but all 128 still have pending visual review and no teacher claim.

The broad filename pass covered 105 JSON/JSONL files under `painter` whose names indicate manifests or programs. Many are run-input, rendered-copy, packaging, judge, or review wrappers duplicating the canonical rows; they were not added as new references. The canonical counts above come from explicit source manifests and explicit image roots, with duplicate hashes separated from declared-but-unrendered program hashes. No secrets or connection details were written.

## Reuse for the scaled run

Immediately reusable local assets are the 432 procedural v1/v2 images and programs, the 128 hash-matched COCO images, and the 12 hash-matched preserved Commons images. The v3 manifest and its generators provide 2,304 additional procedural references for Linux rendering, but the current declared split is 1,920/384; freeze the exact 2,400-train/240-dev mapping before training rather than silently relabeling or treating pending rows as validated. The ASTRA and manual-mirror pools are expansion leads: their source URLs and hashes are retained, while missing local payloads and noncanonical teacher contracts remain explicit.

## Audit artifacts

- [`inventory.json`](inventory.json): canonical v1/v2/v3 plus preserved 12-reference counts.
- [`inventory-with-coco.json`](inventory-with-coco.json): canonical inventory plus the 128-image COCO pool.
- [`real-pool-inventory.json`](real-pool-inventory.json): COCO pool hash check (128/128 matches).
- [`astra-inventory.json`](astra-inventory.json): ASTRA reference-manifest metadata inventory (66 declared, 0 local files).
- [`real-reference-pool.jsonl`](real-reference-pool.jsonl): normalized preserved Commons pool.
- [`astra-real-reference-candidates.jsonl`](astra-real-reference-candidates.jsonl): normalized ASTRA declarations with missing-payload status.
