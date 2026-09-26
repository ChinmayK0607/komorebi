# Diverse teacher data campaign

## Objective and baseline

Build about 500 distinct, rendered prompt-to-painting examples: 250 text prompts and 250 visual references. The target is a beautiful, recognizable painting that preserves the requested subject and composition. A renderer-valid program alone is not a positive example. The student baseline is the public photo-SFT step-512 adapter; the later 39-first-paint Astra SFT regressed on the frozen 28-photo comparison (19/28 versus 14/28 first-turn valid canvases), so this campaign makes no training-improvement claim.

The frozen 28 COCO128 development references remain excluded from all authoring. The earlier 100 COCO128 train references can supply correction states, but repeated scenes do not count as new reference coverage. For new visual inputs, draw from a separate source pool and reserve a fixed held-out set before teacher authoring. Each example binds the exact input bytes, program, rendered canvas, teacher identity, revision/turn, and source license or provenance by hash.

## Mix and quality bar

| Input | Simple | Intermediate | Complex | Target distinct pairs |
| --- | ---: | ---: | ---: | ---: |
| Text prompt | 70 | 100 | 80 | 250 |
| Visual reference | 70 | 100 | 80 | 250 |

Simple cases establish form and palette; intermediate cases combine two to four spatially related objects; complex cases include interiors, streets, vehicles, animals, people, overlap, and depth. Each bucket should vary objects, scale, lighting, viewpoint, color, and composition. Both `gpt-5.6-sol` high and `gpt-6-astra` high contribute. Record the model for each program, rather than treating the two teachers as interchangeable.

For every candidate, check the exact program contract, render it on Linux with the pinned renderer, and retain a reference-or-prompt/program/canvas contact sheet. Review visual quality and faithfulness; exclude blank, malformed, grotesque, copied, and merely recolored examples. Record both strengths and remaining weaknesses. Generate correction turns against the *actual* prior canvas when useful; keep the full state and all turns, but count a correction as a new pair only when its distinct input state and improved canvas are documented. Candidate, renderer-valid, visually admitted, and training-export counts are separate.

## Production loop

Author disjoint batches while previous batches render and receive visual review. Keep source programs and raw receipts outside Git; put only the small manifest, script, and experiment record here. Publish completed bundles to the public Hugging Face teacher dataset and verify a credential-free download against their SHA-256 before removing any local copy. Do not rent a GPU for authoring or rendering; use a finite Codex Cloud Linux job for rendering. A GPU is needed only for a later training run.

The first Sol/Astra batches are pilot candidates, not 500 completed examples. Scale the authoring batch only after rendered samples show that the program contract and aesthetic bar are being met. If a category repeatedly fails, use a targeted correction pass rather than filling the quota with weak examples. When the dataset is ready, compare SFT against the exact step-512 initializer on frozen photo and text prompts, then use pairwise visual feedback for RL. No scheduled watcher is part of this campaign.

## Source-pool note

An HTTPS snapshot of COCO 2017 validation captions from `HamGangster/coco_2017_caption_validation` at revision `57b44af916dcc91ec05b8c1a86977e7a5a65764e` contains 5,000 image records and has SHA-256 `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`. `prepare_reference_pool.py` deterministically reserves 250 candidate train photos and 50 held-out photos from license IDs 4, 5, 7 and 8, with no ID overlap against the earlier COCO128 training or frozen development references. The first 24 train JPEGs have been fetched and hashed; 250 are **not** yet teacher demonstrations. The official image endpoint is `https://s3.amazonaws.com/images.cocodataset.org/val2017/<12-digit-id>.jpg`. Captions provide rough source selection only; visual inspection has already corrected misleading subject categories.

`prepare_openverse_pool.py` provides a second, CC0-only stream from [Openverse](https://docs.openverse.org/packages/js/api_client/index.html), keeping every thumbnail's exact hash, creator, source URL, license and search response receipt. Its initial 28 queries selected 92 candidate train images and 11 held-out images; the first 24 train thumbnails were downloaded. One item tagged as a photograph is visibly an etching, and another is printed textile art, so source type also requires visual review. These two pools are alternatives to mix, not additive quotas. The target remains about 250 distinct visual-input teacher pairs across sources.
