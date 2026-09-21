# COCO capability preparation

This run checks the current Qwen painter before any expanded data generation or training. It compares the unadapted base with the pinned contract-quality step 24 adapter on the same twelve fresh native-painting tasks. The preparation command has no model, renderer, judge, network call, or training side effect.

`eval-manifest.json` contains 12 byte-preserved JPG references selected from the existing 28-image COCO128 `dev` split. The selection is deterministic (`coco-capability-20260922`) and uses the greedy score recorded in each `selection_reason`: 100 points per newly covered annotation category, 30 for a new object-count bucket, and 10 for a new primary family, with ascending source ID as the tie-break. It covers 35 annotation categories, 12 primary families, and all four object-count buckets (`one`, `two_to_four`, `five_to_nine`, `ten_plus`). Every selected source group is dev-only and unique; the 100 train rows are excluded.

The source manifest hash and every copied reference hash are recorded. The COCO/Ultralytics documentation links describe dataset attribution and license documentation at the dataset/package level. The archive does not provide a per-image license ID or creator record for these rows, so the manifest records `per_image_license_status=unverified_not_provided_in_archive` and makes no blanket individual-rights claim. This is a reference-only capability check with no held-out or unseen-pretraining claim.

## Prepared files

- `eval-manifest.json`: native `fresh` taskset manifest, with source classes, original annotations, provenance, hashes, and selection reasons.
- `base.toml`: 12 tasks for the actual `base` served-model and taskset labels (the pinned base repository is Qwen/Qwen3.8-27B).
- `sft.toml`: the identical taskset and sampling settings for the actual `sft` served-model and taskset labels (the pinned adapter is recorded as contract-quality step24).
- `prepare_coco_capability.py`: deterministic importer/config generator.
- `build_gallery.py`: report-only gallery builder for reference, base first/final, and SFT first/final artifacts.
- `make_contact_sheets.py`: node-side Pillow helper that turns the gallery's recorded paths into twelve five-panel PNG rows plus an overview.

Both configs use seed 918200, temperature 1, one rollout per reference, six turns, 8192 max tokens per response, 32768 context, and thinking disabled. They set `training=false` in the manifest protocol and do not include any judge call.

The evaluator copy is frozen. Its uploaded manifest is SHA256 `874695bf596b5cb7db8940e1dfb6a0f3d62ab3050524a392b763b4ac257b4e94`; do not rerun the importer with `--force` or replace that manifest after collection starts. The current replay-aligned local template is SHA256 `8a23d11ce59d65cc84b2c595177cc37b6803768c60c8c704b1b5a5e5e5f45cf7`. An intermediate local correction was SHA256 `c594ffd4034f73b6c560e5029ae221aeceb8fdcdeed4d798f58e3fac8f448333`. All three manifests have the same twelve IDs, source groups, image bytes, image hashes, annotations, and selection reasons; the hash changes are metadata-only, aligning protocol/config policy labels and filenames with the server's actual `base`/`sft` IDs.

## Node invocation

The prepared copy is at `/root/painter/coco-capability`; its launch configs are `base.toml` and `sft.toml`, with policy labels `base` and `sft`. With the reviewed inference server already running on the local endpoint, launch both matched policies concurrently:

```sh
source /root/painter/env.sh
export PYTHONPATH=/root/painter PAINTER_LOCAL_API_KEY=unused
(/root/painter/inference-env/bin/eval @ /root/painter/coco-capability/base.toml --no-serve --no-push > /root/painter/coco-capability/base.eval.log 2>&1) &
BASE_PID=$!
(/root/painter/inference-env/bin/eval @ /root/painter/coco-capability/sft.toml --no-serve --no-push > /root/painter/coco-capability/sft.eval.log 2>&1) &
SFT_PID=$!
wait "$BASE_PID"
BASE_STATUS=$?
wait "$SFT_PID"
SFT_STATUS=$?
printf 'base=%s sft=%s\n' "$BASE_STATUS" "$SFT_STATUS"
```

The existing pinned server exposes the model IDs `base` and `sft`; those IDs must match the config values. Preserve the raw `evaluation-rollouts` and native trace files before releasing the node. No runtime was performed while preparing this record.

## Gallery after evidence collection

The gallery reads only preserved native traces and canvas files. It reports actual turn/status/error/token-usage metadata and leaves missing, invalid, censored, and duplicate policy cells visible:

```sh
python3 /root/painter/painter/runs/coco-capability-20260922/build_gallery.py \
  --evidence-root /root/painter \
  --reference-manifest /root/painter/coco-capability/eval-manifest.json \
  --outdir /root/painter/coco-capability/gallery
```

The resulting `gallery/index.html` links the reference and the first valid plus final recorded canvas for each policy. `gallery/metadata.json` is the machine-readable audit record. The gallery labels the actual `base` traces as “Base unadapted” and `sft` traces as “SFT step24”; it does not relabel either policy as RL. No quality comparator is invented when a trace or artifact is absent.

For a compact visual review on the Linux node, use the gallery metadata as the sole path source:

```sh
/root/painter/inference-env/bin/python /root/painter/painter/runs/coco-capability-20260922/make_contact_sheets.py \
  --metadata /root/painter/coco-capability/gallery/metadata.json \
  --output-dir /root/painter/coco-capability/gallery/contact-sheets
```

This writes one row per case (`*.png`), an `overview.png`, and `index.json`. It uses only the recorded reference, base-first, base-final, SFT-first, and SFT-final paths. Missing, invalid, or unreadable artifacts become explicit panels; it never searches turn directories or picks a final by filename ordering. Run this image-producing helper on the Linux node, after evidence collection.

## Larger-context primary rerun

The checked-in `base.toml` and `sft.toml` now match the primary rerun: 98304 context metadata, eight concurrent evaluations per policy, and 3600-second rollout timeout. Server settings are `EVAL_MAX_MODEL_LEN=98304 EVAL_CONCURRENCY=8 EVAL_MAX_BATCHED_TOKENS=32768`. The original preparation helper/manifest protocol records the initial 32k attempt; rerunning it regenerates those initial defaults. To prepare the larger-context configs, call `native_painting_eval.py` with `--context-length 98304 --rollout-timeout 3600` and set top-level `max_concurrent = 8`. Preserve the frozen case records. See PROTOCOL.md and EXPERIMENT.json for the amendment and exact runtime hashes.
