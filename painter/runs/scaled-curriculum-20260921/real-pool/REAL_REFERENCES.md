# Fresh real-reference pool

This directory records a bounded import of the official Ultralytics COCO128
asset for the `scaled-curriculum-20260921` preparation run. It is a
reference-only image pool: it contains no generated painting, model call,
teacher trajectory, SFT target, training result, or claim that a model has
seen these images. `mode: "fresh"` describes the import operation only;
`exposure_status` is explicitly unknown, and the manifest does not claim
unseen pretraining or a held-out evaluation set.

The reviewable manifest is
[`reference-manifest.json`](reference-manifest.json). It has 128 admitted
image rows with exact source-byte SHA-256 hashes, 100 deterministic `train`
rows, and 28 deterministic `dev` rows. Every row has a distinct
`source_group` (`coco128:train2017:<COCO image id>`), and the importer checks
that no source group crosses the split boundary. The split is stable SHA-256
ranking of source groups with seed
`scaled-curriculum-20260921:coco128`; it does not use annotation labels to
choose the boundary.

## Provenance and attribution

The source archive is the Ultralytics release asset
[`coco128.zip`](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip).
Its exact downloaded-byte hash is:

```
61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e
```

Ultralytics' primary [COCO128 dataset documentation](https://docs.ultralytics.com/datasets/detect/coco128)
states that this is the first 128 images from COCO train2017, describes the
80-class detection taxonomy, shows a CC BY 4.0 attribution link for the COCO
dataset, and identifies the same official release URL. The class mapping used by the
importer is the pinned official
[`coco128.yaml`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml)
with SHA-256
`07a083c6edd3098d4c9b7c7e8915c46ccd8dd48c8eb970bfb14d958958392844`
recorded in `manifest-classes.json` and the main manifest. The dataset-level
attribution recorded here is the CC BY 4.0 link shown by the Ultralytics
documentation. The archive has no per-image creator or license identifiers,
so individual image licenses are marked
`unverified_not_provided_in_archive`; no blanket individual reuse right is
inferred. The archive's bundled GPL-3.0 `LICENSE`/`README.txt` cover the
Ultralytics conversion software and do not establish per-image rights.

Each row retains the original YOLO annotation file path and hash, raw source
line text, class IDs, class names, normalized boxes, dataset-level provenance
URLs, and dataset attribution. The `family` field is a builder-compatible primary
class. The complete original multiclass taxonomy is preserved in
`annotation_classes`, `original_annotation_classes`, `member_families`, and
the `annotations` list; no single-class relabeling was performed. Each row
also carries the documented dataset attribution separately from its
`per_image_license_status`.

## Coverage and source irregularities

The 128 distinct image hashes cover 71 of the 80 COCO class names. The union
has 67 classes in train and 46 in dev. Two source images (IDs `250` and `508`)
have no corresponding label file in the downloaded archive. They remain as
image-only reference rows with `annotation_classes: null` and
`family: "unknown_unlabeled"`; no empty-scene or object label was invented.
This is retained in [`manifest-warnings.json`](manifest-warnings.json). The
archive also contains two label files (`656` and `659`) for image IDs absent
from the image directory; they are not attached to another image. The
importer reports all row-level rejections in
[`manifest-rejections.json`](manifest-rejections.json): this run admitted all
128 image rows with zero image-row rejections, and retains those two orphan
label files with their exact hashes. The two missing image-label joins remain
explicit warnings. The official asset documentation uses the same 128 images
for its own train and validation paths; the 100/28 partition here is a fresh
curriculum bookkeeping split, not an official COCO holdout.

The requested scale of 2,000 rows was not synthesized: the verified public
asset provided 128 image files, so the manifest stops at 128. No image was
resized, rendered, duplicated, or treated as a fresh example to reach a
larger count.

Before admission, the importer compared exact hashes and source groups with
both the existing historical real-reference manifest
`../real-reference-pool.jsonl` and the existing scale inventory
`../inventory.json` (procedural rows in that inventory are ignored). It
checked 24 historical hashes across 12 real-reference source groups and found
no overlap. A future rerun records any overlap as a rejection rather than
overwriting a historical evaluation assignment.

The complete machine-readable receipt is
[`manifest-receipt.json`](manifest-receipt.json). Raw archive bytes, images,
and labels remain under the ignored `raw/` subtree so the Git surface contains
only the importer and small reviewable provenance manifests.

The import is reproducible from the repository root with the pinned source
hashes and both historical exclusion inputs:

```bash
python3 painter/import_reference_pool.py \
  --output painter/runs/scaled-curriculum-20260921/real-pool \
  --historical painter/runs/scaled-curriculum-20260921/real-reference-pool.jsonl \
  --historical painter/runs/scaled-curriculum-20260921/inventory.json
```
