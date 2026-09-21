"""Import a bounded real-image reference pool with exact provenance.

This importer is intentionally metadata-first and reference-only.  It downloads
the official Ultralytics COCO128 archive, hashes each source image, preserves
the YOLO annotations that are present, and writes a compact JSON manifest for
the scaled curriculum builder.  It never calls a model, renders a target, or
creates labels for an image whose source label file is absent.

The default output is the ignored ``real-pool/`` subtree of the scaled run.
Small JSON manifests and ``REAL_REFERENCES.md`` remain reviewable.  The source
archive and images stay out of Git by the repository's tracking boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


RUN_ID = "scaled-curriculum-20260921"
DEFAULT_OUTPUT = Path("painter/runs") / RUN_ID / "real-pool"
DEFAULT_ARCHIVE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
DEFAULT_CLASSES_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml"
DEFAULT_DOCS_URL = "https://docs.ultralytics.com/datasets/detect/coco128"
DEFAULT_COCO_URL = "https://cocodataset.org/#home"
DEFAULT_SEED = "scaled-curriculum-20260921:coco128"
DEFAULT_DEV_COUNT = 28
EXPECTED_ARCHIVE_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
EXPECTED_CLASSES_SHA256 = "07a083c6edd3098d4c9b7c7e8915c46ccd8dd48c8eb970bfb14d958958392844"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CLASS_LINE = re.compile(r"^\s*(\d+)\s*:\s*(.*?)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "painter-reference-import/1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return sha256_file(destination)


def _parse_classes(yaml_path: Path) -> dict[int, str]:
    classes: dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            match = CLASS_LINE.match(line)
            if match:
                classes[int(match.group(1))] = match.group(2).strip().strip("'\"")
                continue
            if line and not line[0].isspace():
                break
    if len(classes) != 80 or sorted(classes) != list(range(80)):
        raise ValueError(f"expected 80 contiguous COCO classes, found {len(classes)}")
    return classes


def _parse_labels(label_path: Path, classes: Mapping[int, str]) -> tuple[list[dict[str, Any]] | None, list[str] | None, str]:
    """Return normalized source rows, distinct names, and source status.

    ``None`` means that the source archive has no label file.  That is kept
    distinct from an existing empty label file: the importer never turns a
    missing annotation into an empty-scene claim.
    """
    if not label_path.exists():
        return None, None, "missing_label_file"
    annotations: list[dict[str, Any]] = []
    names: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        if class_id not in classes:
            raise ValueError(f"{label_path}:{line_number}: unknown class id {class_id}")
        coords = [float(value) for value in fields[1:]]
        if any(value < 0 or value > 1 for value in coords):
            raise ValueError(f"{label_path}:{line_number}: normalized coordinates out of range")
        annotations.append({
            "source_line": line_number,
            "class_id": class_id,
            "class_name": classes[class_id],
            "x_center": coords[0],
            "y_center": coords[1],
            "width": coords[2],
            "height": coords[3],
            "source_text": line.strip(),
        })
        if classes[class_id] not in names:
            names.append(classes[class_id])
    return annotations, sorted(names), "present"


def _stable_dev_ids(source_groups: Iterable[str], dev_count: int, seed: str) -> set[str]:
    ranked = sorted(
        source_groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    return set(ranked[:dev_count])


def _hashes_from_history(paths: Iterable[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Read old manifests without touching their image trees."""
    image_hashes: set[str] = set()
    source_groups: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        record: dict[str, Any] = {"path": str(path), "status": "missing"}
        if path.is_file():
            record.update({"status": "read", "sha256": sha256_file(path)})
            try:
                if path.suffix.lower() in {".jsonl", ".ndjson"}:
                    payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    rows = payload.get("references", payload.get("items", payload.get("rows", payload.get("records", []))))
                elif isinstance(payload, list):
                    rows = payload
                else:
                    rows = []
                if isinstance(rows, dict):
                    rows = [rows]
                if not isinstance(rows, list):
                    rows = []
                count = 0
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    # The existing scale inventory also contains procedural
                    # rows.  Only its explicitly identified real-photo rows
                    # participate in the historical real-image exclusion.
                    source_type = str(row.get("source_type", "")).lower()
                    if source_type and source_type not in {"real_photo", "real_reference", "real_reference_image"}:
                        continue
                    source = row.get("source_group", row.get("source_id", row.get("id")))
                    if source not in (None, ""):
                        source_groups.add(str(source))
                    keys = ("image_sha256", "reference_sha256", "raw_source_sha256", "sha256", "actual_sha256")
                    if source_type in {"real_photo", "real_reference", "real_reference_image"}:
                        keys = keys + ("declared_sha256",)
                    for key in keys:
                        value = row.get(key)
                        if isinstance(value, str) and HEX64.fullmatch(value.lower()):
                            image_hashes.add(value.lower())
                    count += 1
                record["rows"] = count
            except (OSError, json.JSONDecodeError, TypeError):
                record["status"] = "unreadable"
        inputs.append(record)
    return image_hashes, source_groups, inputs


def _extract(archive: Path, raw_dir: Path) -> Path:
    dataset_root = raw_dir / "coco128"
    if dataset_root.is_dir():
        return dataset_root
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(raw_dir)
    if not dataset_root.is_dir():
        raise ValueError(f"archive did not contain expected {dataset_root}")
    return dataset_root


def build_manifest(
    *,
    output: Path = DEFAULT_OUTPUT,
    dev_count: int = DEFAULT_DEV_COUNT,
    seed: str = DEFAULT_SEED,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    classes_url: str = DEFAULT_CLASSES_URL,
    docs_url: str = DEFAULT_DOCS_URL,
    historical_paths: Iterable[Path] = (),
    download: bool = True,
    expected_archive_sha256: str | None = EXPECTED_ARCHIVE_SHA256,
    expected_classes_sha256: str | None = EXPECTED_CLASSES_SHA256,
) -> dict[str, Any]:
    """Download (if needed) and build the deterministic 128-row manifest."""
    output = Path(output)
    workspace = Path.cwd().resolve()

    def workspace_path(path: Path) -> str:
        """Use repository-relative paths when possible, absolute otherwise."""
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(workspace))
        except ValueError:
            return str(resolved)
    raw_dir = output / "raw"
    archive_path = raw_dir / "coco128.zip"
    classes_path = raw_dir / "coco128.yaml"
    if download or not archive_path.exists():
        archive_sha256 = _download(archive_url, archive_path)
    else:
        archive_sha256 = sha256_file(archive_path)
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        raise ValueError(f"archive SHA-256 mismatch: expected {expected_archive_sha256}, got {archive_sha256}")
    if download or not classes_path.exists():
        classes_sha256 = _download(classes_url, classes_path)
    else:
        classes_sha256 = sha256_file(classes_path)
    if expected_classes_sha256 and classes_sha256 != expected_classes_sha256:
        raise ValueError(f"classes YAML SHA-256 mismatch: expected {expected_classes_sha256}, got {classes_sha256}")
    dataset_root = _extract(archive_path, raw_dir)
    classes = _parse_classes(classes_path)
    images_dir = dataset_root / "images" / "train2017"
    labels_dir = dataset_root / "labels" / "train2017"
    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"no images found under {images_dir}")
    historical_hashes, historical_sources, history_inputs = _hashes_from_history(historical_paths)
    raw_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_sources: set[str] = set()
    labels_seen: set[str] = set()
    image_stems = {path.stem for path in image_paths}
    for image_path in image_paths:
        image_path = image_path.resolve()
        image_id = image_path.stem
        source_group = f"coco128:train2017:{image_id}"
        image_sha256 = sha256_file(image_path)
        label_path = labels_dir / f"{image_id}.txt"
        try:
            annotations, annotation_classes, annotation_status = _parse_labels(label_path, classes)
        except (OSError, ValueError) as exc:
            rejections.append({"source_group": source_group, "image_path": str(image_path), "reason": "invalid_annotation", "detail": str(exc)})
            continue
        if source_group in historical_sources:
            rejections.append({"source_group": source_group, "image_path": str(image_path), "reason": "historical_source_group_overlap"})
            continue
        if image_sha256 in historical_hashes:
            rejections.append({"source_group": source_group, "image_path": str(image_path), "image_sha256": image_sha256, "reason": "historical_image_hash_overlap"})
            continue
        if image_sha256 in seen_hashes:
            rejections.append({"source_group": source_group, "image_path": str(image_path), "image_sha256": image_sha256, "reason": "duplicate_image_sha256"})
            continue
        seen_hashes.add(image_sha256)
        seen_sources.add(source_group)
        if annotation_classes is None:
            family = "unknown_unlabeled"
            member_families: list[str] = []
            warnings.append({"source_group": source_group, "reason": "missing_label_file", "label_path": str(label_path)})
        else:
            family = annotation_classes[0] if annotation_classes else "empty_scene"
            member_families = annotation_classes
            labels_seen.update(annotation_classes)
        raw_rows.append({
            "id": f"coco128-{image_id}",
            "source_group": source_group,
            "source_id": image_id,
            "dataset": "COCO train2017 via Ultralytics COCO128",
            "dataset_split": "train2017",
            "reference_path": workspace_path(image_path),
            "reference": workspace_path(image_path),
            "image_sha256": image_sha256,
            "image_hash_status": "verified_local",
            "source_url": archive_url,
            "source_image_url": f"https://cocodataset.org/#explore?id={int(image_id)}",
            "provenance_urls": [archive_url, docs_url, DEFAULT_COCO_URL],
            "dataset_license_documentation": "CC BY 4.0 attribution link shown by the Ultralytics COCO128 documentation",
            "per_image_license": None,
            "per_image_license_status": "unverified_not_provided_in_archive",
            "dataset_attribution": "Microsoft COCO Consortium; Ultralytics COCO128 packaging",
            "annotation_format": "ultralytics_yolo_normalized_xywh",
            "annotation_path": workspace_path(label_path),
            "annotation_sha256": sha256_file(label_path) if label_path.exists() else None,
            "annotation_status": annotation_status,
            "annotation_class_ids": sorted({item["class_id"] for item in annotations}) if annotations is not None else None,
            "annotation_classes": annotation_classes,
            "original_annotation_classes": annotation_classes,
            "annotations": annotations,
            "family": family,
            "primary_family": family,
            "member_families": member_families,
            "family_multiclass": member_families,
            "split": "pending_deterministic_assignment",
            "split_basis": f"stable SHA-256 ranking of source_group with seed {seed!r}; {dev_count} dev groups",
            "mode": "fresh",
            "exposure_status": "fresh_import_exposure_unknown",
            "unseen_pretraining_claim": False,
            "heldout_claim": False,
            "reference_kind": "real_photo_image",
            "dataset_role": "reference_only",
            "teacher_claim": False,
            "sft_teacher_status": "no_sft_teacher_claim",
            "historical_eval_excluded": False,
            "visual_review_status": "pending",
            "render_status": "reference_image_imported_no_render",
        })
    # Keep source-package annotation files that cannot be joined to an image
    # as explicit rejections; never attach them to a nearby image ID.
    if labels_dir.is_dir():
        for label_path in sorted(labels_dir.glob("*.txt")):
            if label_path.stem not in image_stems:
                rejections.append({
                    "source_id": label_path.stem,
                    "annotation_path": workspace_path(label_path),
                    "annotation_sha256": sha256_file(label_path),
                    "reason": "orphan_annotation_no_image",
                })
    if dev_count < 0 or dev_count > len(raw_rows):
        raise ValueError(f"dev_count={dev_count} outside admitted row count {len(raw_rows)}")
    dev_ids = _stable_dev_ids((row["source_group"] for row in raw_rows), dev_count, seed)
    for row in raw_rows:
        row["split"] = "dev" if row["source_group"] in dev_ids else "train"
    rows = sorted(raw_rows, key=lambda row: row["source_group"])
    train_groups = {row["source_group"] for row in rows if row["split"] == "train"}
    dev_groups = {row["source_group"] for row in rows if row["split"] == "dev"}
    if train_groups & dev_groups:
        raise AssertionError("source group leaked across train/dev")
    manifest = {
        "schema": "painter-real-reference-manifest-v1",
        "run_id": RUN_ID,
        "mode": "fresh",
        "dataset_role": "reference_only",
        "teacher_claim": False,
        "sft_teacher_status": "no_sft_teacher_claim",
        "exposure_status": "fresh_import_exposure_unknown; do_not_call_unseen_pretraining_or_heldout",
        "unseen_pretraining_claim": False,
        "heldout_claim": False,
        "source": {
            "dataset": "COCO128",
            "description": "First 128 COCO 2017 train images packaged by Ultralytics; image files remain source bytes.",
            "archive_url": archive_url,
            "archive_sha256": archive_sha256,
            "classes_url": classes_url,
            "classes_sha256": classes_sha256,
            "docs_url": docs_url,
            "coco_url": DEFAULT_COCO_URL,
            "dataset_license_documentation": "CC BY 4.0 attribution link shown by the Ultralytics COCO128 documentation",
            "per_image_license_status": "unverified_not_provided_in_archive",
            "dataset_attribution": "Microsoft COCO Consortium; Ultralytics COCO128 packaging",
            "downloaded_archive": workspace_path(archive_path),
            "source_package_license_note": "The archive README/LICENSE cover Ultralytics conversion software under GPL-3.0; they do not establish an individual license for each source image.",
        },
        "split": {"seed": seed, "dev_count": dev_count, "train_count": len(rows) - dev_count, "source_group_disjoint": True},
        "historical_exclusion_inputs": history_inputs,
        "historical_hashes_checked": len(historical_hashes),
        "historical_source_groups_checked": len(historical_sources),
        "requested_count": 2000,
        "available_source_images": len(image_paths),
        "admitted_count": len(rows),
        "references": rows,
    }
    _json_dump(output / "reference-manifest.json", manifest)
    _json_dump(output / "manifest-rejections.json", {"schema": "painter-real-reference-rejections-v1", "count": len(rejections), "rejections": rejections})
    _json_dump(output / "manifest-warnings.json", {"schema": "painter-real-reference-warnings-v1", "count": len(warnings), "warnings": warnings})
    _json_dump(output / "manifest-classes.json", {"schema": "coco128-class-index-v1", "source_url": classes_url, "source_sha256": classes_sha256, "classes": classes})
    _json_dump(output / "manifest-receipt.json", {
        "schema": "painter-real-reference-receipt-v1",
        "archive_url": archive_url,
        "archive_sha256": archive_sha256,
        "archive_path": workspace_path(archive_path),
        "image_count": len(image_paths),
        "admitted_count": len(rows),
        "rejection_count": len(rejections),
        "warning_count": len(warnings),
        "image_sha256_unique": len(seen_hashes),
        "annotation_classes_seen": sorted(labels_seen),
    })
    return {
        "manifest": manifest,
        "rejections": rejections,
        "warnings": warnings,
        "archive_sha256": archive_sha256,
        "image_counts": {"available": len(image_paths), "admitted": len(rows), "train": len(train_groups), "dev": len(dev_groups)},
        "class_counts": dict(sorted(Counter(name for row in rows for name in (row["annotation_classes"] or [])).items())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-count", type=int, default=DEFAULT_DEV_COUNT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--classes-url", default=DEFAULT_CLASSES_URL)
    parser.add_argument("--docs-url", default=DEFAULT_DOCS_URL)
    parser.add_argument("--historical", type=Path, action="append", default=[], help="old JSON reference manifest to exclude by source group or exact image hash")
    parser.add_argument("--no-download", action="store_true", help="reuse an existing raw archive and YAML")
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_ARCHIVE_SHA256)
    parser.add_argument("--expected-classes-sha256", default=EXPECTED_CLASSES_SHA256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_manifest(
        output=args.output,
        dev_count=args.dev_count,
        seed=args.seed,
        archive_url=args.archive_url,
        classes_url=args.classes_url,
        docs_url=args.docs_url,
        historical_paths=args.historical,
        download=not args.no_download,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_classes_sha256=args.expected_classes_sha256,
    )
    print(json.dumps({"image_counts": result["image_counts"], "archive_sha256": result["archive_sha256"], "class_count": len(result["class_counts"]), "warnings": len(result["warnings"]), "rejections": len(result["rejections"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
