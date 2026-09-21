"""Inventory painting curricula without rendering or loading model artifacts.

The inventory is deliberately metadata-first.  It reads JSON/JSONL manifests,
resolves only explicitly named image references, and hashes those images when
they are present.  It never walks a run's full output tree, which keeps a
large rollout collection from being mistaken for a larger reference corpus.

Example::

    python3 painter/data_inventory.py \
      --manifest painter/data/manifest.jsonl \
      --manifest painter/data-v2/manifest.jsonl \
      --manifest painter/data/diversity-v3-20260918/programs.jsonl \
      --image-root painter/data \
      --image-root painter/runs/diversity-sft-20260918/data \
      --output painter/runs/scaled-curriculum-20260921/inventory.json

The output records dataset rows separately from unique image hashes.  A row
with an inline program or a declared hash but no PNG is reported as
``unrendered``; it is not presented as validated visual data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
SPLIT_VALUES = {
    "train", "dev", "validation", "test", "eval", "sealed", "sealed_test",
    "heldout", "holdout",
}


def _load(path: Path) -> Any:
    """Load a bounded metadata file (never an image or archive)."""
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"manifest exceeds 32 MiB metadata bound: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _walk_records(value: Any, *, parent_key: str = "") -> Iterable[dict[str, Any]]:
    """Yield dicts that look like task/reference records from nested metadata."""
    if isinstance(value, dict):
        keys = set(value)
        record_keys = {
            "id", "source_group", "reference", "reference_path", "reference_sha256",
            "target_sha256", "reference_kind", "primary_family", "family", "sample_id",
        }
        if keys & record_keys and any(k in value for k in (
            "id", "sample_id", "source_group", "reference_path", "reference_sha256", "target_sha256"
        )):
            yield value
        for key, child in value.items():
            yield from _walk_records(child, parent_key=key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_records(child, parent_key=parent_key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_hash(value: Any) -> str | None:
    return str(value).lower() if isinstance(value, str) and HEX64.fullmatch(value) else None


def _source_type(row: dict[str, Any]) -> str:
    kind = str(row.get("reference_kind", "")).lower()
    if "procedural" in kind or "synthetic" in kind or "generated" in kind:
        return "procedural"
    # A URL alone is not evidence that the content is a photograph: Commons
    # also hosts illustrations, renders, and diagrams.  Require explicit
    # photo metadata or a photo/real reference kind.
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    if any(key in row for key in ("photo_evidence", "raw_source_sha256", "source_sha1", "download_sha256")):
        return "real_photo"
    if any(key in source for key in ("photo_id", "observation_id", "raw_source_sha256", "download_sha256")):
        return "real_photo"
    if any(word in kind for word in ("photo", "real", "commons")):
        return "real_photo"
    return "unknown"


def _split(row: dict[str, Any]) -> str:
    value = row.get("split")
    if isinstance(value, str) and value.lower() in SPLIT_VALUES:
        return value.lower()
    return "unknown"


def _family(row: dict[str, Any]) -> str:
    value = row.get("family", row.get("primary_family", row.get("category")))
    return str(value) if value not in (None, "") else "unknown"


def _reference_value(row: dict[str, Any]) -> str | None:
    for key in ("reference_path", "reference_png", "reference", "image_path", "image"):
        value = row.get(key)
        # Diversity v3 stores the reference *program* inline under
        # ``programs.reference``; an inline JS program is metadata, not a
        # filesystem path.  Keep it out of Path/stat calls.
        if isinstance(value, str) and value and len(value) <= 4096 and "\n" not in value and "\r" not in value:
            return value
    return None


def _candidate_paths(reference: str, manifest: Path, roots: list[Path]) -> Iterable[Path]:
    raw = Path(reference)
    candidates = [manifest.parent / raw, *[root / raw for root in roots]]
    # Run manifests commonly use data/<source>/<file> while their image root is
    # already the data directory.
    if raw.parts and raw.parts[0] == "data":
        candidates.extend(root / Path(*raw.parts[1:]) for root in roots)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
            yield resolved


def _program_count(row: dict[str, Any]) -> int:
    programs = row.get("programs")
    if isinstance(programs, list):
        return len(programs)
    if isinstance(programs, dict):
        return len(programs)
    return 0


def inventory(manifests: Iterable[str | Path], image_roots: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Return a compact inventory for explicit manifests and image roots."""
    manifest_paths = sorted({Path(p).resolve() for p in manifests})
    roots = [Path(p).resolve() for p in image_roots]
    records: list[dict[str, Any]] = []
    manifest_summaries: list[dict[str, Any]] = []
    seen_row_keys: set[tuple[str, str, int]] = set()

    for manifest in manifest_paths:
        if not manifest.is_file():
            manifest_summaries.append({"path": str(manifest), "status": "missing"})
            continue
        payload = _load(manifest)
        rows = list(_walk_records(payload))
        summary = {
            "path": str(manifest),
            "status": "read",
            "rows_discovered": len(rows),
            "sha256": _sha256(manifest),
        }
        manifest_summaries.append(summary)
        for index, row in enumerate(rows):
            row_id = row.get("id", row.get("sample_id", row.get("source_group", f"row-{index}")))
            key = (str(manifest), str(row_id), index)
            if key in seen_row_keys:
                continue
            seen_row_keys.add(key)
            reference = _reference_value(row)
            declared_hash = (
                _as_hash(row.get("reference_sha256"))
                or _as_hash(row.get("image_sha256"))
                or _as_hash(row.get("target_sha256"))
                or _as_hash(row.get("sha256"))
            )
            declared_hash_type = (
                "reference_sha256" if _as_hash(row.get("reference_sha256")) else
                "image_sha256" if _as_hash(row.get("image_sha256")) else
                "target_sha256" if _as_hash(row.get("target_sha256")) else
                "sha256" if _as_hash(row.get("sha256")) else None
            )
            resolved = next(_candidate_paths(reference, manifest, roots), None) if reference else None
            actual_hash = _sha256(resolved) if resolved else None
            records.append({
                "manifest": str(manifest),
                "row": index,
                "id": str(row_id),
                "source_group": str(row.get("source_group", row.get("sample_id", row_id))),
                "split": _split(row),
                "family": _family(row),
                "source_type": _source_type(row),
                "reference": reference,
                "declared_sha256": declared_hash,
                "declared_hash_type": declared_hash_type,
                "resolved_path": str(resolved) if resolved else None,
                "actual_sha256": actual_hash,
                "hash_match": bool(declared_hash and actual_hash and declared_hash == actual_hash),
                "rendered": bool(actual_hash),
                "program_count": _program_count(row),
                "visual_review_status": row.get("visual_review_status", row.get("visual_screen")),
            })

    # A declared hash is content identity even when the corresponding PNG was
    # intentionally left on the Linux node.  Actual hashes take precedence.
    hash_to_records: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        digest = record["actual_sha256"] or record["declared_sha256"]
        if digest:
            hash_to_records[digest].append(record)

    def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(row[key] for row in rows).items()))

    source_groups = {row["source_group"] for row in records}
    split_groups = {
        split: {row["source_group"] for row in records if row["split"] == split}
        for split in sorted({row["split"] for row in records})
    }
    overlap = {
        left: {right: sorted(split_groups[left] & split_groups[right]) for right in split_groups if right > left}
        for left in split_groups
    }
    nonempty_overlap = {left: values for left, values in overlap.items() if any(values.values())}
    actual_unique = {row["actual_sha256"] for row in records if row["actual_sha256"]}
    declared_only = {row["declared_sha256"] for row in records if row["declared_sha256"] and not row["actual_sha256"]}
    declared_unique = {row["declared_sha256"] for row in records if row["declared_sha256"]}
    declared_reference_unique = {
        row["declared_sha256"] for row in records
        if row["declared_sha256"] and row["declared_hash_type"] in {"reference_sha256", "image_sha256"}
    }
    declared_image_unique = {
        row["declared_sha256"] for row in records
        if row["declared_sha256"] and row["declared_hash_type"] == "image_sha256"
    }
    declared_target_unique = {
        row["declared_sha256"] for row in records
        if row["declared_sha256"] and row["declared_hash_type"] == "target_sha256"
    }

    # Report only compact hash groups; all per-row provenance is retained in
    # records, while large duplicate groups are summarized by count and a few
    # representative paths.
    duplicate_groups = []
    for digest, rows in sorted(hash_to_records.items()):
        if len(rows) > 1:
            duplicate_groups.append({
                "sha256": digest,
                "records": len(rows),
                "source_groups": sorted({row["source_group"] for row in rows}),
                "paths": sorted({row["resolved_path"] for row in rows if row["resolved_path"]})[:4],
            })

    return {
        "schema": "painter-data-inventory-v1",
        "manifest_count": len(manifest_paths),
        "image_roots": [str(root) for root in roots],
        "manifests": manifest_summaries,
        "counts": {
            "task_rows": len(records),
            "source_groups": len(source_groups),
            "actual_unique_image_hashes": len(actual_unique),
            "unique_declared_hashes": len(declared_unique),
            "unique_declared_reference_hashes": len(declared_reference_unique),
            "unique_declared_image_hashes": len(declared_image_unique),
            "unique_declared_target_hashes": len(declared_target_unique),
            "unique_content_hashes": len(actual_unique | declared_unique),
            # Backward-compatible alias.  Consumers should use the explicit
            # image/reference/target counters above because this union can mix
            # image hashes with declared program-target hashes.
            "unique_reference_hashes": len(actual_unique | declared_unique),
            "declared_unrendered_hashes": len(declared_only),
            "rendered_rows": sum(row["rendered"] for row in records),
            "hash_mismatches": sum(bool(row["declared_sha256"] and row["actual_sha256"] and not row["hash_match"]) for row in records),
        },
        "by_source_type": count_by(records, "source_type"),
        "by_family": count_by(records, "family"),
        "by_split": count_by(records, "split"),
        "source_groups_by_split": {split: len(groups) for split, groups in sorted(split_groups.items())},
        "train_eval_source_group_overlap": nonempty_overlap,
        "duplicate_hash_groups": duplicate_groups,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="JSON/JSONL metadata manifest; repeatable")
    parser.add_argument("--image-root", action="append", default=[], help="Explicit root for resolving named images; repeatable")
    parser.add_argument("--output", required=True, help="Output inventory JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = inventory(args.manifest, args.image_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
