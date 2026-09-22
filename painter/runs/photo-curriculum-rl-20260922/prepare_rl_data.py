#!/usr/bin/env python3
"""Prepare the photo/procedural handoff for native painting RL.

This is a metadata-only preparation tool.  It validates source bytes and
hashes, but never opens, renders, composites, or transforms an image.  The
native RL taskset requires a distinct fixed same-source opponent.  Since
those student canvases are produced on the Linux node, the complete source
inventory is written first and a launchable ``scaled-training-manifest.json``
is written only after an explicit opponent-cache index is supplied.

Typical preparation (from the repository root)::

    python painter/runs/photo-curriculum-rl-20260922/prepare_rl_data.py \
      --output painter/runs/photo-curriculum-rl-20260922

After a node-side cache has some validated canvases, materialize a finite
launch slice without falling back to teacher canvases::

    python .../prepare_rl_data.py --output ... \
      --opponent-index opponent-cache/index.json --max-rows 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VERSION = "photo-curriculum-rl-data-v1"
DEFAULT_SEED = 20260922
DEFAULT_PHOTO_MANIFEST = "painter/runs/scaled-curriculum-20260921/real-pool/reference-manifest.json"
DEFAULT_ADMISSION = "painter/runs/photo-curriculum-sft-20260922/launch-records/data/token-audit.json"
DEFAULT_PROCEDURAL_MANIFEST = "painter/runs/scaled-curriculum-20260921/corpus/scaled-data/manifest.jsonl"
DEFAULT_PROCEDURAL_RESULTS = "painter/runs/scaled-curriculum-20260921/corpus/scaled-data/reference-results.jsonl"
# Immutable SHA-256 of the SFT step-512 adapter served to the frozen
# opponent alias.  Keeping this in the generated handoff prevents an alias
# from being silently rebound to another adapter during RL.
FROZEN_OPPONENT_CHECKPOINT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def rows_from(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl(path)
    value = read_json(path)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = None
        for key in ("references", "rows", "training_rows", "tasks"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        if rows is None:
            raise ValueError(f"{path} has no references/rows list")
    else:
        raise ValueError(f"{path} must contain a JSON list or wrapper")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object row")
    return [dict(row) for row in rows]


def resolve(path_value: str | Path, *, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def portable(path: Path, *, manifest_dir: Path) -> str:
    """Return a manifest-relative path usable by native_painting_rl."""
    return os.path.relpath(path, manifest_dir).replace(os.sep, "/")


def keyed(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def family_order(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Round-robin families, with seeded stable order inside each family."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["family"])].append(row)
    for family, bucket in buckets.items():
        bucket.sort(key=lambda row: keyed(seed, str(row["id"])))
    ordered: list[dict[str, Any]] = []
    families = sorted(buckets, key=lambda value: keyed(seed, value))
    while families:
        next_families: list[str] = []
        for family in families:
            bucket = buckets[family]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_families.append(family)
        families = next_families
    return ordered


def balanced_order(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Expose both source kinds in a deterministic round-robin order.

    This preserves the exact 86:2400 source counts and never duplicates a
    row.  The order is a sampler order, not a claim that the two populations
    are equal or equally distributed in the underlying data.
    """
    buckets = {
        "photo": family_order([r for r in rows if r["source_kind"] == "photo"], seed=seed),
        "procedural": family_order([r for r in rows if r["source_kind"] == "procedural"], seed=seed),
    }
    order = ["photo", "procedural"]
    result: list[dict[str, Any]] = []
    while buckets["photo"] or buckets["procedural"]:
        for kind in order:
            if buckets[kind]:
                result.append(buckets[kind].pop(0))
    return result


def rel_input(path: Path, *, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def validate_file(path: Path, expected: str | None, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or a symlink: {path}")
    actual = digest(path)
    if expected and actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, found {actual}")
    return actual


def admitted_photo_rows(
    *, photo_manifest: Path, admission: Path, repo_root: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_rows = rows_from(photo_manifest)
    by_id = {str(row.get("id")): row for row in manifest_rows}
    if len(by_id) != len(manifest_rows):
        raise ValueError("photo source manifest contains duplicate ids")
    admission_value = read_json(admission)
    kept = admission_value.get("admission", {}).get("kept_real_photo_ids", [])
    if not isinstance(kept, list):
        raise ValueError("admission.kept_real_photo_ids must be a list")
    ids = [str(value).removesuffix("__full") for value in kept]
    if len(ids) != 86 or len(set(ids)) != len(ids):
        raise ValueError(f"expected 86 distinct admitted photos, found {len(ids)}")
    rows: list[dict[str, Any]] = []
    for source_id in ids:
        raw = by_id.get(source_id)
        if raw is None:
            raise ValueError(f"admitted photo is absent from source manifest: {source_id}")
        if raw.get("split") != "train":
            raise ValueError(f"admitted photo is not train split: {source_id}")
        reference = resolve(str(raw.get("reference") or raw.get("reference_path")), base=repo_root)
        reference_sha = validate_file(reference, raw.get("image_sha256"), label=f"photo {source_id}")
        rows.append({
            "id": source_id,
            "source_row_id": source_id,
            "source_group": str(raw.get("source_group") or source_id),
            "source_kind": "photo",
            "family": str(raw.get("primary_family") or raw.get("family") or "photo"),
            "split": "train",
            "mode": "fresh",
            "reference": portable(reference, manifest_dir=output_dir),
            "reference_sha256": reference_sha,
            "source_image_sha256": reference_sha,
            "reference_kind": "real_photo",
            "source_path": portable(reference, manifest_dir=output_dir),
            "admitted_from_sft": True,
        })
    return rows, {
        "source_manifest": rel_input(photo_manifest, repo_root=repo_root),
        "source_manifest_sha256": digest(photo_manifest),
        "admission_file": rel_input(admission, repo_root=repo_root),
        "admission_file_sha256": digest(admission),
        "kept_count": len(rows),
        "excluded_photo_dev_count": sum(1 for row in manifest_rows if row.get("split") == "dev"),
        "excluded_photo_not_admitted_count": len(manifest_rows) - len(rows),
    }


def procedural_rows(
    *, manifest: Path, results: Path, repo_root: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = rows_from(manifest)
    result_rows = rows_from(results)
    result_by_id = {str(row.get("id")): row for row in result_rows}
    if len(result_by_id) != len(result_rows):
        raise ValueError("procedural render receipts contain duplicate ids")
    train = [row for row in raw_rows if row.get("split") == "train"]
    dev_count = sum(1 for row in raw_rows if row.get("split") == "dev")
    if len(train) != 2400:
        raise ValueError(f"expected 2400 procedural train rows, found {len(train)}")
    rows: list[dict[str, Any]] = []
    corpus_root = manifest.parent
    for raw in train:
        source_id = str(raw.get("id") or "")
        if not source_id:
            raise ValueError("procedural row has no id")
        result = result_by_id.get(source_id)
        if result is None or result.get("valid") is not True:
            raise ValueError(f"procedural reference has no valid render receipt: {source_id}")
        reference = resolve(str(result.get("reference_png") or raw.get("reference")), base=corpus_root)
        reference_sha = validate_file(reference, result.get("image_sha256"), label=f"procedural {source_id}")
        source_rel = str(result.get("source") or "")
        source_path = resolve(source_rel, base=corpus_root)
        source_sha = validate_file(source_path, raw.get("source_sha256"), label=f"program {source_id}")
        if result.get("source_sha256") and result["source_sha256"] != source_sha:
            raise ValueError(f"procedural source receipt hash mismatch: {source_id}")
        rows.append({
            "id": source_id,
            "source_row_id": source_id,
            "source_group": str(raw.get("source_group") or source_id),
            "source_kind": "procedural",
            "family": str(raw.get("primary_family") or raw.get("family") or "procedural"),
            "split": "train",
            "mode": "fresh",
            "reference": portable(reference, manifest_dir=output_dir),
            "reference_sha256": reference_sha,
            "source_image_sha256": reference_sha,
            "reference_kind": "procedural_known_program",
            "source_path": portable(source_path, manifest_dir=output_dir),
            "source_sha256": source_sha,
            "difficulty": raw.get("difficulty"),
        })
    return rows, {
        "source_manifest": rel_input(manifest, repo_root=repo_root),
        "source_manifest_sha256": digest(manifest),
        "render_receipts": rel_input(results, repo_root=repo_root),
        "render_receipts_sha256": digest(results),
        "kept_count": len(rows),
        "excluded_procedural_dev_count": dev_count,
    }


def load_opponents(path: Path, *, repo_root: Path, output_dir: Path) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = value.get("opponents") or value.get("rows") or value.get("cache") or []
    else:
        raise ValueError("opponent index must be a JSON list or an opponents/rows/cache wrapper")
    if not isinstance(items, list):
        raise ValueError("opponent index entries must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("opponent index contains a non-object entry")
        key = str(item.get("source_group") or item.get("source_row_id") or item.get("id") or "")
        value_path = item.get("opponent") or item.get("opponent_png") or item.get("path") or item.get("png")
        if not key or not isinstance(value_path, str):
            raise ValueError("each opponent entry needs source_group/id and opponent/path")
        path_abs = resolve(value_path, base=path.parent)
        if not path_abs.is_file() and not Path(value_path).is_absolute():
            path_abs = resolve(value_path, base=repo_root)
        actual = validate_file(path_abs, item.get("opponent_sha256") or item.get("sha256"), label=f"opponent {key}")
        result[key] = {
            "opponent": portable(path_abs, manifest_dir=output_dir),
            "opponent_sha256": actual,
            "opponent_source_path": rel_input(path_abs, repo_root=repo_root),
            "opponent_model": item.get("opponent_model", "student-cache"),
        }
    return result


def materialize(
    rows: list[dict[str, Any]], *, opponent_index: Path, output_dir: Path,
    repo_root: Path, max_rows: int | None, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opponents = load_opponents(opponent_index, repo_root=repo_root, output_dir=output_dir)
    candidates: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        opponent = opponents.get(row["source_group"]) or opponents.get(row["id"])
        if opponent is None:
            missing += 1
            continue
        if opponent["opponent_sha256"] == row["reference_sha256"]:
            raise ValueError(f"opponent is identical to reference: {row['id']}")
        materialized = dict(row)
        materialized.update(opponent)
        materialized["opponent_source_group"] = row["source_group"]
        materialized["rollout_group"] = row["source_group"]
        candidates.append(materialized)
    candidates = balanced_order(candidates, seed=seed)
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("--max-rows must be positive")
        candidates = candidates[:max_rows]
    return candidates, {
        "opponent_index": rel_input(opponent_index, repo_root=repo_root),
        "opponent_index_sha256": digest(opponent_index),
        "available_opponents": len(opponents),
        "materialized_rows": len(candidates),
        "missing_opponents": missing,
        "max_rows": max_rows,
        "opponent_policy": "student_cache_only; no teacher substitution",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_node_input_list(
    *, output_dir: Path, repo_root: Path, photo_manifest: Path, admission: Path,
    procedural_manifest: Path, procedural_results: Path,
) -> dict[str, Any]:
    """List byte inputs for a node-side tar; never copies or transforms them."""
    paths: set[Path] = {photo_manifest, admission, procedural_manifest, procedural_results}
    photo_rows = rows_from(photo_manifest)
    admission_value = read_json(admission)
    admitted_photo_ids = {
        str(value).removesuffix("__full")
        for value in admission_value.get("admission", {}).get("kept_real_photo_ids", [])
    }
    for row in photo_rows:
        if row.get("split") == "dev" or (
            row.get("split") == "train" and str(row.get("id")) in admitted_photo_ids
        ):
            paths.add(resolve(str(row.get("reference") or row.get("reference_path")), base=repo_root))
    procedural_rows_all = rows_from(procedural_manifest)
    result_by_id = {str(row.get("id")): row for row in rows_from(procedural_results)}
    reference_counts = Counter()
    program_paths: set[Path] = set()
    for row in procedural_rows_all:
        split = str(row.get("split"))
        if split not in {"train", "dev"}:
            continue
        result = result_by_id.get(str(row.get("id")))
        if result and result.get("reference_png"):
            paths.add(resolve(str(result["reference_png"]), base=procedural_manifest.parent))
            reference_counts[split] += 1
        for program in row.get("programs", []):
            program_path = resolve(str(program), base=procedural_manifest.parent)
            paths.add(program_path)
            program_paths.add(program_path)
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"node input is missing or a symlink: {path}")
    lines = sorted(rel_input(path, repo_root=repo_root) for path in paths)
    (output_dir / "node-inputs.txt").write_text("\n".join(lines) + "\n")
    return {
        "node_input_list": "node-inputs.txt",
        "node_input_files": len(lines),
        "node_input_bytes": sum((repo_root / line).stat().st_size for line in lines),
        "node_reference_counts": {
            "photo_train": sum(
                1 for row in photo_rows if row.get("split") == "train" and str(row.get("id")) in admitted_photo_ids
            ),
            "photo_dev": sum(1 for row in photo_rows if row.get("split") == "dev"),
            "procedural_train": reference_counts["train"],
            "procedural_dev": reference_counts["dev"],
        },
        "node_program_files": len(program_paths),
        "copy_policy": "byte-copy on Linux node only; no local image processing",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[3]
    def input_path(value: str) -> Path:
        candidate = resolve(value, base=Path.cwd())
        if candidate.is_file() or Path(value).is_absolute():
            return candidate
        return resolve(value, base=repo_root)

    photo_manifest = input_path(args.photo_manifest)
    admission = input_path(args.admission)
    procedural_manifest = input_path(args.procedural_manifest)
    procedural_results = input_path(args.procedural_results)
    if args.max_rows is not None and not args.opponent_index:
        raise ValueError("--max-rows requires --opponent-index")
    photo, photo_meta = admitted_photo_rows(
        photo_manifest=photo_manifest, admission=admission, repo_root=repo_root, output_dir=output_dir
    )
    procedural, procedural_meta = procedural_rows(
        manifest=procedural_manifest, results=procedural_results, repo_root=repo_root, output_dir=output_dir
    )
    rows = balanced_order(photo + procedural, seed=args.seed)
    if len(rows) != 2486:
        raise ValueError(f"expected 2486 source rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    groups = [str(row["source_group"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups):
        raise ValueError("source inventory contains duplicate ids or source_group values")
    for order, row in enumerate(rows):
        row["order"] = order
    source_manifest = {
        "schema": VERSION,
        "run_id": "photo-curriculum-rl-20260922",
        "sampler": {
            "name": "seeded_kind_round_robin",
            "seed": args.seed,
            "description": "photo/procedural round-robin, family round-robin within each kind; no duplication",
        },
        "training_scope": {
            "source_rows": len(rows),
            "photo_train": len(photo),
            "procedural_train": len(procedural),
            "photo_dev_excluded": photo_meta["excluded_photo_dev_count"],
            "procedural_dev_excluded": procedural_meta["excluded_procedural_dev_count"],
            "sft_nonpainting_general_excluded": 192,
        },
        "rl_protocol": {
            "max_turns": 6,
            "max_completion_tokens": 8192,
            "context_length": 32768,
            "live_canvas_feedback": True,
            "taskset": "native_painting_rl",
            "opponent_requirement": "distinct fixed same-source_group canvas",
        },
        "inputs": {"photo": photo_meta, "procedural": procedural_meta},
        "rows": rows,
    }
    write_json(output_dir / "source-manifest.json", source_manifest)
    node_inputs = write_node_input_list(
        output_dir=output_dir,
        repo_root=repo_root,
        photo_manifest=photo_manifest,
        admission=admission,
        procedural_manifest=procedural_manifest,
        procedural_results=procedural_results,
    )
    coverage = {
        "schema": VERSION,
        "run_id": source_manifest["run_id"],
        "source_manifest": "source-manifest.json",
        "source_manifest_sha256": digest(output_dir / "source-manifest.json"),
        "counts": dict(Counter(row["source_kind"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "reference_bytes_verified": len(rows),
        "inventory_scope": "source inventory; not a runtime seen-reference count",
        "runtime_seen_refs": None,
        "node_inputs": node_inputs,
        "opponent_status": "deferred; no teacher opponent substitution",
        "opponent_cache_rows": 0,
        "excluded": {
            "photo_dev": photo_meta["excluded_photo_dev_count"],
            "procedural_dev": procedural_meta["excluded_procedural_dev_count"],
            "nonpainting_general_sft_rows": 192,
        },
    }
    write_json(output_dir / "coverage.json", coverage)
    write_json(output_dir / "RL-CONFIG.json", {
        "schema": VERSION,
        "taskset": "native_painting_rl",
        "manifest": "source-manifest.json",
        "expected_count": 2486,
        "source_manifest": "source-manifest.json",
        "preserve_manifest_order": True,
        "lazy_opponent": True,
        "opponent_cache_root": "frozen-opponents",
        "opponent_seed": DEFAULT_SEED,
        "max_turns": 6,
        "max_completion_tokens": 8192,
        "context_length": 32768,
        "live_canvas_feedback": True,
        "opponent_policy": "frozen_midpoint_student_cache_only",
        "opponent_model_alias": "sft-step512",
        "opponent_checkpoint_sha256": FROZEN_OPPONENT_CHECKPOINT_SHA256,
        "opponent_protocol_version": "native-painting-rl-v1",
        "opponent_render_protocol": "native-painting-v1",
        "opponent_decoding_config": {},
        "opponent_generator": "frozen_opponent:generate_frozen_opponent",
        "opponent_model": "SFT step512 adapter on the same base model",
        "teacher_substitution": False,
        "launch_note": "Launch source-manifest.json lazily with the pinned sft-step512 checkpoint and native generator route; native loader resolves anchors at reward time.",
    })
    handoff = {
        "schema": VERSION,
        "source_manifest": "source-manifest.json",
        "coverage": "coverage.json",
        "launch_manifest": None,
        "lazy_launch_manifest": "source-manifest.json",
        "lazy_launch_note": "Use source-manifest.json with lazy_opponent=true, alias sft-step512, and the pinned checkpoint hash in RL-CONFIG.json.",
        "status": "lazy launch manifest ready; pinned sft-step512 checkpoint",
        "opponent_policy": "student_cache_only; no teacher substitution",
        "materialize_command": "prepare_rl_data.py --opponent-index <node-cache-index.json> [--max-rows N]",
    }
    if args.opponent_index:
        opponent_index = resolve(args.opponent_index, base=Path.cwd())
        launch_rows, opponent_meta = materialize(
            rows, opponent_index=opponent_index, output_dir=output_dir,
            repo_root=repo_root, max_rows=args.max_rows, seed=args.seed
        )
        if not launch_rows:
            raise ValueError("opponent index produced no launchable rows")
        launch = {
            "schema": VERSION,
            "run_id": source_manifest["run_id"],
            "expected_count": len(launch_rows),
            "sampler": source_manifest["sampler"],
            "rl_protocol": source_manifest["rl_protocol"],
            "opponents": opponent_meta,
            "training_rows": launch_rows,
        }
        write_json(output_dir / "scaled-training-manifest.json", launch)
        handoff.update({
            "launch_manifest": "scaled-training-manifest.json",
            "status": "launchable validated student-opponent slice",
            "opponents": opponent_meta,
        })
        coverage.update({
            "source_manifest_sha256": digest(output_dir / "source-manifest.json"),
            "opponent_status": "validated student cache slice",
            "opponent_cache_rows": len(launch_rows),
        })
        write_json(output_dir / "coverage.json", coverage)
    else:
        write_json(output_dir / "scaled-training-manifest.pending.json", {
            "schema": VERSION,
            "status": "pending opponent cache; not accepted by native_painting_rl",
            "source_manifest": "source-manifest.json",
            "training_rows": [],
        })
    write_json(output_dir / "RL-HANDOFF.json", handoff)
    return {
        "output": str(output_dir),
        "source_rows": len(rows),
        "photo": len(photo),
        "procedural": len(procedural),
        "launch_rows": handoff.get("opponents", {}).get("materialized_rows", 0),
        "status": handoff["status"],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="new RL run directory")
    p.add_argument("--photo-manifest", default=DEFAULT_PHOTO_MANIFEST)
    p.add_argument("--admission", default=DEFAULT_ADMISSION)
    p.add_argument("--procedural-manifest", default=DEFAULT_PROCEDURAL_MANIFEST)
    p.add_argument("--procedural-results", default=DEFAULT_PROCEDURAL_RESULTS)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--opponent-index", default=None, help="JSON index of validated student canvases")
    p.add_argument("--max-rows", type=int, default=None, help="deterministic finite launch slice")
    return p


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
