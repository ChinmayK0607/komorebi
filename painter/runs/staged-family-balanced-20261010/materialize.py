#!/usr/bin/env python3
"""Materialize the staged run's verified native manifests.

This is a node-free handoff step.  It accepts the prior run's native train and
heldout manifests plus an asset root containing their prompts, frozen anchor
PNGs, optional casebriefs, and anchor provenance.  The prior manifests contain
absolute paths from another checkout; those paths are treated as names only and
are resolved below the explicitly supplied asset root.

The output is self-contained: both native manifests live at the output root
and use paths relative to that root.  ``painter.native_painting_rl`` resolves a
relative asset first beside the manifest, so no relocation shim is needed.
No model, renderer, judge, network, credentials, or Git state is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable


RUN_ROOT = Path(__file__).resolve().parent
DEFAULT_CURRICULUM = RUN_ROOT / "curriculum-manifest.json"

CURRICULUM_SCHEMA = "staged-family-balanced-curriculum-v1"
NATIVE_SCHEMA = "native-painting-rl-structured-flow-v1"
PROVENANCE_SCHEMA = "structured-flow-anchor-provenance-v1"
MATERIALIZATION_SCHEMA = "staged-family-balanced-materialization-v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
)
EXPECTED_TRAIN_ROWS = 96
EXPECTED_HELDOUT_ROWS = 32
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MaterializationError(ValueError):
    """Raised when an input cannot be admitted into the staged run."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require_sha(value: Any, description: str) -> str:
    if not _is_sha256(value):
        raise MaterializationError(f"{description} must be a lowercase SHA-256 digest")
    return str(value)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(f"{description} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{description} must contain a JSON object")
    return value


def _resolve_input_file(path: Path, description: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(f"{description} is not a regular file: {path}")
    return path.absolute()


def _check_id(value: Any, description: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise MaterializationError(f"{description} is not a safe row identifier: {value!r}")
    return value


def _rows(document: dict[str, Any], description: str) -> list[dict[str, Any]]:
    value = document.get("rows")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise MaterializationError(f"{description} must contain a rows list")
    return [dict(row) for row in value]


def _curriculum_rows(curriculum: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if curriculum.get("schema") != CURRICULUM_SCHEMA:
        raise MaterializationError("curriculum schema is not the staged-family-balanced schema")
    run_id = curriculum.get("run_id")
    prior_run_id = curriculum.get("prior_run_id")
    if not isinstance(run_id, str) or not isinstance(prior_run_id, str) or not run_id or not prior_run_id:
        raise MaterializationError("curriculum run_id and prior_run_id are required")

    by_id: dict[str, dict[str, Any]] = {}
    stage_train_ids: list[str] = []
    for stage in curriculum.get("stages", []):
        if not isinstance(stage, dict):
            raise MaterializationError("curriculum stages must contain objects")
        for row in stage.get("train_rows", []):
            if not isinstance(row, dict):
                raise MaterializationError("curriculum train rows must contain objects")
            ident = _check_id(row.get("id"), "curriculum train row id")
            if ident in by_id:
                raise MaterializationError(f"duplicate curriculum row id: {ident}")
            by_id[ident] = dict(row)
            stage_train_ids.append(ident)

    train_ids = curriculum.get("train_row_ids")
    if not isinstance(train_ids, list) or not all(isinstance(value, str) for value in train_ids):
        raise MaterializationError("curriculum train_row_ids must be a list of strings")
    if len(train_ids) != EXPECTED_TRAIN_ROWS or len(set(train_ids)) != EXPECTED_TRAIN_ROWS:
        raise MaterializationError("curriculum must contain exactly 96 unique train row IDs")
    if set(train_ids) != set(stage_train_ids):
        raise MaterializationError("curriculum train_row_ids do not match the stage train rows")

    panel = curriculum.get("matched_heldout_panel")
    heldout_rows = panel.get("rows") if isinstance(panel, dict) else None
    if not isinstance(heldout_rows, list) or not all(isinstance(row, dict) for row in heldout_rows):
        raise MaterializationError("curriculum heldout panel must contain rows")
    if len(heldout_rows) != EXPECTED_HELDOUT_ROWS:
        raise MaterializationError("curriculum must contain exactly 32 heldout rows")
    heldout_ids: list[str] = []
    for row in heldout_rows:
        ident = _check_id(row.get("id"), "curriculum heldout row id")
        if ident in by_id or ident in heldout_ids:
            raise MaterializationError(f"duplicate curriculum row id: {ident}")
        by_id[ident] = dict(row)
        heldout_ids.append(ident)
    return [str(value) for value in train_ids], [dict(row) for row in heldout_rows], by_id


def _assert_below(root: Path, candidate: Path, description: str) -> Path:
    """Validate a candidate without allowing symlink or lexical escape."""

    root = root.absolute()
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise MaterializationError(f"{description} escapes the asset root: {candidate}") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise MaterializationError(f"{description} uses a symlink: {current}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterializationError(f"{description} escapes or is missing below the asset root: {candidate}") from exc
    if not resolved.is_file():
        raise MaterializationError(f"{description} is not a regular file: {candidate}")
    return resolved


def _candidate_paths(root: Path, raw: str) -> list[Path]:
    value = Path(raw)
    if "\x00" in raw or any(part == ".." for part in value.parts):
        raise MaterializationError(f"asset path contains traversal: {raw!r}")
    candidates: list[Path] = []
    parts = list(value.parts)
    if not value.is_absolute():
        relative = value
        if relative.parts and relative.parts[0] == "curriculum":
            candidates.append(root.joinpath(*relative.parts[1:]))
        candidates.append(root / relative)
    else:
        # Absolute paths from the prior checkout are historical labels.  Use
        # only their known asset suffixes under the explicit new root.
        for marker in ("curriculum", "prompts", "anchors", "casebriefs"):
            if marker in parts:
                index = parts.index(marker)
                candidates.append(root.joinpath(*parts[index:]))
        if value.name == "anchor-provenance.json":
            candidates.extend((root / "anchors" / value.name, root / value.name))
        if value.name:
            candidates.append(root / value.name)

    # Some prior casebrief paths include a generated directory immediately
    # before ``casebriefs``.  Preserve that directory when it is present.
    if value.is_absolute() and "casebriefs" in parts:
        index = parts.index("casebriefs")
        if index:
            candidates.append(root.joinpath(*parts[index - 1 :]))

    # A controlled basename fallback supports compact asset roots used by the
    # node handoff.  It is still hash-bound and rejects symlinks/ambiguity.
    if value.name:
        try:
            candidates.extend(root.rglob(value.name))
        except OSError as exc:
            raise MaterializationError(f"cannot inspect asset root: {root}") from exc
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _resolve_asset(root: Path, raw: Any, expected_sha256: str, description: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise MaterializationError(f"{description} path is missing")
    expected = _require_sha(expected_sha256, f"{description} expected hash")
    matches: list[Path] = []
    for candidate in _candidate_paths(root, raw):
        try:
            resolved = _assert_below(root, candidate, description)
        except MaterializationError as exc:
            # A directly named symlink or traversal is a hard failure.  A
            # basename fallback can contain unrelated paths, so only suppress
            # ordinary non-candidates and continue searching.
            if candidate.absolute() == candidate and candidate.exists() and candidate.is_symlink():
                raise
            if "escapes" in str(exc) or "traversal" in str(exc) or "uses a symlink" in str(exc):
                if candidate.exists() or candidate.is_symlink():
                    raise
            continue
        if sha256_file(resolved) == expected and resolved not in matches:
            matches.append(resolved)
    if not matches:
        raise MaterializationError(f"{description} cannot be found with expected hash {expected}")
    if len(matches) != 1:
        raise MaterializationError(f"{description} has ambiguous matching assets: {matches}")
    return matches[0]


def _resolve_expected_asset(root: Path, raw: Any, expected: Path, description: str) -> Path:
    """Resolve a historical asset label to one already hash-verified path.

    This preserves the legacy absolute-path remapping accepted by
    ``_resolve_asset`` without hashing the same shared artifact for every row.
    """

    if not isinstance(raw, str) or not raw:
        raise MaterializationError(f"{description} path is missing")
    expected = expected.resolve(strict=True)
    for candidate in _candidate_paths(root, raw):
        try:
            resolved = _assert_below(root, candidate, description)
        except MaterializationError as exc:
            if candidate.absolute() == candidate and candidate.exists() and candidate.is_symlink():
                raise
            if "escapes" in str(exc) or "traversal" in str(exc) or "uses a symlink" in str(exc):
                if candidate.exists() or candidate.is_symlink():
                    raise
            continue
        if resolved == expected:
            return expected
    raise MaterializationError(f"{description} does not resolve to the verified shared asset")


def _copy_verified(source: Path, destination: Path, expected_sha256: str, description: str) -> str:
    expected = _require_sha(expected_sha256, f"{description} expected hash")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    actual = sha256_file(destination)
    if actual != expected:
        raise MaterializationError(f"{description} changed while copying")
    return actual


def _validate_manifest_header(document: dict[str, Any], split: str, prior_run_id: str) -> dict[str, Any]:
    if document.get("schema") != NATIVE_SCHEMA:
        raise MaterializationError(f"{split} manifest schema is not {NATIVE_SCHEMA}")
    if document.get("run_id") != prior_run_id:
        raise MaterializationError(f"{split} manifest run_id does not match curriculum prior_run_id")
    if document.get("split") != split:
        raise MaterializationError(f"{split} manifest split is incorrect")
    frozen = document.get("frozen_opponent")
    if not isinstance(frozen, dict):
        raise MaterializationError(f"{split} manifest frozen_opponent metadata is missing")
    checkpoint = _require_sha(frozen.get("checkpoint_sha256"), f"{split} checkpoint hash")
    if checkpoint != EXPECTED_CHECKPOINT_SHA256:
        raise MaterializationError(f"{split} manifest is not bound to the verified SFT512 checkpoint")
    _require_sha(frozen.get("provenance_sha256"), f"{split} provenance hash")
    return frozen


def _validate_row_identity(row: dict[str, Any], expected: dict[str, Any], split: str) -> str:
    ident = _check_id(row.get("id"), f"{split} source row id")
    for key in ("source_group", "family", "family_kind", "stage", "split", "variant", "reference_sha256", "prompt_signature"):
        if row.get(key) != expected.get(key):
            raise MaterializationError(f"{split} row {ident} {key} does not match curriculum")
    if row.get("task_id") not in (None, ident):
        raise MaterializationError(f"{split} row {ident} task_id does not match id")
    if row.get("reference_kind") != "text":
        raise MaterializationError(f"{split} row {ident} must use a text reference")
    if row.get("opponent_source_group") != row.get("source_group"):
        raise MaterializationError(f"{split} row {ident} opponent source_group does not match")
    if not isinstance(row.get("anchor_provenance_file"), str) or not row["anchor_provenance_file"]:
        raise MaterializationError(f"{split} row {ident} anchor provenance path is missing")
    _require_sha(row.get("reference_sha256"), f"{split} row {ident} reference hash")
    _require_sha(row.get("opponent_sha256"), f"{split} row {ident} opponent hash")
    checkpoint = row.get("frozen_sft512_checkpoint_sha256")
    if checkpoint != EXPECTED_CHECKPOINT_SHA256:
        raise MaterializationError(f"{split} row {ident} checkpoint hash is not the verified SFT512 hash")
    return ident


def _validate_provenance(
    provenance: dict[str, Any],
    expected_rows: Iterable[dict[str, Any]],
    source_run_id: str,
    checkpoint: str,
) -> None:
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise MaterializationError("anchor provenance schema is incorrect")
    if provenance.get("run_id") != source_run_id:
        raise MaterializationError("anchor provenance run_id does not match source manifests")
    if provenance.get("checkpoint_sha256") != checkpoint:
        raise MaterializationError("anchor provenance checkpoint hash does not match manifests")
    entries = provenance.get("anchors")
    if not isinstance(entries, dict):
        raise MaterializationError("anchor provenance anchors must be an object")
    for row in expected_rows:
        ident = str(row["id"])
        entry = entries.get(ident)
        if not isinstance(entry, dict):
            raise MaterializationError(f"anchor provenance is missing {ident}")
        if entry.get("source_group") != row["source_group"]:
            raise MaterializationError(f"anchor provenance source_group mismatch for {ident}")
        if entry.get("reference_sha256") != row["reference_sha256"]:
            raise MaterializationError(f"anchor provenance reference hash mismatch for {ident}")
        if entry.get("checkpoint_sha256") != checkpoint:
            raise MaterializationError(f"anchor provenance checkpoint mismatch for {ident}")
        if entry.get("png_sha256") != row["opponent_sha256"]:
            raise MaterializationError(f"anchor provenance PNG hash mismatch for {ident}")
        if entry.get("valid") is not True:
            raise MaterializationError(f"anchor provenance entry is not valid for {ident}")
        _require_sha(entry.get("png_sha256"), f"anchor provenance PNG hash for {ident}")


def _relative(path: Path) -> str:
    return PurePosixPath(path.as_posix()).as_posix()


def _output_row(
    row: dict[str, Any],
    split: str,
    prompt_path: str,
    anchor_path: str,
    provenance_path: str,
    casebrief_path: str | None,
) -> dict[str, Any]:
    result = dict(row)
    result["reference"] = prompt_path
    result["source_prompt"] = prompt_path
    result["opponent"] = anchor_path
    result["anchor_provenance_file"] = provenance_path
    if casebrief_path is not None:
        result["casebrief"] = casebrief_path
    else:
        result.pop("casebrief", None)
        result.pop("casebrief_sha256", None)
    for key in (
        "reference",
        "source_prompt",
        "opponent",
        "casebrief",
        "anchor_provenance_file",
        "initial_canvas",
        "initial_program",
        "renderer",
        "renderer_python",
        "judge_module_dir",
    ):
        value = result.get(key)
        if isinstance(value, str) and Path(value).is_absolute():
            raise MaterializationError(f"output row {row.get('id')} retains an absolute {key} path")
    result["split"] = split
    return result


def _materialize_into(
    *,
    train_manifest: Path,
    heldout_manifest: Path,
    curriculum_manifest: Path,
    asset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    curriculum = _read_json(curriculum_manifest, "curriculum manifest")
    train_ids, heldout_expected_rows, expected_by_id = _curriculum_rows(curriculum)
    staged_run_id = str(curriculum["run_id"])
    prior_run_id = str(curriculum["prior_run_id"])
    train_document = _read_json(train_manifest, "prior train manifest")
    heldout_document = _read_json(heldout_manifest, "prior heldout manifest")
    train_frozen = _validate_manifest_header(train_document, "train", prior_run_id)
    heldout_frozen = _validate_manifest_header(heldout_document, "heldout", prior_run_id)
    if train_frozen.get("checkpoint_sha256") != heldout_frozen.get("checkpoint_sha256"):
        raise MaterializationError("train and heldout checkpoint hashes differ")
    checkpoint = str(train_frozen["checkpoint_sha256"])

    train_rows_source = _rows(train_document, "prior train manifest")
    heldout_rows_source = _rows(heldout_document, "prior heldout manifest")
    if len(train_rows_source) != EXPECTED_TRAIN_ROWS or len(heldout_rows_source) != EXPECTED_HELDOUT_ROWS:
        raise MaterializationError("prior manifests must contain exactly 96 train and 32 heldout rows")

    train_by_id: dict[str, dict[str, Any]] = {}
    for row in train_rows_source:
        ident = _validate_row_identity(row, expected_by_id.get(str(row.get("id")), {}), "train")
        if ident in train_by_id:
            raise MaterializationError(f"duplicate train row id: {ident}")
        if ident not in train_ids:
            raise MaterializationError(f"train row is not in curriculum: {ident}")
        train_by_id[ident] = row
    if set(train_by_id) != set(train_ids):
        raise MaterializationError("prior train manifest membership does not equal curriculum train_row_ids")

    heldout_ids = [str(row["id"]) for row in heldout_expected_rows]
    heldout_by_id: dict[str, dict[str, Any]] = {}
    for row in heldout_rows_source:
        ident = _validate_row_identity(row, expected_by_id.get(str(row.get("id")), {}), "heldout")
        if ident in heldout_by_id:
            raise MaterializationError(f"duplicate heldout row id: {ident}")
        if ident not in heldout_ids:
            raise MaterializationError(f"heldout row is not in curriculum: {ident}")
        heldout_by_id[ident] = row
    if set(heldout_by_id) != set(heldout_ids):
        raise MaterializationError("prior heldout manifest membership does not equal curriculum heldout panel")

    ordered_rows = [train_by_id[ident] for ident in train_ids] + [heldout_by_id[ident] for ident in heldout_ids]
    provenance_raw = train_frozen.get("provenance_file")
    if not isinstance(provenance_raw, str) or not provenance_raw:
        raise MaterializationError("train frozen_opponent provenance_file is missing")
    if heldout_frozen.get("provenance_file") != provenance_raw:
        raise MaterializationError("train and heldout provenance paths differ")
    provenance_hash = _require_sha(train_frozen.get("provenance_sha256"), "provenance hash")
    provenance_source = _resolve_asset(asset_root, provenance_raw, provenance_hash, "anchor provenance")
    resolved_provenance_labels: dict[str, Path] = {}
    for row in ordered_rows:
        # The shared artifact is hash-verified above. Resolve each distinct
        # historical label once, then reuse the identity result for every row.
        raw = row.get("anchor_provenance_file")
        if not isinstance(raw, str) or not raw:
            raise MaterializationError(f"anchor provenance file differs for {row['id']}")
        row_provenance = resolved_provenance_labels.get(raw)
        if row_provenance is None:
            row_provenance = _resolve_expected_asset(
                asset_root, raw, provenance_source, f"anchor provenance for {row['id']}"
            )
            resolved_provenance_labels[raw] = row_provenance
        if row_provenance != provenance_source:
            raise MaterializationError(f"anchor provenance file differs for {row['id']}")
    provenance = _read_json(provenance_source, "anchor provenance")
    _validate_provenance(provenance, ordered_rows, prior_run_id, checkpoint)

    # All validation above happens before the destination is populated.  The
    # caller places this function inside an atomic temporary output root.
    provenance_rel = "assets/anchors/anchor-provenance.json"
    provenance_destination = output_root / provenance_rel
    _copy_verified(provenance_source, provenance_destination, provenance_hash, "anchor provenance")

    def materialize_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        output_rows: list[dict[str, Any]] = []
        for row in rows:
            ident = str(row["id"])
            reference_hash = str(row["reference_sha256"])
            opponent_hash = str(row["opponent_sha256"])
            reference_source = _resolve_asset(asset_root, row.get("reference"), reference_hash, f"{split} reference {ident}")
            if row.get("source_prompt") is not None:
                source_prompt = _resolve_asset(asset_root, row.get("source_prompt"), reference_hash, f"{split} source prompt {ident}")
                if sha256_file(source_prompt) != sha256_file(reference_source):
                    raise MaterializationError(f"{split} source prompt differs from reference for {ident}")
            anchor_source = _resolve_asset(asset_root, row.get("opponent"), opponent_hash, f"{split} anchor {ident}")
            prompt_rel = f"assets/prompts/{split}/{ident}.txt"
            anchor_rel = f"assets/anchors/{ident}.png"
            _copy_verified(reference_source, output_root / prompt_rel, reference_hash, f"{split} reference {ident}")
            _copy_verified(anchor_source, output_root / anchor_rel, opponent_hash, f"{split} anchor {ident}")

            casebrief_rel: str | None = None
            if row.get("casebrief") is not None:
                casebrief_hash = _require_sha(row.get("casebrief_sha256"), f"{split} casebrief hash for {ident}")
                casebrief_source = _resolve_asset(asset_root, row.get("casebrief"), casebrief_hash, f"{split} casebrief {ident}")
                casebrief_rel = f"assets/casebriefs/{split}/{ident}.txt"
                _copy_verified(casebrief_source, output_root / casebrief_rel, casebrief_hash, f"{split} casebrief {ident}")
            elif row.get("casebrief_sha256") is not None:
                raise MaterializationError(f"{split} casebrief hash exists without a casebrief path for {ident}")
            output_rows.append(_output_row(row, split, prompt_rel, anchor_rel, provenance_rel, casebrief_rel))
        return output_rows

    train_rows = materialize_rows([train_by_id[ident] for ident in train_ids], "train")
    heldout_rows = materialize_rows([heldout_by_id[ident] for ident in heldout_ids], "heldout")
    source_metadata = {
        "schema": NATIVE_SCHEMA,
        "run_id": staged_run_id,
        "source_run_id": prior_run_id,
        "status": "materialized_with_verified_frozen_sft512_anchors",
        "native_rl_compatible": True,
        "split": "train",
        "frozen_opponent": {
            **{key: value for key, value in train_frozen.items() if key != "provenance_file"},
            "checkpoint_sha256": checkpoint,
            "provenance_file": provenance_rel,
            "provenance_sha256": provenance_hash,
        },
        "source_manifest_sha256": sha256_file(train_manifest),
        "curriculum_manifest_sha256": sha256_file(curriculum_manifest),
    }
    train_output = {**source_metadata, "rows": train_rows}
    heldout_output = {
        **source_metadata,
        "native_rl_compatible": False,
        "split": "heldout",
        "source_manifest_sha256": sha256_file(heldout_manifest),
        "rows": heldout_rows,
        "cases": heldout_rows,
    }
    _atomic_write_json(output_root / "train-manifest.json", train_output)
    _atomic_write_json(output_root / "heldout-manifest.json", heldout_output)

    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "materialization-receipt.json":
            continue
        files.append({"path": _relative(path.relative_to(output_root)), "sha256": sha256_file(path)})
    return {
        "schema": MATERIALIZATION_SCHEMA,
        "run_id": staged_run_id,
        "source_run_id": prior_run_id,
        "status": "materialized_with_verified_frozen_sft512_anchors",
        "counts": {"train_rows": len(train_rows), "heldout_rows": len(heldout_rows), "copied_files": len(files)},
        "checkpoint_sha256": checkpoint,
        "curriculum_manifest_sha256": sha256_file(curriculum_manifest),
        "source_manifests": {
            "train": {"path": train_manifest.name, "sha256": sha256_file(train_manifest)},
            "heldout": {"path": heldout_manifest.name, "sha256": sha256_file(heldout_manifest)},
        },
        "anchor_provenance": {"path": provenance_rel, "sha256": provenance_hash},
        "output_manifests": {
            "train": {"path": "train-manifest.json", "sha256": sha256_file(output_root / "train-manifest.json")},
            "heldout": {"path": "heldout-manifest.json", "sha256": sha256_file(output_root / "heldout-manifest.json")},
        },
        "files": files,
        "loader_compatibility": {
            "relative_paths": True,
            "resolver": "painter.native_painting_rl._resolve_asset",
            "resolution": "manifest-parent-relative paths are supported; no relocation resolver is required",
            "train_manifest": "train-manifest.json",
        },
        "limitations": [
            "This step validates and copies prior evidence only; it does not render, judge, train, contact a provider, or load Prime.",
            "Historical absolute paths inside anchor provenance terminal metadata are preserved as provenance bytes and are not operational manifest paths.",
        ],
    }


def materialize(
    *,
    train_manifest: Path,
    heldout_manifest: Path,
    asset_root: Path,
    output_root: Path,
    curriculum_manifest: Path = DEFAULT_CURRICULUM,
) -> dict[str, Any]:
    """Materialize a verified, relocatable staged train/heldout pair."""

    train_manifest = _resolve_input_file(Path(train_manifest), "prior train manifest")
    heldout_manifest = _resolve_input_file(Path(heldout_manifest), "prior heldout manifest")
    curriculum_manifest = _resolve_input_file(Path(curriculum_manifest), "curriculum manifest")
    asset_root = Path(asset_root).expanduser()
    if asset_root.is_symlink() or not asset_root.is_dir():
        raise MaterializationError(f"asset root is not a regular directory: {asset_root}")
    asset_root = asset_root.resolve(strict=True)
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        receipt = _materialize_into(
            train_manifest=train_manifest,
            heldout_manifest=heldout_manifest,
            curriculum_manifest=curriculum_manifest,
            asset_root=asset_root,
            output_root=temporary,
        )
        _atomic_write_json(temporary / "materialization-receipt.json", receipt)
        try:
            temporary.replace(output_root)
        except FileExistsError as exc:
            raise FileExistsError(f"output root appeared during materialization: {output_root}") from exc
        receipt["receipt_sha256"] = sha256_file(output_root / "materialization-receipt.json")
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, default=DEFAULT_CURRICULUM)
    args = parser.parse_args(argv)
    receipt = materialize(
        train_manifest=args.train_manifest,
        heldout_manifest=args.heldout_manifest,
        asset_root=args.asset_root,
        output_root=args.output_root,
        curriculum_manifest=args.curriculum_manifest,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
