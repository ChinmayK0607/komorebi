#!/usr/bin/env python3
"""Admit reviewed teacher episodes into a staged painting curriculum.

This is an offline, fail-closed exporter.  It never calls a model, a judge,
or a renderer.  A benchmark episode is copied only when it is a selected
model episode, finished with a valid final canvas, and has an explicit
``admitted: true`` record in the reviewed labels file.  The export keeps the
reference, every observed turn, programs, canvases, and render receipts in a
portable artifact tree; failed final episodes are reported as excluded.

The curriculum is ordered by observed episode length, not by token count:

* short: 1--2 observed turns
* medium: 3--5 observed turns
* long: 6--12 observed turns

These are maximums.  A long episode is not made better by padding it, and a
short episode is not accepted without a valid final canvas.  Use a reviewed
labels JSONL file such as::

    {"job_id":"...", "admitted":true, "quality_score":4, "reason":"..."}

The labels file is intentionally required by default.  This prevents a
benchmark artifact from silently becoming training data before visual review.
For a ``turn_limit`` episode the label must additionally contain one of
``final_quality: true``, ``complete: true`` or ``finished: true``; a valid
canvas at the cap alone is not treated as a completed demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "painter.ai-gateway-teachers.curriculum.v1"
LABEL_SCHEMA = "painter.ai-gateway-teachers.review-labels.v1"
STAGES = (("short", 1, 2), ("medium", 3, 5), ("long", 6, 12))
ACCEPTED_STATUSES = frozenset(("complete", "turn_limit"))
SECRET_PATTERN = re.compile(r"(?:sk-or-v1-|sk-[A-Za-z0-9_-]{16,}|OPENROUTER_API_KEY|AI_GATEWAY_API_KEY)", re.I)


class CurriculumError(ValueError):
    """Raised for malformed or unsafe curriculum inputs."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CurriculumError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CurriculumError(f"invalid JSON file {path}: {exc}") from exc


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _relative(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.replace("\\", "/")


def _resolve_under(root: Path, value: Any, *, role: str) -> Path | None:
    relative = _relative(value)
    if relative is None:
        return None
    candidate = Path(relative)
    if candidate.is_absolute():
        raise CurriculumError(f"{role} path must be relative: {value!r}")
    result = (root / candidate).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise CurriculumError(f"{role} path escapes its root: {value!r}") from exc
    return result


def _check_no_secret(value: Any, *, location: str) -> None:
    """Reject credentials before copying provider responses into an export."""

    if isinstance(value, str):
        if SECRET_PATTERN.search(value):
            raise CurriculumError(f"possible credential in {location}; refusing to export it")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _check_no_secret(key, location=f"{location}.<key>")
            _check_no_secret(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _check_no_secret(child, location=f"{location}[{index}]")


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    episode_root = root / "episodes"
    if not episode_root.is_dir():
        raise CurriculumError(f"episode directory is missing: {episode_root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(episode_root.glob("*/episode.json")):
        row = _load_json(path)
        if not isinstance(row, dict):
            raise CurriculumError(f"episode is not an object: {path}")
        row = dict(row)
        row["_path"] = path
        rows.append(row)
    if not rows:
        raise CurriculumError(f"no episode receipts found under {episode_root}")
    return rows


def _load_reference_manifest(path: Path | None, *, required: bool = False) -> dict[str, dict[str, Any]]:
    if path is None:
        if required:
            raise CurriculumError("a separate training reference manifest is required")
        return {}
    value = _load_json(path)
    entries = value.get("references") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise CurriculumError(f"reference manifest has no references list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            result[entry["id"]] = entry
    return result


def _load_labels(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Load explicit review labels from JSONL or a JSON list/map."""

    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CurriculumError(f"invalid labels JSONL at line {line_number}: {exc}") from exc
        parsed = rows

    if isinstance(parsed, dict) and isinstance(parsed.get("labels"), list):
        parsed = parsed["labels"]
    elif isinstance(parsed, dict):
        # A map keyed by job_id is accepted for convenient manual annotation.
        parsed = [dict(value, job_id=key) for key, value in parsed.items() if isinstance(value, dict)]
    if not isinstance(parsed, list):
        raise CurriculumError("labels must be a JSON list, JSONL, or an object with a labels list")

    labels: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(parsed, start=1):
        if not isinstance(value, dict):
            raise CurriculumError(f"label {index} is not an object")
        job_id = value.get("job_id") or value.get("episode_id")
        if not isinstance(job_id, str) or not job_id:
            raise CurriculumError(f"label {index} has no job_id")
        if job_id in labels:
            raise CurriculumError(f"duplicate review label for {job_id}")
        labels[job_id] = dict(value)
    return labels, _sha(path)


def _status_failure(row: Mapping[str, Any]) -> str | None:
    status = str(row.get("status") or "unknown")
    if status not in ACCEPTED_STATUSES:
        return f"episode_status:{status}"
    if row.get("last_turn_invalid") is True:
        return "episode_ended_with_invalid_turn"
    if str(row.get("status")) == "turn_limit":
        review = row.get("_review_label")
        if not isinstance(review, Mapping) or not any(review.get(key) is True for key in ("final_quality", "complete", "finished")):
            return "turn_limit_requires_reviewed_final_quality"
    if not row.get("final_valid_canvas"):
        return "missing_final_valid_canvas"
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        return "missing_turn_history"
    final_value = row.get("final_valid_canvas")
    matches = [
        turn for turn in turns
        if isinstance(turn, dict)
        and turn.get("canvas") == final_value
        and isinstance(turn.get("render"), dict)
        and turn["render"].get("valid") is True
    ]
    if not matches:
        return "final_canvas_not_backed_by_valid_turn"
    # episode.json lives at ROOT/episodes/JOB/episode.json.
    source_root = Path(row["_path"]).parent.parent.parent
    final_canvas = _resolve_under(source_root, row.get("final_valid_canvas"), role="final canvas")
    if final_canvas is None or not final_canvas.is_file() or final_canvas.is_symlink():
        return "final_canvas_missing"
    return None


def _observed_turns(row: Mapping[str, Any]) -> int:
    turns = row.get("turns")
    if not isinstance(turns, list):
        return 0
    return len(turns)


def _stage(turns: int) -> str | None:
    for name, lower, upper in STAGES:
        if lower <= turns <= upper:
            return name
    return None


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned[:120] or "episode"


def _copy_file(source: Path, target: Path, *, root: Path, role: str) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise CurriculumError(f"{role} is missing or symlinked: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    digest = _sha(source)
    if _sha(target) != digest:
        raise CurriculumError(f"hash changed while copying {role}: {source}")
    return {"path": target.relative_to(root).as_posix(), "sha256": digest, "bytes": target.stat().st_size}


def _copy_episode_artifacts(
    row: Mapping[str, Any], source_root: Path, reference_root: Path, output: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Copy the source receipt and all observed artifacts, preserving evidence."""

    episode_path = Path(row["_path"])
    episode_dir = episode_path.parent
    target_dir = output / "episodes" / _safe_name(str(row.get("job_id") or episode_dir.name))
    copied: list[str] = []

    # Scan every artifact that could be copied before creating the output
    # directory or writing even one byte.  A provider response or program
    # containing a credential must fail closed without leaving a partial
    # curriculum behind.
    artifact_sources: list[Path] = []
    for source in sorted(episode_dir.iterdir()):
        if not source.is_file() or source.is_symlink() or source.name == "episode.json":
            continue
        if not re.fullmatch(r"turn-\d+\.(?:json|js|png|jpe?g)", source.name):
            continue
        artifact_sources.append(source)
        if source.suffix.lower() == ".json":
            _check_no_secret(_load_json(source), location=f"artifact:{source}")
        else:
            try:
                sample = source.read_bytes()
            except OSError as exc:
                raise CurriculumError(f"cannot read episode artifact: {source}") from exc
            if SECRET_PATTERN.search(sample.decode("utf-8", errors="ignore")):
                raise CurriculumError(f"possible credential in artifact:{source}; refusing to export it")

    source_episode = _copy_file(episode_path, target_dir / "episode.json", root=output, role="episode receipt")
    copied.append(source_episode["path"])

    # Preserve only the known episode evidence file types.  In particular,
    # never copy an arbitrary node-private file from a result directory.
    for source in artifact_sources:
        copied_file = _copy_file(source, target_dir / source.name, root=output, role=f"episode artifact {source.name}")
        copied.append(copied_file["path"])

    reference_value = _relative(row.get("reference_image"))
    reference_source = _resolve_under(reference_root, reference_value, role="reference") if reference_value else None
    if reference_source is None or not reference_source.is_file():
        raise CurriculumError(f"reference image is missing for {row.get('job_id')}: {reference_value}")
    reference_target = output / "references" / f"{_safe_name(str(row.get('reference_id') or 'reference'))}{reference_source.suffix.lower() or '.img'}"
    reference_info = _copy_file(reference_source, reference_target, root=output, role="reference image")
    declared_reference_sha = row.get("reference_sha256")
    if not isinstance(declared_reference_sha, str) or reference_info["sha256"] != declared_reference_sha:
        raise CurriculumError(f"reference hash mismatch for {row.get('job_id')}")
    copied.append(reference_info["path"])

    return {
        "source_episode": episode_path.relative_to(source_root).as_posix(),
        "source_reference": reference_source.relative_to(reference_root).as_posix(),
        "reference": reference_info,
        "episode_receipt": source_episode,
        "artifacts": sorted(copied),
    }, copied


def _rewrite_record(row: Mapping[str, Any], labels: Mapping[str, Any], stage: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create an auditable JSONL record without changing the teacher actions."""

    # Keep the original receipt in the episode artifact.  This compact index
    # repeats the full turn data so downstream SFT tooling can stream JSONL
    # without inventing supervision or re-reading the receipt.
    _check_no_secret(row, location=f"episode:{row.get('job_id')}")
    _check_no_secret(labels, location=f"review:{row.get('job_id')}")
    return {
        "schema": SCHEMA,
        "job_id": row.get("job_id"),
        "teacher_model": row.get("model"),
        "reference_id": row.get("reference_id"),
        "track": (row.get("settings") or {}).get("track") or row.get("track"),
        "stage": stage,
        "observed_turns": _observed_turns(row),
        "status": row.get("status"),
        "reference": evidence["reference"],
        "episode_receipt": evidence["episode_receipt"],
        "artifacts": evidence["artifacts"],
        "turns": row.get("turns"),
        "first_valid_canvas": row.get("first_valid_canvas"),
        "final_valid_canvas": row.get("final_valid_canvas"),
        "review": dict(labels),
        "source": {
            "source_episode": evidence["source_episode"],
            "reference_sha256": row.get("reference_sha256"),
            "episode_sha256": _sha(Path(row["_path"])),
        },
    }


def _validate_turn_artifacts(row: Mapping[str, Any], source_root: Path) -> None:
    """Check every path/hash referenced by a retained turn before copying."""

    turns = row.get("turns")
    if not isinstance(turns, list):
        return
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, Mapping):
            raise CurriculumError(f"turn {index} in {row.get('job_id')} is not an object")
        for key in ("canvas", "current_canvas_after", "program"):
            value = turn.get(key)
            if not value:
                continue
            path = _resolve_under(source_root, value, role=f"turn {index} {key}")
            if path is None or not path.is_file() or path.is_symlink():
                raise CurriculumError(f"turn {index} {key} is missing for {row.get('job_id')}")
            if key in ("canvas", "current_canvas_after"):
                declared = []
                for container in (turn, turn.get("render") if isinstance(turn.get("render"), Mapping) else {}):
                    for hash_key in ("canvas_sha256", "current_canvas_sha256", "png_sha256"):
                        if isinstance(container.get(hash_key), str):
                            declared.append(container[hash_key])
                if declared and _sha(path) not in declared:
                    raise CurriculumError(f"turn {index} canvas hash mismatch for {row.get('job_id')}")
        render = turn.get("render")
        receipt_value = render.get("receipt") if isinstance(render, Mapping) else None
        if isinstance(receipt_value, str):
            receipt = _resolve_under(source_root, receipt_value, role=f"turn {index} receipt")
            if receipt is None or not receipt.is_file() or receipt.is_symlink():
                raise CurriculumError(f"turn {index} receipt is missing for {row.get('job_id')}")


def build_curriculum(
    source_root: Path,
    output: Path,
    *,
    model: str,
    labels_path: Path,
    reference_manifest: Path | None = None,
    holdout_manifest: Path | None = None,
    reference_root: Path | None = None,
    split: str | None = None,
    track: str | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.resolve()
    if output == source_root or source_root in output.parents:
        raise CurriculumError("output must not be the source results directory or inside it")
    if not model:
        raise CurriculumError("selected teacher model is required")
    labels, labels_sha = _load_labels(labels_path.resolve())
    manifest_entries = _load_reference_manifest(reference_manifest.resolve() if reference_manifest else None, required=True)
    holdout_entries = _load_reference_manifest(holdout_manifest.resolve() if holdout_manifest else None, required=True)
    holdout_hashes = {
        str(entry.get("sha256")) for entry in holdout_entries.values()
        if isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64
    }
    resolved_reference_root = (reference_root or source_root).resolve()
    rows = _episode_rows(source_root)
    output.mkdir(parents=True, exist_ok=False)
    accepted: dict[str, int] = {name: 0 for name, _, _ in STAGES}
    excluded: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    copied: list[str] = []
    seen_jobs: set[str] = set()

    for row in rows:
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            excluded.append({"job_id": job_id, "reason": "missing_job_id"})
            continue
        if job_id in seen_jobs:
            raise CurriculumError(f"duplicate episode job_id: {job_id}")
        seen_jobs.add(job_id)
        row_track = (row.get("settings") or {}).get("track") or row.get("track")
        if row.get("model") != model:
            continue
        if track is not None and row_track != track:
            continue
        label = labels.get(job_id)
        if label is None:
            excluded.append({"job_id": job_id, "reason": "missing_review_label"})
            continue
        if label.get("admitted") is not True:
            excluded.append({"job_id": job_id, "reason": "review_not_admitted", "label": label})
            continue
        row["_review_label"] = label
        # Sanitize the parsed episode and reviewed label before any artifact
        # copy.  The receipt itself is one of the exported artifacts.
        _check_no_secret(row, location=f"episode:{job_id}")
        _check_no_secret(label, location=f"review:{job_id}")
        failure = _status_failure(row)
        if failure:
            excluded.append({"job_id": job_id, "reason": failure})
            continue
        turn_count = _observed_turns(row)
        stage = _stage(turn_count)
        if stage is None:
            excluded.append({"job_id": job_id, "reason": f"observed_turns_outside_1_to_12:{turn_count}"})
            continue
        reference_id = row.get("reference_id")
        manifest_entry = manifest_entries.get(reference_id) if reference_id else None
        if manifest_entry is None:
            excluded.append({"job_id": job_id, "reason": "reference_missing_training_manifest"})
            continue
        if split is not None:
            source_split = manifest_entry.get("source_split")
            declared_split = manifest_entry.get("split")
            if (source_split is not None and declared_split is not None and source_split != declared_split):
                raise CurriculumError(f"conflicting reference split for {reference_id}")
            if (source_split if source_split is not None else declared_split) != split:
                excluded.append({"job_id": job_id, "reason": "reference_split_mismatch"})
                continue
        expected = manifest_entry.get("sha256")
        actual = row.get("reference_sha256")
        if not isinstance(expected, str) or expected != actual:
            raise CurriculumError(f"reference hash mismatch for {job_id}")
        reference_sha = row.get("reference_sha256")
        if reference_sha in holdout_hashes:
            raise CurriculumError(
                f"reference {reference_id!r} for {job_id} overlaps the evaluation holdout; "
                "generate curriculum demonstrations from separate training references"
            )
        _validate_turn_artifacts(row, source_root)
        evidence, files = _copy_episode_artifacts(row, source_root, resolved_reference_root, output)
        record = _rewrite_record(row, label, stage, evidence)
        records.append(record)
        copied.extend(files)
        accepted[stage] += 1

    if not records:
        # Remove the empty output only if it contains nothing; this keeps a
        # failed invocation from looking like an empty training curriculum.
        try:
            output.rmdir()
        except OSError:
            pass
        raise CurriculumError("no reviewed, valid episodes matched the selected teacher")

    for stage, _, _ in STAGES:
        stage_records = [record for record in records if record["stage"] == stage]
        stage_records.sort(key=lambda record: (int(record["observed_turns"]), str(record["reference_id"]), str(record["job_id"])))
        stage_path = output / "stages" / f"{stage}.jsonl"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        with stage_path.open("w", encoding="utf-8") as stream:
            for record in stage_records:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "schema": SCHEMA,
        "selected_teacher_model": model,
        "source_root": str(source_root),
        "reference_manifest": str(reference_manifest.resolve()) if reference_manifest else None,
        "holdout_manifest": str(holdout_manifest.resolve()) if holdout_manifest else None,
        "reference_root": str(resolved_reference_root),
        "reference_split": split,
        "track": track,
        "labels": {"path": str(labels_path.resolve()), "sha256": labels_sha, "count": len(labels)},
        "stages": {name: {"turn_min": lower, "turn_max": upper, "accepted": accepted[name]} for name, lower, upper in STAGES},
        "accepted_total": len(records),
        "excluded_total": len(excluded),
        "excluded": excluded,
        "records_sha256": _json_sha([{"job_id": row["job_id"], "stage": row["stage"], "episode_sha256": row["source"]["episode_sha256"]} for row in records]),
        "copied_file_count": len(set(copied)),
        "admission_rule": "explicit admitted:true review label AND valid final canvas AND complete/turn_limit status AND reference absent from evaluation holdout",
        "stage_rule": "observed episode turns: short=1-2, medium=3-5, long=6-12",
    }
    _check_no_secret(summary, location="curriculum summary")
    (output / "curriculum-manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "excluded.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in excluded) + ("\n" if excluded else ""), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="collected benchmark results directory containing episodes/")
    parser.add_argument("--output", type=Path, required=True, help="new curriculum export directory")
    parser.add_argument("--model", required=True, help="exact benchmark model ID selected as teacher")
    parser.add_argument("--labels", type=Path, required=True, help="reviewed JSONL/JSON admission labels")
    parser.add_argument("--reference-manifest", type=Path, required=True, help="separate training reference manifest")
    parser.add_argument("--holdout-manifest", type=Path, required=True, help="evaluation refs.json; overlapping references are rejected")
    parser.add_argument("--reference-root", type=Path, help="root containing the references/ path from episode receipts")
    parser.add_argument("--split", help="only admit references whose manifest source_split matches this value")
    parser.add_argument("--track", choices=("quality", "speed"), help="only admit episodes from one track")
    args = parser.parse_args(argv)
    try:
        summary = build_curriculum(
            args.root, args.output, model=args.model, labels_path=args.labels,
            reference_manifest=args.reference_manifest, holdout_manifest=args.holdout_manifest,
            reference_root=args.reference_root,
            split=args.split, track=args.track,
        )
    except CurriculumError as exc:
        parser.exit(2, f"curriculum export failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "accepted_total": summary["accepted_total"], "stages": summary["stages"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
