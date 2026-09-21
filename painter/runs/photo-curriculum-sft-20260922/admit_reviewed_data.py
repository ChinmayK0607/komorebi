#!/usr/bin/env python3
"""Admit only reviewed, hash-bound real-photo rows to the SFT corpus.

The processor audit is run before visual review.  This command keeps the
audited JSONL row bytes, verifies the audit input hashes, and removes only
unreviewed or weak real-photo rows.  Foundation, retention, and validation
rows are copied byte-for-byte.  A review label is never treated as evidence
unless the final review manifest also binds the same program, reference, and
render hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


REAL_KINDS = {"real_photo_teacher", "real_photo", "photo_teacher"}
ACCEPTED_LABELS = {"keep"}
REJECTED_WORDS = {"weak", "reject", "rejected", "fail", "failed", "no", "uncertain"}
HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def accepted_label(value: Any) -> bool:
    """Accept an explicit keep/strong decision; never infer one from a score."""
    text = norm(value)
    if not text or any(word in text.split() for word in REJECTED_WORDS):
        return False
    return text in ACCEPTED_LABELS


def _first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _flatten_records(raw: Any, inherited_key: str | None = None) -> list[dict[str, Any]]:
    """Read both Astra's id->review object and Sol's review list."""
    if isinstance(raw, list):
        result: list[dict[str, Any]] = []
        for item in raw:
            result.extend(_flatten_records(item))
        return result
    if not isinstance(raw, dict):
        return ([{"id": inherited_key, "decision": raw}] if inherited_key else [])

    direct_id = _first(raw, "id", "record_id", "sample_id", "source_image_id", "image_id")
    direct_decision = _first(raw, "decision", "label", "verdict", "quality", "status", "result")
    if direct_id is not None and direct_decision is not None:
        return [dict(raw, id=direct_id, decision=direct_decision)]
    if inherited_key is not None and direct_decision is not None:
        return [dict(raw, id=inherited_key, decision=direct_decision)]

    result: list[dict[str, Any]] = []
    for key in ("reviews", "labels", "decisions", "items", "records", "results"):
        if key in raw:
            result.extend(_flatten_records(raw[key]))
    # Astra's schema is commonly {"reviews": {"000000000001": "strong"}};
    # accept an id-keyed object even when it is nested under another key.
    if not result:
        for key, value in raw.items():
            if key in {"schema", "generated_by", "metadata", "notes"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                result.append({"id": key, "decision": value})
            elif isinstance(value, dict):
                result.extend(_flatten_records(value, inherited_key=key))
    return result


def _id_aliases(value: Any) -> set[str]:
    text = str(value or "")
    aliases = {text} if text else set()
    match = re.search(r"(?<!\d)(\d{12})(?!\d)", text)
    if match:
        aliases.add(match.group(1))
        aliases.add("coco128-" + match.group(1))
    if text.startswith("coco128-"):
        aliases.add(text.removeprefix("coco128-"))
    return aliases


def _label_hashes(row: dict[str, Any]) -> dict[str, str] | None:
    def find(keys: set[str]) -> str | None:
        values = _nested_values(row, keys)
        return next((value for value in values if HEX64.fullmatch(value)), None)
    values = {
        "program": find({"program", "program_sha256", "program_hash", "teacher_program_sha256", "teacher_source_sha256", "source_sha256"}),
        "reference": find({"reference", "reference_sha256", "reference_hash", "image_sha256", "source_image_sha256"}),
        "render": find({"render", "render_sha256", "render_hash", "png_sha256", "output_sha256", "image_render_sha256"}),
    }
    return values if all(values.values()) else None


def load_decisions(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    decisions: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        for row in _flatten_records(json_read(path)):
            decision = _first(row, "decision", "label", "verdict", "quality", "status", "result")
            ids = set()
            for key in ("id", "record_id", "sample_id", "source_image_id", "image_id", "record_key"):
                ids.update(_id_aliases(row.get(key)))
            if not ids or decision is None:
                continue
            result = {"accepted": accepted_label(decision), "hashes": _label_hashes(row)}
            for ident in ids:
                decisions.setdefault(ident, []).append(result)
    return decisions


def manifest_records(path: Path) -> list[dict[str, Any]]:
    raw = json_read(path)
    records = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected records array")
    return [record for record in records if isinstance(record, dict)]


def manifest_hashes(record: dict[str, Any]) -> dict[str, str] | None:
    reference = record.get("reference") or {}
    teacher = record.get("teacher_source") or {}
    render = record.get("render") or {}
    values = {
        "program": _first(teacher, "sha256") or record.get("source_sha256"),
        "reference": _first(reference, "actual_sha256", "expected_sha256"),
        "render": _first(render, "sha256") or record.get("render_sha256"),
    }
    if any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in values.values()):
        return None
    # A complete record must prove the expected reference, receipt source, and
    # receipt PNG all agree with the copied evidence.
    if reference.get("status") != "valid" or render.get("status") != "valid":
        return None
    if reference.get("expected_sha256") != reference.get("actual_sha256"):
        return None
    receipt = render.get("receipt") or {}
    if receipt and (receipt.get("source_sha256") != values["program"] or receipt.get("png_sha256") != values["render"]):
        return None
    if record.get("pair_status") not in {None, "complete"}:
        return None
    return values


def review_index(records: list[dict[str, Any]], decisions: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        hashes = manifest_hashes(record)
        if hashes is None:
            continue
        aliases: set[str] = set()
        for key in ("id", "record_key", "source_image_id", "source_id"):
            aliases.update(_id_aliases(record.get(key)))
        labels_by_value = {
            (label["accepted"], tuple(sorted((label["hashes"] or {}).items())))
            : label
            for alias in aliases for label in decisions.get(alias, [])
        }
        labels = list(labels_by_value.values())
        # Ambiguous labels, missing label hashes, and stale labels are not
        # admission evidence.  The decision must bind all three current hashes.
        if len(labels) != 1 or not labels[0]["accepted"] or labels[0]["hashes"] != hashes:
            continue
        entry = dict(record, _hashes=hashes)
        for alias in aliases:
            index[alias] = entry
    return index


def _nested_values(mapping: Any, keys: set[str]) -> list[str]:
    if not isinstance(mapping, dict):
        return []
    values: list[str] = []
    for key, value in mapping.items():
        if key.lower() in keys and isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            values.extend(_nested_values(value, keys))
    return values


def candidate_hashes(row: dict[str, Any]) -> dict[str, set[str]]:
    metadata = row.get("teacher_metadata") or {}
    return {
        "program": {str(value) for value in (row.get("program_sha256"), row.get("source_sha256")) if value}
        | set(_nested_values(metadata, {"program_sha256", "teacher_program_sha256", "source_sha256"})),
        "reference": {str(row.get("reference_sha256"))} | set(_nested_values(metadata, {"reference_sha256", "image_sha256", "source_image_sha256"})),
        "render": {str(value) for value in (row.get("render_sha256"), row.get("png_sha256")) if value}
        | set(_nested_values(metadata, {"render_sha256", "png_sha256", "output_sha256", "image_render_sha256"})),
    }


def candidate_aliases(row: dict[str, Any]) -> set[str]:
    metadata = row.get("teacher_metadata") or {}
    aliases: set[str] = set()
    for value in (row.get("id"), row.get("source_group"), metadata.get("id"), metadata.get("assignment_id"), metadata.get("source_image_id"), metadata.get("reference")):
        aliases.update(_id_aliases(value))
    return aliases


def is_real(row: dict[str, Any]) -> bool:
    return row.get("source_kind") in REAL_KINDS or str(row.get("id", "")).startswith("real-")


def row_matches_review(row: dict[str, Any], reviewed: dict[str, Any]) -> bool:
    if not (candidate_aliases(row) & set().union(*(_id_aliases(reviewed.get(key)) for key in ("id", "record_key", "source_image_id", "source_id")))):
        return False
    wanted = reviewed["_hashes"]
    got = candidate_hashes(row)
    # The pre-admission JSONL records the canonical teacher program and
    # reference hashes.  Render bytes are bound by the final review manifest
    # and label hashes above; teacher metadata may carry stale QC render fields
    # from an earlier wrapper, so it is deliberately not used here.
    return all(wanted[key] in got[key] for key in ("program", "reference"))


def read_lines(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]


def row_from_line(line: bytes) -> dict[str, Any]:
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict) or not value.get("id"):
        raise ValueError("candidate JSONL line must be an object with id")
    return value


def audit_hash(audit: dict[str, Any], split: str) -> str | None:
    value = audit.get("dataset_sha256")
    if isinstance(value, dict):
        value = value.get(split)
    if value is None:
        value = audit.get(f"{split}_sha256")
    return value if isinstance(value, str) and HEX64.fullmatch(value) else None


def admit(*, train: Path, validation: Path, audit: Path, review_manifest: Path,
          labels: list[Path], output: Path) -> dict[str, Any]:
    audit_data = json_read(audit)
    if not isinstance(audit_data, dict):
        raise ValueError("pre-admission audit must be an object")
    train_lines, validation_lines = read_lines(train), read_lines(validation)
    for split, path, lines in (("train", train, train_lines), ("validation", validation, validation_lines)):
        expected = audit_hash(audit_data, split)
        actual = sha256_file(path)
        if expected is None or expected != actual:
            raise ValueError(f"{split} candidate SHA-256 does not match pre-admission audit")
    decisions = load_decisions(labels)
    reviewed = review_index(manifest_records(review_manifest), decisions)
    kept_train: list[bytes] = []
    kept_real, rejected_real = [], []
    for line in train_lines:
        row = row_from_line(line)
        if not is_real(row):
            kept_train.append(line)
            continue
        match = next((reviewed[alias] for alias in candidate_aliases(row) if alias in reviewed and row_matches_review(row, reviewed[alias])), None)
        if match is None:
            rejected_real.append(row.get("id"))
        else:
            kept_train.append(line)
            kept_real.append(row.get("id"))
    output.mkdir(parents=True, exist_ok=True)
    train_out, validation_out = output / "candidates-train.jsonl", output / "candidates-validation.jsonl"
    train_out.write_bytes(b"".join(kept_train))
    validation_out.write_bytes(b"".join(validation_lines))
    kept_ids = {row_from_line(line).get("id") for line in kept_train + validation_lines}
    examples = [row for row in audit_data.get("examples", []) if isinstance(row, dict) and row.get("id") in kept_ids]
    expected_examples = len(kept_train) + len(validation_lines)
    if len(examples) != expected_examples:
        raise ValueError(f"pre-admission audit examples cover {len(examples)} rows, expected {expected_examples}")
    audit_out = dict(audit_data)
    audit_out.update({
        "status": "passed",
        "pre_admission_audit_sha256": sha256_file(audit),
        "dataset_sha256": {"train": sha256_file(train_out), "validation": sha256_file(validation_out)},
        "train_rows": len(kept_train), "validation_rows": len(validation_lines),
        "examples": examples,
        "admission": {"review_manifest_sha256": sha256_file(review_manifest),
                      "label_file_sha256": {str(path): sha256_file(path) for path in labels},
                      "kept_real_photo_ids": kept_real, "rejected_real_photo_ids": rejected_real,
                      "reviewed_manifest_records": len(reviewed)},
    })
    (output / "token-audit.json").write_text(json.dumps(audit_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"train_rows": len(kept_train), "validation_rows": len(validation_lines), "kept_real": len(kept_real), "rejected_real": len(rejected_real), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-train", "--train-candidates", dest="train", type=Path, required=True)
    parser.add_argument("--candidates-validation", "--validation-candidates", dest="validation", type=Path, required=True)
    parser.add_argument("--pre-admission-audit", "--audit", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--label", "--labels", dest="labels", type=Path, action="append", required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(train=args.train, validation=args.validation, audit=args.pre_admission_audit,
                           review_manifest=args.review_manifest, labels=args.labels, output=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
