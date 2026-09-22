"""Small, protocol-bound file queue for aesthetic Luna comparisons.

The queue is intentionally separate from the live revision queue.  A request
key includes both the aesthetic protocol version and the three source image
hashes, so an old judge receipt can never satisfy a new aesthetic request.
Exported comparisons contain only neutral ``reference``, ``A``, and ``B``
assets.  The private mapping records which displayed side was the original
candidate and is required when importing the two orientation labels.

No model endpoint is assumed here.  A worker or orchestrator reads the staged
``judge-*.json`` file, compares the actual image files, and writes a
``labels-*.json`` file with the schema from ``aesthetic_judge.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any, Iterable

from aesthetic_judge import (
    JUDGE_MODEL,
    JUDGMENT_FIELDS,
    PROTOCOL_VERSION,
    REASONING_EFFORT,
    reconcile,
    validate_judgment,
    valid_normalized_judgment,
)


SOURCE_ROLES = ("reference", "anchor", "candidate")
DISPLAY_ROLES = ("reference", "A", "B")
HEX64 = frozenset("0123456789abcdef")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def request_key(image_sha256: dict[str, str], protocol_version: str = PROTOCOL_VERSION) -> str:
    """Return the content key for one new-protocol image triple."""

    _validate_hashes(image_sha256, SOURCE_ROLES)
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("protocol_version must be a non-empty string")
    return hashlib.sha256(_canonical({
        "protocol_version": protocol_version,
        "image_sha256": {role: image_sha256[role] for role in SOURCE_ROLES},
    })).hexdigest()


def _validate_hashes(hashes: Any, roles: Iterable[str]) -> None:
    roles = tuple(roles)
    if not isinstance(hashes, dict) or set(hashes) != set(roles):
        raise ValueError(f"image_sha256 must contain exactly {roles}")
    for role in roles:
        value = hashes[role]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in HEX64 for char in value)):
            raise ValueError(f"invalid sha256 for {role}")


def _root_for_request(path: Path) -> Path:
    # <root>/judge-queue/<request-id>/request.json
    return path.resolve().parents[2]


def _resolve_image(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("request image path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise ValueError(f"request image is a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"request image is not a regular file: {path}")
    return path


def validate_request(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Validate request schema, protocol binding, and immutable image bytes."""

    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    protocol = request.get("protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError("aesthetic protocol version mismatch")
    if set(request.get("images", {})) != set(SOURCE_ROLES):
        raise ValueError(f"images must contain exactly {SOURCE_ROLES}")
    hashes = request.get("image_sha256")
    _validate_hashes(hashes, SOURCE_ROLES)
    expected = request_key(hashes, protocol)
    if request.get("id") != expected:
        raise ValueError("request id does not match protocol and image hashes")
    for role in SOURCE_ROLES:
        path = _resolve_image(root, request["images"][role])
        if sha(path) != hashes[role]:
            raise ValueError(f"request image changed: {role}")
    if "created_at" in request:
        value = request["created_at"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("created_at must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError("created_at must be finite")
    return request


def create_or_verify_request(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Install one request once, accepting only an exact new-protocol cache hit."""

    path = Path(path)
    expected = dict(value)
    root = _root_for_request(path)
    validate_request(root, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n")
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise ValueError("existing request is not a regular file")
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError("existing request is invalid JSON") from exc
            validate_request(root, existing)
            immutable = ("id", "protocol_version", "image_sha256")
            if any(existing.get(field) != expected.get(field) for field in immutable):
                raise ValueError("existing request conflicts with content-addressed request")
            # Existing paths are retained, but their bytes were just checked by
            # validate_request; this prevents a stale path from becoming a cache hit.
            return existing
        return expected
    finally:
        temporary.unlink(missing_ok=True)


def _request_sort_key(root: Path, path: Path, request: dict[str, Any]) -> tuple[float, str, str]:
    value = request.get("created_at", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid created_at in request: {path}")
    if not math.isfinite(float(value)):
        raise ValueError(f"invalid created_at in request: {path}")
    return float(value), str(request.get("id", "")), str(path.relative_to(root))


def _load_request(path: Path) -> dict[str, Any]:
    root = _root_for_request(path)
    request = json.loads(path.read_text())
    return validate_request(root, request)


def _copy_verified(source: Path, destination: Path, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or sha(destination) != digest:
            raise ValueError(f"existing neutral asset differs: {destination}")
        return
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    try:
        if sha(temporary) != digest:
            raise ValueError(f"copied asset hash mismatch: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def export(root: Path, batch: str, limit: int = 24) -> dict[str, Any]:
    """Export pending requests into neutral, hash-bound orientation pairs."""

    root = Path(root).resolve()
    if not isinstance(batch, str) or not batch or "/" in batch or batch in {".", ".."}:
        raise ValueError("simple batch name required")
    if limit < 1:
        raise ValueError("limit must be positive")
    destination = root / "reward-batches" / batch
    if destination.exists():
        raise ValueError(f"batch already exists: {destination}")
    pending: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "judge-queue").glob("*/request.json")):
        if path.with_name("assigned.json").exists() or path.with_name("receipt.json").exists():
            continue
        request = _load_request(path)
        pending.append((path, request))
    pending.sort(key=lambda item: _request_sort_key(root, *item))
    pending = pending[:limit]
    if not pending:
        return {"batch": batch, "requests": 0, "pairs": 0}

    destination.mkdir(parents=True)
    pairs: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for request_path, request in pending:
        source = {
            role: _resolve_image(root, request["images"][role])
            for role in SOURCE_ROLES
        }
        for candidate_side in ("A", "B"):
            # Do not encode the orientation in the public ID.  Candidate-side
            # mapping remains private-key data only, so a worker cannot infer
            # which display side is the policy candidate from a filename.
            pair_id = hashlib.sha256(_canonical({
                "protocol_version": PROTOCOL_VERSION,
                "request_hash": request["id"],
                "orientation": candidate_side,
            })).hexdigest()
            folder = destination / pair_id
            folder.mkdir()
            source_for_display = {
                "reference": source["reference"],
                "A": source["candidate"] if candidate_side == "A" else source["anchor"],
                "B": source["anchor"] if candidate_side == "A" else source["candidate"],
            }
            display_hashes = {
                role: request["image_sha256"][
                    "reference" if role == "reference" else
                    ("candidate" if role == candidate_side else "anchor")
                ]
                for role in DISPLAY_ROLES
            }
            for role in DISPLAY_ROLES:
                _copy_verified(source_for_display[role], folder / f"{display_hashes[role]}.png", display_hashes[role])
            pair = {
                "id": pair_id,
                "reference": f"{pair_id}/{display_hashes['reference']}.png",
                "A": f"{pair_id}/{display_hashes['A']}.png",
                "B": f"{pair_id}/{display_hashes['B']}.png",
                "image_sha256": display_hashes,
                "protocol_version": PROTOCOL_VERSION,
            }
            pairs.append(pair)
            private.append({
                "id": pair_id,
                "request_id": request["id"],
                "request_hash": request["id"],
                "protocol_version": PROTOCOL_VERSION,
                "candidate_side": candidate_side,
                "image_sha256": display_hashes,
                "source_image_sha256": request["image_sha256"],
            })
        write(request_path.with_name("assigned.json"), {
            "batch": batch,
            "protocol_version": PROTOCOL_VERSION,
            "request_hash": request["id"],
        })

    write(destination / "pairs.json", pairs)
    write(destination / "private-key.json", private)
    # A descriptive alias makes the neutral-to-source mapping explicit for
    # importers while private-key.json remains compatible with the older queue.
    write(destination / "mapping.json", private)
    archive = destination.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(destination, arcname=batch)
    receipt = {
        "batch": batch,
        "protocol_version": PROTOCOL_VERSION,
        "archive_sha256": sha(archive),
        "requests": len(pending),
        "pairs": len(pairs),
    }
    write(archive.with_suffix(".receipt.json"), receipt)
    return receipt


def stage(root: Path, batch: str) -> dict[str, Any]:
    """Write one public judge input file per neutral comparison."""

    root = Path(root).resolve()
    destination = root / "reward-batches" / batch
    pairs = json.loads((destination / "pairs.json").read_text())
    private = {row["id"]: row for row in json.loads((destination / "private-key.json").read_text())}
    if len(private) != len(pairs):
        raise ValueError("pair mapping is incomplete or duplicated")
    review: list[dict[str, Any]] = []
    for pair in pairs:
        if set(pair) != {"id", "reference", "A", "B", "image_sha256", "protocol_version"}:
            raise ValueError(f"invalid neutral pair fields: {pair.get('id')}")
        if pair["protocol_version"] != PROTOCOL_VERSION or pair["id"] not in private:
            raise ValueError("neutral pair protocol/mapping mismatch")
        mapping = private[pair["id"]]
        if mapping["protocol_version"] != PROTOCOL_VERSION or mapping["image_sha256"] != pair["image_sha256"]:
            raise ValueError("neutral pair hash mapping mismatch")
        row = {role: str((destination / pair[role]).resolve()) for role in DISPLAY_ROLES}
        for role in DISPLAY_ROLES:
            path = Path(row[role])
            if not path.is_file() or sha(path) != pair["image_sha256"][role]:
                raise ValueError(f"neutral asset changed: {pair['id']}:{role}")
        review.append({"id": pair["id"], **row, "image_sha256": pair["image_sha256"],
                       "protocol_version": PROTOCOL_VERSION})
    write(destination / "review-input.json", review)
    # Every file is one real comparison.  This is the handoff unit for a
    # separately orchestrated Luna-low worker.
    for row in review:
        write(destination / f"judge-{row['id']}.json", {
            "protocol_version": PROTOCOL_VERSION,
            "judge_model": JUDGE_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "comparisons": [row],
        })
    # Keep the aggregate manifest outside the ``judge-*.json`` glob: each file
    # in that glob is one actual comparison handed to one Luna-low worker.
    write(destination / "view-manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "pairs": [row["id"] for row in review],
    })
    return {"batch": batch, "pairs": len(review), "protocol_version": PROTOCOL_VERSION}


def _label_documents(destination: Path, pair_ids: set[str]) -> dict[str, tuple[dict[str, Any], Path]]:
    labels: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(destination.glob("labels-*.json")):
        document = json.loads(path.read_text())
        metadata = {"protocol_version", "judge_model", "reasoning_effort"}
        if not isinstance(document, dict) or not metadata.issubset(document):
            raise ValueError(f"invalid label document fields: {path}")
        if (document["protocol_version"] != PROTOCOL_VERSION
                or document["judge_model"] != JUDGE_MODEL
                or document["reasoning_effort"] != REASONING_EFFORT):
            raise ValueError(f"label document provenance mismatch: {path}")
        if "judgments" in document:
            if set(document) != metadata | {"judgments"} or not isinstance(document["judgments"], list):
                raise ValueError(f"invalid label document fields: {path}")
            rows = document["judgments"]
        elif "judgment" in document:
            # Also accept the one-view envelope used by the interactive Luna
            # handoff: protocol metadata surrounds the exact four-key label.
            allowed = metadata | {"judgment", "pair_id", "id", "comparison_id"}
            if not set(document).issubset(allowed):
                raise ValueError(f"invalid one-view label fields: {path}")
            pair_id = next((document.get(key) for key in ("pair_id", "id", "comparison_id")
                            if document.get(key) is not None), None)
            if not isinstance(pair_id, str) or not isinstance(document["judgment"], dict):
                raise ValueError(f"one-view label is missing pair_id: {path}")
            rows = [{"id": pair_id, **document["judgment"]}]
        else:
            raise ValueError(f"label document has no judgment: {path}")
        for row in rows:
            if not isinstance(row, dict) or set(row) != JUDGMENT_FIELDS | {"id"}:
                raise ValueError(f"label row must contain id plus exactly four judgment fields: {path}")
            pair_id = row["id"]
            if pair_id not in pair_ids:
                raise ValueError(f"unknown comparison id in labels: {pair_id}")
            if pair_id in labels:
                raise ValueError(f"duplicate comparison label: {pair_id}")
            labels[pair_id] = (validate_judgment({key: row[key] for key in JUDGMENT_FIELDS}), path)
    return labels


def receipts(root: Path, batch: str) -> dict[str, Any]:
    """Build provenance-bound receipts once both orientations are labeled."""

    root = Path(root).resolve()
    destination = root / "reward-batches" / batch
    pairs = {row["id"]: row for row in json.loads((destination / "pairs.json").read_text())}
    private = {row["id"]: row for row in json.loads((destination / "private-key.json").read_text())}
    if set(pairs) != set(private):
        raise ValueError("pair mapping mismatch")
    labels = _label_documents(destination, set(pairs))
    grouped: dict[str, dict[str, tuple[str, dict[str, Any], Path]]] = {}
    for pair_id, (label, source_file) in labels.items():
        mapping = private[pair_id]
        side = mapping["candidate_side"]
        request_id = mapping["request_id"]
        if side in grouped.setdefault(request_id, {}):
            raise ValueError(f"duplicate orientation for request: {request_id}")
        grouped[request_id][side] = (pair_id, label, source_file)

    request_count = 0
    ready_count = 0
    excluded_count = 0
    receipt_dir = destination / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    for request_id in sorted(grouped):
        request_count += 1
        views = grouped[request_id]
        if set(views) != {"A", "B"}:
            continue  # missing orientation remains retryable, never a zero
        first_id, first, first_file = views["A"]
        reverse_id, reverse, reverse_file = views["B"]
        first_mapping = private[first_id]
        reverse_mapping = private[reverse_id]
        normalized = reconcile(first, reverse, "A", "B")
        score = normalized["pairwise_win_fraction"]
        if score is None:
            excluded_count += 1
        result = {
            "id": request_id,
            "request_id": request_id,
            "request_hash": request_id,
            "protocol_version": PROTOCOL_VERSION,
            "image_sha256": first_mapping["source_image_sha256"],
            "judge_model": JUDGE_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "candidate_A": first,
            "candidate_B": reverse,
            "judgment": normalized,
            "pairwise_win_fraction": score,
            "reward": score,
            "status": "scored" if score is not None else "excluded",
            "provenance": {
                "protocol_version": PROTOCOL_VERSION,
                "request_hash": request_id,
                "image_sha256": first_mapping["source_image_sha256"],
                "orientations": [
                    {"pair_id": first_id, "candidate_side": "A", "label_file": str(first_file.name),
                     "image_sha256": first_mapping["image_sha256"]},
                    {"pair_id": reverse_id, "candidate_side": "B", "label_file": str(reverse_file.name),
                     "image_sha256": reverse_mapping["image_sha256"]},
                ],
            },
        }
        path = receipt_dir / request_id
        path.mkdir(exist_ok=True)
        write(path / "receipt.json", result)
        ready_count += 1
    summary = {
        "batch": batch,
        "protocol_version": PROTOCOL_VERSION,
        "labels": len(labels),
        "requests": request_count,
        "ready_requests": ready_count,
        "excluded_requests": excluded_count,
    }
    write(destination / "receipt-summary.json", summary)
    return summary


def _request_for_id(root: Path, request_id: str) -> dict[str, Any]:
    path = root / "judge-queue" / request_id / "request.json"
    if not path.is_file():
        raise ValueError(f"missing source request: {request_id}")
    request = json.loads(path.read_text())
    return validate_request(root, request)


def enqueue_comparison(
    root: Path,
    trace_id: str,
    reference: Path,
    anchor: Path,
    candidate: Path,
) -> str:
    """Create or reuse the new-protocol request used by the native env.

    ``trace_id`` is accepted for the taskset hook and retained as non-identity
    metadata when a request is first created.  It is intentionally absent from
    the cache key so identical image triples share one pair of Luna views.
    """

    root = Path(root).resolve()
    paths = {
        "reference": Path(reference),
        "anchor": Path(anchor),
        "candidate": Path(candidate),
    }
    hashes = {role: sha(path) for role, path in paths.items()}
    request_id = request_key(hashes)
    request = {
        "id": request_id,
        "protocol_version": PROTOCOL_VERSION,
        "images": {role: str(path) for role, path in paths.items()},
        "image_sha256": hashes,
        "trace_id": str(trace_id),
    }
    create_or_verify_request(root / "judge-queue" / request_id / "request.json", request)
    return request_id


def score_receipt(
    receipt: dict[str, Any],
    request_id: str,
    reference: Path,
    anchor: Path,
    candidate: Path,
) -> float | None:
    """Verify a receipt against the exact final-canvas image triple.

    This is the small adapter consumed by ``luna_aesthetic_contract``.  It
    returns only the holistic pairwise fraction; excluded/uncertain receipts
    return ``None`` for the caller to mask.
    """

    hashes = {
        "reference": sha(Path(reference)),
        "anchor": sha(Path(anchor)),
        "candidate": sha(Path(candidate)),
    }
    required = {
        "id", "request_id", "request_hash", "protocol_version", "image_sha256",
        "judge_model", "reasoning_effort", "candidate_A", "candidate_B",
        "judgment", "pairwise_win_fraction", "reward", "status", "provenance",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("receipt fields do not match aesthetic protocol")
    if (receipt["id"] != request_id or receipt["request_id"] != request_id
            or receipt["request_hash"] != request_id
            or receipt["protocol_version"] != PROTOCOL_VERSION
            or receipt["judge_model"] != JUDGE_MODEL
            or receipt["reasoning_effort"] != REASONING_EFFORT
            or receipt["image_sha256"] != hashes):
        raise ValueError("receipt request or image provenance mismatch")
    if not valid_normalized_judgment(receipt["judgment"]):
        raise ValueError("invalid normalized aesthetic judgment")
    first = validate_judgment(receipt["candidate_A"])
    reverse = validate_judgment(receipt["candidate_B"])
    normalized = reconcile(first, reverse)
    if receipt["judgment"] != normalized:
        raise ValueError("receipt labels do not match normalized judgment")
    score = normalized["pairwise_win_fraction"]
    if receipt["pairwise_win_fraction"] != score or receipt["reward"] != score:
        raise ValueError("receipt score does not match normalized judgment")
    expected_status = "scored" if score is not None else "excluded"
    if receipt["status"] != expected_status:
        raise ValueError("receipt status does not match score")
    provenance = receipt["provenance"]
    if (not isinstance(provenance, dict)
            or provenance.get("protocol_version") != PROTOCOL_VERSION
            or provenance.get("request_hash") != request_id
            or provenance.get("image_sha256") != hashes):
        raise ValueError("receipt import provenance mismatch")
    return score


def _verify_receipt(receipt: dict[str, Any], request: dict[str, Any], private: dict[str, Any], destination: Path) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    required = {
        "id", "request_id", "request_hash", "protocol_version", "image_sha256",
        "judge_model", "reasoning_effort", "candidate_A", "candidate_B",
        "judgment", "pairwise_win_fraction", "reward", "status", "provenance",
    }
    if set(receipt) != required:
        raise ValueError("receipt fields do not match aesthetic protocol")
    if (receipt["id"] != request["id"] or receipt["request_id"] != request["id"]
            or receipt["request_hash"] != request["id"]
            or receipt["protocol_version"] != PROTOCOL_VERSION
            or receipt["judge_model"] != JUDGE_MODEL
            or receipt["reasoning_effort"] != REASONING_EFFORT
            or receipt["image_sha256"] != request["image_sha256"]):
        raise ValueError("receipt request or judge provenance mismatch")
    if not valid_normalized_judgment(receipt["judgment"]):
        raise ValueError("invalid normalized aesthetic judgment")
    score = receipt["judgment"]["pairwise_win_fraction"]
    if receipt["pairwise_win_fraction"] != score or receipt["reward"] != score:
        raise ValueError("receipt reward does not match normalized judgment")
    if receipt["status"] != ("scored" if score is not None else "excluded"):
        raise ValueError("receipt status does not match score")
    first = validate_judgment(receipt["candidate_A"])
    reverse = validate_judgment(receipt["candidate_B"])
    if reconcile(first, reverse) != receipt["judgment"]:
        raise ValueError("receipt labels do not match normalized judgment")
    provenance = receipt["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
            "protocol_version", "request_hash", "image_sha256", "orientations"}:
        raise ValueError("receipt import provenance is incomplete")
    if (provenance["protocol_version"] != PROTOCOL_VERSION
            or provenance["request_hash"] != request["id"]
            or provenance["image_sha256"] != request["image_sha256"]):
        raise ValueError("receipt import provenance mismatch")
    if not isinstance(provenance["orientations"], list) or len(provenance["orientations"]) != 2:
        raise ValueError("receipt must preserve both orientation mappings")
    orientation_by_side = {item.get("candidate_side"): item for item in provenance["orientations"]}
    if set(orientation_by_side) != {"A", "B"}:
        raise ValueError("receipt orientation mapping is incomplete")
    for side, row in (("A", private["A"]), ("B", private["B"])):
        item = orientation_by_side[side]
        if (item.get("pair_id") != row["id"]
                or item.get("image_sha256") != row["image_sha256"]):
            raise ValueError("receipt orientation pair mismatch")


def install(root: Path, batch: str) -> dict[str, Any]:
    """Install only exact, protocol-valid receipts into the source queue."""

    root = Path(root).resolve()
    destination = root / "reward-batches" / batch
    private_rows = json.loads((destination / "private-key.json").read_text())
    by_request: dict[str, dict[str, dict[str, Any]]] = {}
    for row in private_rows:
        side = row["candidate_side"]
        existing = by_request.setdefault(row["request_id"], {})
        if side in existing:
            raise ValueError("duplicate orientation mapping")
        existing[side] = row
    installed = 0
    for receipt_path in sorted((destination / "receipts").glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        request_id = receipt.get("id")
        request = _request_for_id(root, request_id)
        if set(by_request.get(request_id, {})) != {"A", "B"}:
            raise ValueError("receipt has incomplete orientation mapping")
        _verify_receipt(receipt, request, by_request[request_id], destination)
        target = root / "judge-queue" / request_id / "receipt.json"
        if target.exists():
            existing = json.loads(target.read_text())
            if existing != receipt:
                raise ValueError("existing installed receipt differs")
        else:
            write(target, receipt)
        installed += 1
    return {"batch": batch, "protocol_version": PROTOCOL_VERSION, "installed": installed}


def status(root: Path) -> dict[str, Any]:
    """Return queue counters without touching or scheduling any work."""

    root = Path(root).resolve()
    total = pending = assigned = completed = 0
    for request_path in sorted((root / "judge-queue").glob("*/request.json")):
        total += 1
        folder = request_path.parent
        has_receipt = (folder / "receipt.json").is_file()
        has_assignment = (folder / "assigned.json").is_file()
        if has_receipt:
            completed += 1
        elif has_assignment:
            assigned += 1
        else:
            pending += 1
    return {
        "protocol_version": PROTOCOL_VERSION,
        "total": total,
        "pending": pending,
        "assigned": assigned,
        "completed": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export", "stage", "receipts", "install", "status"))
    parser.add_argument("batch", nargs="?")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    if args.action == "status":
        result = status(args.root)
    elif not args.batch:
        parser.error("batch is required for export, stage, receipts, and install")
    elif args.action == "export":
        result = export(args.root, args.batch, args.limit)
    elif args.action == "stage":
        result = stage(args.root, args.batch)
    elif args.action == "receipts":
        result = receipts(args.root, args.batch)
    else:
        result = install(args.root, args.batch)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
