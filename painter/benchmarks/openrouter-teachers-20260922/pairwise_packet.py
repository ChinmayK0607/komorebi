#!/usr/bin/env python3
"""Build an offline, blinded cross-model pairwise judge packet.

This module only reads episode receipts and image files.  It never calls a
provider or a judge.  The public packet contains an opaque pair id, the exact
reference bytes, and (for eligible pairs) two opaque candidate images.  Model,
track, provider, turn and episode provenance are written to a separate private
mapping.  Pairs containing an invalid candidate remain in that mapping and in
the public coverage counts as censored cases; they are never converted into a
visual loss.

The default input is an artifact directory containing ``episodes/``.  If the
artifact directory is a collected result tree, reference images are resolved
from its parent benchmark directory as well.  Example::

    python pairwise_packet.py \
      --root results-ai-gateway-20260923 \
      --track quality \
      --output results-ai-gateway-20260923/review/pairwise-quality

The output is deterministic for a fixed artifact tree and ``--seed``.  No
timestamps or generated judgments are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "painter.ai-gateway-teachers.pairwise-judge.v1"
PRIVATE_SCHEMA = "painter.ai-gateway-teachers.pairwise-private.v1"
DEFAULT_SEED = "painter-pairwise-ab-v1"
VALID_EPISODE_STATUSES = frozenset({"complete", "turn_limit"})
INVALID_STATUS_CLASSES = {
    "deadline_censored": "service_censored",
    "renderer_error": "service_censored",
    "api_error": "service_censored",
    "context_exhausted": "service_censored",
    "invalid": "model_invalid",
}


class PairwisePacketError(ValueError):
    """Raised when benchmark receipts are unsafe or internally inconsistent."""


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.replace("\\", "/")


def _resolve_under(root: Path, value: Any, *, role: str) -> Path | None:
    """Resolve a receipt path without allowing it to escape its artifact root."""

    relative = _relative_path(value)
    if relative is None:
        return None
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PairwisePacketError(f"{role} path must be relative: {value!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PairwisePacketError(f"{role} path escapes artifact root: {value!r}") from exc
    return resolved


def _image_info(path: Path, *, role: str) -> dict[str, Any]:
    """Verify that ``path`` is a decodable, non-empty image and return facts."""

    if not path.is_file() or path.is_symlink():
        raise PairwisePacketError(f"{role} image is missing or symlinked: {path}")
    if path.stat().st_size <= 0:
        raise PairwisePacketError(f"{role} image is empty: {path}")
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
            width, height = image.size
            image_format = str(image.format or "").upper()
            mime = Image.MIME.get(image_format, "application/octet-stream")
    except Exception as exc:  # Pillow uses several exception types for bad files.
        raise PairwisePacketError(f"{role} image is not a valid image: {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise PairwisePacketError(f"{role} image has invalid dimensions: {path}")
    return {
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "mime": mime,
    }


def _suffix(info: Mapping[str, Any], source: Path, *, candidate: bool = False) -> str:
    # Canvas renders are PNGs in the benchmark.  Keeping the detected format
    # rather than the source filename prevents model names from entering an
    # asset path while still making the copied asset viewable in a browser.
    if str(info.get("format")) == "PNG":
        return ".png"
    if str(info.get("format")) in {"JPEG", "JPG"}:
        return ".jpg"
    return ".img" if candidate else (source.suffix.lower() or ".img")


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    episodes = root / "episodes"
    rows: list[dict[str, Any]] = []
    if not episodes.is_dir():
        raise PairwisePacketError(f"episode directory is missing: {episodes}")
    for path in sorted(episodes.glob("*/episode.json"), key=lambda item: item.parent.name):
        row = _load_json(path)
        if row is None:
            raise PairwisePacketError(f"invalid episode JSON: {path}")
        row = dict(row)
        row["_path"] = path
        rows.append(row)
    return rows


def _reference_root(root: Path, row: Mapping[str, Any], explicit: Path | None) -> Path:
    value = _relative_path(row.get("reference_image"))
    if value is None:
        raise PairwisePacketError(f"episode has no reference_image: {row.get('_path')}")
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend((root, root.parent))
    for base in candidates:
        path = _resolve_under(base, value, role="reference")
        if path is not None and path.is_file():
            return path
    raise PairwisePacketError(f"reference image is missing for episode {row.get('job_id')}: {value}")


def _declared_reference(row: Mapping[str, Any], path: Path) -> dict[str, Any]:
    declared = row.get("reference_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise PairwisePacketError(f"episode {row.get('job_id')} has no valid reference_sha256")
    actual = _sha(path)
    if actual != declared:
        raise PairwisePacketError(
            f"reference hash mismatch for {row.get('job_id')}: expected {declared}, found {actual}"
        )
    info = _image_info(path, role="reference")
    return {"path": path, "sha256": actual, "info": info}


def _manifest_reference(root: Path, reference_root: Path, row: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    """If refs.json is available, check its immutable image identity too."""

    manifest_path = next((candidate for candidate in (reference_root / "refs.json", root / "refs.json", root.parent / "refs.json") if candidate.is_file()), None)
    if manifest_path is None:
        return
    manifest = _load_json(manifest_path)
    if manifest is None:
        raise PairwisePacketError(f"invalid references manifest: {manifest_path}")
    ref_id = row.get("reference_id")
    entries = manifest.get("references")
    entry = next((item for item in entries or [] if isinstance(item, dict) and item.get("id") == ref_id), None)
    if entry is None:
        raise PairwisePacketError(f"reference {ref_id!r} is not present in {manifest_path}")
    if entry.get("sha256") != reference["sha256"]:
        raise PairwisePacketError(f"reference manifest hash mismatch for {ref_id}")
    manifest_image = _relative_path(entry.get("image"))
    if manifest_image is None:
        raise PairwisePacketError(f"reference manifest entry has no image for {ref_id}")
    manifest_path_image = _resolve_under(manifest_path.parent, manifest_image, role="manifest reference")
    if manifest_path_image is None or manifest_path_image.resolve() != Path(reference["path"]).resolve():
        raise PairwisePacketError(f"reference manifest image mismatch for {ref_id}")


def _same_rel(root: Path, left: Any, right: Any) -> bool:
    a = _resolve_under(root, left, role="canvas")
    b = _resolve_under(root, right, role="canvas")
    return a is not None and b is not None and a == b


def _declared_hashes(turn: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for container in (turn, turn.get("render") if isinstance(turn.get("render"), dict) else {}, turn.get("render", {}).get("receipt") if isinstance(turn.get("render"), dict) and isinstance(turn.get("render", {}).get("receipt"), dict) else {}):
        for key in ("canvas_sha256", "current_canvas_sha256", "png_sha256"):
            value = container.get(key)
            if isinstance(value, str) and value:
                values.add(value)
    return values


def _candidate_evidence(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Return conservative validity evidence for one episode's final canvas."""

    status = str(row.get("status") or "unknown")
    base: dict[str, Any] = {
        "valid": False,
        "status": status,
        "failure_class": INVALID_STATUS_CLASSES.get(status, "invalid_episode"),
        "failure_reason": None,
        "source_canvas": None,
        "turn": None,
        "sha256": None,
        "info": None,
    }
    final_value = row.get("final_valid_canvas")
    if status not in VALID_EPISODE_STATUSES:
        base["failure_reason"] = f"episode_status:{status}"
        return base
    if row.get("last_turn_invalid") is True:
        # A retained earlier canvas does not make an episode with an invalid
        # final response eligible.  This catches the original pilot's parser
        # artifact and any later render/parser failure.
        base["failure_reason"] = "episode_ended_with_invalid_turn"
        base["failure_class"] = "model_invalid"
        return base
    canvas = _resolve_under(root, final_value, role="final canvas")
    if canvas is None or not canvas.is_file():
        base["failure_reason"] = "final_canvas_missing"
        return base
    turns = row.get("turns")
    if not isinstance(turns, list):
        base["failure_reason"] = "turn_history_missing"
        return base
    matching: list[tuple[int, Mapping[str, Any]]] = []
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict) or not _same_rel(root, turn.get("canvas"), final_value):
            continue
        if (turn.get("render") or {}).get("valid") is True:
            value = turn.get("turn")
            matching.append((int(value) if isinstance(value, int) else index, turn))
    if not matching:
        base["failure_reason"] = "final_canvas_not_backed_by_valid_turn"
        return base
    turn_number, turn = matching[-1]
    try:
        info = _image_info(canvas, role="final canvas")
    except PairwisePacketError as exc:
        base["failure_reason"] = str(exc)
        return base
    declared = _declared_hashes(turn)
    if declared and info["sha256"] not in declared:
        base["failure_reason"] = "final_canvas_hash_mismatch"
        return base
    base.update({
        "valid": True,
        "failure_class": None,
        "failure_reason": None,
        "source_canvas": canvas,
        "turn": turn_number,
        "sha256": info["sha256"],
        "info": info,
    })
    return base


def _track(row: Mapping[str, Any]) -> str:
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    value = row.get("track") or settings.get("track")
    if not isinstance(value, str) or not value:
        raise PairwisePacketError(f"episode has no track: {row.get('_path')}")
    return value


def _model(row: Mapping[str, Any]) -> str:
    value = row.get("model")
    if not isinstance(value, str) or not value:
        raise PairwisePacketError(f"episode has no model: {row.get('_path')}")
    return value


def _copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha(destination) != _sha(source):
        raise PairwisePacketError(f"copied asset hash mismatch: {source}")


def _asset_descriptor(path: Path, *, relative_to: Path, role: str) -> dict[str, Any]:
    info = _image_info(path, role=role)
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "sha256": info["sha256"],
        "mime": info["mime"],
        "width": info["width"],
        "height": info["height"],
    }


def _portable_source(path: Path | None, *, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")


def _pair_id(seed: str, track: str, reference_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    jobs = sorted((str(left.get("job_id") or ""), str(right.get("job_id") or "")))
    return "pair-" + _hash_text("\0".join((seed, track, reference_id, *jobs)))[:24]


def _oriented(seed: str, pair_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    # The digest bit is independent for every pair and stable across reruns.
    first, second = (left, right) if str(left.get("job_id")) < str(right.get("job_id")) else (right, left)
    reversed_order = int(_hash_text(f"{seed}\0{pair_id}")[-1], 16) % 2 == 1
    if reversed_order:
        return second, first, "reversed"
    return first, second, "sorted"


def _public_candidate(evidence: Mapping[str, Any], descriptor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"valid": bool(evidence.get("valid"))}
    if descriptor is not None and evidence.get("valid"):
        value.update({"path": descriptor["path"], "sha256": descriptor["sha256"], "mime": descriptor["mime"]})
    else:
        value.update({"path": None, "sha256": None, "mime": None})
    return value


def _public_response_schema() -> dict[str, Any]:
    return {
        "winner": ["A", "B", "tie", "uncertain"],
        "confidence": ["high", "medium", "low"],
        "fidelity_note": "short string",
        "aesthetic_note": "short string",
        "preferred_weakness": "short string or none apparent",
        "tie_or_uncertainty_reason": "short string when applicable",
    }


def build_pairwise_packet(
    root: Path,
    output: Path | None = None,
    *,
    track: str | None = None,
    seed: str = DEFAULT_SEED,
    reference_root: Path | None = None,
) -> dict[str, Path | int]:
    """Build one track's packet and private mapping.

    ``root`` is the collected result directory containing ``episodes/``.
    ``track=None`` includes every discovered track in one artifact tree; the
    caller should normally invoke this once per track so quality and speed are
    handed to judges separately.  The public JSON never contains track or
    model/turn/provider/episode metadata.
    """

    root = Path(root).resolve()
    output = Path(output or root / "review" / "pairwise").resolve()
    rows = _episode_rows(root)
    selected: list[dict[str, Any]] = []
    for row in rows:
        row_track = _track(row)
        if track is not None and row_track != track:
            continue
        # Validate model/reference identity before pairing.  An image hash
        # mismatch is fatal: pairing against a silently different reference is
        # worse than producing no packet.
        _model(row)
        ref_path = _reference_root(root, row, reference_root)
        reference = _declared_reference(row, ref_path)
        _manifest_reference(root, ref_path.parent, row, reference)
        row = dict(row)
        row["_track"] = row_track
        row["_reference"] = reference
        row["_reference_root"] = ref_path.parent
        row["_candidate"] = _candidate_evidence(root, row)
        selected.append(row)
    if not selected:
        raise PairwisePacketError(f"no episodes matched track {track!r}" if track else "no episodes found")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    reference_hashes: dict[tuple[str, str], str] = {}
    for row in selected:
        key = (str(row.get("reference_id") or ""), str(row["_track"]))
        row_hash = str(row["_reference"]["sha256"])
        previous_hash = reference_hashes.get(key)
        if previous_hash is not None and previous_hash != row_hash:
            raise PairwisePacketError(f"reference hash differs within group {key!r}")
        reference_hashes[key] = row_hash
        groups.setdefault(key, []).append(row)

    pairs: list[dict[str, Any]] = []
    for (reference_id, row_track), candidates in sorted(groups.items()):
        by_model: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            model = _model(candidate)
            if model in by_model:
                raise PairwisePacketError(
                    f"duplicate model episode in reference/track group {reference_id!r}/{row_track!r}: {model!r}"
                )
            by_model[model] = candidate
        ordered = sorted(candidates, key=lambda item: (str(item.get("job_id") or ""), _model(item)))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                pair_id = _pair_id(seed, row_track, reference_id, left, right)
                candidate_a, candidate_b, orientation = _oriented(seed, pair_id, left, right)
                pair_status = "eligible" if left["_candidate"]["valid"] and right["_candidate"]["valid"] else (
                    "both_invalid_censored" if not left["_candidate"]["valid"] and not right["_candidate"]["valid"] else "one_invalid_censored"
                )
                pairs.append({
                    "id": pair_id,
                    "track": row_track,
                    "reference_id": reference_id,
                    "reference": left["_reference"],
                    "a": candidate_a,
                    "b": candidate_b,
                    "orientation": orientation,
                    "status": pair_status,
                })

    if output.exists() and output.is_file():
        raise PairwisePacketError(f"output must be a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assets_root = output / "assets"
    private_assets_root = output / "private-assets"
    public_comparisons: list[dict[str, Any]] = []
    public_censored: list[dict[str, Any]] = []
    private_pairs: list[dict[str, Any]] = []
    counts = {
        "attempted_pairs": len(pairs),
        "eligible_comparisons": 0,
        "one_invalid_censored": 0,
        "both_invalid_censored": 0,
    }
    for pair in pairs:
        pair_id = str(pair["id"])
        ref = pair["reference"]
        ref_path = Path(ref["path"])
        ref_info = ref["info"]
        pair_assets = assets_root / pair_id
        public_ref_path = pair_assets / f"reference{_suffix(ref_info, ref_path)}"
        _copy_asset(ref_path, public_ref_path)
        public_ref = _asset_descriptor(public_ref_path, relative_to=output, role="copied reference")
        private_ref_path = private_assets_root / pair_id / f"reference{_suffix(ref_info, ref_path)}"
        _copy_asset(ref_path, private_ref_path)

        public_candidates: dict[str, dict[str, Any]] = {}
        private_candidates: dict[str, dict[str, Any]] = {}
        for role, candidate in (("A", pair["a"]), ("B", pair["b"])):
            evidence = candidate["_candidate"]
            private: dict[str, Any] = {
                "role": role,
                "job_id": candidate.get("job_id"),
                "model": candidate.get("model"),
                "provider": (candidate.get("settings") or {}).get("provider"),
                "track": pair["track"],
                "turn": evidence.get("turn"),
                "valid": bool(evidence.get("valid")),
                "episode_status": evidence.get("status"),
                "failure_class": evidence.get("failure_class"),
                "failure_reason": evidence.get("failure_reason"),
                "source_canvas": _portable_source(evidence.get("source_canvas"), root=root),
                "canvas_sha256": evidence.get("sha256"),
                "private_asset": None,
            }
            if evidence.get("valid"):
                source = evidence["source_canvas"]
                asset_suffix = _suffix(evidence["info"], source, candidate=True)
                private_path = private_assets_root / pair_id / f"candidate-{role.lower()}{asset_suffix}"
                _copy_asset(source, private_path)
                private_descriptor = _asset_descriptor(private_path, relative_to=output, role=f"private candidate {role}")
                private["private_asset"] = private_descriptor
                if pair["status"] == "eligible":
                    public_path = pair_assets / f"candidate-{role.lower()}{asset_suffix}"
                    _copy_asset(source, public_path)
                    public_candidates[role] = _public_candidate(
                        evidence,
                        _asset_descriptor(public_path, relative_to=output, role=f"public candidate {role}"),
                    )
            if role not in public_candidates:
                public_candidates[role] = _public_candidate(evidence)
            private_candidates[role] = private

        private_pairs.append({
            "id": pair_id,
            "status": pair["status"],
            "track": pair["track"],
            "reference_id": pair["reference_id"],
            "reference": {
                "source": _portable_source(ref_path, root=root),
                "sha256": ref["sha256"],
                "private_asset": _asset_descriptor(private_ref_path, relative_to=output, role="private reference"),
            },
            "orientation": pair["orientation"],
            "candidates": private_candidates,
        })
        if pair["status"] == "eligible":
            counts["eligible_comparisons"] += 1
            public_comparisons.append({
                "id": pair_id,
                "reference": public_ref,
                "candidate_a": public_candidates["A"],
                "candidate_b": public_candidates["B"],
                "status": "pending",
                "judgment": None,
            })
        elif pair["status"] == "one_invalid_censored":
            counts["one_invalid_censored"] += 1
            public_censored.append({"id": pair_id, "status": "one_invalid_censored", "reference": public_ref, "candidate_a": {"valid": public_candidates["A"]["valid"]}, "candidate_b": {"valid": public_candidates["B"]["valid"]}})
        else:
            counts["both_invalid_censored"] += 1
            public_censored.append({"id": pair_id, "status": "both_invalid_censored", "reference": public_ref, "candidate_a": {"valid": False}, "candidate_b": {"valid": False}})

    public = {
        "schema": SCHEMA,
        "seed": seed,
        "coverage": counts,
        "comparisons": sorted(public_comparisons, key=lambda item: item["id"]),
        "censored": sorted(public_censored, key=lambda item: item["id"]),
        "judgment_schema": _public_response_schema(),
    }
    private = {
        "schema": PRIVATE_SCHEMA,
        "seed": seed,
        "coverage": counts,
        "pairs": sorted(private_pairs, key=lambda item: item["id"]),
    }
    packet_path = output / "judge-packet.json"
    private_path = output / "private-mapping.json"
    packet_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    private_path.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        pass
    return {"packet": packet_path, "private_mapping": private_path, "attempted_pairs": len(pairs), "eligible_comparisons": counts["eligible_comparisons"]}


def build_pairwise_packets(
    root: Path,
    output: Path | None = None,
    *,
    tracks: Sequence[str] | None = None,
    seed: str = DEFAULT_SEED,
    reference_root: Path | None = None,
) -> dict[str, dict[str, Path | int]]:
    """Build separate packets for each track, preserving track isolation."""

    root = Path(root).resolve()
    rows = _episode_rows(root)
    discovered = sorted({_track(row) for row in rows})
    selected_tracks = list(tracks) if tracks is not None else discovered
    if not selected_tracks:
        raise PairwisePacketError("no tracks found")
    base = Path(output or root / "review" / "pairwise").resolve()
    results: dict[str, dict[str, Path | int]] = {}
    for row_track in selected_tracks:
        results[row_track] = build_pairwise_packet(root, base / row_track, track=row_track, seed=seed, reference_root=reference_root)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="collected result directory containing episodes/")
    parser.add_argument("--output", type=Path, help="output directory (default: ROOT/review/pairwise)")
    parser.add_argument("--track", choices=("quality", "speed", "screen", "all"), default="all")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--reference-root", type=Path, help="optional benchmark root containing references/ and refs.json")
    args = parser.parse_args(argv)
    if args.track == "all":
        result = build_pairwise_packets(args.root, args.output, seed=args.seed, reference_root=args.reference_root)
        for row_track, paths in result.items():
            print(f"{row_track}: {paths['packet']}")
    else:
        paths = build_pairwise_packet(args.root, args.output, track=args.track, seed=args.seed, reference_root=args.reference_root)
        print(paths["packet"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
