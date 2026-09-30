#!/usr/bin/env python3
"""Prepare hash-bound CPU replay inputs from saved native-eval traces.

This is a metadata and source-extraction step.  It never invokes the
renderer, reads a recorded canvas path, or treats a missing canvas as a
successful replay target.  The output contains the final code-bearing paint
for each trace whose recorded final canvas has a PNG hash, plus that recorded
hash for a later operator-controlled replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from painter.native_painting_renderer import extract_sketch
except ModuleNotFoundError:  # Direct ``python path/to/prepare_cpu_replay.py`` use.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from painter.native_painting_renderer import extract_sketch


SEED = 0
PROTOCOL_VERSION = "native-painting-eval-v3"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recorded_hash(value: Any) -> str | None:
    if isinstance(value, str) and HEX64.fullmatch(value.lower()):
        return value.lower()
    return None


def _same_recorded_path(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    # These are opaque recorded paths.  Normalise separators only; never
    # resolve or read them because the node-side canvases may be gone.
    return left.replace("\\", "/") == right.replace("\\", "/")


def _trace_info(row: dict[str, Any]) -> dict[str, Any] | None:
    traces = row.get("traces")
    if not isinstance(traces, list) or len(traces) != 1 or not isinstance(traces[0], dict):
        return None
    info = traces[0].get("info")
    return info if isinstance(info, dict) else None


def _target_hash(info: dict[str, Any], turns: list[dict[str, Any]], final_canvas: str) -> str | None:
    for key in ("final_canvas_sha256", "canvas_sha256", "final_png_sha256"):
        candidate = _recorded_hash(info.get(key))
        if candidate:
            return candidate
    for turn in reversed(turns):
        if _same_recorded_path(turn.get("canvas"), final_canvas):
            candidate = _recorded_hash(turn.get("canvas_sha256"))
            if candidate:
                return candidate
    return None


def _code_candidates(turns: list[dict[str, Any]], final_canvas: str, target_hash: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if turn.get("action") != "paint" or turn.get("valid") is not True:
            continue
        canvas = turn.get("canvas")
        canvas_hash = _recorded_hash(turn.get("canvas_sha256"))
        if not (_same_recorded_path(canvas, final_canvas) or canvas_hash == target_hash):
            continue
        reply = turn.get("reply")
        if not isinstance(reply, str):
            continue
        try:
            source = extract_sketch(reply)
        except ValueError:
            continue
        candidates.append({
            "turn": turn.get("turn", index + 1),
            "source": source,
            "canvas_hash": canvas_hash,
        })
    return candidates


def _safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown"))
    return text.strip(".") or "unknown"


def _renderer_identity(renderer: Path) -> dict[str, Any]:
    renderer = renderer.expanduser().resolve()
    if not renderer.is_file():
        raise FileNotFoundError(f"renderer source is missing: {renderer}")
    assets_dir = renderer.parent / "assets"
    assets = {
        path.name: sha256(path)
        for path in sorted(assets_dir.glob("*.js"))
        if path.is_file()
    }
    if not assets:
        raise FileNotFoundError(f"pinned renderer assets are missing: {assets_dir}")
    return {"renderer_sha256": sha256(renderer), "assets_sha256": assets}


def build_replay_manifest(traces_path: Path, output_dir: Path, renderer: Path) -> dict[str, Any]:
    traces_path = traces_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not traces_path.is_file():
        raise FileNotFoundError(f"trace file is missing: {traces_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    programs_dir = output_dir / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with traces_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trace JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"trace row at line {line_number} is not an object")
            info = _trace_info(row)
            if info is None:
                skipped.append({"line": str(line_number), "reason": "missing_single_trace_info"})
                continue
            task_id = info.get("task_id") or info.get("source_group")
            if not isinstance(task_id, str) or not task_id:
                skipped.append({"line": str(line_number), "reason": "missing_task_id"})
                continue
            if task_id in seen_ids:
                raise ValueError(f"duplicate task id in traces: {task_id}")
            seen_ids.add(task_id)
            final_canvas = info.get("final_canvas")
            turns = info.get("turns")
            if info.get("final_canvas_valid") is not True or not isinstance(final_canvas, str) or not isinstance(turns, list):
                skipped.append({"task_id": task_id, "reason": "no_recorded_valid_final_canvas"})
                continue
            turns = [turn for turn in turns if isinstance(turn, dict)]
            target_hash = _target_hash(info, turns, final_canvas)
            if target_hash is None:
                skipped.append({"task_id": task_id, "reason": "final_canvas_hash_unavailable"})
                continue
            candidates = _code_candidates(turns, final_canvas, target_hash)
            if not candidates:
                skipped.append({"task_id": task_id, "reason": "no_code_bearing_paint_for_final_canvas"})
                continue
            chosen = candidates[-1]
            filename = f"{_safe_id(task_id)}.js"
            source_path = programs_dir / filename
            source_bytes = chosen["source"].encode("utf-8")
            source_digest = hashlib.sha256(source_bytes).hexdigest()
            if source_path.is_symlink():
                raise ValueError(f"program output must not be a symlink: {filename}")
            if source_path.exists():
                if sha256(source_path) != source_digest:
                    raise ValueError(f"existing program differs for task id: {task_id}")
            else:
                source_path.write_bytes(source_bytes)
            rows.append({
                "id": task_id,
                "source_group": info.get("source_group") if isinstance(info.get("source_group"), str) else task_id,
                "family": info.get("family") if isinstance(info.get("family"), str) else None,
                "seed": SEED,
                "protocol_version": info.get("protocol_version", PROTOCOL_VERSION),
                "source_file": f"programs/{filename}",
                "source_sha256": source_digest,
                "source_turn": chosen["turn"],
                "target_png_sha256": target_hash,
                "target_hash_basis": "recorded final_canvas/turn canvas_sha256; PNG not present locally",
            })

    manifest: dict[str, Any] = {
        "schema": "text-sft512-cpu-replay-inputs-v1",
        "source_trace_sha256": sha256(traces_path),
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "renderer": _renderer_identity(renderer),
        "summary": {
            "trace_rows": len(rows) + len(skipped),
            "replay_cases": len(rows),
            "skipped_rows": len(skipped),
        },
        "cases": rows,
        "skipped": skipped,
        "limitations": [
            "Inputs contain recorded target PNG hashes; canvases were not available during preparation.",
            "No renderer, browser, model server, or remote service was launched.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_replay_manifest(args.traces, args.output, args.renderer)
    print(json.dumps({"output": str(args.output), "summary": manifest["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
