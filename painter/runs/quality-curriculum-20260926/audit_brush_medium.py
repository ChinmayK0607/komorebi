#!/usr/bin/env python3
"""Recompute static brush-use and render-time facts for the pinned 595 canvases."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from prepare_visual_audit import OUT


BRUSH_PAINT = re.compile(
    r"\bbrush\s*\.\s*(fill|wash|hatch|line|flowLine|spline|rect|circle|arc|"
    r"beginShape|polygon|beginStroke|move|endStroke)\s*\("
)
NATIVE_PAINT = re.compile(
    r"(?<![.\w])(?:fill|stroke|ellipse|rect|circle|line|beginShape|vertex|endShape)\s*\("
)
SCALE = re.compile(r"\bbrush\s*\.\s*scaleBrushes\s*\(")


def timings(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"n": len(ordered), "median_seconds": round(statistics.median(ordered), 2),
            "p90_seconds": round(ordered[int(0.9 * (len(ordered) - 1))], 2)}


def main() -> None:
    rows = [row for shard in range(6)
            for row in json.loads((OUT / f"shard-{shard}.json").read_text())]
    if len(rows) != 595 or len({x["audit_id"] for x in rows}) != len(rows):
        raise ValueError("unexpected or duplicate audit rows")
    counts = Counter()
    by_mode = defaultdict(Counter)
    seconds = defaultdict(list)
    for row in rows:
        canvas = Path(row["canvas_path"])
        if hashlib.sha256(canvas.read_bytes()).hexdigest() != row["canvas_sha256"]:
            raise ValueError(f"canvas changed: {row['audit_id']}")
        program = (canvas.parent / "program.js").read_text()
        meta = json.loads(canvas.with_suffix(".json").read_text())
        duration = meta["painting_seconds"]
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError(f"invalid render duration: {row['audit_id']}")
        brush = bool(BRUSH_PAINT.search(program))
        native = bool(NATIVE_PAINT.search(program))
        scaled = bool(SCALE.search(program))
        counts.update(brush_paint=brush, no_brush_paint=not brush,
                      native_paint=native, mixed_brush_native=brush and native,
                      scale_brushes=scaled)
        by_mode[row["mode"]].update(brush_paint=brush, no_brush_paint=not brush)
        seconds["all"].append(duration)
        seconds[row["mode"]].append(duration)
        seconds["brush_paint" if brush else "no_brush_paint"].append(duration)
    print(json.dumps({"scope": len(rows), "counts": dict(counts),
                      "by_mode": {key: dict(value) for key, value in by_mode.items()},
                      "timings": {key: timings(value) for key, value in seconds.items()}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
