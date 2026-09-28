"""Freeze valid SFT canvases and build a same-source text RL curriculum.

Run after the pinned step-512 policy has been evaluated on all pool prompts.
All exclusions and substitutions are written to an audit; validation membership
is never changed after inspecting an RL output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import read_traces, _trace_data  # noqa: E402


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def usable(canvas: Path) -> tuple[bool, str]:
    if not canvas.is_file() or canvas.is_symlink():
        return False, "canvas_missing"
    try:
        with Image.open(canvas) as image:
            rgb = image.convert("RGB")
            if rgb.width < 128 or rgb.height < 128:
                return False, "canvas_too_small"
            stat = ImageStat.Stat(rgb.resize((128, 128)))
            if max(stat.stddev) < 2.0:
                return False, "nearly_uniform_canvas"
    except (OSError, ValueError):
        return False, "unreadable_canvas"
    return True, "valid"


def build(*, pool: Path, eval_root: Path, output: Path) -> dict:
    source = json.loads(pool.read_text())
    rows = source["cases"]
    if len(rows) != 72 or len({row["id"] for row in rows}) != 72:
        raise ValueError("expected the frozen 72-brief pool")
    traces, errors, files = read_traces(eval_root)
    if errors:
        raise ValueError("trace read errors: " + "; ".join(errors[:3]))
    if not files:
        raise FileNotFoundError("no evaluation traces")
    by_id: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        data, info = _trace_data(trace)
        task_id = data.get("task_id") or data.get("name")
        if isinstance(task_id, str):
            by_id[task_id].append(info)
    output.mkdir(parents=True, exist_ok=True)
    anchors = output / "anchors"
    anchors.mkdir(exist_ok=True)
    decisions = []
    usable_rows = []
    for row in rows:
        brief = pool.parent / row["reference"]
        if sha(brief) != row["reference_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['id']}")
        episodes = by_id.get(row["id"], [])
        reason = "missing_trace"
        canvas = None
        if len(episodes) > 1:
            reason = "duplicate_trace"
        elif episodes:
            info = episodes[0]
            if info.get("service_failure"):
                reason = "service_failure"
            elif not info.get("final_canvas_valid"):
                reason = "invalid_model_output"
            else:
                raw = info.get("final_canvas")
                candidate = Path(raw) if isinstance(raw, str) and raw else None
                if candidate is not None and not candidate.is_absolute():
                    candidate = eval_root / candidate
                if candidate is not None:
                    valid, reason = usable(candidate)
                    if valid:
                        canvas = candidate
        decision = {"id": row["id"], "split": row["split"], "family": row["family"],
                    "reason": reason}
        if canvas is not None:
            destination = anchors / (row["id"] + ".png")
            shutil.copyfile(canvas, destination)
            decision["anchor_sha256"] = sha(destination)
            usable_rows.append({**row, "opponent": str(destination.resolve()),
                                "opponent_sha256": decision["anchor_sha256"],
                                "opponent_source_group": row["source_group"],
                                "reference": str(brief.resolve()), "mode": "fresh",
                                "max_turns": 3})
        decisions.append(decision)
    chosen = []
    for family, needed in (("simple", 24), ("intermediate", 16)):
        initial = [row for row in usable_rows if row["split"] == "train" and row["family"] == family]
        reserves = [row for row in usable_rows if row["split"] == "reserve" and row["family"] == family]
        replacements = reserves[:max(0, needed - len(initial))]
        chosen.extend(initial + [{**row, "split": "train", "reserve_replacement": True} for row in replacements])
        if len(initial) + len(replacements) < needed:
            raise ValueError(f"only {len(initial) + len(replacements)}/{needed} usable {family} train anchors")
    validation = [row for row in usable_rows if row["split"] == "validation"]
    train = output / "train-manifest.json"
    val = output / "validation-manifest.json"
    train.write_text(json.dumps({"rows": chosen}, indent=2, sort_keys=True) + "\n")
    val.write_text(json.dumps({"cases": validation}, indent=2, sort_keys=True) + "\n")
    audit = {"schema": "painter-text-anchor-freeze-v1", "pool_sha256": sha(pool),
             "eval_trace_files": [str(path) for path in files],
             "train_count": len(chosen), "validation_usable": len(validation),
             "validation_expected": 16, "decisions": decisions,
             "train_manifest_sha256": sha(train), "validation_manifest_sha256": sha(val)}
    (output / "anchor-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(pool=args.pool, eval_root=args.eval_root, output=args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"}, sort_keys=True))


if __name__ == "__main__":
    main()
