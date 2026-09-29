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


def build(*, pool: Path, eval_root: Path, output: Path,
          retry_roots: tuple[Path, ...] = (),
          train_counts: tuple[int, int, int] = (24, 16, 0),
          expected_count: int = 72,
          validation_expected: int = 16) -> dict:
    source = json.loads(pool.read_text())
    rows = source["cases"]
    if len(rows) != expected_count or len({row["id"] for row in rows}) != expected_count:
        raise ValueError(f"expected {expected_count} distinct briefs")
    files = []
    by_id: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for attempt, source_root in enumerate((eval_root, *retry_roots)):
        traces, errors, source_files = read_traces(source_root)
        if errors:
            raise ValueError("trace read errors: " + "; ".join(errors[:3]))
        if not source_files:
            raise FileNotFoundError(f"no evaluation traces in attempt {attempt}")
        files.extend(source_files)
        for trace in traces:
            data, info = _trace_data(trace)
            task_id = data.get("task_id") or data.get("name")
            if isinstance(task_id, str):
                by_id[task_id][attempt].append(info)
    output.mkdir(parents=True, exist_ok=True)
    anchors = output / "anchors"
    anchors.mkdir(exist_ok=True)
    decisions = []
    usable_rows = []
    for row in rows:
        brief = pool.parent / row["reference"]
        if sha(brief) != row["reference_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['id']}")
        episodes_by_attempt = by_id.get(row["id"], {})
        reason = "missing_trace"
        canvas = None
        chosen_attempt = None
        attempts = []
        for attempt, source_root in enumerate((eval_root, *retry_roots)):
            episodes = episodes_by_attempt.get(attempt, [])
            outcome = "missing_trace"
            if len(episodes) > 1:
                outcome = "duplicate_trace"
            elif episodes:
                info = episodes[0]
                if info.get("service_failure"):
                    outcome = "service_failure"
                elif not info.get("final_canvas_valid"):
                    outcome = "invalid_model_output"
                else:
                    raw = info.get("final_canvas")
                    candidate = Path(raw) if isinstance(raw, str) and raw else None
                    if candidate is not None and not candidate.is_absolute():
                        candidate = source_root / candidate
                    if candidate is not None:
                        valid, outcome = usable(candidate)
                        if valid and canvas is None:
                            canvas = candidate
                            chosen_attempt = attempt
            attempts.append(outcome)
            if canvas is not None:
                break
            reason = outcome
        decision = {"id": row["id"], "split": row["split"], "family": row["family"],
                    "reason": "valid" if canvas is not None else reason,
                    "attempts": attempts, "chosen_attempt": chosen_attempt}
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
    for family, needed in zip(("simple", "intermediate", "complex"), train_counts):
        if not needed:
            continue
        initial = [row for row in usable_rows if row["split"] == "train" and row["family"] == family]
        reserves = [row for row in usable_rows if row["split"] == "reserve" and row["family"] == family]
        replacements = reserves[:max(0, needed - len(initial))]
        chosen.extend(initial + [{**row, "split": "train", "reserve_replacement": True} for row in replacements])
        if len(initial) + len(replacements) < needed:
            raise ValueError(f"only {len(initial) + len(replacements)}/{needed} usable {family} train anchors")
    validation = [row for row in usable_rows if row["split"] == "validation"]
    reserve = [row for row in usable_rows if row["split"] == "reserve"]
    if sum(row["split"] == "validation" for row in rows) != validation_expected:
        raise ValueError("unexpected validation split size")
    train = output / "train-manifest.json"
    val = output / "validation-manifest.json"
    reserve_path = output / "reserve-manifest.json"
    train.write_text(json.dumps({"rows": chosen}, indent=2, sort_keys=True) + "\n")
    val.write_text(json.dumps({"cases": validation}, indent=2, sort_keys=True) + "\n")
    reserve_path.write_text(json.dumps({"cases": reserve}, indent=2, sort_keys=True) + "\n")
    audit = {"schema": "painter-text-anchor-freeze-v1", "pool_sha256": sha(pool),
             "eval_trace_files": [str(path) for path in files],
             "train_count": len(chosen), "validation_usable": len(validation),
             "validation_expected": validation_expected, "reserve_usable": len(reserve),
             "decisions": decisions,
             "train_counts_requested": dict(zip(("simple", "intermediate", "complex"), train_counts)),
             "train_manifest_sha256": sha(train), "validation_manifest_sha256": sha(val),
             "reserve_manifest_sha256": sha(reserve_path)}
    (output / "anchor-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--retry-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-simple", type=int, default=24)
    parser.add_argument("--train-intermediate", type=int, default=16)
    parser.add_argument("--train-complex", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=72)
    parser.add_argument("--validation-expected", type=int, default=16)
    args = parser.parse_args()
    result = build(pool=args.pool, eval_root=args.eval_root, output=args.output,
                   retry_roots=tuple(args.retry_root),
                   train_counts=(args.train_simple, args.train_intermediate,
                                 args.train_complex), expected_count=args.expected_count,
                   validation_expected=args.validation_expected)
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"}, sort_keys=True))


if __name__ == "__main__":
    main()
