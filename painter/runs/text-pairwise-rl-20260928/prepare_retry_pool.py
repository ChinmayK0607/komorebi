"""Select only briefs with no usable SFT canvas after prior fixed-seed passes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402

from build_anchors import usable  # noqa: E402


def prepare(pool: Path, sources: list[Path], output: Path) -> dict:
    if not sources:
        raise ValueError("at least one prior evaluation root is required")
    rows = json.loads(pool.read_text())["cases"]
    found: dict[str, list[dict]] = defaultdict(list)
    for source in sources:
        traces, errors, files = read_traces(source)
        if errors or not files:
            raise ValueError(f"incomplete prior evaluation {source}: {errors[:2]}")
        for trace in traces:
            data, info = _trace_data(trace)
            task_id = data.get("task_id") or data.get("name")
            if isinstance(task_id, str):
                found[task_id].append(info)
    pending = []
    for row in rows:
        valid = False
        for info in found.get(row["id"], []):
            raw = info.get("final_canvas")
            if not info.get("service_failure") and info.get("final_canvas_valid") and isinstance(raw, str):
                valid = usable(Path(raw))[0]
            if valid:
                break
        if not valid:
            pending.append(row)
    if len(pending) < 2:
        raise ValueError(f"only {len(pending)} briefs need retry; two-GPU shard runner needs at least two")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"cases": pending, "source_pool": str(pool),
                                  "prior_evaluations": [str(path) for path in sources]},
                                 indent=2, sort_keys=True) + "\n")
    return {"retry_count": len(pending), "output": str(output),
            "train": sum(row["split"] == "train" for row in pending),
            "validation": sum(row["split"] == "validation" for row in pending),
            "reserve": sum(row["split"] == "reserve" for row in pending)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pool, args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
