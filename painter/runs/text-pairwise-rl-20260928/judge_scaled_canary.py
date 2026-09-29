"""Check the live two-order visual judge against a blank on one SFT painting."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import read_traces, _trace_data  # noqa: E402
from painter.text_pairwise_margin_judge import judge_pair  # noqa: E402


async def check(eval_root: Path, key_file: Path, output: Path) -> dict:
    traces, errors, files = read_traces(eval_root)
    if errors or not files:
        raise ValueError("baseline traces absent or unreadable")
    for trace in traces:
        data, info = _trace_data(trace)
        canvas = Path(info.get("final_canvas") or "/nonexistent")
        if info.get("final_canvas_valid") and canvas.is_file():
            break
    else:
        raise ValueError("no valid baseline canvas yet")
    brief = Path(data["reference"])
    output.mkdir(parents=True, exist_ok=True)
    blank = output / "blank.png"
    with Image.open(canvas) as image:
        Image.new("RGB", image.size, "white").save(blank)
    result = await judge_pair(brief=brief, anchor=blank, candidate=canvas,
                              key_file=key_file)
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {"id": data["task_id"], "score": result["score"],
               "order_gap": result.get("order_gap")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(**vars(args))), sort_keys=True))


if __name__ == "__main__":
    main()
