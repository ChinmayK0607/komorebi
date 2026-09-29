"""Freeze first-sample SFT canvases for the independent reserve comparison.

Eligibility depends only on the SFT sample. Keep every reserve case in the
audit, including invalid or censored SFT outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usable(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.width < 128 or image.height < 128:
                return False
            return max(ImageStat.Stat(image.resize((128, 128))).stddev) >= 2.0
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.pool.read_text())["cases"]
    if len(cases) != len({case["id"] for case in cases}):
        raise ValueError("duplicate reserve ID")
    if args.output.exists():
        raise FileExistsError(args.output)
    traces, errors, files = read_traces(args.eval_root)
    if errors or not files:
        raise ValueError(f"missing or corrupt SFT traces: {errors[:3]}")
    by_id: dict[str, list[dict]] = {case["id"]: [] for case in cases}
    for trace in traces:
        data, info = _trace_data(trace)
        ident = data.get("task_id") or data.get("name")
        if ident in by_id:
            by_id[ident].append(info)

    args.output.mkdir(parents=True)
    anchors = args.output / "anchors"
    anchors.mkdir()
    eligible, audit = [], []
    for case in cases:
        brief = Path(case["reference"])
        if sha(brief) != case["reference_sha256"]:
            raise ValueError(f"brief hash mismatch: {case['id']}")
        episodes = by_id[case["id"]]
        reason = "trace_count_" + str(len(episodes))
        if len(episodes) == 1:
            info = episodes[0]
            if info.get("service_failure"):
                reason = "service_failure"
            elif not info.get("final_canvas_valid"):
                reason = "invalid_model_output"
            else:
                raw = info.get("final_canvas")
                canvas = Path(raw) if isinstance(raw, str) and raw else None
                if canvas is not None and usable(canvas):
                    destination = anchors / (case["id"] + ".png")
                    shutil.copyfile(canvas, destination)
                    eligible.append({**case, "opponent": str(destination.resolve()),
                                     "opponent_sha256": sha(destination),
                                     "reference": str(brief.resolve()),
                                     "mode": "fresh", "max_turns": 3})
                    reason = "valid"
                else:
                    reason = "missing_canvas"
        audit.append({"id": case["id"], "family": case["family"], "status": reason})
    manifest = args.output / "validation-manifest.json"
    manifest.write_text(json.dumps({"cases": eligible}, indent=2, sort_keys=True) + "\n")
    receipt = {"pool_sha256": sha(args.pool), "source": str(args.eval_root),
               "expected": len(cases), "sft_valid": len(eligible),
               "manifest_sha256": sha(manifest), "cases": audit}
    (args.output / "audit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "cases"}, sort_keys=True))


if __name__ == "__main__":
    main()
