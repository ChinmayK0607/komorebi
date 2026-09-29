"""Copy every saved turn canvas from one native evaluation for public review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import read_traces, _trace_data  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(eval_root: Path, output: Path) -> dict:
    traces, errors, files = read_traces(eval_root)
    if errors or not files:
        raise ValueError(f"missing or corrupt native traces: {errors[:2]}")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    seen = set()
    output.mkdir(parents=True)
    for trace in traces:
        data, info = _trace_data(trace)
        ident = data.get("task_id") or data.get("name")
        if (not isinstance(ident, str) or
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ident) is None or
                ident in seen):
            raise ValueError("duplicate or unsafe evaluation task ID")
        seen.add(ident)
        paths = []
        for turn in info.get("turns", []):
            raw = turn.get("canvas")
            if not isinstance(raw, str) or not raw:
                continue
            source = Path(raw)
            if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".png":
                raise ValueError(f"missing or unsafe turn canvas: {ident}")
            expected = turn.get("canvas_sha256")
            if expected and sha(source) != expected:
                raise ValueError(f"changed turn canvas: {ident}")
            relative = Path(ident) / f"turn-{turn['turn']}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            paths.append({"path": relative.as_posix(), "sha256": sha(target)})
        final = info.get("final_canvas")
        if isinstance(final, str) and final:
            source = Path(final)
            if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".png":
                raise ValueError(f"missing or unsafe final canvas: {ident}")
            relative = Path(ident) / "final.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            paths.append({"path": relative.as_posix(), "sha256": sha(target)})
        rows.append({"id": ident, "valid": bool(info.get("final_canvas_valid")),
                     "canvases": paths})
    manifest = {"schema": "painter-native-eval-canvases-v1",
                "source_trace_files": [str(path) for path in files], "cases": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"cases": len(rows), "valid": sum(row["valid"] for row in rows),
            "canvases": sum(len(row["canvases"]) for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.eval_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
