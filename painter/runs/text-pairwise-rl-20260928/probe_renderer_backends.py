"""Bounded Linux replay of identical programs across renderer backends.

This is a diagnostic, not an evaluation protocol change. It preserves every
receipt and canvas, measures elapsed and painting time, and compares output
pixels against the pinned SwiftShader rendering when both renders are valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a: Path, b: Path) -> dict:
    with Image.open(a) as source, Image.open(b) as target:
        diff = ImageChops.difference(source.convert("RGB"), target.convert("RGB"))
        stat = ImageStat.Stat(diff)
        return {"mean_absolute_channel_difference": sum(stat.mean) / 3,
                "max_absolute_channel_difference": max(channel[1] for channel in diff.getextrema()),
                "identical_pixels": diff.getbbox() is None}


def run(*, sources: Path, renderer: Path, python: Path, browsers: Path,
        output: Path, timeout: int = 180, capture_mode: str = "data_url") -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= timeout <= 180:
        raise ValueError("timeout must be between 1 and 180 seconds")
    if capture_mode not in {"data_url", "to_blob", "screenshot"}:
        raise ValueError("unsupported capture mode")
    rows = json.loads((sources / "manifest.json").read_text())["rows"]
    if not rows:
        raise ValueError("no input programs")
    output.mkdir(parents=True)
    os.chmod(output, 0o777)
    results = []
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers.resolve())
    backends = ["swiftshader", "gl", "vulkan"]
    backend_available = {name: True for name in backends}
    for index, row in enumerate(rows):
        source = sources / row["file"]
        if digest(source) != row["sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        canvases: dict[str, Path] = {}
        for backend in backends:
            if not backend_available[backend]:
                continue
            folder = output / f"{index:02d}-{row['name']}-{backend}"
            folder.mkdir()
            os.chmod(folder, 0o777)
            canvas = folder / "painting.png"
            # Keep the venv's bin/python path. Resolving its symlink selects
            # /usr/bin/python and loses the Playwright installation.
            command = ["/usr/sbin/runuser", "-u", "painter", "--", str(python.absolute()),
                       str(renderer.resolve()), str(source.resolve()), "--output",
                       str(canvas), "--timeout", str(timeout), "--backend", backend,
                       "--capture", capture_mode]
            started = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=timeout + 25, env=env, check=False)
            elapsed = time.monotonic() - started
            receipt_path = canvas.with_suffix(".json")
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {
                "valid": False, "error_code": "missing_receipt"}
            valid = bool(receipt.get("valid") and canvas.is_file())
            if valid:
                canvases[backend] = canvas
            if index == 0 and backend != "swiftshader" and not valid:
                backend_available[backend] = False
            record = {"name": row["name"], "backend": backend,
                      "capture_mode": capture_mode,
                      "source_sha256": row["sha256"], "valid": valid,
                      "elapsed_seconds": elapsed,
                      "painting_seconds": receipt.get("painting_seconds"),
                      "phase_seconds": receipt.get("phase_seconds"),
                      "gpu": receipt.get("gpu"),
                      "error_code": receipt.get("error_code"),
                      "error": str(receipt.get("error") or receipt.get("errors") or "")[:800],
                      "canvas_sha256": digest(canvas) if valid else None,
                      "historical_painting_seconds": row.get("historical_painting_seconds")}
            results.append(record)
            (output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: record[k] for k in ("name", "backend", "valid", "elapsed_seconds", "painting_seconds", "error_code")}), flush=True)
        if "swiftshader" in canvases:
            for record in results:
                if record["name"] == row["name"] and record["backend"] != "swiftshader" and record["valid"]:
                    record["pixel_comparison_to_swiftshader"] = compare(
                        canvases["swiftshader"], canvases[record["backend"]])
    summary = {"schema": "watercolour-backend-probe-v1", "source_count": len(rows),
               "timeout_seconds": timeout, "capture_mode": capture_mode,
               "results": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--browsers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--capture-mode", choices=["data_url", "to_blob", "screenshot"],
                        default="data_url")
    args = parser.parse_args()
    summary = run(**vars(args))
    print(json.dumps({"source_count": summary["source_count"],
                      "results": len(summary["results"])}))


if __name__ == "__main__":
    main()
