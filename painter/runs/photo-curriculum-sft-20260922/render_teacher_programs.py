"""Render teacher ``program.js`` files with the pinned 180-second deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(folder: Path, renderer: Path, renderer_python: Path) -> dict:
    source = folder / "program.js"
    output = folder / "program.png"
    receipt = folder / "program.json"
    source_hash = sha(source)
    if output.is_file() and receipt.is_file():
        try:
            old = json.loads(receipt.read_text())
            if old.get("valid") and old.get("source_sha256") == source_hash and old.get("png_sha256") == sha(output):
                return {"id": folder.name, "valid": True, "resumed": True, "source_sha256": source_hash, "png_sha256": sha(output)}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if output.exists():
        output.rename(output.with_name(output.name + f".previous-{time.time_ns()}"))
    if receipt.exists():
        receipt.rename(receipt.with_name(receipt.name + f".previous-{time.time_ns()}"))
    command = ["/usr/sbin/runuser", "-u", "painter", "--", str(renderer_python), str(renderer), str(source),
               "--output", str(output), "--timeout", "180", "--backend", "swiftshader"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=210, check=False)
        result = json.loads(receipt.read_text()) if receipt.is_file() else {"valid": False, "error": completed.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        result = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    if result.get("valid") and output.is_file():
        result["source_sha256"] = result.get("source_sha256") or source_hash
        result["png_sha256"] = sha(output)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    return {"id": folder.name, "valid": bool(result.get("valid")), "resumed": False,
            "source_sha256": source_hash, "png_sha256": sha(output) if output.is_file() else None,
            "error": result.get("error") or result.get("errors")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teachers", type=Path, required=True)
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--python", dest="renderer_python", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if sys.platform != "linux":
        raise SystemExit("teacher rendering runs on the Linux node")
    folders = sorted(folder for folder in args.teachers.iterdir() if (folder / "program.js").is_file())
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        # ``resolve()`` follows a venv's python symlink to the system binary,
        # which loses the node-local Playwright/PIL environment.  Preserve
        # the supplied executable path while still making it absolute.
        renderer_path = args.renderer.absolute()
        renderer_python_path = args.renderer_python.absolute()
        futures = {pool.submit(render, folder, renderer_path, renderer_python_path): folder.name for folder in folders}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"id": futures[future], "valid": False, "error": f"{type(exc).__name__}: {exc}"})
            (args.teachers / "render-progress.json").write_text(json.dumps({
                "completed": len(results), "expected": len(folders), "valid": sum(bool(row.get("valid")) for row in results),
            }, indent=2) + "\n")
    results.sort(key=lambda row: row["id"])
    (args.teachers / "render-summary.json").write_text(json.dumps({
        "expected": len(folders), "valid": sum(bool(row.get("valid")) for row in results), "records": results,
        "deadline_seconds": 180, "quality_claim": False,
    }, indent=2) + "\n")
    print(json.dumps({"expected": len(folders), "valid": sum(bool(row.get("valid")) for row in results)}))


if __name__ == "__main__":
    main()
