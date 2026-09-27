"""Restore missing procedural reference PNGs from verified p5.brush sources.

The new PNG hashes define a new curriculum revision. The old hashes are kept
in a receipt; they are never silently replaced in the historical manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run(old: Path, out: Path, renderer: Path, python: Path, browsers: Path,
        workers: int, batch_size: int, limit: int = 0) -> dict:
    source = json.loads(old.read_text())
    photos = []
    for row in source["rows"]:
        if row["source_kind"] != "photo":
            continue
        revised = dict(row)
        original = (old.parent / row["reference"]).resolve(strict=True)
        revised["reference"] = os.path.relpath(original, out)
        photos.append(revised)
    procedural = [row for row in source["rows"] if row["source_kind"] == "procedural"]
    if limit:
        procedural = procedural[:limit]
    inputs, images = out / "render-inputs", out / "references"
    inputs.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "painter:painter", str(inputs), str(images)], check=True)
    tasks = []
    for row in procedural:
        program = (old.parent / row["source_path"]).resolve(strict=True)
        if sha(program) != row["source_sha256"]:
            raise ValueError(f"program hash changed: {row['id']}")
        staged = inputs / f"{row['id']}.js"
        if not staged.exists():
            shutil.copy2(program, staged)
        if sha(staged) != row["source_sha256"]:
            raise ValueError(f"staged program changed: {row['id']}")
        target = images / f"{row['id']}.png"
        receipt = target.with_suffix(".json")
        if target.is_file() and receipt.is_file():
            data = json.loads(receipt.read_text())
            if data.get("valid") and data.get("png_sha256") == sha(target):
                continue
            raise ValueError(f"inconsistent prior render: {row['id']}")
        if receipt.is_file() and not target.exists():
            prior = json.loads(receipt.read_text())
            if prior.get("valid") or prior.get("error_code") != "worker_failure":
                raise ValueError(f"non-retryable render receipt: {row['id']}")
            receipt.unlink()
        tasks.append(staged)
    began = time.monotonic()
    for index in range(0, len(tasks), batch_size):
        batch = tasks[index:index + batch_size]
        command = ["/usr/sbin/runuser", "-u", "painter", "--", str(python),
                   str(renderer), *map(str, batch), "--output-dir", str(images),
                   "--workers", str(workers), "--timeout", "120", "--backend", "swiftshader"]
        env = os.environ.copy()
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers)
        call = subprocess.run(command, capture_output=True, text=True, env=env,
                              timeout=180 + 120 * ((len(batch) + workers - 1) // workers))
        receipts = json.loads(call.stdout)
        failed = [row.get("source") for row in receipts if not row.get("valid")]
        if call.returncode or failed:
            reasons = [(row.get("source"), row.get("error_code"), str(row.get("error"))[-220:])
                       for row in receipts if not row.get("valid")]
            raise RuntimeError(f"render batch {index // batch_size} failed: {reasons[:5]}")
        print(json.dumps({"rendered": min(index + len(batch), len(tasks)),
                          "requested": len(tasks),
                          "elapsed_seconds": round(time.monotonic() - began, 1)}), flush=True)
    replacement = {}
    changed = []
    for row in procedural:
        image = images / f"{row['id']}.png"
        receipt = image.with_suffix(".json")
        if not image.is_file() or not receipt.is_file():
            raise FileNotFoundError(image)
        check = json.loads(receipt.read_text())
        new_sha = sha(image)
        if not check.get("valid") or check.get("png_sha256") != new_sha:
            raise ValueError(f"render receipt mismatch: {row['id']}")
        revised = dict(row)
        revised["reference"] = f"references/{row['id']}.png"
        revised["reference_sha256"] = new_sha
        revised["source_image_sha256"] = new_sha
        replacement[row["id"]] = revised
        changed.append({"id": row["id"], "original_sha256": row["reference_sha256"],
                        "new_sha256": new_sha, "program_sha256": row["source_sha256"]})
    rows = [replacement[row["id"]] for row in source["rows"] if row["source_kind"] == "procedural" and row["id"] in replacement]
    rows += photos
    rows.sort(key=lambda row: row["order"])
    for row in rows:
        p = out / row["reference"]
        if sha(p) != row["reference_sha256"]:
            raise ValueError(f"output reference mismatch: {row['id']}")
    dataset = dict(source)
    dataset["rows"] = rows
    dataset["render_revision"] = "restored-p5-brush-node-render-v1"
    save(out / "source-manifest.json", dataset)
    summary = {"old_manifest_sha256": sha(old), "new_manifest_sha256": sha(out / "source-manifest.json"),
               "photos": len(photos), "procedural": len(procedural), "total": len(rows),
               "rendered_in_this_call": len(tasks), "elapsed_seconds": time.monotonic() - began,
               "changed_reference_hashes": sum(x["original_sha256"] != x["new_sha256"] for x in changed),
               "new_reference_hashes": changed}
    save(out / "render-revision.json", summary)
    return {k: v for k, v in summary.items() if k != "new_reference_hashes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--browsers", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.old.resolve(), args.out.resolve(), args.renderer.resolve(),
                         args.python.absolute(), args.browsers.resolve(),
                         args.workers, args.batch_size, args.limit)), flush=True)


if __name__ == "__main__":
    main()
