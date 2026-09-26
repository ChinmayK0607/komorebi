#!/usr/bin/env python3
"""Render twelve new programs against already-public COCO128 photo bytes."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

from prepare_coco128_overlay import BATCH, DATASET, old_archive, public_bytes, sha
from render_teacher_batch_cloud import OUT, ROOT, render_rows


def stage(output: Path) -> tuple[dict, dict]:
    remote = f"curricula/teacher600/{BATCH}/source-public.json"
    public = json.loads(public_bytes("main", remote))
    if (public.get("schema") != "painter.teacher600-public-program-overlay.v1"
            or public.get("batch") != BATCH or public.get("count") != 12
            or public.get("anonymous_hash_verified") is not True
            or public.get("photos_in_archive") is not False):
        raise ValueError("program-only public receipt mismatch")
    raw = public_bytes(public["dataset_commit"], public["path"])
    if len(raw) != public["archive_bytes"] or sha(raw) != public["archive_sha256"]:
        raise ValueError("program-only archive hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    prior_cache = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if (manifest.get("schema") != "painter.teacher600-program-overlay.v1"
                or manifest.get("count") != 12 or len(manifest.get("rows", [])) != 12
                or manifest.get("photos_in_archive") is not False):
            raise ValueError("program-only manifest mismatch")
        for row in manifest["rows"]:
            ident = row["id"]
            if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,96}", ident)
                    or row["program"] != f"programs/{ident}.js"
                    or row["mode"] != "image_to_image"):
                raise ValueError("unsafe program-only row")
            program = archive.extractfile(row["program"]).read()
            if sha(program) != row["program_sha256"]:
                raise ValueError(f"program mismatch: {ident}")
            prior = row["prior_source"]
            key = prior["bundle_sha256"]
            if key not in prior_cache:
                prior_cache[key] = old_archive(prior)
            image_name = row["prior_reference_member"]
            if image_name != f"references/coco128-{ident.removeprefix('astra-ref-seed-')}.jpg":
                raise ValueError("unexpected prior reference member")
            photo = prior_cache[key].extractfile(image_name).read()
            if sha(photo) != row["input_sha256"]:
                raise ValueError(f"prior reference mismatch: {ident}")
            program_path = output / row["program"]
            input_path = output / "inputs" / f"{ident}.jpg"
            program_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.parent.mkdir(parents=True, exist_ok=True)
            program_path.write_bytes(program)
            input_path.write_bytes(photo)
            row["input"] = f"inputs/{ident}.jpg"
    (output / "source-public.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    return manifest, public


def render(run_id: str, workers: int) -> dict:
    runtime = ROOT / ".painter-cloud-runtime"
    python = runtime / "renderer-env/bin/python"
    browser = runtime / "browsers"
    renderer = ROOT / "painter/vendor/integrations/watercolour/renderer.py"
    if sys.platform != "linux" or not python.is_file() or not browser.is_dir():
        raise RuntimeError("Codex Cloud Linux renderer setup is required")
    output = OUT / run_id
    manifest, public = stage(output)
    started = time.monotonic()
    statuses = render_rows(output, manifest["rows"], renderer, python, browser, 600, workers)
    # The exact JPEGs already exist in older public source bundles. Do not
    # republish them from this rights-incomplete campaign source.
    for row in manifest["rows"]:
        (output / "episodes" / row["id"] / "input.jpg").unlink()
    summary = {"schema": "painter.teacher500-render.v1", "batch": BATCH,
               "run_id": run_id, "source_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_dataset_commit": public["dataset_commit"],
               "source_bundle_sha256": public["archive_sha256"],
               "renderer_sha256": sha(renderer.read_bytes()),
               "timeout_seconds": 600, "workers": workers,
               "wall_seconds": round(time.monotonic() - started, 3),
               "teacher_model": manifest["teacher_model"],
               "teacher_reasoning_effort": manifest["teacher_reasoning_effort"],
               "count": len(statuses), "valid": sum(row["valid"] for row in statuses),
               "statuses": statuses, "visual_review_status": "pending",
               "reference_reused_from_public_prior": True,
               "reference_public_source_by_id": {
                   row["id"]: row["prior_source"] for row in manifest["rows"]}}
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", args.run_id) or not 1 <= args.workers <= 8:
        parser.error("invalid run ID or workers")
    summary = render(args.run_id, args.workers)
    print(json.dumps({"batch": BATCH, "valid": summary["valid"], "count": summary["count"]}))


if __name__ == "__main__":
    main()
