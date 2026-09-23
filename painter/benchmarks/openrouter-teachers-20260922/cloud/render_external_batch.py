#!/usr/bin/env python3
"""Render a small, hash-pinned set of externally generated teacher programs.

This is a provider-free rendering step. The generating model never sees these
canvases unless a separate, explicit revision turn is run afterward.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(BENCHMARK))
from run import render_program  # noqa: E402

DATASET = "CK0607/komorebi-painter-teachers"
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")
SAFE_PATH = re.compile(r"[a-zA-Z0-9._/-]+\Z")
SAFE_ID = re.compile(r"[a-zA-Z0-9._-]+\Z")


def public_file(revision: str, path: str, expected: str) -> bytes:
    if not HEX_SHA.fullmatch(revision) or not HEX_SHA.fullmatch(expected):
        raise ValueError("a source hash or revision is invalid")
    if not SAFE_PATH.fullmatch(path) or path.startswith("/") or ".." in Path(path).parts:
        raise ValueError("unsafe public source path")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}?download=true"
    with urlopen(url, timeout=180) as stream:
        data = stream.read(100_001)
    if len(data) > 100_000 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("public source hash mismatch or size limit exceeded")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="immutable HF dataset commit")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path)
    parser.add_argument("--browser-path", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.timeout <= 900:
        parser.error("workers or timeout outside allowed range")
    manifest = json.loads(public_file(args.revision, args.manifest_path, args.manifest_sha256))
    if manifest.get("schema") != "painter.external-teacher-programs.v1":
        parser.error("unexpected input manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 12:
        parser.error("manifest needs 1–12 entries")
    ids = [entry.get("reference_id") for entry in entries]
    if any(not isinstance(item, str) or not SAFE_ID.fullmatch(item) for item in ids) or len(set(ids)) != len(ids):
        parser.error("invalid or duplicate reference IDs")
    renderer_python = args.renderer_python or BENCHMARK.parents[2] / ".painter-cloud-runtime/renderer-env/bin/python"
    renderer = BENCHMARK.parents[1] / "vendor/integrations/watercolour/renderer.py"
    if not renderer_python.is_file() or not renderer.is_file():
        parser.error("prepared Linux renderer is missing")
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; refusing to overwrite evidence")
    output.mkdir(parents=True)
    jobs = []
    for entry in entries:
        reference_id = entry["reference_id"]
        reference = BENCHMARK / "references" / f"{reference_id}.jpg"
        if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != entry["reference_sha256"]:
            raise ValueError(f"reference hash mismatch: {reference_id}")
        program = public_file(args.revision, entry["program_path"], entry["program_sha256"])
        episode = output / "episodes" / reference_id
        episode.mkdir(parents=True)
        source = episode / "turn-01.program.js"
        source.write_bytes(program)
        jobs.append((reference_id, source, episode / "turn-01.png", entry))

    def render_one(job: tuple[str, Path, Path, dict]) -> tuple[str, str, float | None]:
        reference_id, source, canvas, entry = job
        result = render_program(root=output, source=source, output=canvas, renderer=renderer,
                                renderer_python=renderer_python, browser_path=args.browser_path,
                                timeout=args.timeout, run_as_user="painter", local=False)
        status = "valid" if result.get("valid") else str(result.get("error_code") or "invalid")
        record = {"schema": "painter.external-teacher-render.v1", "reference_id": reference_id,
                  "model": manifest["model"], "reasoning_effort": manifest["reasoning_effort"],
                  "source_dataset_commit": args.revision, "source_manifest_sha256": args.manifest_sha256,
                  "program_sha256": entry["program_sha256"], "status": status, "render": result,
                  "limitation": "Offline initial render; the teacher did not see this canvas."}
        (canvas.parent / "episode.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return reference_id, status, result.get("elapsed_seconds")

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_one, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            reference_id, status, elapsed_seconds = future.result()
            counts[status] = counts.get(status, 0) + 1
            print(json.dumps({"progress": f"{index}/{len(jobs)}", "reference_id": reference_id,
                              "status": status, "elapsed_seconds": elapsed_seconds}), flush=True)
    summary = {"schema": "painter.external-teacher-render-summary.v1", "model": manifest["model"],
               "reasoning_effort": manifest["reasoning_effort"], "source_dataset_commit": args.revision,
               "source_manifest_sha256": args.manifest_sha256, "programs_selected": len(jobs),
               "render_workers": args.workers, "timeout_seconds": args.timeout,
               "status_counts": counts, "paid_model_calls": 0}
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
