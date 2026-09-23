#!/usr/bin/env python3
"""Hash-verify a public shard and rerender its timed-out programs without model calls.

The replay is diagnostic evidence. It does not rewrite the original episode or
claim that the teacher saw this canvas and made subsequent corrections.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(BENCHMARK))
from run import render_program  # noqa: E402

DATASET = "CK0607/komorebi-painter-teachers"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def public_bytes(run_id: str) -> tuple[bytes, dict]:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid source run ID")
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/"
    with urlopen(base + f"main/runs/{run_id}/receipt.json?download=true", timeout=60) as stream:
        receipt = json.load(stream)
    if receipt.get("run_id") != run_id or not receipt.get("public_hash_verified"):
        raise ValueError("missing verified public receipt")
    expected_size = int(receipt["bundle_bytes"])
    if not 0 < expected_size <= 1_000_000_000:
        raise ValueError("archive size outside replay bound")
    revision = receipt["dataset_commit"]
    if receipt.get("representation") == "split-base64-text":
        paths = receipt["part_paths"]
        if len(paths) > 4096 or any(not p.startswith(f"runs/{run_id}/parts/") for p in paths):
            raise ValueError("invalid public archive parts")

        def fetch(path: str) -> bytes:
            with urlopen(base + revision + "/" + path + "?download=true", timeout=180) as stream:
                return base64.b64decode(b"".join(stream.read().split()), validate=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            archive = b"".join(pool.map(fetch, paths))
    elif receipt.get("representation") == "archive":
        with urlopen(base + revision + "/" + receipt["bundle_path"] + "?download=true", timeout=180) as stream:
            archive = stream.read(expected_size + 1)
    else:
        raise ValueError("unknown public archive representation")
    if len(archive) != expected_size or hashlib.sha256(archive).hexdigest() != receipt["bundle_sha256"]:
        raise ValueError("public archive hash mismatch")
    return archive, receipt


def timed_out_programs(archive: bytes) -> list[tuple[str, int, bytes, dict]]:
    selected = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = {m.name: m for m in bundle.getmembers() if m.isfile()}
        for name in sorted(members):
            if not re.fullmatch(r"episodes/[A-Za-z0-9._-]+/episode\.json", name):
                continue
            data = bundle.extractfile(members[name]).read()
            episode = json.loads(data)
            if episode.get("status") != "renderer_error" or not episode.get("turns"):
                continue
            turn = episode["turns"][-1]
            if (turn.get("render") or {}).get("error_code") != "render_timeout":
                continue
            number = int(turn["turn"])
            program_name = name.rsplit("/", 1)[0] + f"/turn-{number:02d}.program.js"
            if program_name not in members:
                raise ValueError(f"missing timeout program for {name}")
            program = bundle.extractfile(members[program_name]).read()
            if len(program) > 100_000:
                raise ValueError(f"program exceeds renderer size bound: {program_name}")
            selected.append((name.split("/")[1], number, program, {
                "model": episode["model"],
                "reference_id": episode["reference_id"],
                "original_episode_sha256": hashlib.sha256(data).hexdigest(),
                "original_status": episode["status"],
            }))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path)
    parser.add_argument("--browser-path", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 181 <= args.timeout <= 900:
        parser.error("replay timeout must be between 181 and 900 seconds")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    archive, source_receipt = public_bytes(args.source_run_id)
    programs = timed_out_programs(archive)
    if args.limit is not None:
        programs = programs[:args.limit]
    print(json.dumps({"source_run_id": args.source_run_id, "archive_sha256": source_receipt["bundle_sha256"],
                      "selected_timeouts": len(programs), "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return 0
    renderer_python = args.renderer_python or BENCHMARK.parents[2] / ".painter-cloud-runtime/renderer-env/bin/python"
    renderer = BENCHMARK.parents[1] / "vendor/integrations/watercolour/renderer.py"
    if not renderer_python.is_file() or not renderer.is_file():
        parser.error("prepared cloud renderer is missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, int] = {}
    for index, (job, turn, program, original) in enumerate(programs, 1):
        episode_dir = output / "episodes" / job
        episode_dir.mkdir(parents=True, exist_ok=True)
        source = episode_dir / f"turn-{turn:02d}.program.js"
        source.write_bytes(program)
        canvas = episode_dir / f"turn-{turn:02d}.png"
        result = render_program(root=output, source=source, output=canvas, renderer=renderer,
                                renderer_python=renderer_python, browser_path=args.browser_path,
                                timeout=args.timeout, run_as_user="painter", local=False)
        status = "valid" if result.get("valid") else str(result.get("error_code") or "invalid")
        statuses[status] = statuses.get(status, 0) + 1
        evidence = {"schema": "painter.renderer-replay.v1", "source_run_id": args.source_run_id,
                    "source_archive_sha256": source_receipt["bundle_sha256"], "source_turn": turn,
                    "program_sha256": hashlib.sha256(program).hexdigest(), "timeout_seconds": args.timeout,
                    "source": original, "result": result,
                    "limitation": "Offline render only; subsequent teacher turns were not generated from this canvas."}
        (episode_dir / "replay.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"progress": f"{index}/{len(programs)}", "job": job, "status": status,
                          "elapsed_seconds": result.get("elapsed_seconds")}), flush=True)
    summary = {"schema": "painter.renderer-replay-summary.v1", "source_run_id": args.source_run_id,
               "source_archive_sha256": source_receipt["bundle_sha256"], "timeout_seconds": args.timeout,
               "programs_selected": len(programs), "status_counts": statuses, "paid_model_calls": 0}
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
