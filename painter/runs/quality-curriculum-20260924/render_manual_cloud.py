#!/usr/bin/env python3
"""Render the ten reviewed manual programs on Codex Cloud's CPU renderer.

Run after the existing cloud/setup.sh has prepared the pinned renderer. This
does not call a model, train, grade, or admit a painting to SFT.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
BENCHMARK = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
sys.path.insert(0, str(BENCHMARK))
from run import render_program  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    runtime = ROOT / ".painter-cloud-runtime"
    python = runtime / "renderer-env/bin/python"
    browser = runtime / "browsers"
    renderer = ROOT / "painter/vendor/integrations/watercolour/renderer.py"
    if not python.is_file() or not browser.is_dir() or not renderer.is_file():
        raise SystemExit("Run painter/benchmarks/openrouter-teachers-20260922/cloud/setup.sh first")
    subprocess.run([sys.executable, str(RUN / "manual_paintings.py")], check=True)
    manifest = json.loads((RUN / "manual-program-manifest.json").read_text())
    if manifest.get("count") != 10 or len(manifest.get("programs", [])) != 10:
        raise SystemExit("Expected exactly ten programs")
    output = ROOT / "painter/collected/quality-curriculum-20260924/manual-cloud-render-v1"
    output.mkdir(parents=True, exist_ok=True)
    statuses = []
    for index, row in enumerate(manifest["programs"], 1):
        ident = row["reference_id"]
        source = ROOT / row["program_path"]
        if sha(source) != row["program_sha256"]:
            raise RuntimeError(f"Program hash mismatch: {ident}")
        episode = output / "episodes" / ident
        episode.mkdir(parents=True, exist_ok=True)
        archived_source = episode / "turn-01.program.js"
        archived_source.write_bytes(source.read_bytes())
        canvas = episode / "turn-01.png"
        result = render_program(root=output, source=archived_source, output=canvas,
                                renderer=renderer, renderer_python=python,
                                browser_path=browser, timeout=600,
                                run_as_user="painter", local=False)
        status = {
            "reference_id": ident,
            "reference_sha256": row["reference_sha256"],
            "program_sha256": row["program_sha256"],
            "valid": bool(result.get("valid")),
            "error_code": result.get("error_code"),
            "canvas_sha256": sha(canvas) if canvas.is_file() else None,
            "elapsed_seconds": result.get("elapsed_seconds"),
        }
        (episode / "render-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        statuses.append(status)
        print(json.dumps({"progress": f"{index}/10", **status}, sort_keys=True), flush=True)
    summary = {
        "schema": "painter.manual-render.v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "program_manifest_sha256": sha(RUN / "manual-program-manifest.json"),
        "renderer_sha256": sha(renderer),
        "programs": len(statuses),
        "valid": sum(row["valid"] for row in statuses),
        "statuses": statuses,
        "visual_review_status": "pending",
        "model_calls": 0,
    }
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "valid": summary["valid"], "total": 10}), flush=True)
    return 0 if summary["valid"] == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
