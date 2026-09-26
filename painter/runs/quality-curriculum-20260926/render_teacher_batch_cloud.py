#!/usr/bin/env python3
"""Render one public teacher500 source batch on a finite Codex Cloud Linux job."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "painter/benchmarks/openrouter-teachers-20260922"))
from run import render_program  # noqa: E402


DATASET = "CK0607/komorebi-painter-teachers"
OUT = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_bytes(commit: str, path: str) -> bytes:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{commit}/{path}?download=true"
    with urlopen(url, timeout=180) as source:
        return source.read(100_000_001)


def stage(batch: str, output: Path) -> tuple[dict, dict]:
    receipt_path = f"curricula/teacher500/{batch}/source-public.json"
    receipt = json.loads(public_bytes("main", receipt_path))
    if (receipt.get("batch") != batch or receipt.get("schema") != "painter.teacher500-public-source.v1"
            or receipt.get("anonymous_hash_verified") is not True):
        raise ValueError("public teacher source receipt mismatch")
    raw = public_bytes(receipt["dataset_commit"], receipt["path"])
    if sha(raw) != receipt["archive_sha256"] or len(raw) != receipt["archive_bytes"]:
        raise ValueError("public teacher source archive hash/size mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 101 or sum(m.size for m in members) > 100_000_000:
            raise ValueError("teacher source archive exceeds bounds")
        for member in members:
            parts = Path(member.name).parts
            if (not member.isfile() or member.size > 5_000_000
                    or (member.name != "manifest.json" and
                        (len(parts) != 2 or parts[0] not in {"programs", "inputs", "prompts"}))
                    or any(part in {".", ".."} for part in parts)):
                raise ValueError(f"unexpected teacher source member: {member.name}")
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest.get("schema") != "painter.teacher500-source.v1"
            or manifest.get("batch") != batch or manifest.get("count") != receipt["count"]):
        raise ValueError("teacher source manifest mismatch")
    for row in manifest["rows"]:
        if row["mode"] not in {"text_to_image", "image_to_image"}:
            raise ValueError("unknown teacher input mode")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,96}", row["id"]):
            raise ValueError("unsafe teacher example ID")
        for name, field in (("input", "input_sha256"), ("program", "program_sha256")):
            member = row[name]
            suffix = (".txt" if row["mode"] == "text_to_image" else ".jpg") if name == "input" else ".js"
            expected = f"{'inputs' if name == 'input' else 'programs'}/{row['id']}{suffix}"
            if member != expected:
                raise ValueError("unexpected teacher file path")
            if sha((output / member).read_bytes()) != row[field]:
                raise ValueError(f"teacher source member hash mismatch: {row['id']} {name}")
        if row.get("prompt"):
            expected = f"prompts/{row['id']}.txt"
            if row["prompt"] != expected or sha((output / expected).read_bytes()) != row["prompt_sha256"]:
                raise ValueError(f"teacher prompt path/hash mismatch: {row['id']}")
    (output / "source-public.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return manifest, receipt


def render(batch: str, run_id: str, timeout: int) -> dict:
    runtime = ROOT / ".painter-cloud-runtime"
    python = runtime / "renderer-env/bin/python"
    browser = runtime / "browsers"
    renderer = ROOT / "painter/vendor/integrations/watercolour/renderer.py"
    if sys.platform != "linux" or not python.is_file() or not browser.is_dir():
        raise RuntimeError("Codex Cloud Linux renderer setup is required")
    output = OUT / run_id
    manifest, receipt = stage(batch, output)
    statuses = []
    for index, row in enumerate(manifest["rows"], 1):
        episode = output / "episodes" / row["id"]
        episode.mkdir(parents=True)
        program = output / row["program"]
        canvas = episode / "canvas.png"
        try:
            result = render_program(root=output, source=program, output=canvas,
                                    renderer=renderer, renderer_python=python,
                                    browser_path=browser, timeout=timeout,
                                    run_as_user="painter", local=False)
        except Exception as exc:
            result = {"valid": False, "error_code": f"renderer_exception_{type(exc).__name__}",
                      "elapsed_seconds": None}
        status = {"id": row["id"], "mode": row["mode"],
                  "program_sha256": row["program_sha256"],
                  "input_sha256": row["input_sha256"],
                  "canvas_sha256": sha(canvas.read_bytes()) if canvas.is_file() else None,
                  "valid": result.get("valid") is True, "error_code": result.get("error_code"),
                  "elapsed_seconds": result.get("elapsed_seconds"),
                  "visual_review_status": "pending"}
        (episode / "program.js").write_bytes(program.read_bytes())
        suffix = ".txt" if row["mode"] == "text_to_image" else ".jpg"
        (episode / f"input{suffix}").write_bytes((output / row["input"]).read_bytes())
        if row.get("prompt"):
            (episode / "prompt.txt").write_bytes((output / row["prompt"]).read_bytes())
        (episode / "render-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        statuses.append(status)
        print(json.dumps({"progress": f"{index}/{manifest['count']}", **status}, sort_keys=True), flush=True)
    summary = {"schema": "painter.teacher500-render.v1", "batch": batch, "run_id": run_id,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_dataset_commit": receipt["dataset_commit"],
               "source_bundle_sha256": receipt["archive_sha256"],
               "renderer_sha256": sha(renderer.read_bytes()), "timeout_seconds": timeout,
               "teacher_model": manifest["teacher_model"],
               "teacher_reasoning_effort": manifest["teacher_reasoning_effort"],
               "count": len(statuses), "valid": sum(row["valid"] for row in statuses),
               "statuses": statuses, "visual_review_status": "pending"}
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("run_id")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if (not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", args.batch)
            or not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", args.run_id)
            or not 1 <= args.timeout <= 900):
        parser.error("invalid batch, run ID or timeout")
    summary = render(args.batch, args.run_id, args.timeout)
    print(json.dumps({"batch": args.batch, "valid": summary["valid"], "count": summary["count"]}))


if __name__ == "__main__":
    main()
