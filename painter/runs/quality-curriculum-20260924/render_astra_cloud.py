#!/usr/bin/env python3
"""Render one finite Astra candidate shard on a Codex Cloud Linux VM.

The teacher programs are public hash-verified inputs. This creates evidence for
visual review; validity alone never admits a painting to SFT or RL preference.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "painter/benchmarks/openrouter-teachers-20260922"))
from run import render_program  # noqa: E402

DATASET = "CK0607/komorebi-painter-teachers"
OUT = ROOT / "painter/collected/quality-curriculum-20260924"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_bytes(commit: str, path: str) -> bytes:
    with urlopen(f"https://huggingface.co/datasets/{DATASET}/resolve/{commit}/{path}?download=true", timeout=180) as source:
        return source.read()


def stage(shard: str, wave: int) -> tuple[Path, dict, dict]:
    receipt_path = f"curricula/astra-high-wave{wave}/{shard}/source-receipt.json"
    receipt = json.loads(public_bytes("main", receipt_path))
    if (receipt.get("shard") != shard or receipt.get("wave", 1) != wave
            or receipt.get("public_hash_verified") is not True):
        raise ValueError("source receipt is not verified for this shard")
    raw = base64.b64decode(public_bytes(receipt["dataset_commit"], receipt["bundle_path"]).strip(), validate=True)
    if sha(raw) != receipt["bundle_sha256"] or len(raw) != receipt["bundle_bytes"]:
        raise ValueError("source archive hash or byte count mismatch")
    output = OUT / f"astra-high-wave{wave}-cloud" / shard
    if output.exists() and list(output.glob("episodes/*/render-status.json")):
        raise ValueError("prior render evidence exists; use a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > 80:
            raise ValueError("too many source archive members")
        for member in members:
            parts = Path(member.name).parts
            if (not member.isfile() or member.size > 1_000_000 or len(parts) != 2 and member.name not in {"manifest.json", "agent-notes.json"}
                    or any(part in {".", ".."} for part in parts)
                    or (member.name not in {"manifest.json", "agent-notes.json"} and parts[0] not in {"references", "programs", "prior"})):
                raise ValueError(f"unexpected source archive member: {member.name}")
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(member).read())
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest.get("shard") != shard or manifest.get("wave", 1) != wave
            or manifest.get("count") != receipt["count"]):
        raise ValueError("source manifest disagrees with public receipt")
    notes = output / "agent-notes.json"
    if (sha(notes.read_bytes()) if notes.is_file() else None) != manifest.get("agent_notes_sha256"):
        raise ValueError("agent analysis notes hash mismatch")
    for row in manifest["rows"]:
        for field, digest in (("reference", "reference_sha256"), ("program", "program_sha256"),
                              ("prior_canvas", "prior_canvas_sha256"), ("prior_program", "prior_program_sha256")):
            if row.get(field) is None:
                if row.get(digest) is not None:
                    raise ValueError(f"incomplete prior evidence: {field}")
                continue
            if sha((output / row[field]).read_bytes()) != row[digest]:
                raise ValueError(f"source member hash mismatch: {row['reference_id']} {field}")
    (output / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return output, manifest, receipt


def render(shard: str, wave: int, timeout: int) -> dict:
    runtime = ROOT / ".painter-cloud-runtime"
    python = runtime / "renderer-env/bin/python"
    browser = runtime / "browsers"
    renderer = ROOT / "painter/vendor/integrations/watercolour/renderer.py"
    if sys.platform != "linux" or not python.is_file() or not browser.is_dir():
        raise RuntimeError("Codex Cloud Linux renderer setup is required")
    output, manifest, receipt = stage(shard, wave)
    statuses = []
    for index, row in enumerate(manifest["rows"], 1):
        ident = row["reference_id"]
        episode = output / "episodes" / ident
        episode.mkdir(parents=True, exist_ok=True)
        source = output / row["program"]
        target = episode / "turn-01.png"
        result = render_program(root=output, source=source, output=target,
                                renderer=renderer, renderer_python=python,
                                browser_path=browser, timeout=timeout,
                                run_as_user="painter", local=False)
        canvas_sha = sha(target.read_bytes()) if target.is_file() else None
        status = {"reference_id": ident, "reference_sha256": row["reference_sha256"],
                  "program_sha256": row["program_sha256"], "prior_canvas_sha256": row["prior_canvas_sha256"],
                  "valid": result.get("valid") is True, "canvas_sha256": canvas_sha,
                  "error_code": result.get("error_code"), "elapsed_seconds": result.get("elapsed_seconds")}
        (episode / "turn-01.program.js").write_bytes(source.read_bytes())
        (episode / "reference.jpg").write_bytes((output / row["reference"]).read_bytes())
        if row["prior_canvas"]:
            (episode / "prior.png").write_bytes((output / row["prior_canvas"]).read_bytes())
            (episode / "prior.program.js").write_bytes((output / row["prior_program"]).read_bytes())
        (episode / "render-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        statuses.append(status)
        print(json.dumps({"progress": f"{index}/{manifest['count']}", **status}, sort_keys=True), flush=True)
    summary = {"schema": "painter.astra-high-render.v1", "wave": wave, "shard": shard,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_bundle_sha256": receipt["bundle_sha256"], "source_dataset_commit": receipt["dataset_commit"],
               "renderer_sha256": sha(renderer.read_bytes()), "timeout_seconds": timeout,
               "model_calls_on_cloud": 0, "teacher_model": manifest["teacher_model"],
               "teacher_reasoning_effort": manifest["teacher_reasoning_effort"],
               "cost": "not instrumented for Astra agent generation",
               "count": len(statuses), "valid": sum(row["valid"] for row in statuses),
               "statuses": statuses, "visual_review_status": "pending"}
    (output / "run-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", help="safe group name, e.g. easy-a or new-a")
    parser.add_argument("--wave", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", args.shard) or not 1 <= args.wave <= 99:
        parser.error("invalid wave or shard")
    if not 1 <= args.timeout <= 900:
        parser.error("timeout must be 1..900 seconds")
    summary = render(args.shard, args.wave, args.timeout)
    print(json.dumps({"wave": args.wave, "shard": args.shard, "valid": summary["valid"], "total": summary["count"]}), flush=True)
    return 0 if summary["valid"] == summary["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
