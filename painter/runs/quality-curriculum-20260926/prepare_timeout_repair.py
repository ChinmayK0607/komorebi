#!/usr/bin/env python3
"""Package bounded-cost alternatives for the eight exact 600-second first-paint timeouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from package_teacher_batch import package


ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
RENDERS = POOL / "rendered-cloud"
SELECTED = {
    "openverse-4b9c3343-f23c-4743-8a84-74767dcf8158",
    "openverse-65371c2f-1343-4f50-a0aa-21fbaede7d0d",
    "openverse-e5cf35d5-896c-4a3f-87ad-f94e60b16771",
    "openverse-b6583e05-2982-4660-957b-975afda9bfd4",
    "openverse-466c3cd0-bbc4-4b90-9b59-588b5e3c3ccc",
    "openverse-6b5f8aae-6e0d-4fff-8bd1-42e78e5e5ecc",
    "openverse-53d37187-6c03-4c49-b782-2005a33764b5",
    "coco-val2017-000000559348",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def transform(program: str, model: str) -> str:
    if model == "gpt-6-astra":
        old = "for(let i=0;i<32;i++){let t=i*Math.PI/16;"
        new = "for(let i=0;i<12;i++){let t=i*Math.PI/6;"
        if program.count(old) != 1:
            raise ValueError("Astra timeout did not use the expected oval helper")
        result = program.replace(old, new).replace("brush.fillTexture(0.14,0.08)", "brush.fillTexture(0.08,0.04)")
    elif model == "gpt-5.6-sol":
        replacements = {
            "for(let i=0;i<28;i++){let t=i*Math.PI/14;": "for(let i=0;i<12;i++){let t=i*Math.PI/6;",
            "for(let i=0;i<150;i++)disc": "for(let i=0;i<60;i++)disc",
            "for(let row=0;row<7;row++)": "for(let row=0;row<4;row++)",
            "x+=33+row*3": "x+=52+row*4",
            "for(let k=0;k<12;k++){let t=k*Math.PI/6;": "for(let k=0;k<8;k++){let t=k*Math.PI/4;",
        }
        result = program
        for old, new in replacements.items():
            if result.count(old) != 1:
                raise ValueError(f"Sol timeout did not match expected pattern: {old}")
            result = result.replace(old, new)
    else:
        raise ValueError(model)
    if result == program:
        raise ValueError("no performance change")
    return result


def prepare() -> list[dict]:
    hits = {}
    for summary_path in sorted(RENDERS.glob("*/run-summary.json")):
        summary = json.loads(summary_path.read_text())
        for status in summary["statuses"]:
            if status["id"] not in SELECTED or status.get("error_code") != "render_timeout":
                continue
            if status["id"] in hits:
                raise ValueError(f"duplicate selected timeout: {status['id']}")
            hits[status["id"]] = (summary, status, summary_path.parent / "episodes" / status["id"])
    if set(hits) != SELECTED:
        raise ValueError(f"timeout selection mismatch: {sorted(SELECTED - set(hits))}")
    batches = {"gpt-6-astra": [], "gpt-5.6-sol": []}
    for ident, (summary, status, episode) in sorted(hits.items()):
        batch = summary["batch"]
        author = json.loads((POOL / batch / "manifest.json").read_text())
        original = next(row for row in author["entries"] if row["id"] == ident)
        model = original["model"]
        if model not in batches or status["input_sha256"] != original["reference_sha256"] or status["program_sha256"] != original["program_sha256"]:
            raise ValueError(f"source identity mismatch: {ident}")
        photo = (episode / "input.jpg").read_bytes()
        prompt = (ROOT / original["prompt_path"]).read_bytes() if original.get("prompt_path") else None
        program = (episode / "program.js").read_bytes()
        if (sha(photo) != status["input_sha256"] or sha(program) != status["program_sha256"]
                or (prompt is not None and sha(prompt) != original["prompt_sha256"])):
            raise ValueError(f"source bytes mismatch: {ident}")
        out_name = "astra-timeout-repair-v1" if model == "gpt-6-astra" else "sol-timeout-repair-v1"
        target = POOL / out_name / ident
        target.mkdir(parents=True, exist_ok=True)
        repaired = transform(program.decode(), model).encode()
        files = [("reference.jpg", photo), ("baseline.js", program), ("program.js", repaired)]
        if prompt is not None:
            files.append(("prompt.txt", prompt))
        for name, raw in files:
            (target / name).write_bytes(raw)
        source = {key: original.get(key) for key in (
            "source_id", "source_url", "thumbnail_url", "provider", "query", "title",
            "creator", "creator_url", "attribution", "license", "license_id",
            "license_name", "license_url", "license_version", "tags") if original.get(key) is not None}
        source["source_visual_type"] = original.get("source_visual_type", "photograph")
        entry = {"id": ident, "source_batch": batch, "reference_path": f"{ident}/reference.jpg",
                 "reference_sha256": sha(photo), "prompt_path": f"{ident}/prompt.txt" if prompt is not None else None,
                 "prompt_sha256": sha(prompt) if prompt is not None else None, "old_program_path": f"{ident}/baseline.js",
                 "old_program_sha256": sha(program), "new_program_path": f"{ident}/program.js",
                 "new_program_sha256": sha(repaired), "source": source,
                 "correction": {"audit_disposition": "unrendered performance alternative",
                                "correction": "Reduce polygon sides and decorative particle work after an exact 600-second render timeout; visual quality unreviewed."},
                 "source_run_id": summary["run_id"], "timeout_seconds": summary["timeout_seconds"]}
        batches[model].append(entry)
    results = []
    for model, entries in batches.items():
        out_name = "astra-timeout-repair-v1" if model == "gpt-6-astra" else "sol-timeout-repair-v1"
        source_dir = POOL / out_name
        manifest = {"schema": "painter.teacher600-timeout-repair.v1", "created": "2026-09-27",
                    "model": model, "reasoning_effort": "high", "count": len(entries),
                    "new_distinct_source_count": 0, "entries": entries,
                    "hypothesis": "Fewer polygon sides and decorative elements will complete within the existing 600-second renderer deadline.",
                    "matched_baseline": "Exact original program on the same image, timed out at 600 seconds.",
                    "status": "unrendered_alternatives_not_sft_admitted"}
        (source_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        results.append(package(source_dir))
    return results


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True))
