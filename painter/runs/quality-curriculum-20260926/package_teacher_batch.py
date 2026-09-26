#!/usr/bin/env python3
"""Normalize and hash-verify one Sol/Astra teacher batch for Linux rendering."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "painter"))
from contract import validate_teacher_program  # noqa: E402


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_file(source: Path, relative: str, expected_sha: str | None) -> bytes:
    path = (ROOT / relative if relative.startswith("painter/") else source / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.is_symlink():
        raise ValueError(f"invalid teacher source file: {relative}")
    raw = path.read_bytes()
    if len(raw) > 5_000_000 or (expected_sha and digest(raw) != expected_sha):
        raise ValueError(f"teacher file length/hash mismatch: {relative}")
    return raw


def candidates(source: Path, manifest: dict) -> list[dict]:
    if manifest.get("modality") == "render-conditioned-correction":
        return [{"id": row["id"], "mode": row["mode"],
                 "category": row["category"], "role": "render_conditioned_correction_candidate",
                 "turn_count": row.get("turn_count", 2),
                 "program": row["new_program_path"], "program_sha": row["new_program_sha256"],
                 "input": row["input_path"], "input_sha": row["input_sha256"],
                 "prompt": row.get("prompt_path"), "prompt_sha": row.get("prompt_sha256"),
                 "baseline_program": row["prior_program_path"],
                 "baseline_program_sha": row["prior_program_sha256"],
                 "prior_canvas": row["prior_canvas_path"],
                 "prior_canvas_sha": row["prior_canvas_sha256"],
                 "baseline_batch": row["source_batch"], "prior_run_id": row["prior_run_id"],
                 "notes": row["rationale"], "source_url": row.get("source_url"),
                 "source_metadata": row.get("source_metadata"),
                 "source_visual_type": row.get("source_visual_type"),
                 "license": row.get("license")}
                for row in manifest["items"]]
    if manifest.get("modality") == "image-to-painting-render-conditioned-correction":
        return [{"id": row["id"], "mode": "image_to_image",
                 "category": row["category"], "role": "render_conditioned_correction_candidate",
                 "turn_count": row.get("turn_count", 2), "program": row["new_program_path"],
                 "program_sha": row["new_program_sha256"],
                 "input": row["reference_path"], "input_sha": row["reference_sha256"],
                 "prompt": row["prompt_path"], "prompt_sha": row["prompt_sha256"],
                 "baseline_program": row["prior_program_path"],
                 "baseline_program_sha": row["prior_program_sha256"],
                 "prior_canvas": row["prior_canvas_path"],
                 "prior_canvas_sha": row["prior_canvas_sha256"],
                 "baseline_batch": row["source_batch"],
                 "prior_run_id": row["prior_run_id"],
                 "notes": row["rationale"], "source_url": row["source_url"],
                 "source_metadata": row["source_metadata"],
                 "source_visual_type": "photograph", "license": row["license"]}
                for row in manifest["items"]]
    if "entries" in manifest and manifest.get("new_distinct_source_count") == 0:
        return [{"id": row["id"], "mode": "image_to_image",
                 "category": "static-repair", "role": "unrendered_alternative_candidate",
                 "program": row.get("new_program_path", row.get("program_path")),
                 "program_sha": row.get("new_program_sha256", row.get("program_sha256")),
                 "input": row["reference_path"], "input_sha": row["reference_sha256"],
                 "prompt": row["prompt_path"], "prompt_sha": row["prompt_sha256"],
                 "baseline_program": row.get("old_program_path", row.get("baseline_program_path")),
                 "baseline_program_sha": row.get("old_program_sha256", row.get("baseline_program_sha256")),
                 "baseline_batch": row["source_batch"],
                 "notes": f"{row['correction'].get('audit_disposition', row['correction'].get('baseline_static_risk', 'static repair'))}: "
                          f"{row['correction'].get('correction', row['correction'].get('concrete_structural_correction', ''))}",
                 "source_url": row["source"].get("source_url"),
                 "source_metadata": row["source"],
                 "source_visual_type": row["source"].get("source_visual_type", "photograph"),
                 "license": {"id": row["source"].get("license_id"),
                             "name": row["source"].get("license_name", row["source"].get("license")),
                             "url": row["source"].get("license_url")}}
                for row in manifest["entries"]]
    if ("items" in manifest and manifest.get("distinct_new_prompts") == 0
            and ("parent_audit_sha256" in manifest or "audit_sha256" in manifest)):
        return [{"id": row["id"], "mode": "text_to_image",
                 "category": "static-repair", "role": "unrendered_alternative_candidate",
                 "difficulty": row.get("difficulty"),
                 "program": row["program_file"], "program_sha": row["program_sha256"],
                 "input": row["prompt_file"], "input_sha": row["prompt_sha256"],
                 "baseline_program": row["original_program_evidence_file"],
                 "baseline_program_sha": row["original_program_sha256"],
                 "baseline_batch": row["source_audit_key"].split("/", 1)[0],
                 "notes": "; ".join(row.get("structural_and_aesthetic_corrections",
                                              row.get("specific_corrections", [])))}
                for row in manifest["items"]]
    if "items" in manifest and manifest.get("modality") == "image-to-painting-static-repair":
        return [{"id": row["id"], "mode": "image_to_image",
                 "category": "static-repair", "role": "unrendered_alternative_candidate",
                 "program": row["new_program_file"], "program_sha": row["new_program_sha256"],
                 "input": row["reference_file"], "input_sha": row["reference_sha256"],
                 "prompt": row["new_prompt_file"], "prompt_sha": row["new_prompt_sha256"],
                 "baseline_program": row["old_program_file"],
                 "baseline_program_sha": row["old_program_sha256"],
                 "baseline_batch": row["source_batch"],
                 "notes": row["structural_correction"],
                 "source_url": row["source"].get("source_url"),
                 "source_metadata": row["source"],
                 "source_visual_type": "photograph",
                 "license": {"id": row["source"].get("license_id"),
                             "name": row["source"].get("license_name"),
                             "url": row["source"].get("license_url")}}
                for row in manifest["items"]]
    if "items" in manifest and manifest.get("modality") == "image-to-painting":
        return [{"id": row["id"], "mode": "image_to_image",
                 "category": row["corrected_category"],
                 "program": row["program_file"], "program_sha": row["program_sha256"],
                 "input": row["reference_file"], "input_sha": row["reference_sha256"],
                 "prompt": row["prompt_file"], "prompt_sha": row["prompt_sha256"],
                 "notes": row.get("intended_composition", ""),
                 "difficulty": row.get("difficulty"),
                 "source_url": row["source"].get("source_url"),
                 "source_metadata": row["source"],
                 "source_visual_type": row.get("source_visual_type", "photograph"),
                 "license": {"id": row["source"].get("license_id"),
                             "name": row["source"].get("license_name"),
                             "url": row["source"].get("license_url")}}
                for row in manifest["items"]]
    if "items" in manifest:  # gpt-5.6-sol high text batch
        return [{"id": row["id"], "mode": "text_to_image", "category": row["category"],
                 "program": row["program_file"], "program_sha": row["program_sha256"],
                 "input": row["prompt_file"], "input_sha": row["prompt_sha256"],
                 "notes": row.get("intended_composition", ""),
                 "difficulty": row.get("difficulty")} for row in manifest["items"]]
    if "samples" in manifest:  # gpt-6-astra high text batch
        return [{"id": row["id"], "mode": "text_to_image", "category": row["category"],
                 "program": row["program_path"], "program_sha": row["sha256"]["program.js"],
                 "input": row["prompt_path"], "input_sha": row["sha256"]["prompt.txt"],
                 "notes": row.get("intended_composition", ""),
                 "difficulty": row.get("difficulty")} for row in manifest["samples"]]
    if manifest.get("task_type") == "text_to_image" and "entries" in manifest:
        return [{"id": row["id"], "mode": "text_to_image", "category": row["category"],
                 "program": row["program_path"], "program_sha": row["program_sha256"],
                 "input": row["prompt_path"], "input_sha": row["prompt_sha256"],
                 "notes": row.get("intended_composition", ""),
                 "difficulty": row.get("difficulty")} for row in manifest["entries"]]
    if "entries" in manifest:  # gpt-6-astra high photo batch
        return [{"id": row["id"], "mode": "image_to_image", "category": row.get("observed_category", row["category"]),
                 "program": row["program_path"], "program_sha": row["program_sha256"],
                 "input": row["reference_path"], "input_sha": row["reference_sha256"],
                 "prompt": row.get("prompt_path"), "prompt_sha": row.get("prompt_sha256"),
                 "notes": row.get("plan", ""),
                 "difficulty": row.get("difficulty"),
                 "turn_count": row.get("turn_count", 1),
                 "source_url": row.get("source_url"),
                 "source_metadata": {key: row.get(key) for key in (
                     "source_id", "source_url", "flickr_url", "thumbnail_url",
                     "provider", "query", "title", "creator", "creator_url",
                     "attribution", "license_id", "license_name", "license",
                     "license_url", "license_version", "tags", "captions", "width", "height")
                     if row.get(key) is not None},
                 "source_visual_type": row.get("source_visual_type", "photograph"),
                 "license": {"id": row.get("license_id"),
                             "name": row.get("license_name", row.get("license")),
                             "url": row.get("license_url")},
                 "creator": row.get("creator"), "attribution": row.get("attribution")}
                for row in manifest["entries"]]
    raise ValueError("unknown teacher manifest schema")


def package(source: Path) -> dict:
    source = source.resolve()
    if not source.is_relative_to(ROOT / "painter/collected/quality-curriculum-20260926"):
        raise ValueError("batch must be in the teacher500 collected directory")
    batch = source.name
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", batch):
        raise ValueError("unsafe batch name")
    original_raw = (source / "manifest.json").read_bytes()
    original = json.loads(original_raw)
    model = original.get("author_model", original.get("model"))
    effort = original.get("reasoning_effort")
    if model not in {"gpt-5.6-sol", "gpt-6-astra"} or effort != "high":
        raise ValueError("unexpected teacher model or reasoning effort")
    normalized = []
    members: dict[str, bytes] = {}
    seen: set[str] = set()
    for row in candidates(source, original):
        ident = row["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,96}", ident) or ident in seen:
            raise ValueError(f"unsafe or duplicate teacher ID: {ident}")
        seen.add(ident)
        program = read_file(source, row["program"], row["program_sha"])
        code = program.decode("utf-8")
        validate_teacher_program(code)
        if len(program) > 250_000 or "loadImage(" in code or "fetch(" in code:
            raise ValueError(f"large or external program: {ident}")
        javascript = shutil.which("node") or shutil.which("bun")
        if javascript is None:
            raise RuntimeError("a Node or Bun JavaScript syntax checker is required")
        subprocess.run([javascript, "--check", "-"], input=program, check=True, capture_output=True)
        if row.get("baseline_program"):
            baseline = read_file(source, row["baseline_program"], row["baseline_program_sha"])
            validate_teacher_program(baseline.decode("utf-8"))
            subprocess.run([javascript, "--check", "-"], input=baseline, check=True, capture_output=True)
            if row.get("prior_canvas"):
                prior_canvas = read_file(source, row["prior_canvas"], row["prior_canvas_sha"])
                if not prior_canvas.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"not a PNG prior canvas: {ident}")
                members[f"priors/{ident}.png"] = prior_canvas
                members[f"baselines/{ident}.js"] = baseline
        input_raw = read_file(source, row["input"], row["input_sha"])
        if row["mode"] == "text_to_image":
            input_raw.decode("utf-8")
            input_path = f"inputs/{ident}.txt"
        else:
            if not input_raw.startswith(b"\xff\xd8\xff"):
                raise ValueError(f"not a JPEG reference: {ident}")
            input_path = f"inputs/{ident}.jpg"
        program_path = f"programs/{ident}.js"
        members[input_path] = input_raw
        members[program_path] = program
        prompt_path = None
        prompt_sha = None
        if row.get("prompt"):
            prompt_raw = read_file(source, row["prompt"], row["prompt_sha"])
            prompt_raw.decode("utf-8")
            prompt_path = f"prompts/{ident}.txt"
            prompt_sha = digest(prompt_raw)
            members[prompt_path] = prompt_raw
        normalized.append({"id": ident, "mode": row["mode"], "category": row["category"],
                           "difficulty": row.get("difficulty"),
                           "turn_count": row.get("turn_count", 1),
                           "role": row.get("role", "first_paint_candidate"),
                           "baseline_batch": row.get("baseline_batch"),
                           "baseline_program_sha256": row.get("baseline_program_sha"),
                           "baseline_program": f"baselines/{ident}.js" if row.get("prior_canvas") else None,
                           "prior_canvas": f"priors/{ident}.png" if row.get("prior_canvas") else None,
                           "prior_canvas_sha256": row.get("prior_canvas_sha"),
                           "prior_run_id": row.get("prior_run_id"),
                           "notes": row["notes"], "program": program_path,
                           "program_sha256": digest(program), "input": input_path,
                           "input_sha256": digest(input_raw), "prompt": prompt_path,
                           "prompt_sha256": prompt_sha,
                           "source_url": row.get("source_url"),
                           "source_metadata": row.get("source_metadata"),
                           "source_visual_type": row.get("source_visual_type"),
                           "license": row.get("license"), "creator": row.get("creator"),
                           "attribution": row.get("attribution"),
                           "model": model, "reasoning_effort": effort,
                           "status": "unrendered_unreviewed"})
    if not normalized:
        raise ValueError("empty teacher batch")
    bundle_manifest = {
        "schema": "painter.teacher500-source.v1", "batch": batch,
        "author_manifest_sha256": digest(original_raw), "teacher_model": model,
        "teacher_reasoning_effort": effort, "count": len(normalized), "rows": normalized,
    }
    members["manifest.json"] = (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode()
    archive_path = source / "source.tar.gz"
    with archive_path.open("wb") as file, gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, raw in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(raw), 0, 0o644
                tar.addfile(info, io.BytesIO(raw))
    receipt = {"schema": "painter.teacher500-source-receipt.v1", "batch": batch,
               "count": len(normalized), "source_manifest_sha256": digest(members["manifest.json"]),
               "archive": str(archive_path), "archive_sha256": digest(archive_path.read_bytes()),
               "archive_bytes": archive_path.stat().st_size,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "status": "candidate_unrendered_unreviewed"}
    (source / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.source), sort_keys=True))


if __name__ == "__main__":
    main()
