#!/usr/bin/env python3
"""Package Astra program candidates and the visual state they were correcting.

The public bundle is source evidence, not accepted training data. Its reference
and optional prior-canvas bytes are hashed against the existing run evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
COLLECTED = ROOT / "painter/collected/quality-curriculum-20260924"
WAVE5 = COLLECTED / "mimo-wave5"
sys.path.insert(0, str(ROOT / "painter"))
from contract import validate_teacher_program  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_valid(shard: str, ident: str) -> tuple[Path | None, Path | None, int | None]:
    episodes = list((WAVE5 / shard / "repaired-n12/episodes").glob(f"*--{ident}--*/episode.json"))
    if len(episodes) != 1:
        raise ValueError(f"missing or repeated MiMo episode: {shard} {ident}")
    episode_file = episodes[0]
    episode = json.loads(episode_file.read_text())
    valid = [turn for turn in episode["turns"] if (turn.get("render") or {}).get("valid") and turn.get("canvas")]
    if not valid:
        return None, None, None
    turn = valid[-1]
    canvas = episode_file.parents[2] / turn["canvas"]
    program = episode_file.parents[2] / turn["program"]
    if not canvas.is_file() or not program.is_file():
        raise ValueError(f"missing prior evidence: {ident} T{turn['turn']}")
    if turn.get("current_canvas_sha256") != sha(canvas):
        raise ValueError(f"prior canvas hash mismatch: {ident} T{turn['turn']}")
    return canvas, program, turn["turn"]


def package(shard: str, wave: int) -> dict:
    source = COLLECTED / f"astra-high-wave{wave}" / shard
    wave5 = {row["id"]: row for row in json.loads((HERE / "reference-manifest-wave5.json").read_text())["references"]}
    prepared = {row["id"]: row for row in json.loads((HERE / "reference-manifest-astra-100.json").read_text())["references"]}
    programs = sorted(path for path in source.glob("*.js") if path.is_file())
    if not programs:
        raise ValueError(f"no Astra programs in {source}")
    rows = []
    files: list[tuple[Path, str]] = []
    seen = set()
    for program in programs:
        match = re.fullmatch(r"(coco128-\d{12})(?:\.program)?\.js", program.name)
        if not match:
            raise ValueError(f"unexpected candidate program filename: {program.name}")
        ident = match.group(1)
        if ident in seen or ident not in prepared:
            raise ValueError(f"repeated or non-training reference: {ident}")
        if wave == 1 and (ident not in wave5 or wave5[ident]["shard"] != shard):
            raise ValueError(f"wave-1 program is not from its assigned shard: {ident}")
        seen.add(ident)
        code = program.read_text()
        validate_teacher_program(code)
        if len(code) > 250_000 or "loadImage(" in code or "fetch(" in code:
            raise ValueError(f"large or externally sourced program: {ident}")
        ref = COLLECTED / "astra-high-100/references" / f"{prepared[ident]['source_id']}.jpg"
        if sha(ref) != prepared[ident]["sha256"]:
            raise ValueError(f"reference hash mismatch: {ident}")
        canvas, prior_program, prior_turn = latest_valid(wave5[ident]["shard"], ident) if ident in wave5 else (None, None, None)
        row = {"reference_id": ident, "reference_sha256": sha(ref),
               "reference": f"references/{ident}.jpg", "program_sha256": sha(program),
               "program": f"programs/{ident}.js", "prior_turn": prior_turn,
               "prior_canvas_sha256": sha(canvas) if canvas else None,
               "prior_canvas": f"prior/{ident}.png" if canvas else None,
               "prior_program_sha256": sha(prior_program) if prior_program else None,
               "prior_program": f"prior/{ident}.js" if prior_program else None,
               "split": "train", "review_status": "candidate_unrendered"}
        rows.append(row)
        files.extend(((ref, row["reference"]), (program, row["program"])))
        if canvas:
            files.extend(((canvas, row["prior_canvas"]), (prior_program, row["prior_program"])))
    notes = source / "manifest.json"
    if notes.is_file():
        files.append((notes, "agent-notes.json"))
    manifest = {"schema": "painter.astra-high-source.v1", "wave": wave, "shard": shard,
                "teacher_model": "gpt-6-astra", "teacher_reasoning_effort": "high",
                "count": len(rows), "source_reference_manifest_sha256": sha(HERE / "reference-manifest-astra-100.json"),
                "agent_notes_sha256": sha(notes) if notes.is_file() else None,
                "status": "unrendered_unreviewed", "rows": rows}
    output = source / "source-bundle.tar.gz"
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    import io
    with tarfile.open(output, "w:gz") as archive:
        info = tarfile.TarInfo("manifest.json")
        info.size, info.mtime, info.mode = len(payload), 0, 0o644
        archive.addfile(info, io.BytesIO(payload))
        for path, arcname in sorted(files, key=lambda pair: pair[1]):
            raw = path.read_bytes()
            info = tarfile.TarInfo(arcname)
            info.size, info.mtime, info.mode = len(raw), 0, 0o644
            archive.addfile(info, io.BytesIO(raw))
    receipt = {"schema": "painter.astra-high-source-receipt.v1", "wave": wave, "shard": shard,
               "bundle": str(output), "bundle_sha256": sha(output), "bundle_bytes": output.stat().st_size,
               "count": len(rows), "reference_ids": sorted(seen)}
    (source / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", help="safe group name, e.g. easy-a or new-a")
    parser.add_argument("--wave", type=int, default=1)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", args.shard) or not 1 <= args.wave <= 99:
        parser.error("invalid wave or shard")
    print(json.dumps(package(args.shard, args.wave), sort_keys=True))


if __name__ == "__main__":
    main()
