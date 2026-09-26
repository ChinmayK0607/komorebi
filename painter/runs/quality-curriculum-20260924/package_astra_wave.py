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
    # Three shards have repaired n12 archives. The easy-b archive only has an
    # earlier n4 prefix, so most of its assigned references have no prior
    # canvas at all. Prefer repaired evidence, then the original prefix.
    episode_file = None
    for dirname in ("repaired-n12/episodes", "repaired-n4/episodes", "episodes"):
        episodes = list((WAVE5 / shard / dirname).glob(f"*--{ident}--*/episode.json"))
        if len(episodes) > 1:
            raise ValueError(f"repeated MiMo episode: {shard} {ident}")
        if episodes:
            episode_file = episodes[0]
            break
    if episode_file is None:
        return None, None, None
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


def prior_astra(wave: int, shard: str, ident: str) -> tuple[Path, Path, int]:
    episode = COLLECTED / f"astra-high-wave{wave}-results" / shard / "episodes" / ident
    status = json.loads((episode / "render-status.json").read_text())
    if status.get("reference_id") != ident or status.get("valid") is not True:
        raise ValueError(f"Astra prior is not renderer-valid: {ident}")
    canvas, program = episode / "turn-01.png", episode / "turn-01.program.js"
    if sha(canvas) != status["canvas_sha256"] or sha(program) != status["program_sha256"]:
        raise ValueError(f"Astra prior program/canvas hash mismatch: {ident}")
    return canvas, program, 1


def package(shard: str, wave: int, prior_render_wave: int | None = None,
            prior_render_shard: str | None = None) -> dict:
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
        if prior_render_wave is not None:
            assert prior_render_shard is not None
            canvas, prior_program, prior_turn = prior_astra(prior_render_wave, prior_render_shard, ident)
            prior_source = f"astra-high-wave{prior_render_wave}-{prior_render_shard}"
        elif ident in wave5:
            canvas, prior_program, prior_turn = latest_valid(wave5[ident]["shard"], ident)
            prior_source = f"mimo-wave5-{wave5[ident]['shard']}" if canvas else None
        else:
            canvas, prior_program, prior_turn, prior_source = None, None, None, None
        row = {"reference_id": ident, "reference_sha256": sha(ref),
               "reference": f"references/{ident}.jpg", "program_sha256": sha(program),
               "program": f"programs/{ident}.js", "prior_turn": prior_turn,
               "prior_source": prior_source,
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
    parser.add_argument("--prior-render-wave", type=int)
    parser.add_argument("--prior-render-shard")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", args.shard) or not 1 <= args.wave <= 99:
        parser.error("invalid wave or shard")
    if (args.prior_render_wave is None) != (args.prior_render_shard is None):
        parser.error("prior render wave and shard must be supplied together")
    if args.prior_render_wave is not None and (not 1 <= args.prior_render_wave < args.wave
            or not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", args.prior_render_shard)):
        parser.error("invalid prior render identity")
    print(json.dumps(package(args.shard, args.wave, args.prior_render_wave, args.prior_render_shard), sort_keys=True))


if __name__ == "__main__":
    main()
