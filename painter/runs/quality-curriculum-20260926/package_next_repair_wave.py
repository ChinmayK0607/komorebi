#!/usr/bin/env python3
"""Package the eight inspected correction states with source rights and hashes."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "painter/collected/quality-curriculum-20260926/next-repair-eight-v1"
SOURCE_BATCHES = {
    "t600-013": "astra-openverse-wave6",
    "t600-085": "sol-val2017-wave8",
    "t600-145": "sol-simple-cc0-select-v2",
    "t600-181": "astra-openverse-wave5",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rights(audit_id: str, source_id: str) -> dict:
    batch = SOURCE_BATCHES[audit_id]
    author = json.loads((ROOT / "painter/collected/quality-curriculum-20260926" / batch / "manifest.json").read_text())
    items = author.get("items", author.get("entries", []))
    row = next((item for item in items if item["id"] == source_id), None)
    if row is None:
        raise ValueError(f"missing source/rights metadata: {audit_id}")
    source = row.get("source") if isinstance(row.get("source"), dict) else row
    name = row.get("license_name") or row.get("license") or source.get("license_name")
    url = row.get("license_url") or source.get("license_url")
    source_url = row.get("source_url") or source.get("source_url")
    if not all(isinstance(x, str) and x for x in (name, url, source_url)):
        raise ValueError(f"incomplete photo rights: {audit_id}")
    return {"name": name, "url": url, "source_url": source_url,
            "creator": row.get("creator") or source.get("creator"),
            "attribution": row.get("attribution") or source.get("attribution")}


def pack() -> dict:
    plan_path = HERE / "NEXT_REPAIR_8.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "painter.teacher600-next-repair-wave.v1" or len(plan.get("rows", [])) != 8:
        raise ValueError("unexpected correction wave")
    members: dict[str, bytes] = {}
    rows = []
    for row in plan["rows"]:
        audit_id = row["audit_id"]
        source = (ROOT / row["input_path"]).read_bytes()
        prior = (ROOT / row["canvas_path"]).read_bytes()
        program = (ROOT / row["program_path"]).read_bytes()
        if (sha(source) != row["input_sha256"] or sha(prior) != row["canvas_sha256"]
                or sha(program) != row["program_sha256"]):
            raise ValueError(f"changed exact first-paint state: {audit_id}")
        prompt = ((ROOT / row["input_path"]) if row["mode"] == "text_to_image"
                  else (ROOT / row["canvas_path"]).parent / "prompt.txt").read_bytes()
        ext = ".txt" if row["mode"] == "text_to_image" else ".jpg"
        paths = {"source": f"sources/{audit_id}{ext}", "prior_canvas": f"priors/{audit_id}.png",
                 "prior_program": f"programs/{audit_id}.js", "prompt": f"prompts/{audit_id}.txt"}
        for field, raw in (("source", source), ("prior_canvas", prior),
                           ("prior_program", program), ("prompt", prompt)):
            members[paths[field]] = raw
        rows.append({
            "audit_id": audit_id, "source_id": row["id"], "mode": row["mode"],
            "category": row["category"], "difficulty": row["difficulty"],
            "parent_visual_weakness": row["parent_visual_weakness"],
            **{field: path for field, path in paths.items()},
            "source_sha256": sha(source), "prior_canvas_sha256": sha(prior),
            "prior_program_sha256": sha(program), "prompt_sha256": sha(prompt),
            "rights": rights(audit_id, row["id"]) if row["mode"] == "image_to_image" else None,
        })
    manifest = {"schema": "painter.teacher600-repair-source.v1",
                "plan_sha256": sha(plan_path.read_bytes()), "count": len(rows),
                "status": "correction_states_not_training_targets", "rows": rows}
    members["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    archive_path = OUT / "source.tar.gz"
    with archive_path.open("wb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, raw in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(raw))
    (OUT / "manifest.json").write_bytes(members["manifest.json"])
    receipt = {"schema": "painter.teacher600-repair-source-receipt.v1",
               "archive_sha256": sha(archive_path.read_bytes()),
               "archive_bytes": archive_path.stat().st_size,
               "manifest_sha256": sha(members["manifest.json"]),
               "plan_sha256": manifest["plan_sha256"], "count": len(rows),
               "all_photo_sources_rights_complete": True}
    (OUT / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(pack(), sort_keys=True))
