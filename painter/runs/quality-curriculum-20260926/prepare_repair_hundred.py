#!/usr/bin/env python3
"""Select and package 100 diverse, licensed, hash-pinned correction states.

This prepares a larger finite teacher wave; it neither calls a model nor
labels a candidate as a visual success. Generated images and archives stay out
of Git. The small selection manifest is kept for review and reproducibility.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
POOL = ROOT / "painter/collected/quality-curriculum-20260926"
INVENTORY = POOL / "teacher600-repair-inventory-v1/manifest.json"
OUT = POOL / "repair-hundred-v1"
PLAN = HERE / "REPAIR_HUNDRED_PLAN.json"
EXPECTED_INVENTORY_SHA256 = "d828f108cf7133a1d1e498437488d390252c644331e0b25c8b38d5bdc53dc613"
QUOTAS = {("flash", "text_to_image", "native"): 10,
          ("flash", "text_to_image", "brush"): 25,
          ("flash", "image_to_image", "brush"): 35,
          ("pro", "text_to_image", "native"): 5,
          ("pro", "text_to_image", "brush"): 10,
          ("pro", "image_to_image", "brush"): 15}
SHARDS = {"flash": 7, "pro": 3}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rights_index() -> dict[str, dict]:
    found: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for path in POOL.glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in manifest.get("items", manifest.get("entries", [])):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            source = row.get("source") if isinstance(row.get("source"), dict) else row
            triple = (row.get("license_name") or row.get("license") or source.get("license_name"),
                      row.get("license_url") or source.get("license_url"),
                      row.get("source_url") or source.get("source_url"))
            if all(isinstance(value, str) and value for value in triple):
                found[row["id"]].add(triple)
    return {key: {"name": next(iter(values))[0], "url": next(iter(values))[1],
                  "source_url": next(iter(values))[2]}
            for key, values in found.items() if len(values) == 1}


def complexity(category: str) -> int:
    category = category.lower()
    if any(word in category for word in ("simple", "object", "food", "still", "fruit", "flower", "plant")):
        return 0
    if any(word in category for word in ("people", "person", "crowd", "portrait", "vehicle", "machine",
                                          "transport", "interior", "architecture", "city", "landscape", "scene")):
        return 2
    return 1


def category_family(category: str) -> str:
    return category.lower().replace(" / ", "/").split("/", 1)[0].strip()


def select() -> dict:
    if sha(INVENTORY.read_bytes()) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("repair inventory changed")
    inventory = json.loads(INVENTORY.read_text())
    if len(inventory["rows"]) != 600:
        raise ValueError("unexpected inventory size")
    covered = {"t600-326", "t600-492"}
    covered.update(row["audit_id"] for row in json.loads((HERE / "REPAIR_WAVE_8_REVIEW.json").read_text())["rows"])
    rights = rights_index()
    eligible = [row for row in inventory["rows"]
                if row["audit_id"] not in covered and row.get("canvas_path")
                and (row["mode"] != "image_to_image" or row["id"] in rights)]
    chosen = []
    used = set()
    for (tier, mode, medium), count in QUOTAS.items():
        candidates = [row for row in eligible if row["mode"] == mode
                      and ("brush" if row["actual_brush_paint_call"] else "native") == medium]
        family_count: dict[str, int] = defaultdict(int)
        source_count: dict[str, int] = defaultdict(int)
        for _ in range(count):
            available = [row for row in candidates if row["audit_id"] not in used]
            if not available:
                raise ValueError(f"quota cannot be filled: {tier}/{mode}/{medium}")
            priority = (lambda row: complexity(row["category"])) if tier == "flash" else (lambda row: -complexity(row["category"]))
            row = min(available, key=lambda row: (priority(row), family_count[category_family(row["category"])],
                                                  source_count[row["source_run_id"]],
                                                  sha(("repair-hundred-v1:" + row["audit_id"]).encode())))
            used.add(row["audit_id"])
            chosen.append((tier, row))
            family_count[category_family(row["category"])] += 1
            source_count[row["source_run_id"]] += 1
    if len(chosen) != 100 or len(used) != 100:
        raise ValueError("selection is not 100 distinct tasks")

    # Keep each ten-scene Cloud shard balanced at five text and five photo.
    assignments = {}
    for tier in SHARDS:
        for mode in ("text_to_image", "image_to_image"):
            rows = sorted((row for selected_tier, row in chosen
                           if selected_tier == tier and row["mode"] == mode),
                          key=lambda row: sha(("shard:" + row["audit_id"]).encode()))
            if len(rows) != SHARDS[tier] * 5:
                raise ValueError(f"unbalanced tier/mode: {tier}/{mode}")
            for index, row in enumerate(rows):
                assignments[row["audit_id"]] = f"{tier}-{index % SHARDS[tier]:02d}"

    plan_rows = []
    for tier, row in chosen:
        aid = row["audit_id"]
        prompt_path = row["input_path"] if row["mode"] == "text_to_image" else None
        prompt_text = (None if prompt_path else
                       "Paint the attached reference photograph faithfully as a beautiful, "
                       "coherent painterly image. Preserve the main subjects, their "
                       "positions, proportions, colors, lighting, and spatial relationships. "
                       f"Scene category: {row['category']}.\n")
        plan_rows.append({"audit_id": aid, "tier": tier, "shard": assignments[aid],
                          "mode": row["mode"], "source_id": row["id"],
                          "category": row["category"], "heuristic_complexity": complexity(row["category"]),
                          "original_medium": "brush" if row["actual_brush_paint_call"] else "native",
                          "source_run_id": row["source_run_id"],
                          "source_path": row["input_path"], "source_sha256": row["input_sha256"],
                          "prior_canvas_path": row["canvas_path"], "prior_canvas_sha256": row["canvas_sha256"],
                          "prior_program_path": row["program_path"], "prior_program_sha256": row["program_sha256"],
                          "prompt_path": prompt_path, "prompt_text": prompt_text,
                          "rights": rights[row["id"]] if row["mode"] == "image_to_image" else None})
    result = {"schema": "painter.teacher600-repair-hundred-plan.v1",
              "hypothesis": "More source-conditioned teacher revisions yield visually better canvases across varied subjects and both input modes.",
              "matched_baseline": "Each exact first-paint canvas in the immutable 600-task inventory",
              "inventory_sha256": EXPECTED_INVENTORY_SHA256,
              "selection": "Fixed-hash, category/source-diverse sample after excluding the ten scenes already piloted; quotas pin 50 text/50 photo, 15 native-only redos and 70 Flash/30 Pro.",
              "count": len(plan_rows), "shards": {f"{tier}-{i:02d}": 10 for tier, n in SHARDS.items() for i in range(n)},
              "training_admission": "none_until_rendered_visual_review",
              "rows": sorted(plan_rows, key=lambda row: (row["shard"], row["mode"], row["audit_id"]))}
    return result


def package(plan: dict) -> dict:
    plan_raw = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    PLAN.write_bytes(plan_raw)
    members: dict[str, bytes] = {}
    source_rows = []
    for row in plan["rows"]:
        aid = row["audit_id"]
        paths = {"source": f"sources/{aid}{'.jpg' if row['mode'] == 'image_to_image' else '.txt'}",
                 "prior_canvas": f"priors/{aid}.png", "prior_program": f"programs/{aid}.js",
                 "prompt": f"prompts/{aid}.txt"}
        for key, original_key in (("source", "source"), ("prior_canvas", "prior_canvas"),
                                  ("prior_program", "prior_program"), ("prompt", "prompt")):
            raw = (row["prompt_text"].encode() if key == "prompt" and row["prompt_path"] is None
                   else (ROOT / row[f"{original_key}_path"]).read_bytes())
            expected = row.get(f"{original_key}_sha256")
            if expected is not None and sha(raw) != expected:
                raise ValueError(f"source state changed: {aid}/{key}")
            members[paths[key]] = raw
        source_rows.append({key: row[key] for key in ("audit_id", "tier", "shard", "mode", "source_id",
                                                     "category", "heuristic_complexity", "original_medium", "rights")}
                           | paths
                           | {f"{key}_sha256": sha(members[path]) for key, path in paths.items()})
    manifest = {"schema": "painter.teacher600-repair-hundred-source.v1",
                "plan_sha256": sha(plan_raw), "inventory_sha256": EXPECTED_INVENTORY_SHA256,
                "count": len(source_rows), "rows": source_rows,
                "training_admission": "none_until_rendered_visual_review"}
    members["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    archive_path = OUT / "source.tar.gz"
    with archive_path.open("wb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, raw in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(raw))
    (OUT / "manifest.json").write_bytes(members["manifest.json"])
    receipt = {"schema": "painter.teacher600-repair-hundred-receipt.v1",
               "archive_sha256": sha(archive_path.read_bytes()), "archive_bytes": archive_path.stat().st_size,
               "manifest_sha256": sha(members["manifest.json"]), "plan_sha256": sha(plan_raw),
               "count": len(source_rows), "photo_rights_complete": True,
               "archive_path": str(archive_path)}
    (OUT / "source-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(package(select()), sort_keys=True))
