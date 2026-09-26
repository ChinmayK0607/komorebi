#!/usr/bin/env python3
"""Collect, hash-check, and display matched eight-scene repair candidates."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"
sys.path[:0] = [str(HERE), str(ROOT / "painter/runs/quality-curriculum-20260924")]
from collect_mimo_cloud import unpack_checked  # noqa: E402
from collect_sol_translation_pilot import download_with_hub_fallback  # noqa: E402
from stage_next_repair_wave_cloud import MODELS, PUBLIC, source_archive, unpack  # noqa: E402


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def collect(run_id: str) -> dict:
    if not re.fullmatch(r"teacher600-repair-[a-z0-9-]{1,50}", run_id):
        raise ValueError("unexpected repair wave ID")
    result_root = BASE / run_id
    result_root.mkdir(parents=True, exist_ok=True)
    archive = source_archive(result_root / "source.tar.gz")
    staged = result_root / "source-staging"
    manifest = json.loads((staged / "manifest.json").read_text()) if staged.exists() else unpack(archive, staged)
    expected = {row["audit_id"]: row for row in manifest["rows"]}
    if len(expected) != 8 or manifest["plan_sha256"] != PUBLIC["plan_sha256"]:
        raise ValueError("source manifest mismatch")
    records = []
    cards = []
    for tier, model in MODELS.items():
        root = result_root / tier
        root.mkdir(parents=True, exist_ok=True)
        result_archive = root / "bundle.tar.gz"
        receipt = download_with_hub_fallback(f"{run_id}-{tier}", result_archive)
        unpack_checked(result_archive, root)
        provenance = json.loads((root / "restored-source.json").read_text())
        if (provenance.get("model") != model or provenance.get("tier") != tier
                or provenance.get("source_archive_sha256") != PUBLIC["archive_sha256"]
                or provenance.get("source_manifest_sha256") != sha((staged / "manifest.json").read_bytes())):
            raise ValueError(f"tier provenance mismatch: {tier}")
        episodes = sorted((root / "episodes").glob("*/episode.json"))
        wanted = {row["audit_id"] for row in manifest["rows"]
                  if (row["difficulty"] == "hard") == (tier == "pro")}
        if len(episodes) != len(wanted):
            raise ValueError(f"missing tier episodes: {tier}")
        seen = set()
        for episode_path in episodes:
            ep = episode_path.parent
            state = json.loads(episode_path.read_text())
            aid = state["reference_id"]
            if aid in seen or aid not in wanted or state["model"] != model:
                raise ValueError(f"unexpected episode identity: {tier}/{aid}")
            seen.add(aid)
            row = expected[aid]
            first = ep / "first-paint.png"
            program = ep / "first-paint.js"
            prompt = ep / "source-prompt.txt"
            extension = ".jpg" if row["mode"] == "image_to_image" else ".txt"
            original = ep / f"source-input{extension}"
            for path, field in ((first, "prior_canvas_sha256"),
                                (program, "prior_program_sha256"),
                                (prompt, "prompt_sha256"),
                                (original, "source_sha256")):
                if not path.is_file() or sha(path.read_bytes()) != row[field]:
                    raise ValueError(f"source hash mismatch: {aid}/{path.name}")
            if row["mode"] == "image_to_image" and state.get("reference_sha256") != row["source_sha256"]:
                raise ValueError(f"photo reference binding mismatch: {aid}")
            if row["mode"] == "text_to_image" and state.get("reference_sha256") != row["prior_canvas_sha256"]:
                raise ValueError(f"text prior binding mismatch: {aid}")
            turns = []
            figures = []
            if row["mode"] == "image_to_image":
                figures.append(f'<figure><img src="{tier}/episodes/{escape(ep.name)}/source-input.jpg"><figcaption>Source photograph</figcaption></figure>')
            figures.append(f'<figure><img src="{tier}/episodes/{escape(ep.name)}/first-paint.png"><figcaption>Matched first painting</figcaption></figure>')
            for turn in state["turns"]:
                rendered = turn.get("render") or {}
                record = {"turn": turn["turn"], "api_status": turn.get("api_status"),
                          "valid": bool(rendered.get("valid")),
                          "finish_reason": (turn.get("response") or {}).get("finish_reason"),
                          "total_tokens": ((turn.get("response") or {}).get("usage") or {}).get("total_tokens"),
                          "error": turn.get("render_error") or turn.get("api_error")}
                if record["valid"]:
                    image = ep / f"turn-{turn['turn']:02d}.png"
                    code = ep / f"turn-{turn['turn']:02d}.program.js"
                    renderer_receipt = rendered.get("receipt") or {}
                    if (not image.is_file() or not code.is_file()
                            or sha(image.read_bytes()) != rendered.get("canvas_sha256")
                            or sha(image.read_bytes()) != renderer_receipt.get("png_sha256")
                            or sha(code.read_bytes()) != renderer_receipt.get("source_sha256")):
                        raise ValueError(f"turn render hash mismatch: {aid}/{turn['turn']}")
                    record["canvas_sha256"] = sha(image.read_bytes())
                    figures.append(f'<figure><img src="{tier}/episodes/{escape(ep.name)}/{image.name}"><figcaption>Turn {turn["turn"]} · {record["total_tokens"]} tokens</figcaption></figure>')
                turns.append(record)
            records.append({"audit_id": aid, "tier": tier, "model": model,
                            "mode": row["mode"], "category": row["category"],
                            "status": state["status"], "first_paint_sha256": row["prior_canvas_sha256"],
                            "source_sha256": row["source_sha256"], "turns": turns,
                            "total_tokens": state.get("total_tokens"),
                            "cost_complete": state.get("cost_complete"),
                            "episode_directory": ep.name, "archive_sha256": receipt["sha256"]})
            weakness = escape(row["parent_visual_weakness"])
            brief = escape(prompt.read_text())
            errors = "; ".join(f"Turn {t['turn']}: {t['error'] or t['finish_reason'] or 'invalid'}"
                               for t in turns if not t["valid"])
            cards.append(f'<article><h2>{aid} · {escape(model)} · {escape(state["status"])}</h2>'
                         f'<p><strong>Source:</strong> {brief}</p><p><strong>Known weakness:</strong> {weakness}</p>'
                         f'<div class="grid">{"".join(figures)}</div><p>{escape(errors)}</p></article>')
        if seen != wanted:
            raise ValueError(f"tier selection mismatch: {tier}")
    review = result_root / "review.html"
    review.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Teacher 600 repair wave</title>'
                      '<style>body{font:16px system-ui;background:#16191b;color:#eee;margin:24px}'
                      'article{padding:20px;background:#282c2f;border-radius:12px;margin:22px 0}'
                      '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}'
                      'figure{margin:0}img{width:100%;background:#ddd}figcaption{padding:5px}</style>'
                      '<h1>Eight source-to-paint correction candidates · visual admission pending</h1>'
                      + "".join(cards) + "</html>")
    summary = {"schema": "painter.teacher600-repair-collected.v1", "run_id": run_id,
               "source_archive_sha256": PUBLIC["archive_sha256"],
               "source_manifest_sha256": sha((staged / "manifest.json").read_bytes()),
               "visual_review": "pending", "rows": sorted(records, key=lambda r: r["audit_id"])}
    (result_root / "collection-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"review": str(review), "episodes": len(records),
            "valid_turns": sum(t["valid"] for r in records for t in r["turns"]),
            "invalid_turns": sum(not t["valid"] for r in records for t in r["turns"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="teacher600-repair-eight-v2-20260927")
    args = parser.parse_args()
    print(json.dumps(collect(args.run_id), sort_keys=True))
