#!/usr/bin/env python3
"""Fetch the finite MiMo redo evidence and show exact first-paint/turn pairs."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"
sys.path[:0] = [str(HERE), str(ROOT / "painter/runs/quality-curriculum-20260924")]
from collect_mimo_cloud import unpack_checked  # noqa: E402
from collect_sol_translation_pilot import download_with_hub_fallback  # noqa: E402

MODELS = {"t600-326": "xiaomi/mimo-v2.6-flash", "t600-492": "xiaomi/mimo-v2.6-pro"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def collect(run_id: str) -> dict:
    if not run_id.startswith("sol-redo-") or not run_id.replace("-", "").isalnum():
        raise ValueError("invalid finite run ID")
    plan = json.loads((HERE / "SOL_BRUSH_PILOT_8.json").read_text())
    expected = {r["audit_id"]: r for r in plan["rows"]}
    roots = []
    results = []
    for audit_id, model in MODELS.items():
        root = BASE / run_id / audit_id
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "bundle.tar.gz"
        receipt = download_with_hub_fallback(f"{run_id}-{audit_id}", archive)
        count = unpack_checked(archive, root)
        provenance = json.loads((root / "restored-source.json").read_text())
        episodes = list((root / "episodes").glob("*/episode.json"))
        if len(episodes) != 1 or provenance.get("audit_id") != audit_id or provenance.get("model") != model:
            raise ValueError(f"episode/stage mismatch: {audit_id}")
        ep = episodes[0].parent
        state = json.loads(episodes[0].read_text())
        first = ep / "first-paint.png"
        source_prompt = ep / "source-prompt.txt"
        first_program = ep / "first-paint.js"
        source = expected[audit_id]
        if (state.get("model") != model or state.get("reference_id") != audit_id
                or sha(first.read_bytes()) != source["original_canvas_sha256"]
                or sha(source_prompt.read_bytes()) != source["input_sha256"]
                or sha(first_program.read_bytes()) != source["source_program_sha256"]):
            raise ValueError(f"source/episode hash mismatch: {audit_id}")
        turns = []
        for turn in state.get("turns", []):
            record = {"turn": turn["turn"], "api_status": turn.get("api_status"),
                      "valid": bool((turn.get("render") or {}).get("valid")),
                      "error": turn.get("render_error") or turn.get("api_error"),
                      "total_tokens": (turn.get("response") or {}).get("usage", {}).get("total_tokens")}
            if record["valid"]:
                path = ep / f"turn-{turn['turn']:02d}.png"
                if (not path.is_file() or sha(path.read_bytes()) != turn["render"]["canvas_sha256"]):
                    raise ValueError(f"turn canvas mismatch: {audit_id} {turn['turn']}")
                record["canvas"] = path.name
            turns.append(record)
        result = {"audit_id": audit_id, "model": model, "status": state["status"],
                  "first_paint_sha256": sha(first.read_bytes()), "turns": turns,
                  "archive_sha256": receipt["sha256"], "files": count,
                  "episode_directory": ep.name, "total_tokens": state.get("total_tokens"),
                  "cost_complete": state.get("cost_complete")}
        roots.append(root)
        results.append(result)
    cards = []
    for root, result in zip(roots, results):
        ep = root / "episodes" / result["episode_directory"]
        prefix = f"{result['audit_id']}/episodes/{ep.name}"
        figures = [f'<figure><img src="{escape(prefix)}/first-paint.png"><figcaption>Matched first paint</figcaption></figure>']
        for turn in result["turns"]:
            if turn.get("canvas"):
                figures.append(f'<figure><img src="{escape(prefix)}/{escape(turn["canvas"])}"><figcaption>Turn {turn["turn"]} · valid · {turn["total_tokens"]} tokens</figcaption></figure>')
        failures = [f'Turn {t["turn"]}: {t["error"]}' for t in result["turns"] if not t["valid"]]
        cards.append(f'<article><h2>{escape(result["audit_id"])} · {escape(result["model"])} · {escape(result["status"])}</h2>'
                     f'<pre>{escape((ep / "source-prompt.txt").read_text())}</pre>'
                     f'<div class="grid">{"".join(figures)}</div><p>{escape("; ".join(failures))}</p></article>')
    review = BASE / run_id / "review.html"
    review.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>MiMo teacher redo pilot</title>'
                      '<style>body{font:16px system-ui;background:#151719;color:#eee;margin:24px}'
                      'article{padding:18px;background:#292d30;border-radius:12px;margin:20px 0}'
                      '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}'
                      'figure{margin:0}img{width:100%;background:#ddd}pre{white-space:pre-wrap}</style>'
                      '<h1>Exact first paints and MiMo revisions · candidates, not SFT admissions</h1>'
                      + "".join(cards) + '</html>')
    summary = {"schema": "painter.sol-redo-collected.v1", "run_id": run_id,
               "visual_review": "pending", "rows": results}
    (BASE / run_id / "collection-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"review": str(review), "rows": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="sol-redo-pilot-20260927")
    args = parser.parse_args()
    print(json.dumps(collect(args.run_id), sort_keys=True))
