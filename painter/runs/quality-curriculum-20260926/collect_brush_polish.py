#!/usr/bin/env python3
"""Verify the finite brush-polish result and build a source-matched gallery."""

from __future__ import annotations

import hashlib
from html import escape
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud/brush-polish-four-20260927"
SOURCE = ROOT / "painter/collected/quality-curriculum-20260926/brush-polish-four-20260927/source.tar.gz"
sys.path[:0] = [str(ROOT / "painter/benchmarks/openrouter-teachers-20260922/cloud"),
                str(ROOT / "painter/runs/quality-curriculum-20260924")]
from fetch_results import download  # noqa: E402
from collect_mimo_cloud import unpack_checked  # noqa: E402


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def collect() -> dict:
    public = json.loads((HERE / "BRUSH_POLISH_PUBLIC.json").read_text())
    source_raw = SOURCE.read_bytes()
    if len(source_raw) != public["archive_bytes"] or sha(source_raw) != public["archive_sha256"]:
        raise ValueError("brush source archive mismatch")
    members = {}
    with tarfile.open(fileobj=io.BytesIO(source_raw), mode="r:gz") as archive:
        for entry in archive:
            path = Path(entry.name)
            if not entry.isfile() or path.is_absolute() or ".." in path.parts or len(path.parts) > 2:
                raise ValueError("unsafe source archive member")
            members[entry.name] = archive.extractfile(entry).read()
    if len(members) != 13 or sha(members["manifest.json"]) != public["manifest_sha256"]:
        raise ValueError("source manifest/members mismatch")
    manifest = json.loads(members["manifest.json"])
    receipt = download("brush-polish-four-20260927", OUT / "result-bundle.tar.gz")
    unpack_checked(OUT / "result-bundle.tar.gz", OUT)
    rows = []
    cards = []
    for source_row in manifest["rows"]:
        ident = source_row["id"]
        for field in ("source", "current_canvas", "current_program"):
            if sha(members[source_row[field]]) != source_row[field + "_sha256"]:
                raise ValueError(f"source member hash mismatch: {ident} {field}")
            target = OUT / "source-staging" / source_row[field]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(members[source_row[field]])
        episodes = list((OUT / "episodes").glob(f"*--{ident}--s01/episode.json"))
        if len(episodes) != 1:
            raise ValueError(f"missing/duplicate result episode: {ident}")
        episode_file = episodes[0]
        ep = episode_file.parent
        state = json.loads(episode_file.read_text())
        expected_ref = (source_row["source_sha256"] if source_row["mode"] == "image_to_image"
                        else source_row["current_canvas_sha256"])
        if (state.get("reference_id") != ident or state.get("reference_sha256") != expected_ref
                or state.get("model") != "xiaomi/mimo-v2.6-pro"):
            raise ValueError(f"episode source/model mismatch: {ident}")
        figures = []
        if source_row["mode"] == "image_to_image":
            figures.append(f'<figure><img src="source-staging/{escape(source_row["source"])}"><figcaption>Original photo</figcaption></figure>')
            brief = "Original photograph above"
        else:
            brief = members[source_row["source"]].decode("utf-8")
        figures.append(f'<figure><img src="source-staging/{escape(source_row["current_canvas"])}"><figcaption>User-liked baseline</figcaption></figure>')
        turns = []
        for turn in state["turns"]:
            rendered = turn.get("render") or {}
            valid = bool(rendered.get("valid") and not rendered.get("skipped")
                         and rendered.get("canvas_sha256"))
            item = {"turn": turn["turn"], "valid": valid,
                    "error": turn.get("render_error") or turn.get("api_error"),
                    "tokens": ((turn.get("response") or {}).get("usage") or {}).get("total_tokens")}
            if valid:
                canvas = ep / f"turn-{turn['turn']:02d}.png"
                code = ep / f"turn-{turn['turn']:02d}.program.js"
                renderer_receipt = rendered.get("receipt") or {}
                if (not canvas.is_file() or not code.is_file()
                        or sha(canvas.read_bytes()) != rendered["canvas_sha256"]
                        or sha(canvas.read_bytes()) != renderer_receipt.get("png_sha256")
                        or sha(code.read_bytes()) != renderer_receipt.get("source_sha256")):
                    raise ValueError(f"render hash mismatch: {ident} turn {turn['turn']}")
                item["canvas_sha256"] = sha(canvas.read_bytes())
                item["program_sha256"] = sha(code.read_bytes())
                figures.append(f'<figure><img src="episodes/{escape(ep.name)}/{canvas.name}"><figcaption>Brush polish turn {turn["turn"]}</figcaption></figure>')
            else:
                figures.append(f'<figure><div class="invalid">Turn {turn["turn"]} did not render<br>{escape(str(item["error"]))}</div></figure>')
            turns.append(item)
        cards.append(f'<article><h2>{escape(ident)} · {escape(source_row["category"])}</h2>'
                     f'<p>{escape(brief)}</p><div class="grid">{"".join(figures)}</div></article>')
        rows.append({"id": ident, "mode": source_row["mode"],
                     "baseline_canvas_sha256": source_row["current_canvas_sha256"],
                     "source_sha256": source_row["source_sha256"],
                     "status": state["status"], "total_tokens": state.get("total_tokens"),
                     "turns": turns, "sft_admitted": False})
    if len(rows) != 4:
        raise ValueError("not four complete scene records")
    run_summary = json.loads((OUT / "run-summary.json").read_text())
    summary = {"schema": "painter.brush-polish-collected.v1", "source_archive_sha256": public["archive_sha256"],
               "result_archive_sha256": receipt["sha256"], "result_archive_bytes": receipt["bytes"],
               "renderer_sha256": run_summary["renderer_sha256"],
               "total_tokens": run_summary["total_tokens"],
               "provider_requests": run_summary["network_requests_made_this_invocation"],
               "cost_complete": run_summary["cost_complete"],
               "valid_turns": sum(t["valid"] for row in rows for t in row["turns"]),
               "rows": rows, "visual_review": "pending"}
    (OUT / "collection-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Brush polish: source-matched comparison</title>'
        '<style>body{font:16px system-ui;background:#171a1c;color:#eee;margin:24px}'
        'article{background:#282d30;border-radius:12px;padding:18px;margin:22px 0}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}'
        'figure{margin:0}img,.invalid{width:100%;height:490px;object-fit:contain;background:#e5e4dd;color:#222;box-sizing:border-box}'
        '.invalid{padding:30px}figcaption{padding:5px}p{white-space:pre-wrap}</style>'
        '<h1>Four source-matched brush polish comparisons</h1>'
        '<p>Original source, user-liked baseline, then model turns. Valid rendering is not visual admission.</p>'
        + "".join(cards) + "</html>")
    return {"review": str(OUT / "review.html"), "episodes": len(rows),
            "valid_turns": summary["valid_turns"], "total_tokens": summary["total_tokens"],
            "result_archive_sha256": summary["result_archive_sha256"]}


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
