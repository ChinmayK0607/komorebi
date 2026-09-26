#!/usr/bin/env python3
"""Collect a public Linux render receipt and show reference versus actual canvas."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "painter/benchmarks/openrouter-teachers-20260922/cloud"))
from fetch_results import download  # noqa: E402


OUT = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run_id: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", run_id):
        raise ValueError("invalid render run ID")
    target = OUT / run_id
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "bundle.tar.gz"
    verification = download(run_id, archive)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > 500 or sum(member.size for member in members) > 100_000_000:
            raise ValueError("render evidence exceeds bounds")
        for member in members:
            parts = Path(member.name).parts
            if (not member.isfile() or member.size > 10_000_000
                    or (member.name != "run-summary.json" and
                        (len(parts) != 3 or parts[0] != "episodes" or
                         not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,96}", parts[1])))
                    or any(part in {".", ".."} for part in parts)):
                raise ValueError(f"unsafe render evidence member: {member.name}")
            path = target / member.name
            raw = bundle.extractfile(member).read()
            if path.is_file() and path.read_bytes() != raw:
                raise ValueError(f"existing render evidence differs: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    summary = json.loads((target / "run-summary.json").read_text())
    if (summary.get("schema") != "painter.teacher500-render.v1"
            or summary.get("run_id") != run_id
            or summary.get("count") != len(summary.get("statuses", []))):
        raise ValueError("render summary identity/count mismatch")
    source_receipt = OUT.parent / summary["batch"] / "source-public.json"
    source = json.loads(source_receipt.read_text())
    if (source["archive_sha256"] != summary["source_bundle_sha256"]
            or source["dataset_commit"] != summary["source_dataset_commit"]):
        raise ValueError("render used a different source bundle")
    for row in summary["statuses"]:
        episode = target / "episodes" / row["id"]
        suffix = ".txt" if row["mode"] == "text_to_image" else ".jpg"
        if sha(episode / f"input{suffix}") != row["input_sha256"]:
            raise ValueError(f"render input mismatch: {row['id']}")
        if sha(episode / "program.js") != row["program_sha256"]:
            raise ValueError(f"render program mismatch: {row['id']}")
        canvas = episode / "canvas.png"
        if (sha(canvas) if canvas.is_file() else None) != row["canvas_sha256"]:
            raise ValueError(f"render canvas mismatch: {row['id']}")
        if row["valid"] and not canvas.is_file():
            raise ValueError(f"valid render lacks canvas: {row['id']}")
    (target / "collection-receipt.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return target, summary


def gallery(target: Path, summary: dict) -> Path:
    cards = []
    for row in summary["statuses"]:
        ident = row["id"]
        episode = target / "episodes" / ident
        if row["mode"] == "image_to_image":
            source = f'<figure><img src="episodes/{escape(ident)}/input.jpg" alt="reference"><figcaption>Reference</figcaption></figure>'
        else:
            prompt = escape((episode / "input.txt").read_text())
            source = f'<figure><pre>{prompt}</pre><figcaption>Text prompt</figcaption></figure>'
        if (episode / "canvas.png").is_file():
            canvas = f'<figure><img src="episodes/{escape(ident)}/canvas.png" alt="rendered canvas"><figcaption>Candidate canvas</figcaption></figure>'
        else:
            canvas = '<figure><div class="missing">No canvas</div></figure>'
        cards.append(f'<article><h2>{escape(ident)} · {"renderer-valid" if row["valid"] else escape(str(row.get("error_code") or "invalid"))}</h2>'
                     f'<div class="frames">{source}{canvas}</div><p><a href="episodes/{escape(ident)}/program.js">program</a> · '
                     f'<a href="episodes/{escape(ident)}/render-status.json">render status</a></p></article>')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Teacher render review</title>
<style>body{{font:16px/1.4 system-ui;background:#181b1d;color:#eee;margin:24px}}p{{color:#bcc5c7}}
article{{background:#282d31;padding:16px;margin:18px 0;border-radius:10px}}h2{{font-size:1.1rem}}
.frames{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}figure{{margin:0}}
img,pre,.missing{{box-sizing:border-box;width:100%;height:420px;object-fit:contain;background:#f4f1ea;color:#222}}
pre{{white-space:pre-wrap;overflow:auto;padding:20px}}.missing{{display:grid;place-items:center}}
figcaption{{color:#bbc6c8;padding:4px}}a{{color:#a5d4df}}
@media(max-width:800px){{.frames{{grid-template-columns:1fr}}}}</style>
<h1>Teacher render review · {escape(summary['batch'])}</h1>
<p>{summary['valid']}/{summary['count']} renderer-valid. Renderer validity is not aesthetic quality approval. Inspect each reference and canvas before admitting a demonstration or asking for a correction turn.</p>
{''.join(cards)}</html>'''
    path = target / "review.html"
    path.write_text(html)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    target, summary = collect(args.run_id)
    print(json.dumps({"gallery": str(gallery(target, summary)),
                      "valid": summary["valid"], "count": summary["count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
