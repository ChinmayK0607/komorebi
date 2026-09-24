#!/usr/bin/env python3
"""Build an offline source-versus-canvas review from a verified cloud archive."""

from __future__ import annotations

import html
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
COLLECTED = ROOT / "painter/collected/quality-curriculum-20260924"
ARCHIVE = COLLECTED / "manual-cloud-render-v1.tar.gz"
OUT = COLLECTED / "manual-cloud-render-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not ARCHIVE.is_file():
        raise SystemExit(f"Download the hash-verified public archive first: {ARCHIVE}")
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if not member.isfile() or not path.parts or path.parts[0] not in {"episodes", "run-summary.json"}:
                raise ValueError(f"Unexpected archive member: {member.name}")
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            dest = OUT / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.extractfile(member).read())
    summary = json.loads((OUT / "run-summary.json").read_text())
    if summary.get("programs") != 10 or len(summary.get("statuses", [])) != 10:
        raise ValueError("Expected exactly ten render statuses")
    refs = {row["id"]: row for row in json.loads((RUN / "reference-manifest.json").read_text())["references"]}
    cards = []
    for row in summary["statuses"]:
        ident = row["reference_id"]
        ref = ROOT / refs[ident]["reference_path"]
        canvas = OUT / "episodes" / ident / "turn-01.png"
        if sha(ref) != row["reference_sha256"]:
            raise ValueError(f"Reference hash mismatch: {ident}")
        if row["valid"] and (not canvas.is_file() or sha(canvas) != row["canvas_sha256"]):
            raise ValueError(f"Canvas hash mismatch: {ident}")
        ref_url = Path(os.path.relpath(ref, OUT)).as_posix()
        canvas_url = canvas.relative_to(OUT).as_posix() if canvas.is_file() else ""
        cards.append(f'''<section><h2>{html.escape(ident)} — {html.escape(refs[ident]["description"])}</h2>
<div class="pair"><figure><img src="{html.escape(ref_url)}"><figcaption>Reference</figcaption></figure>
<figure>{f'<img src="{html.escape(canvas_url)}">' if canvas_url else '<p>No canvas</p>'}<figcaption>Manual v1 · {"valid render" if row["valid"] else "render failed"}</figcaption></figure></div>
<p>Program {html.escape(row["program_sha256"][:12])} · canvas {html.escape((row["canvas_sha256"] or "missing")[:12])}</p></section>''')
    page = '''<!doctype html><meta charset="utf-8"><title>Manual painting render review</title>
<style>body{background:#151515;color:#eee;font:16px system-ui;max-width:1500px;margin:auto;padding:25px}section{border-top:1px solid #555;margin:35px 0;padding-top:15px}.pair{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:320px}img{width:100%;height:540px;object-fit:contain;background:#252525}figcaption,p{color:#aaa}</style>
<h1>Manual painting render review</h1><p>These are candidate canvases. Renderer validity does not mean visual acceptance.</p>'''+"\n".join(cards)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
