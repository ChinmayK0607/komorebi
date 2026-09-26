#!/usr/bin/env python3
"""Collect a public matched translation render and verify every painting hash."""

from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys
import tarfile
from urllib.error import HTTPError

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "painter/benchmarks/openrouter-teachers-20260922/cloud"))
from fetch_results import checked_path, download  # noqa: E402


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def download_with_hub_fallback(run_id: str, archive: Path) -> dict:
    try:
        return download(run_id, archive)
    except HTTPError as exc:
        if exc.code != 429:
            raise
    # Anonymous Hub file downloads have a different retry/cache path from the
    # direct resolve URL. The Cloud publisher already verified these public
    # parts; verify the reconstructed archive again here before extraction.
    dataset = "CK0607/komorebi-painter-teachers"
    cache = archive.parent / ".hf-cache"
    receipt_path = hf_hub_download(dataset, f"runs/{run_id}/receipt.json",
                                   repo_type="dataset", token=False, cache_dir=cache)
    receipt = json.loads(Path(receipt_path).read_text())
    if (receipt.get("run_id") != run_id or receipt.get("dataset_repo") != dataset
            or receipt.get("public_hash_verified") is not True):
        raise ValueError("public receipt identity or verification mismatch")
    parts = receipt.get("part_paths")
    if not isinstance(parts, list) or not parts or receipt.get("encoding") != "base64-per-part":
        raise ValueError("expected split Base64 public archive")
    digest = hashlib.sha256()
    total = 0
    with archive.open("wb") as target:
        for path in parts:
            checked_path(path, run_id)
            local = hf_hub_download(dataset, path, repo_type="dataset",
                                    revision=receipt["dataset_commit"], token=False, cache_dir=cache)
            raw = base64.b64decode(b"".join(Path(local).read_bytes().split()), validate=True)
            target.write(raw)
            digest.update(raw)
            total += len(raw)
    if total != receipt["bundle_bytes"] or digest.hexdigest() != receipt["bundle_sha256"]:
        archive.unlink(missing_ok=True)
        raise ValueError("reconstructed public archive hash/size mismatch")
    return {"run_id": run_id, "output": str(archive), "bytes": total,
            "sha256": digest.hexdigest(), "public_hash_verified": True,
            "representation": receipt["representation"]}


def collect(run_id: str, output: Path, plan: Path) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,63}", run_id):
        raise ValueError("unsafe run ID")
    output.mkdir(parents=True, exist_ok=True)
    verification = download_with_hub_fallback(run_id, output / "bundle.tar.gz")
    with tarfile.open(output / "bundle.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 100 or sum(x.size for x in members) > 100_000_000:
            raise ValueError("archive exceeds pilot bounds")
        for member in members:
            parts = Path(member.name).parts
            if (not member.isfile() or member.size > 10_000_000 or ".." in parts
                    or (member.name not in {"run-summary.json", "gallery.html"}
                        and (len(parts) != 3 or parts[0] != "episodes"
                             or not re.fullmatch(r"t600-\d{3}", parts[1])))):
                raise ValueError(f"unexpected archive member: {member.name}")
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.extractfile(member).read()
            if target.exists() and target.read_bytes() != raw:
                raise ValueError(f"existing result differs: {member.name}")
            target.write_bytes(raw)
    summary = json.loads((output / "run-summary.json").read_text())
    planned = json.loads(plan.read_text())
    if (summary.get("schema") != "painter.sol-native-brush-render-pilot.v1"
            or summary.get("run_id") != run_id
            or summary.get("plan_sha256") != sha(plan.read_bytes())
            or len(summary.get("rows", [])) != 8):
        raise ValueError("run summary does not match frozen plan")
    expected = {r["audit_id"]: r for r in planned["rows"]}
    if set(expected) != {r["audit_id"] for r in summary["rows"]}:
        raise ValueError("pilot episode set differs")
    for row in summary["rows"]:
        original = expected[row["audit_id"]]
        episode = output / "episodes" / row["audit_id"]
        for name, key in (("input.txt", "input_sha256"), ("original.js", "source_program_sha256"),
                          ("translated.js", "candidate_program_sha256")):
            if sha((episode / name).read_bytes()) != original[key]:
                raise ValueError(f"pilot source hash mismatch: {row['audit_id']} {name}")
        for label in ("original", "translated"):
            canvas = episode / f"{label}.png"
            if (sha(canvas.read_bytes()) if canvas.is_file() else None) != row["renders"][label]["canvas_sha256"]:
                raise ValueError(f"pilot render hash mismatch: {row['audit_id']} {label}")
        if json.loads((episode / "status.json").read_text()) != row:
            raise ValueError(f"episode status differs from summary: {row['audit_id']}")
    cards = []
    for row in summary["rows"]:
        ident = escape(row["audit_id"])
        episode = output / "episodes" / row["audit_id"]
        prompt = escape((episode / "input.txt").read_text())
        figures = []
        for label in ("original", "translated"):
            render = row["renders"][label]
            body = (f'<img src="episodes/{ident}/{label}.png" alt="{label} painting">'
                    if render["valid"] else
                    f'<div class="missing">{escape(str(render["error_code"] or "No canvas"))}</div>')
            figures.append(f'<figure>{body}<figcaption>{label}: {"valid" if render["valid"] else "invalid"} · {render["seconds"]}s</figcaption></figure>')
        cards.append(f'<article><h2>{ident} · {escape(row["category"])}</h2><pre>{prompt}</pre>'
                     f'<div class="pair">{"".join(figures)}</div>'
                     f'<p>Original reproduced: {row["original_reproduction_matches"]}. '
                     f'<a href="episodes/{ident}/original.js">Original code</a> · '
                     f'<a href="episodes/{ident}/translated.js">Translated code</a></p></article>')
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Sol brush translation pilot</title>'
            '<style>body{font:16px system-ui;background:#171a1d;color:#eee;margin:24px}'
            'article{background:#292f33;padding:18px;margin:20px 0;border-radius:12px}'
            '.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}'
            'figure{margin:0}img,.missing{width:100%;max-height:600px;object-fit:contain;background:#eee}'
            '.missing{height:400px;display:grid;place-items:center;color:#333}'
            'pre{white-space:pre-wrap;background:#1d2225;padding:12px}a{color:#a8dcef}'
            '@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
            '<h1>Sol native → p5.brush · matched render pilot</h1>'
            '<p>Same prompt, seed and renderer. These are unreviewed conversion candidates, not SFT admissions.</p>'
            + ''.join(cards) + '</html>')
    (output / "review.html").write_text(html)
    (output / "collection-receipt.json").write_text(json.dumps(verification, indent=2) + "\n")
    return {"review": str(output / "review.html"), "archive_sha256": verification["sha256"],
            "original_valid": sum(x["renders"]["original"]["valid"] for x in summary["rows"]),
            "translated_valid": sum(x["renders"]["translated"]["valid"] for x in summary["rows"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--plan", type=Path, default=HERE / "SOL_BRUSH_PILOT_8.json")
    args = parser.parse_args()
    output = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud" / args.run_id
    print(json.dumps(collect(args.run_id, output, args.plan)))


if __name__ == "__main__":
    main()
