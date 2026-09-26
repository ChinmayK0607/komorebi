#!/usr/bin/env python3
"""Collect public Astra Linux renders and make a reference/prior/teacher review page."""

from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
COLLECTED = ROOT / "painter/collected/quality-curriculum-20260924"
sys.path.insert(0, str(ROOT / "painter/benchmarks/openrouter-teachers-20260922/cloud"))
from fetch_results import DATASET, download, public_url  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(wave: int, shard: str) -> tuple[Path, dict]:
    run_id = f"astra-high-wave{wave}-{shard}-20260926"
    target = COLLECTED / f"astra-high-wave{wave}-results" / shard
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "bundle.tar.gz"
    verification = download(run_id, archive)
    with urlopen(public_url("main", f"runs/{run_id}/receipt.json"), timeout=60) as response:
        receipt = json.load(response)
    if receipt.get("bundle_sha256") != verification["sha256"] or not receipt.get("public_hash_verified"):
        raise ValueError("result public receipt mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != receipt["file_count"]:
            raise ValueError("result archive member count mismatch")
        for member in members:
            parts = Path(member.name).parts
            if (not member.isfile() or member.size > 10_000_000
                    or member.name != "run-summary.json" and (len(parts) != 3 or parts[0] != "episodes")
                    or any(part in {".", ".."} for part in parts)):
                raise ValueError(f"unsafe result archive member: {member.name}")
            path = target / member.name
            raw = bundle.extractfile(member).read()
            if path.is_file() and path.read_bytes() != raw:
                raise ValueError(f"different existing evidence: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    summary = json.loads((target / "run-summary.json").read_text())
    source_commit = str(summary.get("source_commit", ""))
    if (summary.get("shard") != shard or summary.get("wave", 1) != wave
            or summary.get("count") != len(summary.get("statuses", []))
            or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
            or subprocess.run(["git", "merge-base", "--is-ancestor", "d79bccc", source_commit],
                              cwd=ROOT, check=False).returncode != 0):
        raise ValueError("render summary source/run identity mismatch")
    for row in summary["statuses"]:
        ep = target / "episodes" / row["reference_id"]
        if sha(ep / "reference.jpg") != row["reference_sha256"]:
            raise ValueError(f"reference hash mismatch: {ep}")
        if sha(ep / "turn-01.program.js") != row["program_sha256"]:
            raise ValueError(f"program hash mismatch: {ep}")
        if row.get("prior_canvas_sha256") and sha(ep / "prior.png") != row["prior_canvas_sha256"]:
            raise ValueError(f"prior canvas hash mismatch: {ep}")
        canvas = ep / "turn-01.png"
        if (sha(canvas) if canvas.is_file() else None) != row.get("canvas_sha256"):
            raise ValueError(f"candidate canvas hash mismatch: {ep}")
    (target / "public-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return target, summary


def page(wave: int, entries: list[tuple[str, Path, dict]]) -> Path:
    output = COLLECTED / f"astra-high-wave{wave}-results" / "review.html"
    cards = []
    for shard, root, summary in entries:
        for row in summary["statuses"]:
            ident = row["reference_id"]
            ep = root / "episodes" / ident
            images = [("Reference", f"{shard}/episodes/{ident}/reference.jpg")]
            if (ep / "prior.png").is_file():
                images.append(("Prior MiMo canvas", f"{shard}/episodes/{ident}/prior.png"))
            if (ep / "turn-01.png").is_file():
                images.append(("Astra high candidate", f"{shard}/episodes/{ident}/turn-01.png"))
            frames = "".join(f'<figure><a href="{escape(path)}"><img loading="lazy" src="{escape(path)}" alt="{escape(label)}"></a><figcaption>{escape(label)}</figcaption></figure>'
                             for label, path in images)
            cards.append(f'<article><h2>{escape(ident)} · {escape(shard)} · {"render valid" if row["valid"] else escape(str(row.get("error_code")))}</h2><div class="frames">{frames}</div></article>')
    html = f'''<!doctype html><meta charset="utf-8"><title>Astra wave {wave} render review</title>
<style>body{{font:16px system-ui;background:#191b1e;color:#eee;margin:24px}}p{{color:#bbb}}article{{background:#282b30;padding:14px;margin:18px 0;border-radius:10px}}h2{{font-size:1.1rem}}.frames{{display:flex;gap:16px;overflow-x:auto}}figure{{margin:0;min-width:260px;max-width:500px}}img{{width:100%;height:370px;object-fit:contain;background:#eee}}figcaption{{color:#ccc}}</style>
<h1>Astra high · wave {wave}</h1><p>Renderer validity is not quality approval. Compare composition, likeness, aesthetics and correction to the prior canvas before admitting a turn to training.</p>{''.join(cards)}'''
    output.write_text(html)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--shard", action="append", required=True)
    args = parser.parse_args()
    if not 1 <= args.wave <= 99 or any(not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", s) for s in args.shard):
        parser.error("invalid wave or shard")
    entries = [(s, *collect(args.wave, s)) for s in args.shard]
    print(page(args.wave, entries))


if __name__ == "__main__":
    main()
