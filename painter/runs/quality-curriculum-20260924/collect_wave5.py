#!/usr/bin/env python3
"""Collect hash-verified repaired wave-5 shards and build a turn review page.

This is a read-only public download. It does not accept turns for SFT or call a
model. The gallery deliberately shows invalid and regressive turns as evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
BENCH_CLOUD = HERE.parents[1] / "benchmarks/openrouter-teachers-20260922/cloud"
sys.path.insert(0, str(BENCH_CLOUD))
from fetch_results import download  # noqa: E402
from stage_wave5_cloud import OUT, SHARDS, stage  # noqa: E402

REPAIRED_BASE = "e58fe45526d4827504745c4bc5bd13058994c094"
TOP_LEVEL = {"progress.json", "events.jsonl", "run-summary.json", "gallery.html",
             "restored-source.json", "restored-public-shard.json", "renderer-smoke-result.json"}
EPISODE_FILE = re.compile(r"episodes/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")


def collect(shard: str, prefix: int) -> tuple[Path, dict]:
    root = stage(shard)
    run_id = f"mimo-wave5-{shard}-repaired-20260925-n{prefix}"
    evidence = root / f"repaired-n{prefix}"
    receipt_file = evidence / "public-receipt.json"
    if receipt_file.is_file():
        receipt = json.loads(receipt_file.read_text())
        if receipt.get("run_id") == run_id and receipt.get("public_hash_verified") is True:
            return evidence, receipt
        raise ValueError(f"different prior receipt in {evidence}")
    if evidence.exists():
        raise ValueError(f"partial evidence directory exists: {evidence}")
    archive_path = root / f"{run_id}.tar.gz"
    download(run_id, archive_path)
    from urllib.request import urlopen
    with urlopen(f"https://huggingface.co/datasets/CK0607/komorebi-painter-teachers/resolve/main/runs/{run_id}/receipt.json?download=true", timeout=60) as response:
        receipt = json.load(response)
    source_commit = str(receipt.get("source_commit", ""))
    ancestor = (re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None and
                subprocess.run(["git", "merge-base", "--is-ancestor", REPAIRED_BASE, source_commit],
                               cwd=HERE.parents[2], check=False).returncode == 0)
    if not ancestor or receipt.get("public_hash_verified") is not True:
        raise ValueError(f"source commit or public hash verification mismatch: {run_id}")
    evidence.mkdir(parents=True)
    total = 0
    with tarfile.open(archive_path, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != receipt["file_count"]:
            raise ValueError(f"archive member count mismatch: {run_id}")
        for member in members:
            if (not member.isfile() or member.size > 10_000_000 or
                    any(part in {".", ".."} for part in Path(member.name).parts)):
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.name not in TOP_LEVEL and not EPISODE_FILE.fullmatch(member.name):
                raise ValueError(f"unexpected archive path: {member.name}")
            target = evidence / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(member).read())
            total += member.size
            if total > 1_000_000_000:
                raise ValueError("archive expands beyond 1 GB review bound")
    receipt_file.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    archive_path.unlink()
    return evidence, receipt


def page(prefix: int, entries: list[tuple[str, Path, dict]]) -> Path:
    review = OUT / f"review-n{prefix}"
    review.mkdir(parents=True, exist_ok=True)
    cards = []
    counts: Counter[str] = Counter()
    queue = []
    for shard, evidence, receipt in entries:
        root = OUT / shard
        refs = {row["id"]: row for row in json.loads((root / "refs.json").read_text())["references"]}
        episodes = sorted((evidence / "episodes").glob("*/episode.json"))
        for episode_file in episodes:
            episode = json.loads(episode_file.read_text())
            reference_id = episode["reference_id"]
            reference = refs[reference_id]
            counts[episode.get("status", "unknown")] += 1
            ref_url = f"../{shard}/{reference['image']}"
            turns = []
            queue_turns = []
            preview_href = ref_url
            for turn in episode.get("turns", []):
                number = turn["turn"]
                valid = bool((turn.get("render") or {}).get("valid")) and bool(turn.get("canvas"))
                # The renderer may save a partial PNG even when JavaScript
                # fails or times out. Show it with an invalid label for review;
                # the exporter still rejects it as a supervised target.
                canvas = turn.get("canvas")
                candidate = episode_file.parent / f"turn-{number:02d}.png"
                image_rel = canvas if valid else (str(candidate.relative_to(evidence)) if candidate.is_file() else None)
                href = f"../{shard}/{evidence.name}/{image_rel}" if image_rel else None
                if href:
                    preview_href = href
                error = str(turn.get("render_error") or turn.get("api_error") or "")[:300]
                label = f"T{number}: {'rendered' if valid else 'invalid'}"
                image = (f'<a href="{escape(href)}"><img loading="lazy" src="{escape(href)}" alt="{escape(reference_id)} turn {number}"></a>'
                         if href else '<div class="missing">No valid canvas</div>')
                turns.append(f'<figure>{image}<figcaption>{escape(label)}<small>{escape(error)}</small></figcaption></figure>')
                queue_turns.append({"turn": number, "renderer_valid": valid,
                                    "canvas_sha256": turn.get("current_canvas_sha256") if valid else None,
                                    "quality_review": None, "visual_improvement_over_prior": None})
            cards.append(f'<details><summary><img class="thumb" src="{escape(preview_href)}" alt="latest attempt">'
                         f'<span><b>{escape(reference_id)}</b><br>{escape(shard)} · {escape(episode.get("status", "unknown"))} · {len(turns)} turns</span></summary>'
                         f'<div class="frames"><figure><a href="{escape(ref_url)}"><img loading="lazy" src="{escape(ref_url)}" alt="reference"></a><figcaption>Reference</figcaption></figure>{"".join(turns)}</div></details>')
            queue.append({"shard": shard, "run_id": receipt["run_id"], "reference_id": reference_id,
                          "status": episode.get("status"), "turns": queue_turns})
    html = f"""<!doctype html><meta charset="utf-8"><title>Wave 5 turn review</title>
<style>body{{font:16px system-ui;background:#191b1e;color:#eee;margin:24px}}h1{{font-size:1.6rem}}p{{color:#bbb}}
details{{background:#282b30;border:1px solid #46494e;border-radius:10px;margin:12px 0;padding:12px}}
summary{{cursor:pointer;display:flex;align-items:center;gap:12px}}summary .thumb{{width:96px;height:72px;object-fit:contain;background:#eee}}
.frames{{display:flex;gap:12px;overflow-x:auto;padding-top:12px}}.frames figure:first-child{{position:sticky;left:0;background:#282b30;z-index:1}}
figure{{margin:0;min-width:220px;max-width:330px}}img{{width:100%;height:240px;object-fit:contain;background:#eee}}
figcaption{{font-size:.9rem}}small{{display:block;color:#f4a8a8;overflow-wrap:anywhere}}
.missing{{height:240px;display:grid;place-items:center;background:#433}}a{{color:#a9d8ff}}</style>
<h1>Wave 5 repaired teacher turns · prefix {prefix}</h1><p>{len(queue)} scenes · {escape(str(dict(counts)))}. Each rendered turn is a candidate, not an SFT admission. Inspect reference, composition, beauty and actual improvement before selecting targets.</p>
{''.join(cards)}"""
    output = review / "index.html"
    output.write_text(html)
    (review / "review-queue.json").write_text(json.dumps({"schema": "painter.wave5-review-queue.v1",
                                                        "prefix": prefix, "episodes": queue}, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=int, choices=(4, 8, 12), default=12)
    parser.add_argument("--shard", choices=SHARDS, action="append", help="omit to collect all four shards")
    args = parser.parse_args()
    entries = [(shard, *collect(shard, args.prefix)) for shard in (args.shard or SHARDS)]
    print(page(args.prefix, entries))


if __name__ == "__main__":
    main()
