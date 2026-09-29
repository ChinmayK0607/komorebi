#!/usr/bin/env python3
"""Collect latest hash-verified public prefixes into a visual candidate review.

This makes no model, judge, or renderer calls. It never automatically admits a
teacher turn to training; the result is a reviewable candidate inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import tarfile
import time

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "painter/benchmarks/openrouter-teachers-20260922/cloud"))
from fetch_results import download  # noqa: E402
from stage_cloud import load_archive  # noqa: E402

DATASET = "CK0607/komorebi-painter-teachers"
COLLECTED = ROOT / "painter/collected/diverse-teacher-wave-20260929"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source() -> tuple[dict, dict[str, bytes]]:
    receipt = json.loads((HERE / "source-receipt.json").read_text())
    raw = load_archive(local=(HERE / "source.tar.gz").is_file())
    if len(raw) != receipt["archive_bytes"] or sha(raw) != receipt["archive_sha256"]:
        raise ValueError("local source archive does not match public receipt")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
        manifest_raw = bundle.extractfile("manifest.json").read()
        if sha(manifest_raw) != receipt["manifest_sha256"]:
            raise ValueError("source manifest hash mismatch")
        manifest = json.loads(manifest_raw)
        photos = {}
        for row in manifest["rows"]:
            if row["mode"] != "image_to_paint":
                continue
            name = row["path"]
            if not name.startswith("references/") or ".." in Path(name).parts:
                raise ValueError("unsafe source image member")
            image = bundle.extractfile(name).read()
            if sha(image) != row["sha256"] or len(image) != row["bytes"]:
                raise ValueError("source image hash/size mismatch")
            photos[row["id"]] = image
    return manifest, photos


def published_prefixes(manifest: dict) -> tuple[dict[str, tuple[str, int]], list[str]]:
    files = set(HfApi(token=False).list_repo_files(DATASET, repo_type="dataset"))
    sizes = {row["shard"] for row in manifest["rows"]}
    selected = {}
    missing = []
    for shard in sorted(sizes):
        count = sum(row["shard"] == shard for row in manifest["rows"])
        for prefix in sorted({1, 4, 8, count}, reverse=True):
            if prefix > count:
                continue
            run_id = f"diverse-teacher-20260929-{shard}-n{prefix}"
            if f"runs/{run_id}/receipt.json" in files:
                selected[shard] = (run_id, prefix)
                break
        else:
            missing.append(shard)
    return selected, missing


def verified_bundle(run_id: str, cache: Path) -> tuple[dict, tarfile.TarFile]:
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / f"{run_id}.tar.gz"
    for attempt in range(3):
        try:
            report = download(run_id, archive_path)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    bundle = tarfile.open(archive_path, mode="r:gz")
    members = bundle.getmembers()
    if not members or any(not member.isfile() or member.issym() or member.islnk()
                          or Path(member.name).is_absolute() or ".." in Path(member.name).parts
                          for member in members):
        bundle.close()
        raise ValueError(f"unsafe archive members in {run_id}")
    return report, bundle


def asset(raw: bytes, suffix: str, output: Path, folder: str) -> str:
    digest = sha(raw)
    path = output / folder / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return f"{folder}/{path.name}"


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError("select a fresh review output directory")
    manifest, photos = source()
    rows = {row["id"]: row for row in manifest["rows"]}
    selected, missing = published_prefixes(manifest)
    if not selected:
        raise ValueError("no public result prefixes found")
    output.mkdir(parents=True)
    candidates = {}
    receipts = []
    for shard, (run_id, prefix) in selected.items():
        report, bundle = verified_bundle(run_id, COLLECTED / "verified-archive-cache")
        receipts.append({"shard": shard, "run_id": run_id, "prefix": prefix,
                         "archive_sha256": report["sha256"], "archive_bytes": report["bytes"]})
        try:
            members = {member.name: member for member in bundle}
            episode_names = sorted(name for name in members if name.startswith("episodes/")
                                   and name.endswith("/episode.json"))
            if len(episode_names) != prefix:
                raise ValueError(f"unexpected episode count for {run_id}")
            for name in episode_names:
                episode = json.load(bundle.extractfile(members[name]))
                task_id = episode.get("reference_id")
                if task_id not in rows or rows[task_id]["shard"] != shard or task_id in candidates:
                    raise ValueError(f"unexpected/duplicate task in {run_id}: {task_id}")
                row = rows[task_id]
                if episode.get("reference_sha256") != row["sha256"]:
                    raise ValueError(f"reference hash mismatch: {task_id}")
                expected_model = f"xiaomi/mimo-v2.6-{row['tier']}"
                if episode.get("model") != expected_model:
                    raise ValueError(f"model mismatch: {task_id}")
                turns = episode.get("turns") or []
                if len(turns) > 4:
                    raise ValueError(f"turn limit exceeded: {task_id}")
                turn_rows = []
                folder = name.rsplit("/", 1)[0]
                for number, turn in enumerate(turns, start=1):
                    turn_json = f"{folder}/turn-{number:02d}.json"
                    if turn_json not in members:
                        raise ValueError(f"missing turn receipt: {task_id}/{number}")
                    receipt_raw = bundle.extractfile(members[turn_json]).read()
                    if json.loads(receipt_raw).get("turn") != number:
                        raise ValueError(f"turn receipt mismatch: {task_id}/{number}")
                    transcript = asset(receipt_raw, ".json", output, "transcripts")
                    canvas_name = f"{folder}/turn-{number:02d}.png"
                    program_name = f"{folder}/turn-{number:02d}.program.js"
                    image = None
                    program = None
                    if canvas_name in members:
                        raw = bundle.extractfile(members[canvas_name]).read()
                        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise ValueError(f"bad canvas PNG: {task_id}/{number}")
                        image = asset(raw, ".png", output, "assets")
                    if program_name in members:
                        program = asset(bundle.extractfile(members[program_name]).read(),
                                        ".js", output, "programs")
                    response = turn.get("response") or {}
                    usage = response.get("usage") or {}
                    turn_rows.append({"turn": number, "canvas": image, "program": program,
                                      "transcript": transcript,
                                      "plan": turn.get("plan"),
                                      "render_feedback": turn.get("render_feedback"),
                                      "render_seconds": (turn.get("render") or {}).get("render_seconds"),
                                      "api_seconds": response.get("latency_seconds"),
                                      "render_valid": (turn.get("render") or {}).get("valid") is True,
                                      "finished_without_code": (turn.get("render") or {}).get("finished_without_code") is True,
                                      "api_status": turn.get("api_status"),
                                      "prompt_tokens": usage.get("prompt_tokens"),
                                      "completion_tokens": usage.get("completion_tokens"),
                                      "cost_usd": usage.get("cost")})
                final_name = episode.get("final_valid_canvas")
                final_canvas = None
                if isinstance(final_name, str):
                    matching = [turn["canvas"] for turn in turn_rows
                                if turn["canvas"] and f"turn-{turn['turn']:02d}.png" in final_name]
                    if len(matching) != 1:
                        raise ValueError(f"final canvas missing from turns: {task_id}")
                    final_canvas = matching[0]
                reference_image = None
                if row["mode"] == "image_to_paint":
                    reference_image = asset(photos[task_id], ".jpg", output, "assets")
                valid_turns = [turn for turn in turn_rows if turn["render_valid"] and turn["canvas"]]
                candidates[task_id] = {"id": task_id, "shard": shard, "mode": row["mode"],
                                       "tier": row["tier"], "category": row["category"],
                                       "brief": row.get("task_text"), "reference_image": reference_image,
                                       "source_sha256": row["sha256"], "status": episode.get("status"),
                                       "final_canvas": final_canvas, "turns": turn_rows,
                                       "terminal_turn_valid": bool(turn_rows and turn_rows[-1]["render_valid"]),
                                       "terminal_action": ("finish" if turn_rows and turn_rows[-1]["finished_without_code"]
                                                           else "paint" if turn_rows else None),
                                       "first_valid_turn": valid_turns[0]["turn"] if valid_turns else None,
                                       "valid_turn_count": len(valid_turns),
                                       "distinct_valid_canvases": len({turn["canvas"] for turn in valid_turns}),
                                       "prompt_tokens": sum(turn["prompt_tokens"] or 0 for turn in turn_rows),
                                       "completion_tokens": sum(turn["completion_tokens"] or 0 for turn in turn_rows),
                                       "total_tokens": episode.get("total_tokens"),
                                       "cost_complete": episode.get("cost_complete") is True,
                                       "known_cost_usd": episode.get("known_cost"),
                                       "training_admission": "unreviewed_candidate",
                                       "public_run_id": run_id}
        finally:
            bundle.close()
    by_status = {}
    for case in candidates.values():
        key = str(case["status"])
        by_status[key] = by_status.get(key, 0) + 1
    report = {"schema": "painter.diverse-teacher-review.v2",
              "source_archive_sha256": json.loads((HERE / "source-receipt.json").read_text())["archive_sha256"],
              "source_rows": len(rows), "published_candidates": len(candidates),
              "saved_valid_canvas": sum(bool(case["final_canvas"]) for case in candidates.values()),
              "terminal_turn_valid": sum(case["terminal_turn_valid"] for case in candidates.values()),
              "valid_first_turn": sum(case["first_valid_turn"] == 1 for case in candidates.values()),
              "revised_after_first_valid": sum(case["distinct_valid_canvases"] > 1 for case in candidates.values()),
              "missing_shards": missing, "selected_public_archives": receipts,
              "statuses": by_status, "cost_note": "Missing provider cost fields are unknown, not zero.",
              "candidates": [candidates[key] for key in sorted(candidates)]}
    (output / "candidates.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "index.html").write_text(page(report))
    return {key: report[key] for key in ("source_rows", "published_candidates", "saved_valid_canvas",
                                          "terminal_turn_valid", "missing_shards", "statuses")}


def page(report: dict) -> str:
    cards = []
    for case in report["candidates"]:
        ref = (f'<img loading="lazy" src="{case["reference_image"]}" alt="Reference">'
               if case["reference_image"] else f'<p class="brief">{html.escape(case["brief"] or "")}</p>')
        final = (f'<img loading="lazy" src="{case["final_canvas"]}" alt="Last saved valid canvas">'
                 if case["final_canvas"] else '<p>No valid canvas</p>')
        turns = "".join(
            f'<div class="turn"><strong>Turn {turn["turn"]}</strong><br>'
            + (f'<img loading="lazy" src="{turn["canvas"]}" alt="Turn {turn["turn"]}">' if turn["canvas"]
               else '<em>Finish; retained prior valid canvas</em>' if turn["finished_without_code"]
               else '<em>No canvas</em>')
            + (f'<p><a href="{turn["program"]}">Code</a> · ' if turn["program"] else '<p>')
            + f'<a href="{turn["transcript"]}">Full turn receipt</a></p>'
            + f'<small>{html.escape(str(turn["api_status"]))} · {turn["completion_tokens"] or "?"} tokens · '
            + f'{turn["api_seconds"] if turn["api_seconds"] is not None else "?"}s API · '
            + f'{turn["render_seconds"] if turn["render_seconds"] is not None else "?"}s render</small>'
            + f'<p>{html.escape(str(turn["plan"] or ""))}</p>'
            + f'<p>{html.escape(str(turn["render_feedback"] or ""))}</p></div>'
            for turn in case["turns"])
        title = html.escape(f'{case["id"]} · {case["category"]} · {case["tier"]}')
        cards.append(f'<article data-mode="{case["mode"]}" data-tier="{case["tier"]}" data-search="{title.lower()}">'
                     f'<h2>{title}</h2><p>{html.escape(str(case["status"]))} · {len(case["turns"])} turns · '
                     f'{case["total_tokens"]} tokens · admission unreviewed</p>'
                     f'<div class="pair"><section><h3>Source</h3>{ref}</section><section><h3>Last saved valid canvas</h3>{final}</section></div>'
                     f'<details><summary>Inspect every turn</summary><div class="turns">{turns}</div></details></article>')
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Diverse painting teacher candidates</title><style>body{font:15px system-ui;background:#f4f0e9;color:#242322;margin:0}"
            "header{position:sticky;top:0;background:#f4f0e9ef;padding:15px 20px;border-bottom:1px solid #c8c2b7}"
            "main{max-width:1500px;margin:auto;padding:20px}article{background:white;border:1px solid #ddd3c5;border-radius:10px;padding:15px;margin:20px 0}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;height:450px;object-fit:contain;background:#ede9e2}"
            ".brief{white-space:pre-wrap;font-size:1.2rem;padding:25px}.turns{display:flex;gap:12px;overflow-x:auto}.turn{min-width:250px;width:250px}"
            ".turn img{width:100%}select,input{font:inherit;padding:5px;margin-right:10px}summary{cursor:pointer;padding:10px}"
            "@media(max-width:700px){.pair{grid-template-columns:1fr}.pair img{height:320px}}</style>"
            f'<header><h1>Teacher candidates · {report["published_candidates"]}/{report["source_rows"]} published, '
            f'{report["saved_valid_canvas"]} saved canvases, {report["terminal_turn_valid"]} valid terminal turns</h1>'
            '<p>All turns are preserved. An invalid terminal turn may retain an earlier canvas. No candidate is admitted to training by this gallery.</p>'
            '<label>Mode <select id="mode"><option value="all">All</option><option value="text_to_paint">Text</option><option value="image_to_paint">Photo</option></select></label>'
            '<label>Tier <select id="tier"><option value="all">All</option><option value="flash">Flash</option><option value="pro">Pro</option></select></label>'
            '<label>Find <input id="search" type="search"></label></header><main>' + "".join(cards)
            + "</main><script>const mode=document.getElementById('mode'),tier=document.getElementById('tier'),search=document.getElementById('search');"
              "function filter(){for(const card of document.querySelectorAll('article')){card.hidden=(mode.value!=='all'&&card.dataset.mode!==mode.value)||(tier.value!=='all'&&card.dataset.tier!==tier.value)||!card.dataset.search.includes(search.value.toLowerCase())}}"
              "mode.onchange=tier.onchange=search.oninput=filter;</script></html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))
