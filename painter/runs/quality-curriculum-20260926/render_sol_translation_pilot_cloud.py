#!/usr/bin/env python3
"""Render matched original/brush-translated Sol programs on Codex Cloud Linux."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "painter/benchmarks/openrouter-teachers-20260922")]
from render_teacher_batch_cloud import stage  # noqa: E402
from run import render_program  # noqa: E402
from translate_sol_native import translate  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_one(row: dict, source_root: Path, output: Path, renderer: Path,
               python: Path, browser: Path, timeout: int) -> dict:
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    matches = [r for r in source_manifest["rows"] if r["id"] == row["id"]]
    if len(matches) != 1:
        raise ValueError(f"missing source member: {row['audit_id']}")
    source_row = matches[0]
    program = (source_root / source_row["program"]).read_bytes()
    image = (source_root / source_row["input"]).read_bytes()
    if (source_row["mode"] != "text_to_image" or sha(program) != row["source_program_sha256"]
            or sha(image) != row["input_sha256"]):
        raise ValueError(f"source program/input mismatch: {row['audit_id']}")
    replacement, status = translate(program.decode())
    if status != "candidate" or replacement is None or sha(replacement.encode()) != row["candidate_program_sha256"]:
        raise ValueError(f"translation changed: {row['audit_id']} {status}")
    episode = output / "episodes" / row["audit_id"]
    episode.mkdir(parents=True, exist_ok=False)
    (episode / "input.txt").write_bytes(image)
    (episode / "original.js").write_bytes(program)
    (episode / "translated.js").write_text(replacement)
    results = {}
    for label, filename in (("original", "original.js"), ("translated", "translated.js")):
        target = episode / f"{label}.png"
        started = time.monotonic()
        try:
            receipt = render_program(root=output, source=episode / filename, output=target,
                                     renderer=renderer, renderer_python=python,
                                     browser_path=browser, timeout=timeout,
                                     run_as_user="painter", local=False)
        except Exception as exc:
            receipt = {"valid": False, "error_code": f"renderer_exception_{type(exc).__name__}",
                       "error": str(exc)[:240]}
        results[label] = {"valid": receipt.get("valid") is True and target.is_file(),
                          "canvas_sha256": sha(target.read_bytes()) if target.is_file() else None,
                          "seconds": round(time.monotonic() - started, 3),
                          "error_code": receipt.get("error_code"),
                          "error": str(receipt.get("errors") or receipt.get("error") or "")[:240]}
    result = {"audit_id": row["audit_id"], "id": row["id"], "category": row["category"],
              "source_batch": row["source_batch"], "input_sha256": row["input_sha256"],
              "original_program_sha256": sha(program),
              "translated_program_sha256": sha(replacement.encode()),
              "expected_original_canvas_sha256": row["original_canvas_sha256"],
              "original_reproduction_matches": results["original"]["canvas_sha256"] == row["original_canvas_sha256"],
              "renders": results, "visual_review": "pending"}
    (episode / "status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run(plan: Path, run_id: str, timeout: int, workers: int) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("render only on Codex Cloud Linux")
    source = json.loads(plan.read_text())
    if source.get("schema") != "painter.sol-native-brush-pilot.v1" or len(source["rows"]) != 8:
        raise ValueError("unexpected pilot plan")
    runtime = ROOT / ".painter-cloud-runtime"
    python = runtime / "renderer-env/bin/python"
    browser = runtime / "browsers"
    renderer = ROOT / "painter/vendor/integrations/watercolour/renderer.py"
    if not python.is_file() or not browser.is_dir():
        raise RuntimeError("Codex Cloud renderer setup is missing")
    output = ROOT / "painter/collected/quality-curriculum-20260926/rendered-cloud" / run_id
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "source-staging"
    roots = {}
    receipts = {}
    for batch in sorted({row["source_batch"] for row in source["rows"]}):
        manifest, receipt = stage(batch, staged / batch)
        expected = source["source_receipts"][batch]
        if any(receipt[key] != expected[key] for key in expected):
            raise ValueError(f"public source receipt changed: {batch}")
        roots[batch] = staged / batch
        receipts[batch] = {"dataset_commit": receipt["dataset_commit"],
                           "archive_sha256": receipt["archive_sha256"], "count": manifest["count"]}
    started = time.monotonic()
    outcomes = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(render_one, row, roots[row["source_batch"]], output,
                                   renderer, python, browser, timeout): row["audit_id"]
                   for row in source["rows"]}
        for future in as_completed(pending):
            result = future.result()
            outcomes[result["audit_id"]] = result
            print(json.dumps({"progress": f"{len(outcomes)}/8", "id": result["audit_id"],
                              "original_valid": result["renders"]["original"]["valid"],
                              "translated_valid": result["renders"]["translated"]["valid"]}), flush=True)
    shutil.rmtree(staged)
    result = {"schema": "painter.sol-native-brush-render-pilot.v1", "run_id": run_id,
              "plan_sha256": sha(plan.read_bytes()), "source_receipts": receipts,
              "renderer_sha256": sha(renderer.read_bytes()), "timeout_seconds": timeout,
              "workers": workers, "elapsed_seconds": round(time.monotonic() - started, 3),
              "rows": [outcomes[row["audit_id"]] for row in source["rows"]],
              "training_admission": "none_without_pairwise_visual_review"}
    (output / "run-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    cards = []
    for row in result["rows"]:
        ident = escape(row["audit_id"])
        episode = output / "episodes" / row["audit_id"]
        prompt = escape((episode / "input.txt").read_text())
        figures = []
        for label in ("original", "translated"):
            render = row["renders"][label]
            if render["valid"]:
                body = f'<img src="episodes/{ident}/{label}.png" alt="{label} painting">'
            else:
                body = f'<div class="missing">{escape(str(render["error_code"] or "No canvas"))}</div>'
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
    (output / "gallery.html").write_text(html)
    return {"run_id": run_id, "elapsed_seconds": result["elapsed_seconds"],
            "original_valid": sum(x["renders"]["original"]["valid"] for x in result["rows"]),
            "translated_valid": sum(x["renders"]["translated"]["valid"] for x in result["rows"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "SOL_BRUSH_PILOT_8.json")
    parser.add_argument("--run-id", default="sol-brush-translation-pilot-20260927")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 30 <= args.timeout <= 600:
        raise ValueError("workers/timeout outside pilot bounds")
    print(json.dumps(run(args.plan, args.run_id, args.timeout, args.workers)))


if __name__ == "__main__":
    main()
