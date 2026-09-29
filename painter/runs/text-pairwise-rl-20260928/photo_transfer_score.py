"""Score a seven-photo, three-policy transfer diagnostic and make a gallery."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "painter/runs/quality-oss-rl-20260927"))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402
from gateway_judge import judge_pair, load_key  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_canvases(eval_root: Path, expected_ids: set[str]) -> dict[str, Path | None]:
    traces, errors, files = read_traces(eval_root)
    if errors or not files:
        raise ValueError(f"corrupt or absent native traces: {eval_root}: {errors[:2]}")
    found: dict[str, Path | None] = {}
    for trace in traces:
        data, info = _trace_data(trace)
        ident = data.get("task_id") or data.get("name")
        if ident not in expected_ids:
            continue
        if ident in found:
            raise ValueError(f"duplicate trace for {ident}")
        raw = info.get("final_canvas")
        path = Path(raw) if isinstance(raw, str) and raw else None
        if info.get("final_canvas_valid") and path is not None and path.is_file():
            found[ident] = path
        else:
            found[ident] = None
    if set(found) != expected_ids:
        raise ValueError(f"missing photo traces: {sorted(expected_ids - set(found))}")
    return found


def compare(key: str, reference: Path, candidate: Path | None,
            baseline: Path | None) -> dict:
    if candidate is None and baseline is None:
        return {"status": "both_invalid", "score_candidate": None}
    if candidate is None:
        return {"status": "candidate_invalid", "score_candidate": 0.0}
    if baseline is None:
        return {"status": "baseline_invalid", "score_candidate": 1.0}
    result = judge_pair(key=key, reference=reference, a=candidate, b=baseline)
    return {"status": result["status"], "score_candidate": result["score_A"],
            "judge": result}


def build(manifest: Path, roots: dict[str, Path], key_file: Path,
          output: Path) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    if len(cases) != 7 or len({row["id"] for row in cases}) != 7:
        raise ValueError("expected seven distinct held-out photos")
    expected_ids = {row["id"] for row in cases}
    if set(roots) != {"sft", "rl16", "final"}:
        raise ValueError("expected SFT, step-16, and final policy roots")
    found = {label: final_canvases(path, expected_ids) for label, path in roots.items()}
    images = output / "images"
    images.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in cases:
        ident = row["id"]
        reference = Path(row["reference"])
        if not reference.is_file() or sha(reference) != row["reference_sha256"]:
            raise ValueError(f"changed or missing reference for {ident}")
        saved = images / f"{ident}-reference{reference.suffix.lower()}"
        shutil.copyfile(reference, saved)
        item = {"id": ident, "reference": saved.name,
                "reference_sha256": sha(saved), "paintings": {}}
        for label in ("sft", "rl16", "final"):
            source = found[label][ident]
            if source is None:
                item["paintings"][label] = None
                continue
            target = images / f"{ident}-{label}.png"
            shutil.copyfile(source, target)
            item["paintings"][label] = {"file": target.name, "sha256": sha(target)}
        rows.append(item)
    key = load_key(key_file)

    def score(item: dict, baseline: str) -> tuple[str, str, dict]:
        ident = item["id"]
        ref = images / item["reference"]
        candidate = item["paintings"]["final"]
        old = item["paintings"][baseline]
        result = compare(key, ref, images / candidate["file"] if candidate else None,
                         images / old["file"] if old else None)
        return ident, baseline, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = list(pool.map(lambda pair: score(*pair),
                               [(row, baseline) for row in rows
                                for baseline in ("sft", "rl16")]))
    by_id = {row["id"]: row for row in rows}
    for ident, baseline, result in scores:
        by_id[ident].setdefault("pairwise", {})[f"final_vs_{baseline}"] = result
    cards = []
    for row in rows:
        cells = []
        for label, name in [("reference", row["reference"]),
                            *[(label, row["paintings"][label]["file"]
                               if row["paintings"][label] else None)
                              for label in ("sft", "rl16", "final")]]:
            body = (f'<img loading="lazy" src="images/{html.escape(name)}">'
                    if name else '<div class="missing">No valid canvas</div>')
            cells.append(f"<div><strong>{html.escape(label)}</strong>{body}</div>")
        marks = " · ".join(
            f"final vs {baseline}: {row['pairwise'][f'final_vs_{baseline}']['score_candidate']}"
            for baseline in ("sft", "rl16"))
        cards.append(f"<section><h2>{html.escape(row['id'])}</h2><p>{html.escape(marks)}</p>"
                     f"<div class='row'>{''.join(cells)}</div></section>")
    page = """<!doctype html><meta charset="utf-8"><title>Photo transfer</title>
<style>body{background:#171717;color:#eee;font:16px system-ui;margin:24px}
section{border:1px solid #555;padding:16px;margin:18px 0} .row{display:flex;gap:12px;overflow-x:auto}
.row>div{min-width:280px}img,.missing{display:block;width:280px;height:280px;object-fit:contain;
background:#fff;margin-top:8px}.missing{background:#333;display:grid;place-items:center}</style>
<h1>Held-out photo to painting transfer</h1><p>Seven photos; identical four-turn, seed-918200
sampling for SFT step 512, RL step 16, and the text-RL continuation.</p>
""" + "\n".join(cards)
    (output / "index.html").write_text(page)
    summary = {"schema": "text-rl-photo-transfer-v1", "cases": len(rows),
               "valid": {label: sum(row["paintings"][label] is not None for row in rows)
                         for label in roots}, "rows": rows}
    (output / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"cases": 7, "valid": summary["valid"], "output": str(output / "index.html")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sft", required=True, type=Path)
    parser.add_argument("--rl16", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest,
                           {"sft": args.sft, "rl16": args.rl16, "final": args.final},
                           args.key_file, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
