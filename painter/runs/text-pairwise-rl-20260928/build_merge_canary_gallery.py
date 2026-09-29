"""Portable per-turn gallery for the matched four-policy text-paint canary."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting_report import _trace_data, read_traces  # noqa: E402


POLICIES = ("sft", "rl", "mix025", "mix050")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest: Path, eval_root: Path, output: Path) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    ids = [row["id"] for row in cases]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("expected eight unique matched cases")
    if output.exists():
        raise FileExistsError(output)
    traces: dict[str, dict[str, dict]] = {}
    for policy in POLICIES:
        records, errors, files = read_traces(eval_root / policy)
        if errors or not files:
            raise ValueError(f"{policy} traces missing or corrupt: {errors[:2]}")
        found: dict[str, dict] = {}
        for trace in records:
            data, info = _trace_data(trace)
            ident = data.get("task_id") or data.get("name")
            if ident in ids:
                if ident in found:
                    raise ValueError(f"duplicate trace: {policy}/{ident}")
                found[ident] = info
        if set(found) != set(ids):
            raise ValueError(f"{policy} completed {len(found)}/8 cases")
        traces[policy] = found
    output.mkdir(parents=True)
    images = output / "images"
    programs = output / "programs"
    images.mkdir()
    programs.mkdir()
    cards: list[str] = []
    receipts: list[dict] = []
    for row in cases:
        ident = row["id"]
        brief = (manifest.parent / row["reference"]).read_text().strip()
        cells: list[str] = []
        for policy in POLICIES:
            info = traces[policy][ident]
            turns = info.get("turns") or []
            turn_html: list[str] = []
            turn_receipts: list[dict] = []
            for index, turn in enumerate(turns, 1):
                source = Path(turn["canvas"]) if turn.get("canvas") else None
                valid = bool(turn.get("valid") and source and source.is_file())
                url = None
                code_url = None
                if valid:
                    target = images / f"{ident}-{policy}-t{index}.png"
                    shutil.copyfile(source, target)
                    url = f"images/{target.name}"
                    program = source.parent / "program.js"
                    if program.is_file() and not turn.get("no_change"):
                        code = programs / f"{ident}-{policy}-t{index}.js"
                        shutil.copyfile(program, code)
                        code_url = f"programs/{code.name}"
                caption = f"turn {index} · {'valid' if valid else 'invalid'}"
                if turn.get("no_change"):
                    caption += " · no change"
                if isinstance(turn.get("render_seconds"), (int, float)):
                    caption += f" · {turn['render_seconds']:.1f}s render"
                view = (f"<a href='{url}'><img loading='lazy' src='{url}' alt='{ident} {policy} turn {index}'></a>"
                        if url else "<div class='missing'>No valid canvas</div>")
                if code_url:
                    caption += f" · <a href='{code_url}'>program</a>"
                turn_html.append(f"<figure>{view}<figcaption>{caption}</figcaption></figure>")
                turn_receipts.append({"turn": index, "valid": valid,
                                      "no_change": bool(turn.get("no_change")),
                                      "render_seconds": turn.get("render_seconds"),
                                      "image": url, "code": code_url,
                                      "image_sha256": sha(images / Path(url).name) if url else None})
            final_valid = bool(info.get("final_canvas_valid"))
            cells.append(f"<div class='policy'><h3>{policy} · {'valid' if final_valid else 'invalid final'}</h3>"
                         + "".join(turn_html) + "</div>")
            receipts.append({"id": ident, "policy": policy,
                             "final_canvas_valid": final_valid, "turns": turn_receipts})
        cards.append(f"<section><h2>{html.escape(ident)}</h2><p class='brief'>{html.escape(brief)}</p>"
                     + "<div class='grid'>" + "".join(cells) + "</div></section>")
    page = """<!doctype html><html><head><meta charset='utf-8'>
<title>Matched text-paint merge canary</title><style>
body{font:16px system-ui;background:#151515;color:#eee;margin:0;padding:24px}
section{background:#222;border:1px solid #444;border-radius:12px;margin:24px 0;padding:20px}
h1,h2,h3{margin:0 0 10px}.brief{max-width:1000px;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:14px;overflow-x:auto}
.policy{min-width:260px;border:1px solid #555;border-radius:8px;padding:10px}
figure{margin:10px 0}img,.missing{width:100%;aspect-ratio:1;object-fit:contain;background:#fff}
.missing{display:grid;place-items:center;background:#383838;color:#bbb}
figcaption{font-size:13px;color:#ccc;margin-top:5px}a{color:#a9ceff}
</style></head><body><h1>Matched text-paint merge canary</h1>
<p>Eight selected development briefs · same base, settings, seed, and two-turn cap.
SFT, RL step 16, and exact 25%/50% RL adapter blends. Invalid turns remain visible.
This selected set is not held-out proof of a general gain.</p>
""" + "\n".join(cards) + "</body></html>"
    (output / "index.html").write_text(page)
    receipt = {"schema": "matched-text-paint-merge-gallery-v1",
               "manifest_sha256": sha(manifest), "case_count": len(cases),
               "policies": list(POLICIES), "rows": receipts}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manifest, args.eval_root, args.output)
    print(json.dumps({"case_count": result["case_count"],
                      "policies": result["policies"]}))


if __name__ == "__main__":
    main()
