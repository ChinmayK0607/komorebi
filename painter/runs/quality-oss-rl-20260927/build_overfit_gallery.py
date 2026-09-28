"""Create an offline, matched four-scene SFT/RL painting review gallery."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(*, manifest: Path, rollout_root: Path, outdir: Path, step: int,
          latest_episode: bool = False) -> dict:
    cases = json.loads(manifest.read_text())["cases"]
    if len(cases) != 4:
        raise ValueError("matched gallery requires exactly four cases")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for case in cases:
        case_id = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
            raise ValueError(f"unsafe case id: {case_id!r}")
        reference = Path(case["reference"])
        if not reference.is_file() or sha(reference) != case["reference_sha256"]:
            raise ValueError(f"reference changed: {case_id}")
        ref_name = f"{case_id}-reference{reference.suffix.lower()}"
        shutil.copy2(reference, outdir / ref_name)
        cells = [("Reference", ref_name, [])]
        record = {"id": case_id, "reference_sha256": case["reference_sha256"]}
        for policy, label in (("sft-step512", "SFT step 512"), (f"rl-step-{step}", f"RL step {step}")):
            episodes = [p for p in (rollout_root / policy / case_id).glob("*") if p.is_dir()]
            if len(episodes) != 1 and not (latest_episode and episodes):
                raise ValueError(f"expected one {policy} episode for {case_id}, found {len(episodes)}")
            episode = max(episodes, key=lambda path: path.stat().st_mtime_ns) if latest_episode else episodes[0]
            turns = []
            attempted_turns = []
            for turn in sorted(episode.glob("turn-*"), key=lambda p: int(p.name.split("-")[-1])):
                receipt = turn / "painting.json"
                image = turn / "painting.png"
                program = turn / "program.js"
                if not receipt.is_file():
                    continue
                data = json.loads(receipt.read_text())
                evidence = outdir / "evidence" / case_id / policy / turn.name
                evidence.mkdir(parents=True, exist_ok=True)
                shutil.copy2(receipt, evidence / "painting.json")
                if program.is_file():
                    shutil.copy2(program, evidence / "program.js")
                attempted_turns.append({
                    "turn": int(turn.name.split("-")[-1]), "valid": bool(data.get("valid")),
                    "program_sha256": sha(program) if program.is_file() else None,
                    "receipt_sha256": sha(receipt),
                    "program": str((evidence / "program.js").relative_to(outdir)) if program.is_file() else None,
                    "receipt": str((evidence / "painting.json").relative_to(outdir)),
                    "errors": data.get("errors") or data.get("error"),
                })
                if not data.get("valid") or not image.is_file():
                    continue
                name = f"{case_id}-{policy}-{turn.name}.png"
                shutil.copy2(image, outdir / name)
                turns.append({"turn": int(turn.name.split("-")[-1]), "image": name,
                              "sha256": sha(image), "render_seconds": data.get("painting_seconds")})
            final = turns[-1]["image"] if turns else None
            record[policy] = {"episode": str(episode), "valid_turns": turns,
                              "attempted_turns": attempted_turns,
                              "final": final, "final_sha256": turns[-1]["sha256"] if turns else None}
            cells.append((label, final, turns))
        rows.append(record)
        pieces = []
        for label, image, turns in cells:
            image_html = (f'<img src="{html.escape(image)}" loading="lazy" alt="{html.escape(label)} for {html.escape(case_id)}">'
                          if image else '<div class="missing">No valid canvas</div>')
            links = " ".join(f'<a href="{html.escape(item["image"])}">turn {item["turn"]}</a>' for item in turns)
            policy = ("sft-step512" if label == "SFT step 512" else
                      f"rl-step-{step}" if label == f"RL step {step}" else None)
            attempts = record.get(policy, {}).get("attempted_turns", []) if policy else []
            links += " " + " ".join(
                f'<a href="{html.escape(item["receipt"])}">turn {item["turn"]} '
                f'{"receipt" if item["valid"] else "error"}</a>'
                for item in attempts
            )
            pieces.append(f'<div class="cell"><h3>{html.escape(label)}</h3>{image_html}<div class="turns">{links}</div></div>')
        cards.append(f'<section><h2>{html.escape(case_id)}</h2><div class="grid">{"".join(pieces)}</div></section>')
    report = {"schema": "matched-overfit-gallery-v1", "step": step, "rows": rows}
    (outdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Four-reference painting overfit review</title><style>
body{font:16px system-ui,sans-serif;background:#141619;color:#ececec;margin:0;padding:24px;line-height:1.45}
main{max-width:1650px;margin:auto}section{margin:24px 0 46px;padding:16px;background:#202328;border-radius:12px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.cell{min-width:0}
.cell img{width:100%;height:auto;max-height:660px;object-fit:contain;background:white;border-radius:8px}
h1,h2,h3{margin:0 0 10px}h3{font-size:1rem}.turns{margin-top:8px;min-height:24px}
.turns a{color:#9dccff;margin-right:12px}.missing{height:360px;display:grid;place-items:center;background:#333}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1>Four-reference painting overfit review</h1>
<p>Same references, four-turn cap, sampling seed 918200, and pinned p5.brush renderer. Evaluate final canvases first; turn links show the path.</p>
""" + "".join(cards) + "</main></body></html>"
    (outdir / "index.html").write_text(page)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--latest-episode", action="store_true",
                        help="select the newest episode when a matched policy was rerun")
    args = parser.parse_args()
    report = build(manifest=args.manifest, rollout_root=args.rollout_root,
                   outdir=args.outdir, step=args.step,
                   latest_episode=args.latest_episode)
    print(json.dumps({"cases": len(report["rows"]), "gallery": str(args.outdir / "index.html")}))


if __name__ == "__main__":
    main()
