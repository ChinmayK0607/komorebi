#!/usr/bin/env python3
"""Build a local reference / MiMo Pro / Luna high comparison from verified evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cloud"))
from replay_renderer import public_bytes  # noqa: E402

REFS = ("coco128-000000000110", "coco128-000000000247", "coco128-000000000520")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="luna-high-initial-20260924")
    parser.add_argument("--api-run-id", action="append", default=[],
                        help="paid Gateway Luna high episode; repeat for more references")
    parser.add_argument("--revision-run-id", help="provider-free visual revision render")
    parser.add_argument("--benchmark", type=Path, default=HERE)
    parser.add_argument("--selection-gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, receipt = public_bytes(args.run_id)
    api_archives = [public_bytes(run_id) for run_id in args.api_run_id]
    revision_archive, revision_receipt = public_bytes(args.revision_run_id) if args.revision_run_id else (None, None)
    selection = json.loads((args.selection_gallery / "manifest.json").read_text())
    output = args.output.resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    revised = {}
    if revision_archive is not None:
        with tarfile.open(fileobj=io.BytesIO(revision_archive), mode="r:gz") as revision_bundle:
            revision_members = {m.name: m for m in revision_bundle.getmembers() if m.isfile()}
            for ref in REFS:
                episode = json.load(revision_bundle.extractfile(revision_members[f"episodes/{ref}/episode.json"]))
                canvas_name = f"episodes/{ref}/turn-01.png"
                target = None
                if canvas_name in revision_members:
                    target = assets / f"{ref}-luna-revised.png"
                    data = revision_bundle.extractfile(revision_members[canvas_name]).read()
                    expected = episode.get("render", {}).get("receipt", {}).get("png_sha256")
                    if expected and hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError(f"revised Luna PNG hash mismatch: {ref}")
                    target.write_bytes(data)
                revised[ref] = {"canvas": f"assets/{target.name}" if target else None,
                                "status": episode.get("status"),
                                "elapsed_seconds": episode.get("render", {}).get("elapsed_seconds")}
    api_episodes = {}
    for api_archive, api_receipt in api_archives:
        with tarfile.open(fileobj=io.BytesIO(api_archive), mode="r:gz") as api_bundle:
            api_members = {m.name: m for m in api_bundle.getmembers() if m.isfile()}
            for name in api_members:
                if not name.startswith("episodes/") or not name.endswith("/episode.json"):
                    continue
                episode = json.load(api_bundle.extractfile(api_members[name]))
                if episode.get("model") != "openai/gpt-6-luna":
                    continue
                ref = episode["reference_id"]
                if ref in api_episodes:
                    raise ValueError(f"multiple paid Luna episodes for {ref}")
                canvas_name = episode.get("final_valid_canvas")
                target = None
                if canvas_name and canvas_name in api_members:
                    target = assets / f"{ref}-luna-api.png"
                    target.write_bytes(api_bundle.extractfile(api_members[canvas_name]).read())
                api_episodes[ref] = {
                    "canvas": f"assets/{target.name}" if target else None,
                    "status": episode.get("status"),
                    "turns": len(episode.get("turns", [])),
                    "tokens": episode.get("total_tokens"),
                    "active_seconds": episode.get("active_seconds"),
                    "cost_complete": episode.get("cost_complete"),
                    "total_cost": episode.get("total_cost") if episode.get("cost_complete") else None,
                    "receipt": api_receipt,
                }
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        members = {m.name: m for m in bundle.getmembers() if m.isfile()}
        summary = json.load(bundle.extractfile(members["run-summary.json"]))
        for ref in REFS:
            key = f"quality|{ref}|xiaomi/mimo-v2.6-pro"
            mimo = selection["candidates"][key]
            # The offline replay may recover the interrupted last program.
            replay = selection["replayed"].get(key)
            choice = replay if replay and replay.get("canvas") else mimo
            reference_source = args.benchmark / "references" / f"{ref}.jpg"
            ref_target = assets / f"{ref}.jpg"
            shutil.copyfile(reference_source, ref_target)
            mimo_target = None
            if choice.get("canvas"):
                source = args.selection_gallery / choice["canvas"]
                mimo_target = assets / f"{ref}-mimo.png"
                shutil.copyfile(source, mimo_target)
            episode_path = f"episodes/{ref}/episode.json"
            episode = json.load(bundle.extractfile(members[episode_path]))
            canvas_path = f"episodes/{ref}/turn-01.png"
            luna_target = None
            if canvas_path in members:
                luna_target = assets / f"{ref}-luna.png"
                data = bundle.extractfile(members[canvas_path]).read()
                png_hash = episode.get("render", {}).get("receipt", {}).get("png_sha256")
                if png_hash and hashlib.sha256(data).hexdigest() != png_hash:
                    raise ValueError(f"Luna PNG hash mismatch: {ref}")
                luna_target.write_bytes(data)
            rows.append({"reference_id": ref, "reference": f"assets/{ref}.jpg",
                         "mimo": f"assets/{mimo_target.name}" if mimo_target else None,
                         "mimo_status": choice.get("status"),
                         "mimo_offline_replay": bool(replay and replay.get("canvas")),
                         "luna": f"assets/{luna_target.name}" if luna_target else None,
                         "luna_status": episode.get("status"),
                         "luna_elapsed_seconds": episode.get("render", {}).get("elapsed_seconds"),
                         "luna_revised": revised.get(ref),
                         "luna_api": api_episodes.get(ref)})
    evidence = {"schema": "painter.luna-high-comparison.v1", "luna_receipt": receipt,
                "luna_revision_receipt": revision_receipt,
                "luna_summary": summary, "luna_api_receipts": [receipt for _, receipt in api_archives],
                "rows": rows,
                "limitation": "Luna Codex-agent sketches are first passes and are not budget-matched to MiMo Pro. Paid Luna API episodes use the same reference and quality-turn cap, but only one sample per reference. Recovered MiMo canvases are offline replays the teacher did not see. This is a visual feasibility comparison, not a measured API cost/quality ranking."}
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    cards = []
    for row in rows:
        panels = []
        for label, path, note in [
            ("Reference", row["reference"], "COCO photo"),
            ("MiMo Pro", row["mimo"], f"{row['mimo_status']}" + (" · offline replay" if row["mimo_offline_replay"] else "")),
            ("Luna high", row["luna"], f"{row['luna_status']} · first pass"),
            *(("Luna revised", row["luna_revised"]["canvas"],
                f"{row['luna_revised']['status']} · visual revision")
               for _ in (0,) if row["luna_revised"]),
            *(("Luna high API", row["luna_api"]["canvas"],
                f"{row['luna_api']['status']} · {row['luna_api']['turns']} turns")
               for _ in (0,) if row["luna_api"]),
        ]:
            image = f'<a href="{html.escape(path)}" target="_blank"><img src="{html.escape(path)}" alt="{html.escape(label)} for {row["reference_id"]}"></a>' if path else '<div class="missing">No canvas</div>'
            panels.append(f'<div class="panel"><h3>{html.escape(label)}</h3>{image}<p>{html.escape(note)}</p></div>')
        cards.append(f'<section><h2>{html.escape(row["reference_id"])}</h2><div class="grid">{"".join(panels)}</div></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Luna high / MiMo Pro painter comparison</title><style>
    body{background:#171a1e;color:#f4f2ec;font:16px system-ui;margin:0 auto;padding:24px;max-width:1800px}h1{margin-bottom:4px}p{color:#c6c6c6}section{border-top:1px solid #555;padding:25px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:16px}.panel{background:#23282d;padding:14px;border-radius:12px}.panel img{width:100%;height:auto;max-height:70vh;object-fit:contain;background:#111}.panel h3{margin:0 0 10px}.panel p{margin:7px 0}.missing{height:300px;display:grid;place-items:center;color:#aaa}@media(max-width:800px){.grid{grid-template-columns:1fr}}
    </style><h1>Luna high vs MiMo Pro</h1><p>Same three references. Luna's Codex-agent sketches have an initial and optional visually revised pass; paid Gateway Luna episodes and MiMo Pro had multiple benchmark turns. Click any image to inspect full size. This is a visual feasibility check, not a definitive cost ranking.</p>''' + "".join(cards) + "</html>"
    (output / "index.html").write_text(page)
    print(json.dumps({"output": str(output / "index.html"), "luna_status_counts": summary["status_counts"], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
