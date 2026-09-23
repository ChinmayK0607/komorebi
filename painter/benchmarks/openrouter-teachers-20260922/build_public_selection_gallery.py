#!/usr/bin/env python3
"""Build a local, offline model-selection gallery from verified public archives.

No model, renderer, or judge calls occur here. Timed-out programs are rendered
separately on Linux and their verified replay archives can be added later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
from urllib.request import urlopen


DATASET = "CK0607/komorebi-painter-teachers"
RUN_NAME = re.compile(r"(?:full-(?:quality|speed)-\d\d|renderer-replay-full-quality-\d\d)-20260923\Z")
CANVAS = re.compile(r"episodes/[A-Za-z0-9._-]+/turn-\d\d\.png\Z")
EPISODE = re.compile(r"episodes/[A-Za-z0-9._-]+/episode\.json\Z")
REPLAY = re.compile(r"episodes/[A-Za-z0-9._-]+/replay\.json\Z")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def public_receipt(run_id: str) -> dict:
    if not RUN_NAME.fullmatch(run_id):
        raise ValueError(f"unexpected run ID: {run_id}")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/runs/{run_id}/receipt.json?download=true"
    with urlopen(url, timeout=30) as response:
        receipt = json.load(response)
    if receipt.get("run_id") != run_id or receipt.get("public_hash_verified") is not True:
        raise ValueError(f"unverified public receipt: {run_id}")
    return receipt


def verified_archives(paths: list[Path]):
    for path in paths:
        run_id = path.name.removesuffix(".tar.gz")
        receipt = public_receipt(run_id)
        if path.stat().st_size != receipt["bundle_bytes"] or sha_file(path) != receipt["bundle_sha256"]:
            raise ValueError(f"local archive does not match public receipt: {path}")
        yield run_id, path, receipt


def write_asset(bundle: tarfile.TarFile, members: dict, name: str, output: Path) -> str | None:
    if not CANVAS.fullmatch(name) or name not in members:
        return None
    source = bundle.extractfile(members[name])
    if source is None:
        return None
    data = source.read()
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > 10_000_000:
        return None
    digest = hashlib.sha256(data).hexdigest()
    destination = output / "assets" / f"{digest}.png"
    if not destination.exists():
        destination.write_bytes(data)
    return f"assets/{digest}.png"


def build(benchmark: Path, archives: list[Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    config = json.loads((benchmark / "config.json").read_text())
    references = json.loads((benchmark / "refs.json").read_text())["references"]
    model_ids = config["models"]
    refs = []
    for ref in references:
        path = benchmark / ref["image"]
        if not path.is_file() or sha_file(path) != ref["sha256"]:
            raise ValueError(f"reference missing or hash mismatch: {ref['id']}")
        target = output / "assets" / path.name
        if not target.exists():
            shutil.copyfile(path, target)
        refs.append({"id": ref["id"], "category": ref.get("category", ""), "image": f"assets/{path.name}"})

    candidates: dict[str, dict] = {}
    replayed: dict[str, dict] = {}
    sources = []
    for run_id, path, receipt in verified_archives(archives):
        sources.append({"run_id": run_id, "sha256": receipt["bundle_sha256"],
                        "dataset_commit": receipt["dataset_commit"]})
        with tarfile.open(path, "r:gz") as bundle:
            members = {member.name: member for member in bundle if member.isfile()}
            for name, member in members.items():
                if EPISODE.fullmatch(name):
                    episode = json.load(bundle.extractfile(member))
                    track = (episode.get("settings") or {}).get("track")
                    model, reference = episode.get("model"), episode.get("reference_id")
                    if track not in ("quality", "speed") or model not in model_ids:
                        continue
                    key = f"{track}|{reference}|{model}"
                    if key in candidates:
                        raise ValueError(f"duplicate model/reference/track: {key}")
                    canvas_name = episode.get("final_valid_canvas")
                    canvas = write_asset(bundle, members, canvas_name, output) if isinstance(canvas_name, str) else None
                    status = str(episode.get("status"))
                    candidates[key] = {"status": status, "canvas": canvas,
                                       "terminal": bool(canvas and status in ("complete", "turn_limit")),
                                       "turns": len(episode.get("turns") or []),
                                       "tokens": episode.get("total_tokens"), "run_id": run_id}
                elif REPLAY.fullmatch(name):
                    evidence = json.load(bundle.extractfile(member))
                    if not (evidence.get("result") or {}).get("valid"):
                        continue
                    source = evidence.get("source") or {}
                    model, reference = source.get("model"), source.get("reference_id")
                    if model not in model_ids:
                        continue
                    turn = int(evidence["source_turn"])
                    image_name = name.rsplit("/", 1)[0] + f"/turn-{turn:02d}.png"
                    image = write_asset(bundle, members, image_name, output)
                    if image:
                        replayed[f"quality|{reference}|{model}"] = {
                            "canvas": image, "turn": turn, "run_id": run_id,
                            "note": "Offline render of a saved response; the teacher did not see this canvas."}

    report = {"schema": "painter.public-teacher-selection.v1", "models": model_ids,
              "references": refs, "candidates": candidates, "replayed": replayed,
              "sources": sources, "limitation": "Only publicly hash-verified archives are included; partial and offline replay paintings are marked separately."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    payload = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("/* DATA */ null", payload)
    (output / "index.html").write_text(page)
    return {"output": str(output / "index.html"), "sources": len(sources),
            "candidates": len(candidates), "replayed": len(replayed),
            "quality": sum(key.startswith("quality|") for key in candidates),
            "speed": sum(key.startswith("speed|") for key in candidates)}


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Painter teacher selection</title>
<style>
:root{font:15px/1.4 system-ui;background:#f5f3ef;color:#25231f}*{box-sizing:border-box}body{margin:0}
header{position:sticky;top:0;background:#f5f3efef;backdrop-filter:blur(10px);padding:14px 22px;border-bottom:1px solid #c9c5bd;z-index:2}
h1{font-size:1.45rem;margin:0 0 4px}p{margin:4px 0 12px;color:#5b5750}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
select,button{font:inherit;padding:7px 10px;border:1px solid #b7b3ab;background:white;border-radius:5px}button{cursor:pointer}
main{max-width:1800px;margin:auto;padding:18px 22px}.summary{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.pill{background:#eae5dc;border-radius:20px;padding:5px 10px}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}
.card{background:white;border:1px solid #d9d5cf;border-radius:8px;overflow:hidden;min-width:0}.card h2{font-size:1rem;margin:0;padding:9px;border-bottom:1px solid #eee;overflow-wrap:anywhere}
.frame{height:250px;display:grid;place-items:center;background:#ebe8e2}.frame img{width:100%;height:100%;object-fit:contain;cursor:zoom-in}
.info{padding:9px;min-height:92px;font-size:.86rem;color:#5b5750;overflow-wrap:anywhere}.badge{display:inline-block;border-radius:4px;background:#e6e4e0;padding:2px 5px;color:#413e39}
.terminal{background:#dcefdc;color:#24522a}.partial{background:#fff0d0;color:#775009}.failed{background:#f4dede;color:#8b3030}
.replay{border-top:1px solid #ddd;padding:8px}.replay img{width:100%;max-height:180px;object-fit:contain;cursor:zoom-in}
#viewer{border:0;background:#141414e8;color:white;max-width:100vw;max-height:100vh;padding:12px}#viewer img{display:block;max-width:94vw;max-height:86vh;object-fit:contain;margin:auto}
#viewer p{text-align:center;color:white}#viewer button{float:right}footer{margin:20px 0;color:#5b5750}
@media(max-width:1200px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.frame{height:190px}}
</style><header><h1>Choose the painting teacher</h1><p>Reference beside five models. Green = valid final painting; amber = last valid canvas before an error; red = no valid canvas. Offline rerenders are labeled separately.</p>
<div class="controls"><label>Track <select id="track"><option value="quality">Quality · up to 12 turns</option><option value="speed">Speed · up to 3 turns</option></select></label>
<label>Reference <select id="reference"></select></label><button id="prev">← Previous</button><button id="next">Next →</button></div></header>
<main><div id="summary" class="summary"></div><div id="grid" class="grid"></div><footer>Only hash-verified public archives are included. Missing quality shards and unrendered outputs remain visible as missing evidence. This gallery does not score or rank models.</footer></main>
<dialog id="viewer"><button id="close">Close</button><img id="large" alt="Full-size painting"><p id="caption"></p></dialog>
<script>const data=/* DATA */ null;
const track=document.getElementById('track'), reference=document.getElementById('reference'), grid=document.getElementById('grid'), summary=document.getElementById('summary');
const viewer=document.getElementById('viewer'), large=document.getElementById('large'), caption=document.getElementById('caption');
const counts={};for(const model of data.models){counts[model]={quality:{attempted:0,terminal:0,canvas:0},speed:{attempted:0,terminal:0,canvas:0}}}
for(const [key,row] of Object.entries(data.candidates)){const [t,,m]=key.split('|');if(!counts[m]||!counts[m][t])continue;const c=counts[m][t];c.attempted++;c.terminal+=row.terminal?1:0;c.canvas+=row.canvas?1:0}
for(const ref of data.references){const opt=document.createElement('option');opt.value=ref.id;opt.textContent=`${ref.id.slice(-3)} · ${ref.category}`;reference.append(opt)}
function openImage(path,label){large.src=path;caption.textContent=label;viewer.showModal()}
document.getElementById('close').onclick=()=>viewer.close();viewer.onclick=e=>{if(e.target===viewer)viewer.close()};
function card(title,image,status,details,extra){const section=document.createElement('section');section.className='card';
const h=document.createElement('h2');h.textContent=title;section.append(h);const frame=document.createElement('div');frame.className='frame';
if(image){const img=document.createElement('img');img.src=image;img.loading='lazy';img.alt=title;img.onclick=()=>openImage(image,title);frame.append(img)}else{frame.textContent='No verified canvas'}section.append(frame);
const info=document.createElement('div');info.className='info';const badge=document.createElement('span');badge.className='badge '+status;badge.textContent=status==='terminal'?'Final':status==='partial'?'Partial':'Missing';info.append(badge,document.createElement('br'),document.createTextNode(details));section.append(info);
if(extra){const box=document.createElement('div');box.className='replay';const label=document.createElement('strong');label.textContent='Offline timeout replay · turn '+extra.turn;box.append(label);const img=document.createElement('img');img.src=extra.canvas;img.loading='lazy';img.onclick=()=>openImage(extra.canvas,title+' · offline replay');box.append(img);section.append(box)}return section}
function render(){const t=track.value,id=reference.value,ref=data.references.find(x=>x.id===id);if(!ref)return;grid.replaceChildren();grid.append(card('Reference · '+id.slice(-3),ref.image,'terminal',ref.category,null));
for(const model of data.models){const key=`${t}|${id}|${model}`,row=data.candidates[key],replay=data.replayed[key],name=model.split('/').pop();
let label=row?`${row.status} · ${row.turns} turns · ${row.tokens??'?'} tokens`:'No published episode yet';if(row)label+=` · ${row.run_id}`;
grid.append(card(name,row?.canvas??null,row?.terminal?'terminal':row?.canvas?'partial':'failed',label,replay))}
summary.replaceChildren();for(const model of data.models){const c=counts[model][t],pill=document.createElement('span');pill.className='pill';pill.textContent=`${model.split('/').pop()}: ${c.terminal}/${c.attempted} final, ${c.canvas} any canvas`;summary.append(pill)}
history.replaceState(null,'','#'+t+'/'+id)}
track.onchange=render;reference.onchange=render;
function shift(n){reference.selectedIndex=(reference.selectedIndex+n+reference.options.length)%reference.options.length;render()}
document.getElementById('prev').onclick=()=>shift(-1);document.getElementById('next').onclick=()=>shift(1);
const hash=location.hash.slice(1).split('/');if(['quality','speed'].includes(hash[0]))track.value=hash[0];if(data.references.some(x=>x.id===hash[1]))reference.value=hash[1];else{let best=data.references[0],score=-1;for(const ref of data.references){let n=data.models.filter(m=>data.candidates[`quality|${ref.id}|${m}`]?.terminal).length;if(n>score){score=n;best=ref}}reference.value=best.id}render();
</script></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--speed-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.quality_dir.glob("full-quality-??-20260923.tar.gz"))
    paths += sorted(args.speed_dir.glob("full-speed-??-20260923.tar.gz"))
    if args.replay_dir:
        paths += sorted(args.replay_dir.glob("renderer-replay-full-quality-??-20260923.tar.gz"))
    if not paths:
        parser.error("no verified archive candidates found")
    print(json.dumps(build(args.benchmark, paths, args.output), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
