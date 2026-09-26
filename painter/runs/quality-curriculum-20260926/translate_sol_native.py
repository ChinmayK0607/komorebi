#!/usr/bin/env python3
"""Make *candidate* brush versions of the two exact Sol helper templates.

This is deliberately narrow. The source and output programs remain separate;
each output needs a fresh render and visual comparison before SFT admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from audit_brush_medium import BRUSH_PAINT


AUDIT = Path(__file__).resolve().parents[2] / "collected/quality-curriculum-20260926/rendered-cloud/teacher600-visual-audit-v1"
NATIVE_MARKS = re.compile(r"(?<![.\w])(?:fill|stroke|ellipse|rect|circle|line|beginShape|vertex|endShape|arc|text|triangle|quad)\s*\(")

TEMPLATES = {
    "function pg(c,p){noStroke();fill(c);beginShape();for(const q of p)vertex(q[0],q[1]);endShape(CLOSE);}":
        "function pg(c,p){brush.noStroke();brush.noWash();brush.fill(c,255);brush.fillTexture(.10,.08);brush.polygon(p);}",
    "function pgon(c,p){noStroke();fill(c);beginShape();for(const q of p)vertex(q[0],q[1]);endShape(CLOSE);}":
        "function pgon(c,p){brush.noStroke();brush.noWash();brush.fill(c,255);brush.fillTexture(.10,.08);brush.polygon(p);}",
    "function wa(c,a,p){let z=color(c);z.setAlpha(a);pg(z,p);}":
        "function wa(c,a,p){brush.noStroke();brush.noWash();brush.fill(c,a);brush.fillTexture(.10,.08);brush.polygon(p);}",
    "function wash(c,a,p){let z=color(c);z.setAlpha(a);pgon(z,p);}":
        "function wash(c,a,p){brush.noStroke();brush.noWash();brush.fill(c,a);brush.fillTexture(.10,.08);brush.polygon(p);}",
    "function wash(c,a,p){let cc=color(c);cc.setAlpha(a);pgon(cc,p);}":
        "function wash(c,a,p){brush.noStroke();brush.noWash();brush.fill(c,a);brush.fillTexture(.10,.08);brush.polygon(p);}",
    "function ov(c,x,y,w,h){noStroke();fill(c);ellipse(x,y,w,h);}":
        "function ov(c,x,y,w,h){brush.noStroke();brush.noWash();brush.fill(c,255);brush.fillTexture(.10,.08);let p=[];for(let i=0;i<24;i++){let t=TWO_PI*i/24;p.push([x+cos(t)*w/2,y+sin(t)*h/2]);}brush.polygon(p);}",
    "function oval(c,x,y,w,h){noStroke();fill(c);ellipse(x,y,w,h);}":
        "function oval(c,x,y,w,h){brush.noStroke();brush.noWash();brush.fill(c,255);brush.fillTexture(.10,.08);let p=[];for(let i=0;i<24;i++){let t=TWO_PI*i/24;p.push([x+cos(t)*w/2,y+sin(t)*h/2]);}brush.polygon(p);}",
    "function ln(c,w,a,b,d,e){stroke(c);strokeWeight(w);line(a,b,d,e);noStroke();}":
        'function ln(c,w,a,b,d,e){brush.noFill();brush.noWash();brush.set("pen",c,w);brush.line(a,b,d,e);}',
    "function seg(c,w,a,b,d,e){stroke(c);strokeWeight(w);line(a,b,d,e);noStroke();}":
        'function seg(c,w,a,b,d,e){brush.noFill();brush.noWash();brush.set("pen",c,w);brush.line(a,b,d,e);}',
    "function seg(c,w,x1,y1,x2,y2){stroke(c);strokeWeight(w);line(x1,y1,x2,y2);noStroke();}":
        'function seg(c,w,x1,y1,x2,y2){brush.noFill();brush.noWash();brush.set("pen",c,w);brush.line(x1,y1,x2,y2);}',
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def translate(source: str) -> tuple[str | None, str]:
    if BRUSH_PAINT.search(source):
        return None, "already_uses_brush"
    output = source
    changes = 0
    for old, new in TEMPLATES.items():
        if old in output:
            output = output.replace(old, new)
            changes += 1
    if changes != 4:
        return None, f"unknown_helper_template:{changes}"
    if NATIVE_MARKS.search(output):
        return None, "direct_native_marks_need_manual_translation"
    if not BRUSH_PAINT.search(output):
        return None, "no_brush_mark_after_translation"
    return "// Candidate translation; render and visually review before training.\n" + output, "candidate"


def collect(audit: Path, output: Path, limit: int | None = None) -> dict:
    rows = [r for index in range(6) for r in json.loads((audit / f"shard-{index}.json").read_text())]
    candidates = []
    for row in rows:
        source_path = Path(row["canvas_path"]).parent / "program.js"
        source = source_path.read_bytes()
        if BRUSH_PAINT.search(source.decode()):
            continue
        program, status = translate(source.decode())
        candidates.append((row, source, program, status))
    if len(candidates) != 75:
        raise ValueError(f"expected 75 native-only programs, got {len(candidates)}")
    if limit is not None:
        candidates = candidates[:limit]
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for row, source, program, status in candidates:
        batch = row["chosen_run_id"].removeprefix("teacher600-").removesuffix("-20260927")
        if batch == "sol-runtime-repair":
            batch = "sol-runtime-repair-v1"
        item = {"audit_id": row["audit_id"], "id": row["id"],
                "source_batch": batch,
                "mode": row["mode"], "category": row["category"],
                "input_sha256": row["input_sha256"], "source_program_sha256": sha(source),
                "original_canvas_sha256": row["canvas_sha256"], "status": status}
        if program:
            target = output / row["audit_id"] / "program.js"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(program)
            item.update(candidate_program_sha256=sha(target.read_bytes()), candidate_program=str(target))
        report.append(item)
    manifest = {"schema": "painter.sol-native-brush-candidates.v1", "scope": len(report),
                "source_audit_manifest_sha256": sha((audit / "manifest.json").read_bytes()),
                "rows": report, "training_admission": "none_without_new_render_and_visual_review"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"scope": len(report), "candidate": sum(x["status"] == "candidate" for x in report),
            "manual": sum(x["status"] != "candidate" for x in report), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(collect(args.audit, args.output, args.limit)))


if __name__ == "__main__":
    main()
