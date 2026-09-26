#!/usr/bin/env python3
"""Create a readable reference / first paint / revision comparison gallery."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


NODE_ROOT = Path("/root/painter/astra-firstpaint")
POLICIES = ("baseline", "trained")


def local_image(evidence: Path, output: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(NODE_ROOT)
        except ValueError:
            return None
    candidate = evidence / path
    if not candidate.is_file() and not path.is_absolute():
        candidate = evidence / "painter/eval-prep" / path
    if not candidate.is_file() or not candidate.resolve().is_relative_to(evidence.resolve()):
        return None
    return os.path.relpath(candidate, output.parent)


def traces(evidence: Path, policy: str) -> dict[str, dict]:
    paths = sorted((evidence / "eval" / policy / "results").glob("*/traces.jsonl"))
    if len(paths) != 1:
        return {}
    found = {}
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        for trace in episode.get("traces", []):
            info = trace.get("info") or {}
            case_id = info.get("task_id")
            if case_id:
                if case_id in found:
                    raise ValueError(f"duplicate {policy} trace: {case_id}")
                found[case_id] = trace
    return found


def picture(src: str | None, label: str, valid: bool | None) -> str:
    badge = "valid" if valid is True else "invalid" if valid is False else "missing"
    if src is None:
        return f'<div class="painting missing"><span>{html.escape(label)} · {badge}<br>No saved canvas</span></div>'
    url = html.escape(src, quote=True)
    return (f'<a class="painting" href="{url}" target="_blank">'
            f'<span>{html.escape(label)} · {badge}</span><img loading="lazy" src="{url}"></a>')


def build(evidence: Path, output: Path) -> dict:
    evidence = evidence.resolve()
    output = output.resolve()
    manifest = json.loads((evidence / "painter/eval-prep/eval-manifest.json").read_text())
    cases = manifest["cases"]
    policies = {policy: traces(evidence, policy) for policy in POLICIES}
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for case in cases:
        case_id = case["id"]
        reference = local_image(evidence, output, case.get("reference"))
        columns = [f'<div class="reference"><h3>Reference</h3>{picture(reference,"photo",reference is not None)}</div>']
        for policy in POLICIES:
            trace = policies[policy].get(case_id)
            info = (trace or {}).get("info") or {}
            turns = info.get("turns") or []
            first = turns[0] if turns else {}
            second = turns[1] if len(turns) > 1 else {}
            first_src = local_image(evidence, output, first.get("canvas"))
            second_src = local_image(evidence, output, second.get("canvas"))
            final_valid = info.get("final_canvas_valid") is True
            columns.append(f'<div class="policy"><h3>{html.escape(policy.title())} '
                           f'<small>{"final valid" if final_valid else "final invalid / missing"}</small></h3>'
                           f'{picture(first_src,"first paint",first.get("valid"))}'
                           f'{picture(second_src,"revision",second.get("valid"))}</div>')
        cards.append(f'<section class="case" id="{html.escape(case_id,quote=True)}">'
                     f'<h2>{html.escape(case_id)} <small>{html.escape(str(case.get("primary_family") or case.get("family") or ""))}</small></h2>'
                     f'<div class="columns">{"".join(columns)}</div></section>')
    counts = {policy: {"traces": len(found), "final_valid": sum((trace.get("info") or {}).get("final_canvas_valid") is True for trace in found.values())}
              for policy, found in policies.items()}
    summary = " · ".join(f'{policy}: {counts[policy]["traces"]}/{len(cases)} cases, {counts[policy]["final_valid"]} valid final canvases'
                         for policy in POLICIES)
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>Astra first-paint SFT comparison</title>'
            '<style>body{font:16px/1.4 system-ui;background:#161a1c;color:#f0f0ee;margin:0;padding:20px}'
            'h1{margin:0 0 6px}p{color:#c4c9c9}a{color:#a9d5ff}.summary{position:sticky;top:0;background:#263034;padding:12px;z-index:2;border-radius:8px}'
            '.case{background:#22282b;margin:24px 0;padding:16px;border-radius:12px}h2{margin:0 0 12px;font-size:18px}'
            'h2 small,h3 small{font-weight:400;color:#a8b4b6}.columns{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px}'
            '.policy,.reference{min-width:0}.policy h3,.reference h3{margin:0 0 8px;font-size:16px}'
            '.painting{display:block;position:relative;background:#101315;border-radius:8px;margin:0 0 10px;min-height:230px;text-decoration:none;color:#fff;overflow:hidden}'
            '.painting img{width:100%;height:280px;object-fit:contain;display:block}.painting span{display:block;padding:5px 8px;background:#344148}'
            '.painting.missing{display:flex;align-items:center;justify-content:center;text-align:center;color:#b8b8b8}'
            '@media(max-width:1050px){.columns{grid-template-columns:repeat(2,minmax(260px,1fr))}}'
            '@media(max-width:590px){.columns{grid-template-columns:1fr}}</style></head><body>'
            '<h1>Reference → paint → revision</h1><p>Matched frozen photo references; each policy sees its own first canvas. '
            'Click a painting for the full image. Validity is mechanical, not an aesthetic judgment.</p>'
            f'<div class="summary">{html.escape(summary)}</div>{"".join(cards)}</body></html>')
    output.write_text(page)
    return {"cases": len(cases), "policies": counts, "output": str(output)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.evidence, args.output)))


if __name__ == "__main__":
    main()
