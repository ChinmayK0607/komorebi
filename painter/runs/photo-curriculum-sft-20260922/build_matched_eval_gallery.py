#!/usr/bin/env python3
"""Build a four-column reference/base/checkpoint evaluation gallery.

The input traces are evidence only.  A canvas is copied only when its trace
marks ``final_canvas_valid`` true, the exact case id and reference hash match,
and the recorded final-canvas file exists.  Turn canvases are never promoted
to final artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "matched-photo-eval-gallery-v1"
POLICIES = ("base", "midpoint512", "final1024")
DEFAULT_NODE_ROOT = "/root/painter"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def short_hash(value: Any) -> str:
    return f"{value[:12]}…" if isinstance(value, str) and len(value) > 16 else (value or "—")


def safe(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).lstrip(".")
    return result or "unknown"


def path_candidates(value: Any, *, evidence_root: Path, base_dir: Path | None = None) -> Iterable[Path]:
    if not isinstance(value, (str, Path)) or not str(value) or str(value).startswith("data:"):
        return
    raw = Path(str(value)).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        try:
            relative = raw.relative_to(DEFAULT_NODE_ROOT)
        except ValueError:
            relative = None
        if relative is not None:
            candidates.extend((evidence_root / relative, evidence_root / "evidence" / relative))
    else:
        if base_dir is not None:
            candidates.append(base_dir / raw)
        candidates.extend((evidence_root / raw, evidence_root / "evidence" / raw))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def resolve_file(value: Any, *, evidence_root: Path, base_dir: Path | None = None) -> Path | None:
    return next((path for path in path_candidates(value, evidence_root=evidence_root, base_dir=base_dir) if path.is_file()), None)


def load_cases(manifest: Path) -> list[dict[str, Any]]:
    raw = read_json(manifest)
    cases = raw if isinstance(raw, list) else raw.get("cases", []) if isinstance(raw, dict) else []
    if not isinstance(cases, list):
        raise ValueError("evaluation manifest cases must be a list")
    result = []
    for index, value in enumerate(cases):
        if not isinstance(value, dict):
            raise ValueError(f"evaluation manifest case {index + 1} is not an object")
        case = dict(value)
        case["id"] = first(case, "id", "task_id", "case_id") or f"case-{index + 1:03d}"
        case["reference_sha256"] = first(case, "reference_sha256", "image_sha256")
        result.append(case)
    return result


def trace_stream(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    if path.suffix == ".zst":
        try:
            import zstandard as zstd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"cannot read compressed trace stream without zstandard: {path}") from exc
        handle = path.open("rb")
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(handle), encoding="utf-8")
    return path.open(encoding="utf-8")


def trace_objects(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    traces: list[dict[str, Any]] = []
    errors: list[str] = []
    with trace_stream(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
                continue
            nested = record.get("traces") if isinstance(record, dict) else None
            values = nested if isinstance(nested, list) else [record]
            for trace in values:
                if isinstance(trace, dict) and isinstance(trace.get("info"), dict):
                    item = dict(trace)
                    item["_trace_file"] = str(path)
                    item["_trace_line"] = line_number
                    traces.append(item)
    return traces, errors


def expand_trace_input(value: Path) -> list[Path]:
    if value.is_file():
        return [value.resolve()]
    if value.is_dir():
        return sorted({path.resolve() for pattern in ("traces.jsonl", "traces.jsonl.gz", "traces.jsonl.zst") for path in value.rglob(pattern)})
    return []


def load_policy_traces(inputs: list[Path]) -> dict[str, Any]:
    files: list[Path] = []
    for value in inputs:
        files.extend(expand_trace_input(value))
    files = sorted(set(files))
    by_id: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    file_hashes = []
    for file in files:
        file_digest = sha256(file)
        file_hashes.append({"file": file.name, "sha256": file_digest})
        traces, parse_errors = trace_objects(file)
        errors.extend(parse_errors)
        for trace in traces:
            trace["_trace_sha256"] = file_digest
            info = trace["info"]
            task_id = first(info, "task_id", "id")
            if not task_id:
                task = trace.get("task")
                task_id = first(task.get("data"), "task_id", "id") if isinstance(task, dict) else None
            if task_id:
                by_id.setdefault(str(task_id), []).append(trace)
    return {"files": file_hashes, "traces": by_id, "errors": errors}


def usage_summary(trace: dict[str, Any]) -> dict[str, Any]:
    prompt = completion = total = 0
    finishes: Counter[str] = Counter()
    calls = trace.get("calls") if isinstance(trace.get("calls"), list) else []
    for call in calls:
        if not isinstance(call, dict):
            continue
        reason = call.get("finish_reason")
        if reason is not None:
            finishes[str(reason)] += 1
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        prompt += int(first(usage, "prompt_tokens", "input_tokens") or 0)
        completion += int(first(usage, "completion_tokens", "output_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    if not total:
        total = prompt + completion
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    turns = info.get("turns") if isinstance(info.get("turns"), list) else []
    lengths = [turn.get("turn") for turn in turns if isinstance(turn, dict) and isinstance(turn.get("turn"), int)]
    return {"calls": len(calls), "turns": len(turns), "last_turn": max(lengths) if lengths else None,
            "finish_reasons": dict(finishes), "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def reference_for_case(case: dict[str, Any], manifest: Path, references: Path | None, evidence_root: Path) -> tuple[Path | None, str | None, str | None]:
    values = [case.get("reference") or case.get("reference_png"), f"{case['id']}.jpg", f"{case['id']}.png"]
    roots = [references] if references else []
    roots.append(manifest.parent / "references")
    for root in roots:
        if root:
            for value in values:
                candidate = root / Path(str(value)).name
                if candidate.is_file():
                    actual = sha256(candidate)
                    expected = case.get("reference_sha256")
                    status = "valid" if expected and actual == expected else ("hash_mismatch" if expected else "unverified")
                    return candidate.resolve(), status, actual
    direct = resolve_file(case.get("reference") or case.get("reference_png"), evidence_root=evidence_root, base_dir=manifest.parent)
    if direct:
        actual = sha256(direct)
        expected = case.get("reference_sha256")
        return direct, "valid" if expected and actual == expected else ("hash_mismatch" if expected else "unverified"), actual
    return None, "missing", None


def copy_file(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(source.read_bytes())
    return {"path": output, "sha256": sha256(output), "bytes": output.stat().st_size}


def policy_case(case: dict[str, Any], policy: str, traces: dict[str, Any], reference_status: str, evidence_root: Path, output: Path) -> dict[str, Any]:
    matches = traces["traces"].get(str(case["id"]), [])
    base = {"status": "missing_trace", "error": "no trace with exact task_id", "trace_file": None, "trace_line": None, "trace_sha256": None,
            "source_sha256": None, "render_sha256": None, "asset": None, "usage": {}, "duplicate_trace_count": len(matches)}
    if not matches:
        return base
    trace = matches[-1]
    info = trace["info"]
    usage = usage_summary(trace)
    base.update(trace_file=Path(trace["_trace_file"]).name, trace_line=trace["_trace_line"], trace_sha256=trace.get("_trace_sha256"), usage=usage)
    trace_reference_hash = first(info, "reference_sha256", "image_sha256")
    if reference_status != "valid":
        base.update(status="reference_mismatch", error=f"reference status is {reference_status}")
        return base
    if not trace_reference_hash or trace_reference_hash != case.get("reference_sha256"):
        base.update(status="reference_mismatch", error=f"trace reference_sha256 {trace_reference_hash or 'missing'} does not match manifest")
        return base
    final_canvas = info.get("final_canvas")
    canvas_valid = info.get("final_canvas_valid") is True
    canvas_path = resolve_file(final_canvas, evidence_root=evidence_root, base_dir=Path(trace["_trace_file"]).parent)
    if not canvas_valid:
        base.update(status="invalid_final_canvas", error=f"final_canvas_valid is not true; no final artifact selected (failure_kind={info.get('failure_kind') or 'unspecified'})")
        return base
    if canvas_path is None:
        base.update(status="artifact_missing", error="final_canvas_valid is true but recorded final_canvas file is missing")
        return base
    actual_hash = sha256(canvas_path)
    target = output / "assets" / f"{safe(case['id'])}--{safe(policy)}.png"
    copied = copy_file(canvas_path, target)
    base.update(status="valid", error=None, source_sha256=trace_reference_hash, render_sha256=actual_hash,
                asset={"role": policy, "path": str(copied["path"].relative_to(output)), "sha256": copied["sha256"], "bytes": copied["bytes"]})
    return base


def make_contacts(records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    if sys.platform != "linux":
        raise RuntimeError("--contact-sheets is Linux-only; do not compose image artifacts on this host")
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("--contact-sheets requires Pillow on the Linux node") from exc
    contact_dir = output / "contacts"
    contact_dir.mkdir(parents=True, exist_ok=True)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    cell_width, cell_height, label_height = 600, 600, 30
    sheets = []
    for number, start in enumerate(range(0, len(records), 5), 1):
        batch = records[start:start + 5]
        sheet = Image.new("RGB", (cell_width * 4, (cell_height + label_height) * len(batch)), "#f3f1ec")
        for row_index, record in enumerate(batch):
            for column, (label, asset) in enumerate((("reference", record["reference"].get("asset")),
                                                       ("base", record["policies"]["base"].get("asset")),
                                                       ("midpoint512", record["policies"]["midpoint512"].get("asset")),
                                                       ("final1024", record["policies"]["final1024"].get("asset")))):
                x, y = column * cell_width, row_index * (cell_height + label_height)
                cell = Image.new("RGBA", (cell_width, cell_height + label_height), "white")
                draw = ImageDraw.Draw(cell)
                status = "available" if asset else "missing/failed"
                draw.text((8, 7), f"{record['id']} · role: {label} · {status}", fill="black")
                if asset:
                    with Image.open(output / asset["path"]) as source:
                        fitted = ImageOps.contain(source.convert("RGBA"), (cell_width, cell_height), method=resample)
                    cell.alpha_composite(fitted, ((cell_width - fitted.width) // 2, label_height + (cell_height - fitted.height) // 2))
                else:
                    draw.text((12, label_height + 20), record["policies"].get(label, {}).get("error", "unavailable"), fill="#555")
                sheet.paste(cell.convert("RGB"), (x, y))
        path = contact_dir / f"contact-{number:03d}.png"
        sheet.save(path, format="PNG", optimize=False)
        sheets.append({"file": str(path.relative_to(output)), "sha256": sha256(path), "record_ids": [r["id"] for r in batch],
                       "pair_count": len(batch), "width": sheet.width, "height": sheet.height})
    return sheets



def build_html(manifest: dict[str, Any]) -> str:
    # Keep the manifest in the standalone file so opening index.html via file://
    # never needs fetch(), a local server, or a dependency. Escape '<' before
    # placing JSON in a script element; metadata and asset paths remain exact.
    data = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Matched photo evaluation</title>
<style>
:root{color-scheme:light;--ink:#20252a;--muted:#68717a;--line:#d7dce1;--panel:#fff;--soft:#f2f4f6;--accent:#245e7b;--warn:#9a6200;--bad:#9b3a32}
*{box-sizing:border-box}body{margin:0;background:#eef1f3;color:var(--ink);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}body.modal-open{overflow:hidden}
button,select{font:inherit}button,select{border:1px solid #bbc4ca;border-radius:6px;background:#fff;color:var(--ink);padding:7px 10px}button{cursor:pointer}button:hover,button:focus-visible,select:focus-visible{border-color:var(--accent);outline:3px solid #b9d9e8;outline-offset:1px}.button-primary{background:#245e7b;color:#fff;border-color:#245e7b}
main{width:min(100% - 28px,1900px);margin:0 auto;padding:18px 0 48px}.app-header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:10px}.kicker{margin:0 0 2px;color:var(--accent);font-size:11px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}.app-header h1{font-size:clamp(20px,2.4vw,31px);line-height:1.1;margin:0}.lede{margin:5px 0 0;color:var(--muted);font-size:13px}.counter{white-space:nowrap;font-weight:700;color:#35414a}.summary,.toolbar,.case-panel,.overview-panel,.secondary{background:var(--panel);border:1px solid var(--line);border-radius:10px}.summary{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:8px 12px;color:var(--muted);font-size:12px}.summary strong{color:var(--ink)}.summary .quality-note{color:var(--warn)}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:9px 12px;margin:10px 0}.toolbar .spacer{flex:1}.policy-controls{display:flex;gap:5px;flex-wrap:wrap}.policy-controls button[aria-pressed=true]{background:#e1f0f6;border-color:#245e7b;color:#16445b;font-weight:700}.case-panel{padding:12px}.case-heading{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:0 2px 10px}.case-heading h2{font-size:18px;margin:0}.case-family{color:var(--muted);font-size:12px}.comparison{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}.canvas-card{min-width:0;margin:0}.canvas-card h3{font-size:13px;margin:0 0 5px}.role{color:var(--muted);font-weight:500}.image-stage{height:clamp(390px,68vh,820px);display:grid;place-items:center;background:#e8edf0;border:1px solid var(--line);border-radius:7px;overflow:hidden}.image-stage img{display:block;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;cursor:zoom-in}.canvas-caption{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:6px 1px 0;color:var(--muted);font-size:12px}.status{display:inline-flex;align-items:center;min-height:22px;padding:2px 7px;border-radius:999px;background:#e7edf0;color:#45525a}.status.valid{background:#e3f2e8;color:#22613a}.status.failed{background:#fae8e5;color:var(--bad);font-weight:700}.missing-stage{height:100%;width:100%;display:grid;align-content:center;justify-items:center;gap:5px;padding:20px;text-align:center;color:var(--bad)}.missing-stage strong{font-size:16px}.missing-stage span{color:var(--muted);font-size:12px}.evidence{margin:12px 2px 0;border-top:1px solid var(--line);padding-top:9px;color:var(--muted)}.evidence summary,.secondary summary{cursor:pointer;font-weight:650;color:#45525a}.evidence pre{max-height:360px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);padding:10px;border-radius:6px;font-size:11px;color:#343c41}.overview-panel{padding:12px}.overview-heading{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px}.overview-heading h2{font-size:17px;margin:0}.overview-grid{display:grid;gap:10px}.overview-case{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fbfcfc;cursor:pointer}.overview-case:hover,.overview-case:focus-visible{border-color:var(--accent);outline:3px solid #b9d9e8;outline-offset:1px}.overview-case h3{font-size:12px;margin:0 0 6px;overflow-wrap:anywhere}.overview-cells{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.overview-cell{min-width:0}.overview-cell label{display:block;color:var(--muted);font-size:10px;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.overview-thumb{height:150px;display:grid;place-items:center;background:#e8edf0;border:1px solid var(--line);border-radius:4px;overflow:hidden;color:var(--bad);font-size:11px;text-align:center;padding:5px}.overview-thumb img{display:block;width:100%;height:100%;object-fit:contain}.secondary{margin-top:10px;padding:9px 12px}.contact-links{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}.contact-links a{color:var(--accent)}[hidden]{display:none!important}
.lightbox{position:fixed;inset:0;z-index:10;display:grid;grid-template-rows:auto 1fr;background:rgba(8,13,17,.94);padding:12px}.lightbox-toolbar{display:flex;align-items:center;gap:7px;color:#fff;min-height:42px}.lightbox-toolbar .lightbox-label{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.lightbox-toolbar button{background:#fff;color:#1e2529}.lightbox-stage{min-height:0;overflow:auto;display:block;padding:18px}.lightbox-image{display:block;margin:0 auto;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;cursor:zoom-out}.lightbox-image.actual{max-width:none;max-height:none;width:auto;height:auto;cursor:zoom-out}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:700px){main{width:min(100% - 18px,1900px);padding-top:10px}.app-header{align-items:flex-start;flex-direction:column;gap:5px}.lede{display:none}.comparison{grid-template-columns:1fr}.image-stage{height:60vh;min-height:300px}.overview-cells{grid-template-columns:repeat(2,minmax(0,1fr))}.overview-thumb{height:120px}.toolbar .spacer{display:none}}
</style>
</head>
<body>
<main>
  <header class="app-header">
    <div><p class="kicker">matched photo evaluation</p><h1>Reference vs. generated painting</h1><p class="lede">Artifact and trace evidence only; no visual quality claim.</p></div>
    <div class="counter" id="case-counter" aria-live="polite"></div>
  </header>
  <section class="summary" aria-label="Evaluation summary">
    <span><strong id="summary-denominator"></strong> cases</span><span><strong id="summary-references"></strong> valid references</span><span><strong id="summary-cells"></strong> valid final canvases</span><span><strong id="summary-failures"></strong> missing or failed cells</span><span class="quality-note">Failures stay in the case list.</span>
  </section>
  <section class="toolbar" aria-label="Gallery controls">
    <button type="button" id="previous-case" aria-label="Previous case">← Previous</button>
    <button type="button" id="next-case" aria-label="Next case">Next →</button>
    <label for="case-select">Case <select id="case-select"></select></label>
    <span class="spacer"></span>
    <div class="policy-controls" id="policy-controls" role="group" aria-label="Candidate checkpoint"></div>
    <button type="button" id="overview-toggle" class="button-primary" aria-expanded="false">Overview</button>
  </section>
  <section id="focus-view" class="case-panel" tabindex="-1" aria-label="Focused case"></section>
  <section id="overview-view" class="overview-panel" aria-label="All cases overview" hidden></section>
  <section class="secondary"><details><summary>Existing contact-sheet files</summary><div class="contact-links" id="contact-links"></div></details></section>
</main>
<div id="lightbox" class="lightbox" role="dialog" aria-modal="true" aria-labelledby="lightbox-label" hidden>
  <div class="lightbox-toolbar"><span id="lightbox-label" class="lightbox-label"></span><button type="button" id="lightbox-fit">Fit</button><button type="button" id="lightbox-actual">100%</button><button type="button" id="lightbox-minus" aria-label="Zoom out">−</button><span id="lightbox-zoom">100%</span><button type="button" id="lightbox-plus" aria-label="Zoom in">+</button><button type="button" id="lightbox-close">Close</button></div>
  <div id="lightbox-stage" class="lightbox-stage"><img id="lightbox-image" class="lightbox-image" alt=""></div>
</div>
<script id="gallery-data" type="application/json">__MANIFEST_DATA__</script>
<script>
(function () {
  "use strict";
  var manifest = JSON.parse(document.getElementById("gallery-data").textContent);
  var records = Array.isArray(manifest.records) ? manifest.records : [];
  var policies = [["base", "Base"], ["midpoint512", "Midpoint 512"], ["final1024", "Final 1024"]];
  var focus = document.getElementById("focus-view");
  var overview = document.getElementById("overview-view");
  var caseSelect = document.getElementById("case-select");
  var policyControls = document.getElementById("policy-controls");
  var counter = document.getElementById("case-counter");
  var index = 0;
  var selectedPolicy = "midpoint512";
  var lightbox = document.getElementById("lightbox");
  var lightboxImage = document.getElementById("lightbox-image");
  var lightboxLabel = document.getElementById("lightbox-label");
  var lightboxScale = 1;
  var lightboxMode = "fit";
  var lastFocus = null;
  function esc(value) {
    return String(value == null ? "—" : value).replace(/[&<>"']/g, function (char) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char];
    });
  }
  function labelFor(key) {
    for (var n = 0; n < policies.length; n += 1) if (policies[n][0] === key) return policies[n][1];
    return key;
  }
  function current() { return records[index] || null; }
  function assetPath(value) { return value && value.asset && value.asset.path ? String(value.asset.path) : null; }
  function failureText(value, isReference) {
    if (!value) return "unavailable";
    if (isReference) return value.status === "valid" ? "verified reference" : "reference unavailable";
    return ({invalid_final_canvas:"final canvas marked invalid",artifact_missing:"recorded canvas file missing",missing_trace:"no exact trace",reference_mismatch:"reference hash mismatch"}[value.status] || "no valid final painting");
  }
  function imageMarkup(value, alt, isReference) {
    var path = assetPath(value);
    if (path) return '<img data-open="' + esc(path) + '" data-alt="' + esc(alt) + '" src="' + esc(path) + '" alt="' + esc(alt) + '" role="button" tabindex="0" aria-label="Open ' + esc(alt) + '">';
    return '<div class="missing-stage"><strong>' + esc(isReference ? "reference unavailable" : "no valid final painting") + '</strong><span>' + esc(failureText(value, isReference)) + '</span></div>';
  }
  function panelMarkup(record, key, value, isReference) {
    var title = isReference ? "Reference" : labelFor(key);
    var status = value && value.status === "valid" ? "valid" : "failed";
    var statusText = value && value.status === "valid" ? "verified image" : failureText(value, isReference);
    return '<figure class="canvas-card"><h3>' + esc(title) + ' <span class="role">' + (isReference ? "source" : "candidate") + '</span></h3><div class="image-stage">' + imageMarkup(value, record.id + " " + title, isReference) + '</div><figcaption class="canvas-caption"><span class="status ' + status + '">' + esc(statusText) + '</span></figcaption></figure>';
  }
  function setSummary() {
    document.getElementById("summary-denominator").textContent = manifest.denominator == null ? records.length : manifest.denominator;
    document.getElementById("summary-references").textContent = (manifest.reference_valid_count == null ? 0 : manifest.reference_valid_count) + "/" + (manifest.denominator == null ? records.length : manifest.denominator);
    document.getElementById("summary-cells").textContent = (manifest.valid_cell_count == null ? 0 : manifest.valid_cell_count) + "/" + ((manifest.denominator == null ? records.length : manifest.denominator) * 3);
    document.getElementById("summary-failures").textContent = manifest.missing_or_failed_count == null ? 0 : manifest.missing_or_failed_count;
  }
  function setSelectors() {
    caseSelect.innerHTML = records.map(function (record, n) { return '<option value="' + n + '">' + esc(String(n + 1).padStart(2, "0")) + " · " + esc(record.id) + "</option>"; }).join("");
    policyControls.innerHTML = policies.map(function (item) { return '<button type="button" data-policy="' + item[0] + '" aria-pressed="' + (item[0] === selectedPolicy) + '">' + esc(item[1]) + "</button>"; }).join("");
    caseSelect.value = String(index);
    policyControls.querySelectorAll("button[data-policy]").forEach(function (button) { button.addEventListener("click", function () { selectedPolicy = button.dataset.policy; showFocusedCase(); }); });
  }
  function attachImageOpeners(root) {
    root.querySelectorAll("img[data-open]").forEach(function (image) { image.addEventListener("click", function (event) { event.stopPropagation(); openLightbox(image.dataset.open, image.dataset.alt); }); image.addEventListener("keydown", function (event) { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); event.stopPropagation(); openLightbox(image.dataset.open, image.dataset.alt); } }); });
  }
  function renderFocus() {
    var record = current();
    if (!record) { focus.innerHTML = '<p class="missing-stage">No cases in manifest.</p>'; return; }
    var candidate = record.policies[selectedPolicy] || {};
    counter.textContent = "Case " + (index + 1) + " / " + records.length;
    caseSelect.value = String(index);
    policyControls.querySelectorAll("button[data-policy]").forEach(function (button) { button.setAttribute("aria-pressed", String(button.dataset.policy === selectedPolicy)); });
    var evidence = {reference: record.reference, policy: candidate, policy_name: selectedPolicy};
    focus.innerHTML = '<div class="case-heading"><h2>' + esc(record.id) + '</h2><span class="case-family">' + esc(record.family || "matched case") + '</span></div><div class="comparison">' + panelMarkup(record, "reference", record.reference, true) + panelMarkup(record, selectedPolicy, candidate, false) + '</div><details class="evidence"><summary>Checksums, trace, finish, tokens and failure evidence</summary><pre>' + esc(JSON.stringify(evidence, null, 2)) + "</pre></details>";
    attachImageOpeners(focus);
  }
  function overviewCell(value, title, alt) {
    var path = assetPath(value);
    if (path) return '<div class="overview-cell"><label>' + esc(title) + '</label><div class="overview-thumb"><img loading="lazy" data-open="' + esc(path) + '" data-alt="' + esc(alt) + '" src="' + esc(path) + '" alt="' + esc(alt) + '" role="button" tabindex="0" aria-label="Open ' + esc(alt) + '"></div></div>';
    return '<div class="overview-cell"><label>' + esc(title) + '</label><div class="overview-thumb"><span>' + esc(title === "Reference" ? "reference unavailable" : "no valid final painting") + "</span></div></div>";
  }
  function renderOverview() {
    overview.innerHTML = '<div class="overview-heading"><h2>All cases · four-role overview</h2><span class="case-family">Select a row to return to the focused viewer.</span></div><div class="overview-grid">' + records.map(function (record, n) { return '<article class="overview-case" tabindex="0" data-case-index="' + n + '" aria-label="Open case ' + esc(record.id) + '"><h3>' + esc(String(n + 1).padStart(2, "0")) + ' · ' + esc(record.id) + '</h3><div class="overview-cells">' + overviewCell(record.reference, "Reference", record.id + " reference") + overviewCell(record.policies.base, "Base", record.id + " Base") + overviewCell(record.policies.midpoint512, "Midpoint 512", record.id + " Midpoint 512") + overviewCell(record.policies.final1024, "Final 1024", record.id + " Final 1024") + "</div></article>"; }).join("") + "</div>";
    overview.querySelectorAll(".overview-case").forEach(function (card) {
      var open = function () { index = Number(card.dataset.caseIndex); showFocusedCase(); focus.focus(); };
      card.addEventListener("click", open);
      card.addEventListener("keydown", function (event) { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); } });
    });
    attachImageOpeners(overview);
  }
  var showingOverview = false;
  function showFocusedCase() { showingOverview = false; overview.hidden = true; focus.hidden = false; var button = document.getElementById("overview-toggle"); button.setAttribute("aria-expanded", "false"); button.textContent = "Overview"; renderFocus(); }
  function move(delta) { if (!records.length) return; index = (index + delta + records.length) % records.length; showFocusedCase(); }
  function toggleOverview() {
    showingOverview = !showingOverview;
    overview.hidden = !showingOverview;
    focus.hidden = showingOverview;
    var button = document.getElementById("overview-toggle");
    button.setAttribute("aria-expanded", String(showingOverview));
    button.textContent = showingOverview ? "Focused case" : "Overview";
    if (showingOverview) renderOverview(); else { renderFocus(); focus.focus(); }
  }
  function applyLightbox() {
    lightboxImage.classList.toggle("actual", lightboxMode !== "fit");
    if (lightboxMode === "fit") { lightboxImage.style.removeProperty("width"); lightboxImage.style.removeProperty("height"); } else if (lightboxImage.naturalWidth) { lightboxImage.style.width = Math.round(lightboxImage.naturalWidth * lightboxScale) + "px"; lightboxImage.style.height = Math.round(lightboxImage.naturalHeight * lightboxScale) + "px"; }
    document.getElementById("lightbox-zoom").textContent = Math.round(lightboxScale * 100) + "%";
  }
  function openLightbox(src, alt) {
    lastFocus = document.activeElement;
    lightbox.hidden = false;
    document.body.classList.add("modal-open");
    lightboxImage.onload = applyLightbox;
    lightboxImage.src = src;
    lightboxImage.alt = alt || "painting";
    lightboxLabel.textContent = alt || "painting";
    lightboxMode = "fit";
    lightboxScale = 1;
    applyLightbox();
    document.getElementById("lightbox-close").focus();
  }
  function closeLightbox() {
    lightbox.hidden = true;
    document.body.classList.remove("modal-open");
    lightboxImage.removeAttribute("src");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  document.getElementById("previous-case").addEventListener("click", function () { move(-1); });
  document.getElementById("next-case").addEventListener("click", function () { move(1); });
  caseSelect.addEventListener("change", function (event) { index = Number(event.target.value); showFocusedCase(); });
  document.getElementById("overview-toggle").addEventListener("click", toggleOverview);
  document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
  document.getElementById("lightbox-fit").addEventListener("click", function () { lightboxMode = "fit"; lightboxScale = 1; applyLightbox(); });
  document.getElementById("lightbox-actual").addEventListener("click", function () { lightboxMode = "actual"; lightboxScale = 1; applyLightbox(); });
  document.getElementById("lightbox-minus").addEventListener("click", function () { lightboxMode = "actual"; lightboxScale = Math.max(.25, lightboxScale - .25); applyLightbox(); });
  document.getElementById("lightbox-plus").addEventListener("click", function () { lightboxMode = "actual"; lightboxScale = Math.min(4, lightboxScale + .25); applyLightbox(); });
  lightbox.addEventListener("click", function (event) { if (event.target === lightbox) closeLightbox(); });
  document.addEventListener("keydown", function (event) {
    if (!lightbox.hidden) {
      if (event.key === "Escape") { event.preventDefault(); closeLightbox(); return; }
      if (event.key === "Tab") {
        var modalItems = Array.prototype.slice.call(lightbox.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex=\"-1\"])"));
        if (modalItems.length) { var first = modalItems[0]; var last = modalItems[modalItems.length - 1]; if (!lightbox.contains(document.activeElement)) { event.preventDefault(); first.focus(); } else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); } else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); } }
      }
      return;
    }
    var target = event.target;
    if (target && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].indexOf(target.tagName) !== -1)) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); } else if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
  });
  var contactLinks = document.getElementById("contact-links");
  contactLinks.innerHTML = (manifest.contact_sheets || []).map(function (sheet) { return '<a href="' + esc(sheet.file) + '">' + esc(sheet.file) + " · " + esc(sheet.sha256) + "</a>"; }).join("") || "No contact sheets recorded.";
  setSummary(); setSelectors(); renderFocus();
}());
</script>
</body>
</html>
'''
    return template.replace("__MANIFEST_DATA__", data)

def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    evidence_root = (args.evidence_root or Path.cwd()).resolve()
    reference_root = args.references.resolve() if args.references else None
    output = args.output.resolve()
    (output / "assets").mkdir(parents=True, exist_ok=True)
    cases = load_cases(manifest_path)
    policy_inputs: dict[str, list[Path]] = {policy: [] for policy in POLICIES}
    for policy, value in (("base", args.base or []), ("midpoint512", args.midpoint512 or []), ("final1024", args.final1024 or [])):
        policy_inputs[policy].extend(value)
    for item in args.trace or []:
        if "=" not in item:
            raise ValueError(f"--trace expects POLICY=PATH: {item}")
        policy, raw_path = item.split("=", 1)
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}; expected one of {', '.join(POLICIES)}")
        policy_inputs[policy].append(Path(raw_path))
    traces = {policy: load_policy_traces(values) for policy, values in policy_inputs.items()}
    records = []
    for case in cases:
        ref_path, ref_status, ref_actual = reference_for_case(case, manifest_path, reference_root, evidence_root)
        reference = {"status": ref_status, "expected_sha256": case.get("reference_sha256"), "actual_sha256": ref_actual, "asset": None, "error": None}
        if ref_path and ref_status == "valid":
            target = output / "assets" / f"{safe(case['id'])}--reference{ref_path.suffix.lower() or '.jpg'}"
            copied = copy_file(ref_path, target)
            reference["asset"] = {"role": "reference", "path": str(copied["path"].relative_to(output)), "sha256": copied["sha256"], "bytes": copied["bytes"]}
        elif ref_status == "missing":
            reference["error"] = "reference file missing"
        elif ref_status == "unverified":
            reference["error"] = "reference hash is missing from manifest; no reference image selected"
        else:
            reference["error"] = "reference hash does not match manifest; no reference image selected"
        records.append({"id": case["id"], "family": case.get("family"), "reference": reference,
                        "policies": {policy: policy_case(case, policy, traces[policy], ref_status, evidence_root, output) for policy in POLICIES}})
    contacts = make_contacts(records, output) if args.contact_sheets else []
    summaries = {}
    for policy in POLICIES:
        cells = [record["policies"][policy] for record in records]
        finish = Counter()
        total_usage = Counter()
        for cell in cells:
            finish.update(cell.get("usage", {}).get("finish_reasons", {}))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[key] += cell.get("usage", {}).get(key, 0)
        summaries[policy] = {"denominator": len(records), "valid_final_canvas_count": sum(c["status"] == "valid" for c in cells),
                            "status_counts": dict(Counter(c["status"] for c in cells)), "finish_reason_counts": dict(finish),
                            "finish_length": {"turn_counts": dict(Counter(c.get("usage", {}).get("turns") for c in cells)),
                                              "max_turn": max((c.get("usage", {}).get("last_turn") or 0 for c in cells), default=0)},
                            "token_usage": dict(total_usage), "trace_errors": traces[policy]["errors"], "trace_files": traces[policy]["files"]}
    output_manifest = {"schema": SCHEMA, "quality_claim": False, "manifest_file": manifest_path.name,
                       "denominator": len(records), "reference_valid_count": sum(r["reference"]["status"] == "valid" for r in records),
                       "valid_cell_count": sum(r["policies"][p]["status"] == "valid" for r in records for p in POLICIES),
                       "missing_or_failed_count": sum(r["policies"][p]["status"] != "valid" for r in records for p in POLICIES),
                       "policies": summaries, "contact_sheets": contacts, "records": records}
    write_json(output / "manifest.json", output_manifest)
    (output / "index.html").write_text(build_html(output_manifest), encoding="utf-8")
    return output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--base", action="append", type=Path)
    parser.add_argument("--baseline", dest="base", action="append", type=Path)
    parser.add_argument("--midpoint512", action="append", type=Path)
    parser.add_argument("--final1024", action="append", type=Path)
    parser.add_argument("--trace", action="append", help="POLICY=TRACE_FILE_OR_DIRECTORY; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args()
    if not any((args.base, args.midpoint512, args.final1024, args.trace)):
        parser.error("provide at least one policy trace via --base/--midpoint512/--final1024/--trace")
    try:
        manifest = build(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.resolve()), "denominator": manifest["denominator"],
                      "valid_cells": manifest["valid_cell_count"], "contact_sheets": len(manifest["contact_sheets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
