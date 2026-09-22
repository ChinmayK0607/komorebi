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


def cell_html(record: dict[str, Any], label: str, value: dict[str, Any]) -> str:
    asset = value.get("asset")
    image = f'<img loading="lazy" src="{esc(asset["path"])}" alt="{esc(record["id"])} {esc(label)}">' if asset else '<div class="missing">no verified final canvas</div>'
    return f'<figure>{image}<figcaption><b>role: {esc(label)}</b><br>status: {esc(value.get("status"))}<br>sha256: <code>{esc(value.get("render_sha256") or "—")}</code><br>{esc(value.get("error") or "verified final canvas")}</figcaption></figure>'


def build_html(manifest: dict[str, Any]) -> str:
    links = "".join(f'<a href="{esc(sheet["file"])}"><img src="{esc(sheet["file"])}" alt="contact batch {sheet["pair_count"]}"><br>batch {sheet["pair_count"]} pairs · {esc(sheet["sha256"])}</a>' for sheet in manifest["contact_sheets"])
    rows = []
    for record in manifest["records"]:
        policies = record["policies"]
        rows.append(f'''<article class="case"><h2>{esc(record["id"])}</h2><p class="meta">reference status: {esc(record["reference"]["status"])} · expected sha256: <code>{esc(record["reference"].get("expected_sha256") or "—")}</code> · actual: <code>{esc(record["reference"].get("actual_sha256") or "—")}</code></p><div class="grid"><figure>{f'<img loading="lazy" src="{esc(record["reference"]["asset"]["path"])}" alt="{esc(record["id"])} reference">' if record["reference"].get("asset") else '<div class="missing">reference unavailable</div>'}<figcaption><b>role: reference</b><br>sha256: <code>{esc(record["reference"].get("actual_sha256") or "—")}</code></figcaption></figure>{cell_html(record, "base", policies["base"])}{cell_html(record, "midpoint512", policies["midpoint512"])}{cell_html(record, "final1024", policies["final1024"])}</div><details><summary>Trace, finish, token and failure evidence</summary><pre>{esc(json.dumps({"policies": policies, "reference": record["reference"]}, indent=2, sort_keys=True))}</pre></details></article>''')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched photo evaluation</title><style>body{{margin:0;padding:24px;background:#f3f1ec;color:#252525;font:14px/1.45 system-ui,sans-serif}}main{{max-width:1800px;margin:auto}}.note,.case,.contacts a,.stat{{background:white;border:1px solid #d5d1c8;border-radius:8px;padding:12px}}.note{{border-left:4px solid #bd8b00}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}.stat{{min-width:130px;text-align:center}}.case{{margin:16px 0}}h1,h2{{margin:0 0 6px}}.meta{{color:#666;overflow-wrap:anywhere}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}figure{{margin:0;min-width:0}}figure img,.missing{{display:block;width:100%;height:auto;max-height:600px;object-fit:contain;background:#eef0f2;border:1px solid #ddd}}.missing{{min-height:150px;display:grid;place-items:center;color:#777}}figcaption{{padding:6px 0;color:#555;font-size:.82rem;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f3ef;padding:10px;border-radius:6px;font-size:.76rem}}.contacts{{margin:18px 0}}.contact-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}.contacts a{{display:block;color:#225f8c;text-decoration:none}}.contacts img{{width:100%;height:auto}}@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:600px){{body{{padding:12px}}.grid{{grid-template-columns:1fr}}}}</style></head><body><main><h1>Matched photo evaluation</h1><p>Reference, baseline, midpoint-512, and final-1024 artifacts from preserved native traces. This gallery reports artifact and trace evidence only; it makes no visual quality claim and never substitutes a turn canvas for a missing final canvas.</p><p class="note">{manifest["denominator"]} case denominator · reference valid {manifest["reference_valid_count"]}/{manifest["denominator"]} · complete policy cells {manifest["valid_cell_count"]}/{manifest["denominator"] * 3}</p><div class="stats"><div class="stat"><b>{manifest["denominator"]}</b><br>case denominator</div><div class="stat"><b>{manifest["reference_valid_count"]}</b><br>valid references</div><div class="stat"><b>{manifest["valid_cell_count"]}</b><br>valid final canvases</div><div class="stat"><b>{manifest["missing_or_failed_count"]}</b><br>missing/failed cells</div></div>{f'<section class="contacts"><h2>Contact sheets</h2><div class="contact-grid">{links}</div></section>' if links else ''}{''.join(rows)}</main></body></html>\n'''


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
