#!/usr/bin/env python3
"""Export verified teacher evidence and a local static review gallery.

This command does not render programs.  It copies reference/render bytes into
the output directory, records their hashes and receipt evidence, and (on the
Linux compute node) makes five-pair PNG contact sheets with Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


SCHEMA = "teacher-review-v1"
CANDIDATES = (("program.json", "program.png"), ("program.js.json", "program.js.png"))
SHEET_PAIRS = 5
THUMB = 600
LABEL_H = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def error_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def receipt_errors(receipt: Any) -> list[str]:
    if not isinstance(receipt, dict):
        return []
    values: list[Any] = []
    if receipt.get("error") is not None:
        values.append(receipt["error"])
    raw = receipt.get("errors")
    if isinstance(raw, list):
        values.extend(raw)
    elif raw is not None:
        values.append(raw)
    return [error_text(value) for value in values]


def source_id(row: dict[str, Any]) -> str | None:
    for value in (row.get("source_id"), row.get("id"), row.get("reference")):
        match = re.search(r"(?<!\d)(\d{12})(?!\d)", str(value or ""))
        if match:
            return match.group(1)
    return None


def safe(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).lstrip(".")
    return result or "unknown"


def load_assignments(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment_path in paths:
        raw = json_read(assignment_path)
        teacher = text(raw.get("teacher")) if isinstance(raw, dict) else None
        teacher = teacher or assignment_path.stem
        references = raw.get("references") if isinstance(raw, dict) else None
        if isinstance(references, dict):
            references = [dict(value, id=key) if isinstance(value, dict) else {"id": key, "reference": value}
                          for key, value in references.items()]
        if not isinstance(references, list):
            raise ValueError(f"{assignment_path}: expected references array or object")
        for index, reference in enumerate(references):
            if not isinstance(reference, dict):
                raise ValueError(f"{assignment_path}: reference {index + 1} is not an object")
            row = dict(reference)
            row["id"] = text(row.get("id")) or text(row.get("assignment_id")) or source_id(row)
            if not row["id"]:
                raise ValueError(f"{assignment_path}: reference {index + 1} has no id/source id")
            row.update(teacher=teacher, assignment_file=assignment_path.name, assignment_index=index)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["teacher"]), str(row["id"]), row["assignment_file"], row["assignment_index"]))
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        key = f"{row['teacher']}/{row['id']}"
        seen[key] = seen.get(key, 0) + 1
        row.update(record_index=index, record_key=key if seen[key] == 1 else f"{key}#{seen[key]}")
    return rows


def find_reference(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    image_id = source_id(row)
    if not image_id:
        return {"id": None, "path": None, "tried": [], "status": "missing_source_id", "error": "no 12-digit source image id"}
    tried = [f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
    for name in tried:
        candidate = root / name
        if candidate.is_file():
            return {"id": image_id, "path": candidate, "tried": tried, "status": "found", "error": None}
    return {"id": image_id, "path": None, "tried": tried, "status": "missing",
            "error": f"reference not found under reference-root (tried {', '.join(tried)})"}


def teacher_dir(root: Path, assignment_id: str) -> tuple[Path | None, list[Path]]:
    tried = [root / assignment_id, root / "teachers" / assignment_id,
             root / "accepted-candidates" / "teachers" / assignment_id]
    return next((candidate for candidate in tried if candidate.is_dir()), None), tried


def summarize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": receipt.get("valid") is True,
        "source_sha256": first(receipt, "source_sha256", "sourceHash"),
        "png_sha256": first(receipt, "png_sha256", "render_sha256", "output_sha256", "image_sha256"),
        "requested_angle_backend": receipt.get("requested_angle_backend"),
        "canvas_contract": receipt.get("canvas_contract"),
        "telemetry_enabled": receipt.get("telemetry_enabled"),
        "network_enabled": receipt.get("network_enabled"),
        "browser_sandbox_enabled": receipt.get("browser_sandbox_enabled"),
        "finished": receipt.get("finished"),
        "timed_out": receipt.get("timed_out"),
        "error_code": receipt.get("error_code"),
        "errors": receipt_errors(receipt),
        "assets": receipt.get("assets") if isinstance(receipt.get("assets"), dict) else {},
        "gpu": receipt.get("gpu") if isinstance(receipt.get("gpu"), dict) else None,
    }


def check_candidate(folder: Path, program_hash: str | None, receipt_name: str, render_name: str) -> dict[str, Any]:
    receipt_path, render_path = folder / receipt_name, folder / render_name
    reasons: list[str] = []
    check: dict[str, Any] = {"receipt_file": receipt_name, "render_file": render_name,
                             "receipt_present": receipt_path.is_file(), "render_present": render_path.is_file(),
                             "receipt_valid": None, "receipt_source_sha256": None,
                             "receipt_png_sha256": None, "actual_png_sha256": None,
                             "receipt_errors": [], "reasons": reasons, "valid": False}
    receipt: dict[str, Any] | None = None
    if not check["receipt_present"]:
        reasons.append(f"missing receipt {receipt_name}")
    if not check["render_present"]:
        reasons.append(f"missing render {render_name}")
    if check["receipt_present"]:
        try:
            value = json_read(receipt_path)
            if not isinstance(value, dict):
                raise ValueError("receipt must be a JSON object")
            receipt = value
            check["receipt_valid"] = receipt.get("valid") is True
            check["receipt_source_sha256"] = first(receipt, "source_sha256", "sourceHash")
            check["receipt_png_sha256"] = first(receipt, "png_sha256", "render_sha256", "output_sha256", "image_sha256")
            check["receipt_errors"] = receipt_errors(receipt)
            if not check["receipt_valid"]:
                reasons.append("receipt valid is not true")
            if check["receipt_errors"]:
                reasons.append("receipt error: " + "; ".join(check["receipt_errors"]))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"invalid receipt JSON: {exc}")
    if program_hash is None:
        reasons.append("current program.js is missing; source hash cannot be checked")
    elif check["receipt_source_sha256"] != program_hash:
        reasons.append(f"receipt source_sha256 {check['receipt_source_sha256'] or 'missing'} does not match current program.js {program_hash}")
    if check["render_present"]:
        try:
            check["actual_png_sha256"] = sha256(render_path)
        except OSError as exc:
            reasons.append(f"cannot hash render: {exc}")
    if check["render_present"] and check["receipt_png_sha256"] != check["actual_png_sha256"]:
        reasons.append(f"receipt png_sha256 {check['receipt_png_sha256'] or 'missing'} does not match current PNG {check['actual_png_sha256'] or 'missing'}")
    check["valid"] = not reasons
    if receipt is not None:
        check["receipt"] = summarize_receipt(receipt)
    return check


def copy_asset(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return {"path": output, "sha256": sha256(output), "bytes": output.stat().st_size}


def build_record(row: dict[str, Any], reference_root: Path, teachers_root: Path, output: Path) -> dict[str, Any]:
    ref = find_reference(reference_root, row)
    folder, tried_dirs = teacher_dir(teachers_root, row["id"])
    program = folder / "program.js" if folder else None
    program_hash = sha256(program) if program and program.is_file() else None
    checks = [check_candidate(folder, program_hash, receipt, render) for receipt, render in CANDIDATES] if folder else []
    selected = next((check for check in checks if check["valid"]), None)
    expected_ref_hash = first(row, "image_sha256", "reference_sha256", "source_image_sha256", "source_sha256")
    actual_ref_hash = sha256(ref["path"]) if ref["path"] else None
    if not ref["path"]:
        ref_status, ref_error = ref["status"], ref["error"]
    elif expected_ref_hash is None:
        ref_status, ref_error = "found_unverified", "assignment did not provide an expected reference image hash"
    elif actual_ref_hash == expected_ref_hash:
        ref_status, ref_error = "valid", None
    else:
        ref_status, ref_error = "hash_mismatch", f"reference hash {actual_ref_hash} does not match assignment hash {expected_ref_hash}"
    if folder is None:
        render_status, render_error = "missing_teacher", f"teacher directory not found (tried {', '.join(str(p) for p in tried_dirs)})"
    elif program_hash is None:
        render_status, render_error = "missing_program", "program.js is missing"
    elif selected:
        render_status, render_error = "valid", None
    else:
        render_status, render_error = "invalid_receipt", "no receipt/render pair passed validity, current-program source hash, and PNG hash checks"
    stem = f"{safe(row['teacher'])}--{safe(row['id'])}--{row['record_index'] + 1:03d}"
    assets: dict[str, dict[str, Any]] = {}
    if ref["path"]:
        copied = copy_asset(ref["path"], output / "assets" / f"{stem}--reference{ref['path'].suffix.lower() or '.jpg'}")
        assets["reference"] = {"role": "reference", "path": str(copied["path"].relative_to(output)), "sha256": copied["sha256"], "bytes": copied["bytes"]}
    if selected:
        copied = copy_asset(folder / selected["render_file"], output / "assets" / f"{stem}--teacher.png")
        assets["teacher"] = {"role": "teacher", "path": str(copied["path"].relative_to(output)), "sha256": copied["sha256"], "bytes": copied["bytes"]}
    return {
        "record_index": row["record_index"], "record_key": row["record_key"], "id": row["id"], "teacher": row["teacher"],
        "family": row.get("family"), "classes": row.get("classes") if isinstance(row.get("classes"), list) else [],
        "split": row.get("split"), "source_group": row.get("source_group"), "assignment_file": row["assignment_file"],
        "assignment_index": row["assignment_index"], "source_image_id": ref["id"],
        "reference": {"file": f"{ref['id']}{ref['path'].suffix}" if ref["path"] else None, "status": ref_status,
                      "expected_sha256": expected_ref_hash, "actual_sha256": actual_ref_hash, "error": ref_error, "tried": ref["tried"]},
        "teacher_source": {"file": "program.js" if program_hash else None, "sha256": program_hash, "status": "found" if program_hash else "missing"},
        "render": {"status": render_status, "error": render_error, "sha256": selected["actual_png_sha256"] if selected else None,
                   "receipt_file": selected["receipt_file"] if selected else None, "file": selected["render_file"] if selected else None,
                   "receipt": selected.get("receipt") if selected else None,
                   "candidate_checks": [{key: check.get(key) for key in ("receipt_file", "render_file", "receipt_present", "render_present", "receipt_valid", "receipt_source_sha256", "receipt_png_sha256", "actual_png_sha256", "receipt_errors", "reasons", "valid")} for check in checks]},
        "pair_status": "complete" if ref_status == "valid" and render_status == "valid" else ("render_valid_reference_unverified" if render_status == "valid" else render_status),
        "source_sha256": program_hash,
        "render_sha256": selected["actual_png_sha256"] if selected else None,
        "receipt_assets": (selected.get("receipt") or {}).get("assets", {}) if selected else {},
        "assets": assets,
    }


def make_contact_sheets(records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    if sys.platform != "linux":
        raise RuntimeError("--contact-sheets is Linux-only; build metadata/gallery without it on this host")
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("--contact-sheets requires Pillow on the Linux compute node") from exc
    eligible = [row for row in records if row["pair_status"] == "complete" and "reference" in row["assets"] and "teacher" in row["assets"]]
    contact_dir = output / "contacts"
    contact_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[dict[str, Any]] = []
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

    def cell(image_path: Path, label: str) -> Image.Image:
        with Image.open(image_path) as source:
            source = source.convert("RGBA")
            fitted = ImageOps.contain(source, (THUMB, THUMB), method=resample)
        canvas = Image.new("RGBA", (THUMB, LABEL_H + THUMB), "white")
        canvas.alpha_composite(fitted, ((THUMB - fitted.width) // 2, LABEL_H + (THUMB - fitted.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), label, fill="black")
        return canvas.convert("RGB")

    for number, start in enumerate(range(0, len(eligible), SHEET_PAIRS), 1):
        batch = eligible[start:start + SHEET_PAIRS]
        sheet = Image.new("RGB", (THUMB * 2, (LABEL_H + THUMB) * len(batch)), "#f3f1ec")
        for row_index, row in enumerate(batch):
            y = row_index * (LABEL_H + THUMB)
            ref = output / row["assets"]["reference"]["path"]
            render = output / row["assets"]["teacher"]["path"]
            sheet.paste(cell(ref, f"{row['teacher']} / {row['id']} · role: reference"), (0, y))
            sheet.paste(cell(render, f"{row['teacher']} / {row['id']} · role: teacher"), (THUMB, y))
        file = contact_dir / f"contact-{number:03d}.png"
        sheet.save(file, format="PNG", optimize=False)
        sheets.append({"role": "contact_sheet", "batch": number, "file": str(file.relative_to(output)), "sha256": sha256(file),
                       "record_ids": [row["id"] for row in batch], "record_keys": [row["record_key"] for row in batch],
                       "pair_count": len(batch), "width": sheet.width, "height": sheet.height,
                       "thumb": {"width": THUMB, "height": THUMB, "native_bytes_preserved": True}})
    return sheets


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pre(value: Any) -> str:
    return esc(json.dumps(value, indent=2, sort_keys=True))


def card(row: dict[str, Any]) -> str:
    receipt = row["render"].get("receipt")
    status = "ok" if row["render"]["status"] == "valid" else "bad"
    def image(kind: str, label: str) -> str:
        asset = row["assets"].get(kind)
        return f'<img loading="lazy" src="{esc(asset["path"])}" alt="{esc(label)}">' if asset else f'<div class="missing">{esc(label)} unavailable</div>'
    evidence = {"receipt": receipt, "candidate_checks": row["render"]["candidate_checks"]}
    return f'''<article class="card {status}"><div class="head"><h2>{esc(row["teacher"])} · {esc(row["id"])}</h2><span>{esc(row["pair_status"])}</span></div>
<p class="meta">family: {esc(row.get("family") or "—")} · assignment: {esc(row["assignment_file"])} · reference: {esc(row["reference"]["status"])} · render: {esc(row["render"]["status"])}</p>
<div class="pair"><figure>{image("reference", f'{row["id"]} reference')}<figcaption><b>role: reference</b><br>sha256: <code>{esc(row["reference"]["actual_sha256"] or row["reference"]["expected_sha256"] or "—")}</code></figcaption></figure>
<figure>{image("teacher", f'{row["id"]} teacher render')}<figcaption><b>role: teacher</b><br>sha256: <code>{esc(row["render"]["sha256"] or "—")}</code></figcaption></figure></div>
<dl><dt>current program.js sha256</dt><dd><code>{esc(row["teacher_source"]["sha256"] or "—")}</code></dd><dt>reference expected sha256</dt><dd><code>{esc(row["reference"]["expected_sha256"] or "—")}</code></dd></dl>
<details><summary>Shader receipt validity, errors, and candidate checks</summary><p>reference: {esc(row["reference"]["error"] or "hash verified")}</p><p>render: {esc(row["render"]["error"] or "receipt, current source, and PNG hash verified")}</p><pre>{pre(evidence)}</pre></details></article>'''


def build_html(manifest: dict[str, Any]) -> str:
    contacts = "".join(f'<a href="{esc(sheet["file"])}"><img src="{esc(sheet["file"])}" alt="contact batch {sheet["batch"]}"><br>batch {sheet["batch"]} · {sheet["pair_count"]} pairs · {esc(sheet["sha256"])}</a>' for sheet in manifest["contact_sheets"])
    cards = "\n".join(card(row) for row in manifest["records"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Teacher evidence review</title><style>
body{{margin:0;padding:24px;background:#f3f1ec;color:#252525;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto}}.note,.stat,.card,.contacts a{{background:white;border:1px solid #d5d1c8;border-radius:8px;padding:12px}}.note{{border-left:4px solid #bd8b00}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}}.stat{{min-width:110px;text-align:center}}.card{{margin:16px 0;border-left:5px solid #b34b42}}.card.ok{{border-left-color:#45875e}}.head{{display:flex;justify-content:space-between;gap:12px;align-items:center}}h1,h2{{margin:0 0 6px}}.meta{{color:#666}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:0}}figure img,.missing{{width:100%;height:auto;max-height:600px;object-fit:contain;background:#eef0f2;border:1px solid #ddd}}.missing{{min-height:120px;display:grid;place-items:center;color:#777}}figcaption{{padding:6px 0;color:#555;font-size:.85rem}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;overflow-wrap:anywhere;font-size:.85rem}}dd{{margin:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f3ef;padding:10px;border-radius:6px;font-size:.78rem}}.contacts{{margin:18px 0}}.contact-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}}.contacts a{{color:#225f8c;text-decoration:none;display:block}}.contacts img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{body{{padding:12px}}.pair{{grid-template-columns:1fr}}dl{{grid-template-columns:1fr}}.head{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><main><h1>Teacher evidence review</h1><p>Reference-to-teacher artifacts from the assignment manifests. This page reports byte hashes and renderer receipt evidence only; it makes no visual quality claim and never counts a stale or failed receipt as a valid teacher render.</p><p class="note">{manifest["record_count"]} assignment rows · {manifest["complete_count"]} complete verified pairs · {manifest["valid_render_count"]} valid render receipts · {manifest["reference_found_count"]} reference files found.</p><div class="stats"><div class="stat"><b>{manifest["record_count"]}</b><br>rows</div><div class="stat"><b>{manifest["complete_count"]}</b><br>complete</div><div class="stat"><b>{manifest["valid_render_count"]}</b><br>valid receipts</div><div class="stat"><b>{manifest["reference_found_count"]}</b><br>references found</div></div>{f'<section class="contacts"><h2>Contact sheets</h2><div class="contact-grid">{contacts}</div></section>' if contacts else ''}<h2>All assignment rows</h2>{cards}</main></body></html>\n'''


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    (output / "assets").mkdir(parents=True, exist_ok=True)
    rows = load_assignments(args.assignments)
    records = [build_record(row, args.reference_root.resolve(), args.teachers.resolve(), output) for row in rows]
    contacts = make_contact_sheets(records, output) if args.contact_sheets else []
    manifest = {"schema": SCHEMA, "generated_by": "build_teacher_review.py", "quality_claim": False,
                "assignment_files": [path.name for path in args.assignments], "record_count": len(records),
                "complete_count": sum(row["pair_status"] == "complete" for row in records),
                "valid_render_count": sum(row["render"]["status"] == "valid" for row in records),
                "reference_found_count": sum(row["reference"]["status"] in ("valid", "found_unverified", "hash_mismatch") for row in records),
                "contact_sheets_requested": bool(args.contact_sheets), "contact_sheets": contacts, "records": records}
    json_write(output / "manifest.json", manifest)
    (output / "index.html").write_text(build_html(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, action="append", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheets", action="store_true", help="compose PNG sheets; Linux compute node only")
    args = parser.parse_args()
    try:
        manifest = build(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.resolve()), "records": manifest["record_count"],
                      "complete": manifest["complete_count"], "valid_renders": manifest["valid_render_count"],
                      "contact_sheets": len(manifest["contact_sheets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
