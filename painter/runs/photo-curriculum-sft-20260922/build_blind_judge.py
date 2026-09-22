#!/usr/bin/env python3
"""Make deterministic, checkpoint-blind A/B/reference triptychs for Luna.

This reads the manifest produced by ``build_matched_eval_gallery.py``.  It
never renders or invents a canvas: a cell is copied only when the manifest
marks its artifact valid and the copied bytes still match their recorded hash.
The checkpoint mapping is written separately from the judge-facing gallery.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "blind-photo-judge-v1"
POLICIES = ("midpoint512", "final1024")
THUMB = 600
LABEL_H = 30


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).lstrip(".")
    return result or "unknown"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        raise ValueError("gallery manifest must be an object with records")
    return raw


def valid_asset(record: dict[str, Any], role: str, gallery_root: Path) -> tuple[Path | None, str]:
    asset = record.get("asset") if isinstance(record, dict) else None
    if record.get("status") != "valid":
        return None, str(record.get("error") or f"{role} has no valid final painting")
    if not isinstance(asset, dict) or not asset.get("path"):
        return None, f"{role} is marked valid but has no copied asset"
    path = gallery_root / str(asset["path"])
    if not path.is_file():
        return None, f"{role} asset is missing: {asset['path']}"
    actual = digest(path)
    expected = asset.get("sha256") or record.get("render_sha256")
    if expected and actual != expected:
        return None, f"{role} asset hash mismatch: {actual} != {expected}"
    return path, "verified"


def valid_reference(reference: dict[str, Any], gallery_root: Path) -> tuple[Path | None, str]:
    if reference.get("status") != "valid":
        return None, str(reference.get("error") or "reference is not hash-verified")
    asset = reference.get("asset")
    if not isinstance(asset, dict) or not asset.get("path"):
        return None, "reference has no copied asset"
    path = gallery_root / str(asset["path"])
    if not path.is_file():
        return None, f"reference asset is missing: {asset['path']}"
    actual = digest(path)
    expected = asset.get("sha256") or reference.get("actual_sha256")
    if expected and actual != expected:
        return None, f"reference asset hash mismatch: {actual} != {expected}"
    return path, "verified"


def assignment(case_id: str, seed: str) -> tuple[str, str, str]:
    value = hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).digest()[0]
    a, b = POLICIES if value % 2 == 0 else (POLICIES[1], POLICIES[0])
    return a, b, hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).hexdigest()


def copy_verified(source: Path, output: Path, expected: str | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(source.read_bytes())
    actual = digest(output)
    if expected and actual != expected:
        raise ValueError(f"copied asset hash mismatch: {actual} != {expected}")
    return {"path": str(output), "sha256": actual, "bytes": output.stat().st_size}


def make_cell(source: Path | None, label: str, missing: str, Image: Any, ImageDraw: Any, ImageOps: Any, resample: Any) -> Any:
    cell = Image.new("RGBA", (THUMB, THUMB + LABEL_H), "white")
    draw = ImageDraw.Draw(cell)
    draw.text((8, 7), label, fill="black")
    if source is None:
        draw.text((12, LABEL_H + 20), "no valid final painting", fill="#555")
        return cell.convert("RGB")
    with Image.open(source) as image:
        fitted = ImageOps.contain(image.convert("RGBA"), (THUMB, THUMB), method=resample)
    cell.alpha_composite(fitted, ((THUMB - fitted.width) // 2, LABEL_H + (THUMB - fitted.height) // 2))
    return cell.convert("RGB")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "linux":
        raise RuntimeError("blind PNG composition is Linux-only; run this helper on the compute node")
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("blind PNG composition requires Pillow on the Linux node") from exc
    manifest_path = args.manifest.resolve()
    gallery_root = manifest_path.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "triptychs").mkdir(parents=True, exist_ok=True)
    (output / "baseline").mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(manifest_path)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    mapping_rows: list[dict[str, Any]] = []
    visible_rows: list[dict[str, Any]] = []

    for ordinal, record in enumerate(manifest["records"], 1):
        case_id = str(record.get("id") or f"case-{ordinal:03d}")
        reference_path, reference_reason = valid_reference(record.get("reference") or {}, gallery_root)
        policy_a, policy_b, assignment_hash = assignment(case_id, args.seed)
        candidates: dict[str, dict[str, Any]] = {}
        for policy in POLICIES:
            policy_record = record.get("policies", {}).get(policy) or {}
            path, reason = valid_asset(policy_record, policy, gallery_root)
            candidates[policy] = {"path": path, "reason": reason, "status": "valid" if path else "missing_or_failed",
                                  "render_sha256": policy_record.get("render_sha256"), "source_sha256": policy_record.get("source_sha256"),
                                  "trace_sha256": policy_record.get("trace_sha256")}
        visible = []
        for label, policy in (("candidate A", policy_a), ("candidate B", policy_b)):
            visible.append({"label": label, "policy": policy, "path": candidates[policy]["path"], "reason": candidates[policy]["reason"]})

        triptych_path = output / "triptychs" / f"case-{ordinal:03d}-{safe(case_id)}.png"
        sheet = Image.new("RGB", (THUMB * 3, THUMB + LABEL_H), "#f3f1ec")
        sheet.paste(make_cell(reference_path, f"{case_id} · role: reference", reference_reason, Image, ImageDraw, ImageOps, resample), (0, 0))
        for column, cell in enumerate(visible, 1):
            sheet.paste(make_cell(cell["path"], f"{case_id} · role: {cell['label']}", cell["reason"], Image, ImageDraw, ImageOps, resample), (column * THUMB, 0))
        sheet.save(triptych_path, format="PNG", optimize=False)

        baseline_record = record.get("policies", {}).get("base") or {}
        baseline_path, baseline_reason = valid_asset(baseline_record, "base", gallery_root)
        baseline_asset = None
        if baseline_path:
            copied = copy_verified(baseline_path, output / "baseline" / f"case-{ordinal:03d}-{safe(case_id)}.png", baseline_record.get("render_sha256"))
            baseline_asset = {"path": str(Path(copied["path"]).relative_to(output)), "sha256": copied["sha256"], "bytes": copied["bytes"]}
        mapping_rows.append({
            "ordinal": ordinal, "case_id": case_id, "assignment_hash": assignment_hash,
            "reference": {"status": "valid" if reference_path else "missing_or_failed", "sha256": record.get("reference", {}).get("actual_sha256"),
                           "asset": record.get("reference", {}).get("asset", {}).get("path") if reference_path else None, "reason": reference_reason},
            "candidate_a": {"visible_label": "candidate A", "policy": policy_a, "status": candidates[policy_a]["status"],
                            "render_sha256": candidates[policy_a]["render_sha256"], "source_sha256": candidates[policy_a]["source_sha256"], "trace_sha256": candidates[policy_a]["trace_sha256"], "reason": candidates[policy_a]["reason"]},
            "candidate_b": {"visible_label": "candidate B", "policy": policy_b, "status": candidates[policy_b]["status"],
                            "render_sha256": candidates[policy_b]["render_sha256"], "source_sha256": candidates[policy_b]["source_sha256"], "trace_sha256": candidates[policy_b]["trace_sha256"], "reason": candidates[policy_b]["reason"]},
            "baseline": {"status": "valid" if baseline_asset else "missing_or_failed", "render_sha256": baseline_record.get("render_sha256"),
                         "source_sha256": baseline_record.get("source_sha256"), "trace_sha256": baseline_record.get("trace_sha256"),
                         "asset": baseline_asset, "reason": baseline_reason},
            "triptych": {"path": str(triptych_path.relative_to(output)), "sha256": digest(triptych_path), "width": sheet.width, "height": sheet.height},
        })
        visible_rows.append({"ordinal": ordinal, "case_id": case_id, "triptych": str(triptych_path.relative_to(output)),
                             "baseline": str(baseline_asset["path"]) if baseline_asset else None,
                             "baseline_status": "valid" if baseline_asset else baseline_reason})

    mapping = {"schema": SCHEMA, "quality_claim": False, "seed": args.seed, "checkpoint_identities_hidden_in_gallery": True,
               "case_count": len(mapping_rows), "mapping_rows": mapping_rows}
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    (output / "index.html").write_text(build_html(visible_rows), encoding="utf-8")
    return mapping


def build_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        baseline = f'<p><a href="{esc(row["baseline"])}">separate baseline image</a></p>' if row["baseline"] else f'<p>separate baseline: {esc(row["baseline_status"])}</p>'
        cards.append(f'<article><h2>case {row["ordinal"]}: {esc(row["case_id"])}</h2><img src="{esc(row["triptych"])}" alt="{esc(row["case_id"])} blind reference and candidate A/B triptych">{baseline}</article>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind photo judge</title><style>body{{margin:0;padding:24px;background:#f3f1ec;color:#252525;font:14px/1.45 system-ui,sans-serif}}main{{max-width:1850px;margin:auto}}.note,article{{background:white;border:1px solid #d5d1c8;border-radius:8px;padding:12px}}article{{margin:16px 0}}article img{{display:block;max-width:100%;height:auto;background:#eef0f2}}h1,h2{{margin:0 0 6px}}a{{color:#225f8c}}</style></head><body><main><h1>Blind photo judge</h1><p class="note">Each image shows reference, candidate A, and candidate B. Candidate checkpoint identities are stored only in mapping.json. Missing or failed outputs say “no valid final painting”; no canvas is fabricated.</p>{''.join(cards)}</main></body></html>\n'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="matched gallery manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="photo-curriculum-sft-20260922-blind-v1")
    args = parser.parse_args()
    try:
        mapping = build(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output.resolve()), "cases": mapping["case_count"], "mapping": "mapping.json"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
