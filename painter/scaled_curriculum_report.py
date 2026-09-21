"""Build a static preview of the scaled painter curriculum references.

This is a metadata-only gallery builder.  It does not render, resize, generate,
judge, or modify any image.  Every displayed image is resolved from the
existing procedural preview or the hash-verified real-reference manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "painter/runs/scaled-curriculum-20260921"
DEFAULT_REAL_COUNT = 24


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(value: Any, *, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [base_dir / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _rel(path: Path, *, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _safe_link(path: Path | None, *, output_dir: Path, label: str) -> str:
    if path is None or not path.is_file():
        return f'<span class="missing">{html.escape(label)} missing</span>'
    target = html.escape(_rel(path, output_dir=output_dir), quote=True)
    return f'<a href="{target}">{html.escape(label)}</a>'


def _image(path: Path | None, *, output_dir: Path, alt: str) -> str:
    if path is None or not path.is_file():
        return f'<div class="missing">{html.escape(alt)} missing</div>'
    target = html.escape(_rel(path, output_dir=output_dir), quote=True)
    return (
        f'<a href="{target}"><img loading="lazy" src="{target}" '
        f'alt="{html.escape(alt, quote=True)}"></a>'
    )


def _provenance_links(urls: Iterable[Any]) -> str:
    links = []
    for value in urls:
        if isinstance(value, str) and value:
            links.append(f'<a href="{html.escape(value, quote=True)}">source</a>')
    return " · ".join(links)


def _integer_field(data: dict[str, Any], *names: str) -> int | None:
    """Return the first non-negative integer field, without treating bool as int."""
    for name in names:
        value = data.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _load_render_summary(path: Path) -> dict[str, Any]:
    """Normalize a renderer-produced summary without inventing completion."""
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"render summary is not an object: {path}")
    # Permit a small wrapper while keeping the renderer's original fields intact.
    data = raw.get("summary") if isinstance(raw.get("summary"), dict) else raw
    expected = _integer_field(data, "expected", "total", "render_expected")
    completed = _integer_field(data, "completed", "records_completed", "rendered")
    valid = _integer_field(data, "valid", "accepted", "rendered_valid")
    failed = _integer_field(data, "failed", "invalid", "errors")
    explicit_complete = data.get("complete")
    complete: bool | None
    if isinstance(explicit_complete, bool):
        complete = explicit_complete
    elif expected is not None and completed is not None and valid is not None:
        complete = completed == expected and valid == expected and (failed in (None, 0))
    else:
        complete = None
    status = data.get("status")
    if not isinstance(status, str) or not status:
        if complete is True:
            status = "complete per supplied render summary"
        elif complete is False:
            status = "partial or failed per supplied render summary"
        else:
            status = "reported by supplied render summary"
    return {
        "path": path.resolve(),
        "expected": expected,
        "completed": completed,
        "valid": valid,
        "failed": failed,
        "complete": complete,
        "status": status,
        "timestamp": data.get("finished_utc") or data.get("completed_utc") or data.get("timestamp_utc"),
        "raw_schema": raw.get("schema"),
    }


def _load_procedural(run_dir: Path) -> list[dict[str, Any]]:
    selection_path = run_dir / "preview/selection.json"
    rows = _read_json(selection_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"procedural preview selection is empty or invalid: {selection_path}")
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"procedural preview row is not an object: {row!r}")
        family = str(row.get("family") or row.get("id") or "unknown")
        image = _resolve_repo_path(row.get("image"), base_dir=run_dir)
        if image is None:
            image = _resolve_repo_path(f"preview/{family}.png", base_dir=run_dir)
        if image is None:
            raise FileNotFoundError(f"procedural preview image missing for {row.get('id')}")
        expected = row.get("sha256")
        if expected and _sha256(image) != expected:
            raise ValueError(f"procedural preview hash mismatch for {row.get('id')}: {image}")
        program = _resolve_repo_path(row.get("program"), base_dir=run_dir)
        if program is None:
            program = _resolve_repo_path(f"preview/{family}.js", base_dir=run_dir)
        output.append(
            {
                "id": row.get("id"),
                "family": family,
                "composition": row.get("composition"),
                "image": image,
                "program": program,
                "sha256": expected,
                "source_kind": "procedural_known_program",
                "curriculum_kind": "training reference target",
            }
        )
    return output


def _load_real(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "real-pool/reference-manifest.json"
    raw = _read_json(manifest_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("references"), list):
        raise ValueError(f"real reference manifest has no references list: {manifest_path}")
    rows: list[dict[str, Any]] = []
    for row in raw["references"]:
        if not isinstance(row, dict):
            raise ValueError("real reference manifest contains a non-object row")
        image = _resolve_repo_path(row.get("reference") or row.get("reference_path"), base_dir=ROOT)
        if image is None:
            raise FileNotFoundError(f"real reference image missing for {row.get('id')}")
        expected = row.get("image_sha256") or row.get("sha256")
        if expected and _sha256(image) != expected:
            raise ValueError(f"real reference hash mismatch for {row.get('id')}: {image}")
        item = dict(row)
        item["image"] = image
        item["source_kind"] = "real_photo"
        item["curriculum_kind"] = "training reference target"
        rows.append(item)
    return raw, rows


def _diverse_real(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(rows):
        return list(rows)
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    covered: set[str] = set()

    def families(row: dict[str, Any]) -> set[str]:
        values = row.get("annotation_classes") or row.get("member_families") or []
        if isinstance(values, str):
            values = [values]
        out = {str(value) for value in values if value}
        primary = row.get("primary_family") or row.get("family")
        if primary:
            out.add(str(primary))
        return out or {"unknown_unlabeled"}

    # Cover as many annotation classes as possible in deterministic manifest
    # order before filling the remaining slots.
    for row in rows:
        row_id = str(row.get("id"))
        row_families = families(row)
        if row_id not in chosen_ids and not (row_families & covered):
            chosen.append(row)
            chosen_ids.add(row_id)
            covered.update(row_families)
            if len(chosen) == count:
                return chosen
    for row in rows:
        row_id = str(row.get("id"))
        if row_id not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(row_id)
            if len(chosen) == count:
                return chosen
    return chosen


def _card(row: dict[str, Any], *, output_dir: Path, real: bool) -> str:
    image = _image(row["image"], output_dir=output_dir, alt=str(row.get("id")))
    if real:
        classes = row.get("annotation_classes") or row.get("member_families") or []
        if isinstance(classes, str):
            classes = [classes]
        class_text = ", ".join(str(value) for value in classes) if classes else "unknown_unlabeled"
        urls = list(row.get("provenance_urls") or [])
        for value in (row.get("source_url"), row.get("source_image_url")):
            if value and value not in urls:
                urls.append(value)
        dataset_license = row.get("dataset_license_documentation")
        if not dataset_license:
            dataset_license = "unrecorded"
        per_image_status = row.get("per_image_license_status") or "unrecorded"
        details = (
            f"<div class=\"meta\"><strong>{html.escape(str(row.get('primary_family') or row.get('family') or 'unknown_unlabeled'))}</strong> · "
            f"source_id {html.escape(str(row.get('source_id') or row.get('id')))} · "
            f"split {html.escape(str(row.get('split') or 'unknown'))}</div>"
            f"<div>annotation classes: {html.escape(class_text)}</div>"
            f"<div>dataset license documentation: {html.escape(str(dataset_license))}</div>"
            f"<div>per-image license status: {html.escape(str(per_image_status))}</div>"
            f"<div>{_provenance_links(urls)}</div>"
            f"<div class=\"hash\">sha256: {html.escape(str(row.get('image_sha256') or 'unrecorded'))}</div>"
        )
    else:
        details = (
            f"<div class=\"meta\"><strong>{html.escape(str(row.get('family')))}</strong> · "
            f"source id {html.escape(str(row.get('id')))} · "
            f"composition {html.escape(str(row.get('composition') or 'unrecorded'))}</div>"
            f"<div>{_safe_link(row.get('program'), output_dir=output_dir, label='source program')}</div>"
            f"<div class=\"hash\">sha256: {html.escape(str(row.get('sha256') or 'unrecorded'))}</div>"
        )
    return (
        f"<article class=\"card\">{image}<div class=\"caption\">"
        f"<h3>{html.escape(str(row.get('id')))}</h3>"
        f"<div class=\"kind\">{html.escape(str(row['curriculum_kind']))} · {html.escape(str(row['source_kind']))}</div>"
        f"{details}</div></article>"
    )


def build_gallery(
    run_dir: Path = DEFAULT_RUN,
    output: Path | None = None,
    *,
    real_count: int = DEFAULT_REAL_COUNT,
    render_summary: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    output = (output or run_dir / "review/index.html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    procedural = _load_procedural(run_dir)
    real_manifest, real_rows = _load_real(run_dir)
    selected_real = _diverse_real(real_rows, real_count)
    render_expected = None
    render_started = None
    supplied_render_summary: dict[str, Any] | None = None
    experiment_path = run_dir / "EXPERIMENT.json"
    if experiment_path.is_file():
        experiment = _read_json(experiment_path)
        preparation = experiment.get("preparation") if isinstance(experiment, dict) else {}
        if isinstance(preparation, dict):
            render_expected = preparation.get("render_expected")
            render_started = preparation.get("render_started_utc")
    if render_summary is not None:
        supplied_render_summary = _load_render_summary(Path(render_summary).resolve())
        if supplied_render_summary["expected"] is not None:
            render_expected = supplied_render_summary["expected"]
    summary = {
        "schema": "scaled-curriculum-review-v1",
        "run_dir": str(run_dir),
        "output": str(output),
        "procedural_preview_count": len(procedural),
        "real_manifest_count": len(real_rows),
        "real_displayed_count": len(selected_real),
        "real_manifest_admitted_count": real_manifest.get("admitted_count"),
        "real_manifest_available_source_images": real_manifest.get("available_source_images"),
        "bulk_render_expected": render_expected,
        "bulk_render_started_utc": render_started,
        "bulk_render_status": (
            supplied_render_summary["status"]
            if supplied_render_summary is not None
            else "not asserted; no completed bulk-render receipt was supplied to this gallery"
        ),
        "bulk_render_summary_path": (
            str(supplied_render_summary["path"]) if supplied_render_summary is not None else None
        ),
        "bulk_render_completed": (
            supplied_render_summary["completed"] if supplied_render_summary is not None else None
        ),
        "bulk_render_valid": supplied_render_summary["valid"] if supplied_render_summary is not None else None,
        "bulk_render_failed": supplied_render_summary["failed"] if supplied_render_summary is not None else None,
        "bulk_render_complete": supplied_render_summary["complete"] if supplied_render_summary is not None else None,
        "bulk_render_timestamp": supplied_render_summary["timestamp"] if supplied_render_summary is not None else None,
    }
    page = _html(summary, procedural, selected_real, output.parent)
    output.write_text(page, encoding="utf-8")
    return summary


def _html(
    summary: dict[str, Any],
    procedural: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    procedural_cards = "".join(_card(row, output_dir=output_dir, real=False) for row in procedural)
    real_cards = "".join(_card(row, output_dir=output_dir, real=True) for row in real_rows)
    s = lambda value: html.escape(str(value)) if value is not None else "unknown"
    if summary.get("bulk_render_summary_path"):
        summary_path = Path(str(summary["bulk_render_summary_path"]))
        render_link = _safe_link(summary_path, output_dir=output_dir, label="supplied render summary")
        counts = (
            f"completed {s(summary.get('bulk_render_completed'))}/{s(summary.get('bulk_render_expected'))}; "
            f"valid {s(summary.get('bulk_render_valid'))}; failed {s(summary.get('bulk_render_failed'))}"
        )
        render_text = (
            f"<strong>Bulk render:</strong> {html.escape(str(summary['bulk_render_status']))}; {counts}. "
            f"{render_link}"
        )
        if summary.get("bulk_render_timestamp"):
            render_text += f"; timestamp {s(summary['bulk_render_timestamp'])}"
    else:
        render_text = (
            f"<strong>Bulk render:</strong> planned expected count {s(summary['bulk_render_expected'])}; "
            "status is not asserted because no completed bulk-render receipt was supplied. "
            f"Preparation start recorded at {s(summary['bulk_render_started_utc'])}."
        )
    return f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scaled painter curriculum reference preview</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:24px;background:#f3f0e9;color:#27241f}}
h1{{font-size:1.6rem;margin-bottom:.3rem}}h2{{margin-top:2.2rem}}
.notice,.summary{{background:#fff;border:1px solid #d8d1c4;border-radius:8px;padding:14px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}}
.card{{background:#fff;border:1px solid #d8d1c4;border-radius:8px;overflow:hidden}}
.card img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#eee8dd}}
.caption{{padding:11px}}h3{{margin:.1rem 0 .4rem;font-size:1rem;overflow-wrap:anywhere}}
.kind,.meta{{font-size:.9rem;color:#665b4c}}.hash{{font:11px ui-monospace,monospace;color:#807566;overflow-wrap:anywhere;margin-top:5px}}
.missing{{padding:18px;color:#a23c2f;font-style:italic}}a{{color:#715426}}small{{color:#766b5d}}
</style></head><body>
<h1>Scaled painter curriculum reference preview</h1>
<div class="notice"><strong>Reference targets for training preparation.</strong> These are source references and procedural targets, not student paintings, model results, or quality judgments. The page performs no rendering or image editing.</div>
<div class="summary"><strong>Inventory:</strong> {s(summary['procedural_preview_count'])} procedural previews displayed · {s(summary['real_displayed_count'])} real photos displayed from {s(summary['real_manifest_count'])} admitted manifest rows (deterministic diversity selection).<br>
<strong>Manifests:</strong> <a href="../preview/selection.json">procedural selection.json</a> · <a href="../real-pool/reference-manifest.json">real-pool reference-manifest.json</a><br>
{render_text}</div>
<h2>Procedural preview targets</h2><div class="grid">{procedural_cards}</div>
<h2>Real-photo reference pool sample</h2><p>Each card retains source ID, split, annotation classes, license, hash, and provenance links from the real-pool manifest. The underlying manifest has {s(summary['real_manifest_count'])} locally hash-verified rows; this page shows {s(summary['real_displayed_count'])} for quick inspection.</p>
<div class="grid">{real_cards}</div>
</body></html>'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--real-count", type=int, default=DEFAULT_REAL_COUNT)
    parser.add_argument("--all-real", action="store_true", help="display all admitted real manifest rows")
    parser.add_argument("--render-summary", type=Path, default=None, help="optional renderer-produced JSON summary")
    args = parser.parse_args(argv)
    summary = build_gallery(
        args.run_dir,
        args.output,
        real_count=(10**9 if args.all_real else args.real_count),
        render_summary=args.render_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
