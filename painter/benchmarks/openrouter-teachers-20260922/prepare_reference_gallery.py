#!/usr/bin/env python3
"""Build a provenance-rich static gallery for the benchmark references.

This helper only reads the reference manifest and hashes the existing JPG bytes.
It never decodes, resizes, composites, copies, renders, or calls a model.  The
HTML points at the original reference files with relative links so the gallery
stays small and remains useful when the benchmark directory is archived.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REFS = ROOT / "refs.json"
DEFAULT_OUTPUT = ROOT / "reference-gallery.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    if value is None:
        value = ""
    return html.escape(str(value), quote=True)


def _load_and_validate(refs_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    refs_path = refs_path.resolve()
    with refs_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("reference manifest must be a JSON object")
    entries = document.get("references")
    if not isinstance(entries, list):
        raise ValueError("reference manifest has no references list")
    declared_count = document.get("count", len(entries))
    if len(entries) != 40 or declared_count != 40:
        raise ValueError(f"expected exactly 40 references, found {len(entries)}")

    manifest_root = refs_path.parent
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"reference {index} is not an object")
        reference_id = entry.get("id")
        image_name = entry.get("image")
        expected_hash = entry.get("sha256")
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError(f"reference {index} has no id")
        if reference_id in seen_ids:
            raise ValueError(f"duplicate reference id: {reference_id}")
        seen_ids.add(reference_id)
        if not isinstance(image_name, str) or not image_name:
            raise ValueError(f"{reference_id}: missing image path")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"{reference_id}: malformed sha256")

        raw_image_path = manifest_root / image_name
        if raw_image_path.is_symlink():
            raise ValueError(f"{reference_id}: image must not be a symlink: {image_name}")
        image_path = raw_image_path.resolve()
        try:
            image_path.relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError(f"{reference_id}: image escapes manifest directory") from exc
        if image_path.is_symlink() or not image_path.is_file():
            raise ValueError(f"{reference_id}: missing regular image file: {image_name}")
        actual_hash = sha256(image_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                f"{reference_id}: sha256 mismatch (manifest {expected_hash}, file {actual_hash})"
            )
        validated.append(dict(entry, _image_path=image_path, _actual_sha256=actual_hash))
    return document, validated


def _provenance(entry: dict[str, Any]) -> str:
    fields = (
        ("Dataset", entry.get("dataset")),
        ("Source split", entry.get("source_split")),
        ("Source group", entry.get("source_group")),
        ("Annotated objects", entry.get("annotated_object_count")),
        ("Annotation classes", entry.get("annotation_classes")),
        ("Prior exposure", entry.get("prior_exposure")),
        ("License status", entry.get("per_image_license_status")),
        ("Selection reason", entry.get("selection_reason")),
        ("SHA-256", entry.get("sha256")),
    )
    rows = "".join(
        f"<dt>{_escape(label)}</dt><dd>{_escape(value) or '&mdash;'}</dd>"
        for label, value in fields
    )
    source_url = entry.get("source_url")
    if source_url:
        safe_url = _escape(source_url)
        rows += (
            f'<dt>Source URL</dt><dd><a href="{safe_url}" target="_blank" '
            f'rel="noopener">{safe_url}</a></dd>'
        )
    return f"<dl>{rows}</dl>"


def _card(entry: dict[str, Any], output_dir: Path) -> str:
    image_path: Path = entry["_image_path"]
    relative_image = Path(os.path.relpath(image_path, output_dir)).as_posix()
    reference_id = _escape(entry["id"])
    category = _escape(entry.get("category"))
    alt = f"Reference {reference_id}; {category}" if category else f"Reference {reference_id}"
    return f"""
      <article class="card" id="{reference_id}">
        <div class="card-heading">
          <h3>{reference_id}</h3>
          <span class="badge">{category}</span>
        </div>
        <a class="image-link" href="{_escape(relative_image)}" target="_blank" rel="noopener">
          <img src="{_escape(relative_image)}" alt="{_escape(alt)}" loading="lazy">
        </a>
        <details>
          <summary>Source provenance</summary>
          {_provenance(entry)}
        </details>
      </article>
"""


def render_gallery(refs_path: Path = DEFAULT_REFS, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    document, entries = _load_and_validate(refs_path)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry.get("category") or "Uncategorized")].append(entry)

    sections: list[str] = []
    for category in sorted(grouped):
        category_entries = sorted(grouped[category], key=lambda item: str(item["id"]))
        cards = "".join(_card(entry, output_dir) for entry in category_entries)
        sections.append(
            f'<section class="category"><h2>{_escape(category)} '
            f'<span>({len(category_entries)} references)</span></h2><div class="grid">{cards}</div></section>'
        )

    manifest_hash = document.get("source_manifest_sha256") or "not recorded"
    html_document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Gateway teacher benchmark references</title>
<style>
:root {{ color-scheme: light; --ink:#1f2937; --muted:#596579; --line:#dbe2ea; --card:#fff; --wash:#f4f7fb; --accent:#255db5; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--wash); color:var(--ink); font:16px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:28px clamp(16px,3vw,44px) 64px; }}
header {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:24px clamp(18px,3vw,36px); margin-bottom:30px; }}
h1 {{ margin:0 0 8px; font-size:clamp(1.65rem,3vw,2.35rem); line-height:1.15; }}
.lede {{ margin:0 0 18px; color:var(--muted); max-width:80ch; }}
.stats {{ display:flex; flex-wrap:wrap; gap:8px 22px; margin:0; color:var(--muted); }}
.stats strong {{ color:var(--ink); }}
.category {{ margin-top:34px; }}
.category h2 {{ margin:0 0 14px; font-size:1.35rem; }}
.category h2 span {{ color:var(--muted); font-weight:500; font-size:.95rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr)); gap:18px; }}
.card {{ min-width:0; background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; box-shadow:0 2px 8px #1f29370d; }}
.card-heading {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin-bottom:10px; }}
.card h3 {{ margin:0; font-size:1rem; overflow-wrap:anywhere; }}
.badge {{ color:var(--muted); font-size:.78rem; text-align:right; }}
.image-link {{ display:block; border-radius:9px; background:#eef2f7; outline-offset:3px; }}
.image-link:focus-visible {{ outline:3px solid var(--accent); }}
img {{ display:block; width:100%; height:clamp(260px,26vw,380px); object-fit:contain; border-radius:9px; }}
details {{ margin-top:12px; border-top:1px solid var(--line); padding-top:10px; }}
summary {{ color:var(--accent); cursor:pointer; font-weight:650; }}
dl {{ display:grid; grid-template-columns:minmax(110px,.5fr) 1fr; gap:5px 12px; margin:12px 0 0; font-size:.86rem; }}
dt {{ color:var(--muted); }}
dd {{ margin:0; overflow-wrap:anywhere; }}
a {{ color:var(--accent); }}
@media (max-width:600px) {{ main {{ padding-inline:12px; }} .grid {{ grid-template-columns:1fr; }} img {{ height:300px; }} dl {{ grid-template-columns:1fr; gap:1px; }} dt {{ margin-top:7px; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>AI Gateway teacher benchmark references</h1>
<p class="lede">Reference images for provenance and prompt-set review. This page contains no model outputs or judgments. Open an image to inspect its original full-resolution JPG; the gallery does not transform image bytes.</p>
<p class="stats"><span><strong>{len(entries)}</strong> references</span><span><strong>{len(grouped)}</strong> categories</span><span>manifest SHA-256: <strong>{_escape(manifest_hash)}</strong></span><span>generated from <strong>{_escape(Path(refs_path).name)}</strong></span></p>
</header>
{"".join(sections)}
</main>
</body>
</html>
"""
    output_path.write_text(html_document, encoding="utf-8")
    return {
        "status": "written",
        "output": str(output_path),
        "references": len(entries),
        "categories": len(grouped),
        "manifest_sha256": manifest_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS, help="reference manifest JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="HTML output path")
    args = parser.parse_args()
    print(json.dumps(render_gallery(args.refs, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
