#!/usr/bin/env python3
"""Build offline galleries and a stable blind packet from benchmark artifacts.

The normal gallery is provenance-rich and may name model/track.  The blind
packet copies reference, first-valid, and final-valid images under opaque IDs
and deliberately contains no model or reference IDs.  Nothing in this file
calls a judge or a model.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "episodes").glob("*/episode.json")):
        row = _load(path)
        if row:
            row["_path"] = path
            rows.append(row)
    return rows


def _link(root: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = root / value
    return _rel(path, root) if path.is_file() else None


def build_reference_gallery(root: Path, output: Path | None = None) -> Path:
    """Build a selection-only gallery without requiring any episode output."""

    refs_doc = _load(root / "refs.json") or {}
    refs = refs_doc.get("references") or []
    cards = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        image = root / str(ref.get("image", ""))
        if not image.is_file():
            continue
        rel = _rel(image, root)
        caption = f"{ref.get('id', '')} · {ref.get('category', '')}"
        cards.append(
            f'<figure><img loading="lazy" src="{html.escape(rel)}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}<br><code>{html.escape(ref.get("sha256", ""))}</code></figcaption></figure>'
        )
    page = _page(
        "Reference gallery",
        "<h1>AI Gateway teacher benchmark references</h1>"
        "<p>Offline selection preview. No API calls or generated outputs are used.</p>"
        '<section class="grid">' + "".join(cards) + "</section>",
    )
    output = output or root / "reference-gallery.html"
    output.write_text(page, encoding="utf-8")
    return output


def build_gallery(root: Path, output: Path | None = None, *, build_blind: bool = True) -> Path:
    """Build the model-labeled episode gallery and optional blind judge packet."""

    rows = _episode_rows(root)
    cards: list[str] = []
    blind_comparisons: list[dict[str, Any]] = []
    blind_map: dict[str, Any] = {}
    blind_root = root / "review" / "blind"
    if build_blind:
        blind_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        episode_path = row["_path"]
        episode_dir = episode_path.parent
        turns = row.get("turns") or []
        reference = _link(root, row.get("reference_image"))
        first = _link(root, row.get("first_valid_canvas"))
        final = _link(root, row.get("final_valid_canvas"))
        turn_html: list[str] = []
        for turn in turns:
            number = int(turn.get("turn", len(turn_html) + 1))
            canvas = _link(root, turn.get("canvas"))
            program = _link(root, turn.get("program"))
            raw = _rel(episode_dir / f"turn-{number:02d}.json", root) if (episode_dir / f"turn-{number:02d}.json").is_file() else None
            response = turn.get("response") or {}
            usage = response.get("usage") or {}
            validity = "valid" if (turn.get("render") or {}).get("valid") else "invalid/unrendered"
            image = f'<img loading="lazy" src="{html.escape(canvas)}" alt="turn {number}">' if canvas else "<div class=missing>No valid render</div>"
            links = " · ".join(
                f'<a href="{html.escape(target)}">{label}</a>' for label, target in (("program", program), ("raw response", raw)) if target
            )
            turn_html.append(
                f'<figure>{image}<figcaption>Turn {number}: {html.escape(validity)} · '
                f'{html.escape(str(usage.get("total_tokens") or "?"))} tokens · {links}</figcaption></figure>'
            )
        reference_image = f'<img loading="lazy" src="{html.escape(reference)}" alt="reference image">' if reference else "<div class=missing>Missing reference</div>"
        first_image = f'<img loading="lazy" src="{html.escape(first)}" alt="first valid canvas">' if first else "<div class=missing>Missing</div>"
        final_image = f'<img loading="lazy" src="{html.escape(final)}" alt="final valid canvas">' if final else "<div class=missing>Missing</div>"
        cards.append(
            f'<article><h2>{html.escape(str(row.get("track")))} · {html.escape(str(row.get("model")))} · '
            f'{html.escape(str(row.get("reference_id")))}</h2><p>Status: <code>{html.escape(str(row.get("status")))}</code> · '
            f'{html.escape(str(row.get("total_tokens", "?")))} tokens · known cost {html.escape(str(row.get("known_cost", row.get("total_cost", "?"))))}</p>'
            f'<div class="compare"><figure>{reference_image}<figcaption>Reference</figcaption></figure>'
            f'<figure>{first_image}<figcaption>First valid canvas</figcaption></figure>'
            f'<figure>{final_image}<figcaption>Final valid canvas</figcaption></figure></div>'
            f'<div class="turns">{"".join(turn_html)}</div></article>'
        )
        if build_blind:
            identity = f"{row.get('job_id')}|{row.get('reference_sha256')}"
            blind_id = "blind-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            destination = blind_root / blind_id
            destination.mkdir(parents=True, exist_ok=True)
            ref = root / str(row.get("reference_image", ""))
            comparison = {"id": blind_id, "protocol_version": "painter.ai-gateway-teachers.blind.v1", "images": {}}
            for role, source_value in (("reference", row.get("reference_image")), ("first", row.get("first_valid_canvas")), ("final", row.get("final_valid_canvas"))):
                source = root / str(source_value) if source_value else None
                if source and source.is_file():
                    target = destination / f"{role}{source.suffix.lower() or '.png'}"
                    shutil.copyfile(source, target)
                    comparison["images"][role] = {"path": _rel(target, root), "sha256": _sha(target)}
            blind_comparisons.append(comparison)
            blind_map[blind_id] = {"job_id": row.get("job_id"), "model": row.get("model"), "track": row.get("track"), "reference_id": row.get("reference_id")}
    page = _page(
        "AI Gateway teacher benchmark gallery",
        "<h1>AI Gateway teacher benchmark</h1>"
        "<p>Generated from local episode receipts. Raw response and program links remain available before a render exists.</p>"
        + "".join(cards),
    )
    output = output or root / "gallery.html"
    output.write_text(page, encoding="utf-8")
    if build_blind:
        review = root / "review"
        review.mkdir(parents=True, exist_ok=True)
        (review / "blind-map.json").write_text(json.dumps({"schema": "painter.ai-gateway-teachers.blind-map.v1", "created_at": time.time(), "items": blind_map}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (review / "judge-packet.json").write_text(json.dumps({"schema": "painter.ai-gateway-teachers.blind.v1", "comparisons": blind_comparisons}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>body{{font:15px/1.45 system-ui;margin:2rem;background:#f4f0e8;color:#292722}}"
        "article,figure{background:#fff;padding:1rem;border-radius:8px}article{margin:1rem 0}"
        ".grid,.compare,.turns{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}"
        "img{max-width:100%;display:block}figcaption{margin-top:.5rem}.missing{padding:4rem 1rem;background:#eee;text-align:center}"
        "code{overflow-wrap:anywhere}</style>" + body
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--no-blind", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.references_only:
        print(build_reference_gallery(args.root, args.output))
    else:
        print(build_gallery(args.root, args.output, build_blind=not args.no_blind))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
