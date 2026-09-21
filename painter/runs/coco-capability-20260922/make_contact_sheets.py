"""Make Pillow contact sheets from the recorded capability-gallery metadata.

The gallery builder is the source of truth for first/final canvas paths.  This
helper never searches episode directories or chooses a file by filename order:
each slot is either the exact recorded artifact or an explicit unavailable
panel.  Invoke it on the Linux node after the paired evaluations finish.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


SLOTS = (
    ("Reference", "reference", None, None),
    ("Base unadapted · first", "first_canvas", "base", "first_canvas"),
    ("Base unadapted · final", "final_canvas", "base", "final_canvas"),
    ("SFT step24 · first", "first_canvas", "sft", "first_canvas"),
    ("SFT step24 · final", "final_canvas", "sft", "final_canvas"),
)
CELL_WIDTH = 320
CELL_HEIGHT = 360
HEADER_HEIGHT = 58


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return text or "case"


def _remap_recorded_path(value: Any, *, node_root: Path | None, evidence_root: Path | None) -> Path | None:
    """Resolve an exact recorded path, with an explicit node-root remap."""

    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if node_root is None or evidence_root is None or not path.is_absolute():
        return None
    try:
        relative = path.resolve().relative_to(node_root.resolve())
    except ValueError:
        return None
    candidate = evidence_root.resolve() / relative
    return candidate.resolve() if candidate.is_file() else None


def _slot_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the five exact metadata slots without filesystem inference."""

    if not isinstance(case.get("policies"), dict):
        raise ValueError(f"case {case.get('id')} has no policy metadata")
    slots: list[dict[str, Any]] = []
    for label, key, policy, recorded_key in SLOTS:
        if policy is None:
            value = case.get("reference")
            status = "reference"
        else:
            record = case["policies"].get(policy)
            if not isinstance(record, dict):
                raise ValueError(f"case {case.get('id')} is missing actual {policy} metadata")
            value = record.get(recorded_key)
            status = record.get("status")
        slots.append({"label": label, "recorded_path": value, "status": status})
    return slots


def _wrap(text: str, width: int = 30) -> str:
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines) or "unavailable"


def _draw_case(case: dict[str, Any], *, output: Path, node_root: Path | None, evidence_root: Path | None) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised on the node
        raise RuntimeError("Pillow is required on the Linux node for contact sheets") from exc

    slots = _slot_records(case)
    image = Image.new("RGB", (CELL_WIDTH * len(slots), CELL_HEIGHT), (245, 242, 236))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()
    resolved_slots: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        left = index * CELL_WIDTH
        draw.rectangle((left, 0, left + CELL_WIDTH - 1, CELL_HEIGHT - 1), outline=(190, 183, 171), width=1)
        draw.multiline_text((left + 8, 8), _wrap(slot["label"], 27), fill=(38, 35, 31), font=font, spacing=2)
        path = _remap_recorded_path(slot.get("recorded_path"), node_root=node_root, evidence_root=evidence_root)
        available = path is not None
        if available:
            try:
                with Image.open(path) as source:
                    panel = source.convert("RGB")
                panel.thumbnail((CELL_WIDTH - 20, CELL_HEIGHT - HEADER_HEIGHT - 34), Image.Resampling.LANCZOS)
                x = left + (CELL_WIDTH - panel.width) // 2
                y = HEADER_HEIGHT + (CELL_HEIGHT - HEADER_HEIGHT - 28 - panel.height) // 2
                image.paste(panel, (x, y))
                caption = slot.get("status") or "recorded artifact"
            except (OSError, ValueError) as exc:
                available = False
                caption = f"artifact unreadable: {type(exc).__name__}"
        else:
            caption = f"unavailable (status={slot.get('status') or 'missing'})"
        if not available:
            draw.rectangle(
                (left + 10, HEADER_HEIGHT, left + CELL_WIDTH - 11, CELL_HEIGHT - 30),
                fill=(225, 219, 209),
                outline=(184, 174, 160),
            )
            draw.multiline_text(
                (left + 20, HEADER_HEIGHT + 24),
                _wrap(caption, 24),
                fill=(110, 53, 44),
                font=font_small,
                spacing=3,
            )
        draw.text((left + 8, CELL_HEIGHT - 20), str(caption), fill=(83, 76, 67), font=font_small)
        resolved_slots.append(
            {
                "label": slot["label"],
                "recorded_path": slot.get("recorded_path"),
                "resolved_path": str(path) if path else None,
                "available": available,
                "status": slot.get("status"),
            }
        )

    case_id = str(case.get("id") or "case")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{_safe_name(case_id)}.png"
    image.save(target, format="PNG", optimize=False)
    return {"case_id": case_id, "path": str(target), "slots": resolved_slots}


def make_contact_sheets(
    *, metadata_path: Path, output_dir: Path, node_root: Path | None = Path("/root/painter"), evidence_root: Path | None = None
) -> dict[str, Any]:
    metadata_path = metadata_path.resolve()
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "coco-capability-gallery-v1":
        raise ValueError("metadata must be generated by the base/sft capability gallery")
    if raw.get("policy_labels") != {"base": "base", "sft": "sft"}:
        raise ValueError("metadata must use actual base/sft policy labels")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("metadata cases must be a list of objects")
    if len(cases) != 12:
        raise ValueError(f"expected twelve capability cases, found {len(cases)}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = [
        _draw_case(case, output=output_dir, node_root=node_root, evidence_root=evidence_root)
        for case in cases
    ]

    # A single overview makes the twelve per-case rows easy to inspect while
    # retaining the individual PNGs for exact per-case review.
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required on the Linux node for contact sheets") from exc
    overview = Image.new("RGB", (CELL_WIDTH * len(SLOTS), CELL_HEIGHT * len(sheets)), (245, 242, 236))
    draw = ImageDraw.Draw(overview)
    font = ImageFont.load_default()
    for index, sheet in enumerate(sheets):
        with Image.open(sheet["path"]) as row:
            overview.paste(row.convert("RGB"), (0, index * CELL_HEIGHT))
        draw.text((6, index * CELL_HEIGHT + CELL_HEIGHT - 4), str(sheet["case_id"]), fill=(30, 27, 24), font=font)
    overview_path = output_dir / "overview.png"
    overview.save(overview_path, format="PNG", optimize=False)
    result = {
        "schema": "coco-capability-contact-sheets-v1",
        "metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "case_count": len(sheets),
        "overview": str(overview_path),
        "sheets": sheets,
        "notes": [
            "Every image slot comes from the gallery metadata's recorded reference/first/final path.",
            "Unavailable or unreadable artifacts are shown explicitly; no directory-order fallback is used.",
        ],
    }
    (output_dir / "index.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="gallery/metadata.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, default=Path("/root/painter"))
    parser.add_argument("--evidence-root", type=Path, default=None, help="optional collected-root remap for recorded node paths")
    args = parser.parse_args(argv)
    result = make_contact_sheets(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        node_root=args.node_root,
        evidence_root=args.evidence_root,
    )
    print(json.dumps({"overview": result["overview"], "case_count": result["case_count"], "output_dir": result["output_dir"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
