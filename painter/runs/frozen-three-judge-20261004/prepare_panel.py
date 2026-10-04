"""Freeze a small, diversified three-judge frozen-image panel.

The source galleries are existing local evidence.  This preparation step only
reads their HTML/JSON and PNGs; it never renders images or contacts a model.
The resulting pairs file is deliberately explicit so a later launch cannot
silently change a source image, prompt, or historical checkpoint label.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN = Path(__file__).resolve().parent
OLD_GALLERY = RUN.parent / "text-grounded-pilot-20260930" / "local-evidence" / "gallery"
PRO_REVIEW = RUN.parent / "text-rl16-pro-20261002" / "local-evidence" / "review"

# Deliberately selected for shape, count, scene, object, and detail diversity.
# These names are selection metadata only; the judge receives the brief and
# labeled LEFT/RIGHT panels, never these checkpoint labels.
SFT_OLD_RL16 = (
    "t600-309",  # harvester
    "t600-516",  # fan
    "t600-375",  # pantry
    "t600-377",  # octopus
    "t600-403",  # horseshoe
    "t600-413",  # beekeeper
    "t600-425",  # seahorse
    "t600-433",  # hand-crank drill
    "t600-439",  # clock
    "t600-542",  # yarn / knitting needles
    "t600-560",  # sand yacht
    "t600-591",  # kingfisher
)
SFT_GROUNDED_RL24 = (
    "t600-316",  # cranes
    "t600-375",  # pantry
    "t600-342",  # beach cabin
    "t600-370",  # ferret
    "t600-373",  # gutter repair
    "t600-466",  # train
    "t600-516",  # fan
    "t600-487",  # cellist
)
DIRECT_RL16_RL24 = (
    "t600-309",  # harvester
    "t600-413",  # beekeeper
)
SFT_NEW_PRO12 = (
    "t600-478",  # observatory
    "t600-516",  # fan
)


# These are provenance records for the three explicit labels selected by this
# panel.  They preserve the existing user evidence without adding labels.  The
# two fan rows are correlated views of one preference statement, not two
# independent human accuracy examples.
HUMAN_LABEL_PROVENANCE = {
    "sft-oldrl16-02": {
        "source_date": "2026-09-30",
        "source_artifact": "painter/runs/judge-alignment-20260930/PROTOCOL.md",
        "user_quote": (
            "Explicit preferences are step 16 over step 24 for the beekeeper and "
            "kingfisher, and the frozen SFT anchor over the other fan paintings."
        ),
        "correlation_note": "Same fan preference statement as sft-groundedrl24-07; correlated, not independent.",
    },
    "sft-groundedrl24-07": {
        "source_date": "2026-09-30",
        "source_artifact": "painter/runs/judge-alignment-20260930/PROTOCOL.md",
        "user_quote": (
            "Explicit preferences are step 16 over step 24 for the beekeeper and "
            "kingfisher, and the frozen SFT anchor over the other fan paintings."
        ),
        "correlation_note": "Same fan preference statement as sft-oldrl16-02; correlated, not independent.",
    },
    "oldrl16-groundedrl24-02": {
        "source_date": "2026-09-30",
        "source_artifact": "painter/runs/judge-alignment-20260930/PROTOCOL.md",
        "user_quote": (
            "Explicit preferences are step 16 over step 24 for the beekeeper and "
            "kingfisher, and the frozen SFT anchor over the other fan paintings."
        ),
        "correlation_note": "Direct beekeeper preference; no independent-label claim beyond this saved statement.",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_id(path: Path) -> str:
    """Opaque stable identity; no source filename is sent to the judge."""

    return "img_" + sha256(path)[:20]


def _parse_old_gallery(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    section_re = re.compile(r"<section>(.*?)</section>", re.S)
    result: dict[str, dict[str, Any]] = {}
    for section_match in section_re.finditer(text):
        section = section_match.group(1)
        heading = re.search(r"<h2>(.*?)</h2>", section, re.S)
        brief_match = re.search(r"<p class='brief'>(.*?)</p>", section, re.S)
        if not heading or not brief_match:
            continue
        case_id = html.unescape(re.sub(r"\s+", " ", heading.group(1)).strip()).split(" · ", 1)[0]
        brief = html.unescape(re.sub(r"\s+", " ", brief_match.group(1)).strip())
        images: dict[str, str] = {}
        for src in re.findall(r"<img\b[^>]*\bsrc='([^']+)'", section):
            src = html.unescape(src)
            name = Path(src).name
            if name.endswith("-frozen SFT.png"):
                role = "sft"
            elif name.endswith("-RL-step-16.png"):
                role = "oldRL16"
            elif name.endswith("-pilot-step-24.png"):
                role = "oldgroundedRL24"
            else:
                continue
            images[role] = str((path.parent / src).resolve())
        if case_id in result:
            raise ValueError(f"duplicate gallery case: {case_id}")
        result[case_id] = {"brief": brief, "images": images}
    if len(result) != 47:
        raise ValueError(f"expected 47 old gallery sections, found {len(result)}")
    return result


def _load_pro(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "pro-step12-results-gallery-v1":
        raise ValueError("unexpected Pro gallery schema")
    result: dict[str, dict[str, Any]] = {}
    for case in value.get("cases", []):
        case_id = case.get("id")
        if not isinstance(case_id, str) or case_id in result:
            raise ValueError(f"invalid or duplicate Pro case: {case_id!r}")
        rel = case.get("last_valid_png")
        image = (path.parent / rel).resolve() if isinstance(rel, str) else None
        if image is not None and (image.is_symlink() or not image.is_file()):
            raise ValueError(f"missing Pro last valid PNG for {case_id}: {rel}")
        result[case_id] = {
            "brief": case.get("prompt_text"),
            "image": str(image) if image is not None else None,
            "turn": case.get("last_valid_turn"),
        }
    return result


def _image_record(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "image_id": image_id(path),
    }


def _pair(*, pair_id: str, case_id: str, brief: str, left: Path, right: Path,
          left_label: str, right_label: str, family: str, note: str,
          historical_human_preference: str | None = None) -> dict[str, Any]:
    if not brief or len(brief) > 4000:
        raise ValueError(f"invalid brief for {pair_id}")
    value = {
        "pair_id": pair_id,
        "case_id": case_id,
        "group_id": case_id,
        "family": family,
        "selection_note": note,
        "brief": brief,
        "brief_sha256": hashlib.sha256(brief.encode("utf-8")).hexdigest(),
        "left": _image_record(left, left_label),
        "right": _image_record(right, right_label),
    }
    if historical_human_preference is not None:
        value["historical_human_preference"] = historical_human_preference
    return value


def build_pairs(*, old_gallery: Path = OLD_GALLERY / "index.html",
                pro_gallery: Path = PRO_REVIEW / "gallery.json") -> dict[str, Any]:
    old = _parse_old_gallery(old_gallery)
    pro = _load_pro(pro_gallery)
    pairs: list[dict[str, Any]] = []

    for index, case_id in enumerate(SFT_OLD_RL16, 1):
        case = old[case_id]
        pairs.append(_pair(
            pair_id=f"sft-oldrl16-{index:02d}", case_id=case_id,
            brief=case["brief"], left=Path(case["images"]["sft"]),
            right=Path(case["images"]["oldRL16"]), left_label="SFT512 frozen opponent",
            right_label="oldRL16", family="SFT512_vs_oldRL16",
            note="Matched frozen SFT anchor versus historical RL16 gallery image.",
            historical_human_preference=(
                "LEFT" if case_id == "t600-516" else None
            ),
        ))

    for index, case_id in enumerate(SFT_GROUNDED_RL24, 1):
        case = old[case_id]
        pairs.append(_pair(
            pair_id=f"sft-groundedrl24-{index:02d}", case_id=case_id,
            brief=case["brief"], left=Path(case["images"]["sft"]),
            right=Path(case["images"]["oldgroundedRL24"]), left_label="SFT512 frozen opponent",
            right_label="oldgroundedRL24", family="SFT512_vs_oldgroundedRL24",
            note="Matched frozen SFT anchor versus historical grounded RL24 gallery image.",
            historical_human_preference=(
                "LEFT" if case_id == "t600-516" else None
            ),
        ))

    for index, case_id in enumerate(DIRECT_RL16_RL24, 1):
        case = old[case_id]
        pairs.append(_pair(
            pair_id=f"oldrl16-groundedrl24-{index:02d}", case_id=case_id,
            brief=case["brief"], left=Path(case["images"]["oldRL16"]),
            right=Path(case["images"]["oldgroundedRL24"]), left_label="oldRL16",
            right_label="oldgroundedRL24", family="oldRL16_vs_oldgroundedRL24",
            note="Direct historical RL16 versus grounded RL24 comparison; no SFT anchor.",
            historical_human_preference=("LEFT" if case_id == "t600-413" else None),
        ))

    for index, case_id in enumerate(SFT_NEW_PRO12, 1):
        case = old[case_id]
        pro_case = pro.get(case_id)
        if not pro_case or not pro_case.get("image"):
            raise ValueError(f"selected Pro case has no last valid saved PNG: {case_id}")
        if pro_case["brief"] != case["brief"]:
            raise ValueError(f"brief mismatch between old and Pro evidence: {case_id}")
        pairs.append(_pair(
            pair_id=f"sft-newpro12-{index:02d}", case_id=case_id,
            brief=case["brief"], left=Path(case["images"]["sft"]),
            right=Path(pro_case["image"]), left_label="SFT512 frozen opponent",
            right_label="newPro12 (last valid saved partial)", family="SFT512_vs_newPro12",
            note="Frozen SFT versus Pro step12 last valid saved partial; not a completed terminal-action claim.",
        ))

    if len(pairs) != 24:
        raise AssertionError(f"expected 24 pairs, found {len(pairs)}")
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("duplicate pair ID")
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        provenance = HUMAN_LABEL_PROVENANCE.get(pair["pair_id"])
        if provenance is not None:
            pair["human_label_provenance"] = dict(provenance)
        grouped[pair["group_id"]].update((pair["left"]["image_id"], pair["right"]["image_id"]))
    return {
        "schema": "frozen-three-judge-panel-v1",
        "selection_note": (
            "Selected frozen-image diagnostic panel: 12 SFT512/oldRL16, "
            "8 SFT512/oldgroundedRL24, 2 direct oldRL16/oldgroundedRL24, "
            "and 2 SFT512/newPro12 (last valid saved partial). This is not a "
            "random sample and has no invented gold labels."
        ),
        "source_evidence": {
            "old_gallery_html": str(old_gallery.resolve()),
            "old_gallery_html_sha256": sha256(old_gallery),
            "pro_gallery_json": str(pro_gallery.resolve()),
            "pro_gallery_json_sha256": sha256(pro_gallery),
            "credit_assignment_grouping": "case_id/group_id with fixed SFT anchors where available",
            "credit_assignment_group_count": len(grouped),
            "groups_with_four_distinct_images": sum(len(images) >= 4 for images in grouped.values()),
            "complete_four_image_on_policy_rollout_groups": 0,
            "credit_assignment_limitation": (
                "Saved local evidence has one four-image same-brief group and several smaller groups; "
                "it has no complete four-image on-policy rollout groups for GRPO advantage validation. "
                "The saved rows compare different checkpoints under the same brief, so no "
                "GRPO/group-normalized credit assignment is validated. "
                "No images were generated to fill that gap."
            ),
        },
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-gallery", type=Path, default=OLD_GALLERY / "index.html")
    parser.add_argument("--pro-gallery", type=Path, default=PRO_REVIEW / "gallery.json")
    parser.add_argument("--output", type=Path, default=RUN / "local-evidence" / "pairs.json")
    args = parser.parse_args(argv)
    panel = build_pairs(old_gallery=args.old_gallery, pro_gallery=args.pro_gallery)
    args.output.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(panel['pairs'])} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
