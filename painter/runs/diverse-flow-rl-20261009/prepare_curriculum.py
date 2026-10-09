"""Prepare the diverse-flow text-to-paint curriculum.

This controller writes deterministic prompt and manifest metadata only.  It
does not call a model, render an image, download a checkpoint, or make a
quality claim.  The curriculum has ten semantic families per stage: six
training families with four variants each and four heldout families with two
variants each.  Thus the balanced proposal contains 96 training and 32
heldout rows.  Family assignment happens before variants are expanded, so
subject families cannot leak across the split.

The ``materialize`` and ``stage-manifests`` commands are the node handoff.
They require real, hash-verified frozen SFT512 anchor files and preserve the
native painting RL row fields.  ``pilot-manifests`` is retained as a
compatibility alias for materializing the complete 96/32 balanced selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_ID = "diverse-flow-rl-20261009"
SCHEMA = "diverse-flow-text-paint-proposal-v1"
# Reuse the shared native loader's established schema; RUN_ID keeps this
# curriculum's manifests distinct from the older structured-flow run.
NATIVE_SCHEMA = "native-painting-rl-structured-flow-v1"
TRAIN_VARIANTS_PER_FAMILY = 4
HELDOUT_VARIANTS_PER_FAMILY = 2
TRAIN_FAMILIES_PER_STAGE = 6
HELDOUT_FAMILIES_PER_STAGE = 4
TRAIN_PER_STAGE = TRAIN_FAMILIES_PER_STAGE * TRAIN_VARIANTS_PER_FAMILY
HELDOUT_PER_STAGE = HELDOUT_FAMILIES_PER_STAGE * HELDOUT_VARIANTS_PER_FAMILY
TRAIN_COUNT = TRAIN_PER_STAGE * 4
HELDOUT_COUNT = HELDOUT_PER_STAGE * 4
# Kept as a compatibility name for callers that only need the maximum
# family expansion.  Heldout families use HELDOUT_VARIANTS_PER_FAMILY.
VARIANTS_PER_FAMILY = TRAIN_VARIANTS_PER_FAMILY
REPLAY_FRACTION = 0.25
MAX_TURNS = 6
ANCHOR_POLICY = "frozen-sft512"
ANCHOR_MODEL_ALIAS = "sft-step512"
ANCHOR_PROVENANCE_FILE = "anchor-provenance.json"
# This is the established provenance wire identity emitted by the native
# anchor generator and consumed by the curriculum admission loader. Keep it
# separate from the run's native manifest schema below.
ANCHOR_PROVENANCE_SCHEMA = "structured-flow-anchor-provenance-v1"
PILOT_TRAIN_COUNT = TRAIN_COUNT
PILOT_HELDOUT_COUNT = HELDOUT_COUNT
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


# Families are grouped by stage and ordered deterministically.  The first six
# in each stage are train and the last four are heldout.  Every subject label
# is unique at the family level, including the 16 heldout labels, so the
# heldout set covers exactly 16 distinct subject families.
BASE_FAMILIES: tuple[dict[str, Any], ...] = (
    {"id": "single-apple", "stage": 1, "kind": "single", "subject": "ripe apple", "materials": ("warm red skin", "golden fruit", "painted apple")},
    {"id": "single-vase", "stage": 1, "kind": "single", "subject": "small vase", "materials": ("aged brass", "painted tin", "warm copper")},
    {"id": "single-orb", "stage": 1, "kind": "single", "subject": "smooth orb", "materials": ("chalky ceramic", "pearly glass", "rose stone")},
    {"id": "single-lantern", "stage": 1, "kind": "single", "subject": "small lantern", "materials": ("cream ceramic", "blue porcelain", "painted earthenware")},
    {"id": "single-pear", "stage": 1, "kind": "single", "subject": "ripe pear", "materials": ("warm green skin", "golden fruit", "painted pear")},
    {"id": "single-shell", "stage": 1, "kind": "single", "subject": "spiral seashell", "materials": ("chalky shell", "pearly shell", "rose shell")},
    {"id": "single-hourglass", "stage": 1, "kind": "single", "subject": "glass hourglass", "materials": ("clear glass", "smoky glass", "amber glass")},
    {"id": "single-bell", "stage": 1, "kind": "single", "subject": "hand bell", "materials": ("polished bronze", "aged brass", "painted metal")},
    {"id": "single-pumpkin", "stage": 1, "kind": "single", "subject": "small pumpkin", "materials": ("orange rind", "painted gourd", "muted ochre rind")},
    {"id": "single-feather", "stage": 1, "kind": "single", "subject": "blue feather", "materials": ("soft down", "painted quill", "iridescent barb")},

    {"id": "connected-cup-saucer", "stage": 2, "kind": "connected", "subject": "a cup resting on a saucer", "materials": ("ceramic cup and porcelain saucer", "painted stoneware cup and ivory porcelain saucer", "blue ceramic cup and cream porcelain saucer")},
    {"id": "connected-bird-branch", "stage": 2, "kind": "connected", "subject": "a bird perched on a branch with feet, body, and tail", "materials": ("painted wood", "warm feathers", "soft fabric")},
    {"id": "connected-boat-sail", "stage": 2, "kind": "connected", "subject": "a boat with hull, mast, and sail", "materials": ("painted timber", "white canvas", "weathered fiberglass")},
    {"id": "connected-teapot", "stage": 2, "kind": "connected", "subject": "a teapot with body, lid, handle, and spout", "materials": ("honey ceramic", "painted lacquer", "dark copper")},
    {"id": "connected-umbrella", "stage": 2, "kind": "connected", "subject": "an umbrella with canopy, ribs, and handle", "materials": ("striped cloth", "bright nylon", "waxed canvas")},
    {"id": "connected-watering-can", "stage": 2, "kind": "connected", "subject": "a watering can with body, handle, and spout", "materials": ("painted enamel", "galvanized metal", "warm copper")},
    {"id": "connected-bicycle", "stage": 2, "kind": "connected", "subject": "a bicycle with frame, wheels, and handlebars", "materials": ("painted steel", "brushed aluminum", "dark carbon")},
    {"id": "connected-backpack", "stage": 2, "kind": "connected", "subject": "a backpack with body, flap, and shoulder straps", "materials": ("woven canvas", "red nylon", "soft leather")},
    {"id": "connected-windmill", "stage": 2, "kind": "connected", "subject": "a windmill with tower, cap, and four blades", "materials": ("white timber", "painted boards", "weathered stone")},
    {"id": "connected-kettle", "stage": 2, "kind": "connected", "subject": "a kettle with body, lid, handle, and spout", "materials": ("enameled steel", "hammered copper", "matte ceramic")},

    {"id": "pair-apple-pear", "stage": 3, "kind": "pair", "subject": "an apple beside a pear", "relation": "Keep a small clear gap between the apple and pear while showing both resting on one surface; place their shared cast shadow beneath the pair.", "materials": ("real red apple and green pear", "warm red apple and golden pear", "ripe apple and amber pear")},
    {"id": "pair-book-mug", "stage": 3, "kind": "pair", "subject": "a closed book beside a mug", "relation": "Keep a small clear gap between the book and mug while showing both resting on one surface; place their shared cast shadow beneath the pair.", "materials": ("linen-bound book and ceramic mug", "leather book and glazed mug", "paperback book and painted mug")},
    {"id": "pair-rock-flower", "stage": 3, "kind": "pair", "subject": "a small rock beside one flower", "relation": "Keep the flower separate from the rock with a visible gap; show both resting on soil with distinct but softly overlapping shadows.", "materials": ("granite and fresh petals", "painted clay and silk", "slate and paper")},
    {"id": "pair-sun-cloud", "stage": 3, "kind": "pair", "subject": "a low sun beneath one cloud", "relation": "Keep the sun and cloud separated in the sky with a clear gap; use a soft atmospheric transition rather than a grounded cast shadow.", "materials": ("golden light and blue-gray cloud", "peach light and slate cloud", "amber light and violet-gray cloud")},
    {"id": "pair-lamp-table", "stage": 3, "kind": "pair", "subject": "a table lamp on a small table", "relation": "Place the lamp on top of the table; preserve contact at its base and show the tabletop supporting it, with the cast shadow beneath the table.", "materials": ("wood and linen", "brass and oak", "painted metal and glass")},
    {"id": "pair-vase-branch", "stage": 3, "kind": "pair", "subject": "a vase holding one branch", "relation": "Keep the branch entering the vase opening so the vase contains its base; place the cast shadow beneath the vase and give the branch no independent grounded shadow.", "materials": ("matte clay", "clear glass", "painted porcelain")},
    {"id": "pair-boot-lace", "stage": 3, "kind": "pair", "subject": "a boot with one loose lace beside it", "relation": "Keep the loose lace clearly outside the boot while preserving its contact with the boot opening; show a single grounded shadow beneath the pair.", "materials": ("worn leather and cotton", "painted leather and rope", "soft suede and linen")},
    {"id": "pair-basket-bread", "stage": 3, "kind": "pair", "subject": "a basket beside a round loaf of bread", "relation": "Keep the loaf outside the basket with a visible gap; show both resting on the same cloth with distinct shadows.", "materials": ("woven reed and baked crust", "painted wicker and warm bread", "dark basket and pale loaf")},
    {"id": "pair-sunhat-towel", "stage": 3, "kind": "pair", "subject": "a sunhat on a folded beach towel", "relation": "Place the sunhat on top of the folded towel; preserve their contact and show the towel supporting the hat with one broad shadow.", "materials": ("straw and cotton", "painted fiber and linen", "woven paper and terry cloth")},
    {"id": "pair-jar-spoon", "stage": 3, "kind": "pair", "subject": "a jar beside a wooden spoon", "relation": "Keep the spoon beside the jar with a small clear gap; show both on one counter with separate soft shadows.", "materials": ("clear glass and maple", "painted ceramic and beech", "amber glass and walnut")},

    {"id": "scene-window-plant", "stage": 4, "kind": "scene", "subject": "a sunlit window with one potted plant", "materials": ("rough timber", "painted boards", "warm plaster and wood")},
    {"id": "scene-market-stall", "stage": 4, "kind": "scene", "subject": "a small market stall with one awning", "materials": ("striped canvas", "painted wood", "corrugated metal")},
    {"id": "scene-lighthouse-cliff", "stage": 4, "kind": "scene", "subject": "a lighthouse on a low cliff", "materials": ("white plaster", "weathered stone", "painted concrete")},
    {"id": "scene-tram-street", "stage": 4, "kind": "scene", "subject": "a tram passing one row of street trees", "materials": ("painted steel", "brushed metal", "aged red paint")},
    {"id": "scene-orchard-fence", "stage": 4, "kind": "scene", "subject": "a low orchard fence before three fruit trees", "materials": ("whitewashed wood", "warm cedar", "weathered boards")},
    {"id": "scene-bridge-willows", "stage": 4, "kind": "scene", "subject": "a small bridge beneath two willow trees", "materials": ("weathered timber", "painted boards", "dark rope and wood")},
    {"id": "scene-hill-cabin", "stage": 4, "kind": "scene", "subject": "a tiny cabin on a low hill", "materials": ("warm plaster", "terracotta tile", "pale stone")},
    {"id": "scene-pond-reeds", "stage": 4, "kind": "scene", "subject": "a quiet pond with reeds and one stepping stone", "materials": ("muddy bank", "smooth stone", "wooden edge")},
    {"id": "scene-dock-buoy", "stage": 4, "kind": "scene", "subject": "a short dock with one floating buoy", "materials": ("weathered timber", "painted boards", "dark rope and wood")},
    {"id": "scene-garden-bench", "stage": 4, "kind": "scene", "subject": "a garden path ending at one wooden bench", "materials": ("mossy stone", "painted wood", "warm brick")},
)

PALETTES: tuple[tuple[str, str], ...] = (
    ("warm ochre", "ochre, umber, cream, and a small vermilion accent"),
    ("muted harbor", "slate blue, muted teal, pale sand, and warm cream"),
    ("spring garden", "leaf green, coral, warm ivory, and a restrained blue-gray"),
    ("blue hour", "blue-gray, indigo, muted lavender, and pale apricot"),
    ("golden studio", "golden yellow, cocoa, pale peach, and charcoal"),
)
ORIENTATIONS: tuple[tuple[str, str], ...] = (
    ("quiet square", "a square view with comfortable margins"),
    ("wide landscape", "a wide landscape view with comfortable margins"),
    ("tall portrait", "a tall portrait view with comfortable margins"),
)
FLOW_GUIDANCE: dict[int, str] = {
    1: "Make one complete first pass and inspect the rendered canvas. Revise only if a concrete revision improves the brief; otherwise finish.",
    2: "Paint the connected construction and inspect attachment and overlap. Revise only if a concrete revision improves the brief; otherwise finish.",
    3: "Paint both objects and inspect their relation and silhouettes. Revise only if a concrete revision improves the brief; otherwise finish.",
    4: "Paint the simplified scene and inspect its three depth bands. Revise only if a concrete revision improves the brief; otherwise finish.",
}
FLOW_HINT = (
    "Plan before writing code: identify the subject hierarchy, placement, and largest shapes first; then make explicit canvas, palette, and brush choices. "
    "Revise by preserving correct structure and changing one control at a time. After observing the actual canvas, either return one meaningful complete painting revision or respond FINISHED when it satisfies the brief. "
    "For a paint action, return only executable painting code in the assistant turn."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_number(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:16], 16)


def _safe_id(value: str) -> None:
    if not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"unsafe case identifier: {value!r}")


def _family_table() -> dict[str, dict[str, Any]]:
    table = {str(row["id"]): dict(row) for row in BASE_FAMILIES}
    if len(table) != len(BASE_FAMILIES):
        raise AssertionError("family IDs must be unique")
    for row in table.values():
        if int(row["stage"]) not in range(1, 5) or row.get("kind") not in {"single", "connected", "pair", "scene"}:
            raise AssertionError(f"invalid family metadata: {row}")
    return table


def _split_families() -> tuple[dict[int, list[str]], dict[str, str]]:
    by_stage: dict[int, list[str]] = {stage: [] for stage in range(1, 5)}
    for family in BASE_FAMILIES:
        by_stage[int(family["stage"])].append(str(family["id"]))
    assignment: dict[str, str] = {}
    for stage, families in by_stage.items():
        expected = TRAIN_FAMILIES_PER_STAGE + HELDOUT_FAMILIES_PER_STAGE
        if len(families) != expected:
            raise AssertionError(f"stage {stage} must contain {expected} families")
        for family in families[:TRAIN_FAMILIES_PER_STAGE]:
            assignment[family] = "train"
        for family in families[TRAIN_FAMILIES_PER_STAGE:]:
            assignment[family] = "heldout"
    return by_stage, assignment


def _brief(family: Mapping[str, Any], palette: tuple[str, str], orientation: tuple[str, str], material: str) -> str:
    stage = int(family["stage"])
    palette_name, palette_text = palette
    orientation_name, orientation_text = orientation
    subject = str(family["subject"])
    if stage == 1:
        body = f"Paint exactly one {subject} as the clear subject. Place it centrally with a readable silhouette, a grounded shadow, and one restrained highlight. Use {material}; keep the background simple and empty."
    elif stage == 2 and family["id"] == "connected-cup-saucer":
        body = f"Paint exactly two objects in one supporting construction: {subject}. Show the cup resting on, and visibly contacting, the saucer without fusing the two forms; give the pair a grounded shadow and one restrained highlight. Use {material}; keep the background simple and empty."
    elif stage == 2:
        body = f"Paint exactly one connected construction: {subject}. Make every attachment point and overlap readable, with a grounded shadow and one restrained highlight. Use {material}; keep the background simple and empty."
    elif stage == 3:
        body = f"Paint exactly two objects: {subject}. {family['relation']} Preserve the stated contact, containment, or separation instead of inventing another relationship. Use {material}; keep the background quiet with no extra objects."
    else:
        body = f"Paint a simplified scene containing only {subject}. Establish one clear foreground, one middle distance, and a calm background shape; preserve the subject relationships without tiny clutter. Use {material}."
    return (
        f"{body} The composition uses {orientation_text}; the intended view is {orientation_name}. "
        f"Use the {palette_name} palette ({palette_text}) with clear value separation. Keep comfortable canvas margins so the main forms stay inside the central 80 percent. "
        "Use warm, readable painterly p5.brush marks with a few deliberate shapes and edges; do not add text, borders, or unrelated subjects."
    )


def generate_rows(seed: int = 20261009) -> list[dict[str, Any]]:
    """Generate the deterministic 96/32 proposal rows without writing files."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _family_table()
    _, assignment = _split_families()
    rows: list[dict[str, Any]] = []
    for family in BASE_FAMILIES:
        family_id = str(family["id"])
        split = assignment[family_id]
        stage = int(family["stage"])
        variants = TRAIN_VARIANTS_PER_FAMILY if split == "train" else HELDOUT_VARIANTS_PER_FAMILY
        for variant in range(variants):
            palette = PALETTES[(stage + variant + (0 if split == "train" else 2)) % len(PALETTES)]
            orientation = ORIENTATIONS[(variant + stage - 1) % len(ORIENTATIONS)]
            material = tuple(family["materials"])[variant % len(tuple(family["materials"]))]
            case_id = f"df-s{stage}-{split[:2]}-{family_id}-{variant:02d}"
            _safe_id(case_id)
            prompt = _brief(family, palette, orientation, material)
            rows.append({
                "id": case_id,
                "task_id": case_id,
                "source_group": case_id,
                "family": family_id,
                "base_family_id": family_id,
                "subject": str(family["subject"]),
                "family_kind": str(family["kind"]),
                "stage": stage,
                "split": split,
                "variant": variant,
                "palette": palette[0],
                "orientation": orientation[0],
                "material": material,
                "mode": "fresh",
                "reference_kind": "text",
                "reference": f"prompts/{split}/stage-{stage}/{case_id}.txt",
                "reference_sha256": sha256_bytes((prompt + "\n").encode("utf-8")),
                "anchor_policy": ANCHOR_POLICY,
                "anchor_model_alias": ANCHOR_MODEL_ALIAS,
                "anchor_required": True,
                "anchor_status": "pending",
                "admission_status": "awaiting_frozen_sft512_anchor",
                "flow_guidance": FLOW_GUIDANCE[stage],
                "max_turns": MAX_TURNS,
                "order": _stable_number(seed, stage, family_id, variant),
                "prompt_signature": sha256_bytes("|".join((family_id, str(stage), split, palette[0], orientation[0], material, str(variant))).encode()),
                "prompt_text": prompt,
            })
    rows.sort(key=lambda row: (int(row["stage"]), 0 if row["split"] == "train" else 1, str(row["base_family_id"]), int(row["variant"])))
    if len(rows) != TRAIN_COUNT + HELDOUT_COUNT or len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("curriculum must contain 128 unique rows")
    if sum(row["split"] == "train" for row in rows) != TRAIN_COUNT or sum(row["split"] == "heldout" for row in rows) != HELDOUT_COUNT:
        raise AssertionError("curriculum split counts are unbalanced")
    if len({row["prompt_signature"] for row in rows}) != len(rows):
        raise AssertionError("prompt variants must be unique")
    return rows


def _write_prompt_files(rows: Sequence[Mapping[str, Any]], root: Path) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    for row in rows:
        relative = Path(str(row["reference"]))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(row["prompt_text"]).strip() + "\n", encoding="utf-8")
        actual = sha256_file(target)
        if actual != row["reference_sha256"]:
            raise AssertionError(f"prompt hash mismatch after write: {row['id']}")
        written.append({"path": str(relative), "sha256": actual, "bytes": target.stat().st_size})
    return written


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "prompt_text"}


def _count_receipt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"total": len(rows), "by_stage": {}, "by_split": {}, "by_base_family": {}}
    fields = (("stage", "by_stage"), ("split", "by_split"), ("base_family_id", "by_base_family"))
    for row in rows:
        for field, bucket_name in fields:
            key = str(row[field])
            bucket = result[bucket_name]
            bucket[key] = bucket.get(key, 0) + 1
    return result


def build_pilot_plan(train_rows: Sequence[Mapping[str, Any]], heldout_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select all rows in fixed stage/family order for the 96/32 pilot."""

    def select(rows: Sequence[Mapping[str, Any]], split: str, variants: int) -> list[str]:
        selected: list[str] = []
        for stage in range(1, 5):
            stage_rows = [row for row in rows if int(row["stage"]) == stage and row["split"] == split]
            families = sorted({str(row["base_family_id"]) for row in stage_rows})
            expected_families = TRAIN_FAMILIES_PER_STAGE if split == "train" else HELDOUT_FAMILIES_PER_STAGE
            if len(families) != expected_families:
                raise ValueError(f"{split} needs {expected_families} families at stage {stage}, found {len(families)}")
            for family in families:
                family_rows = sorted((row for row in stage_rows if str(row["base_family_id"]) == family), key=lambda row: int(row["variant"]))
                if len(family_rows) != variants:
                    raise ValueError(f"family {family} must have {variants} variants")
                selected.extend(str(row["id"]) for row in family_rows)
        return selected

    train_ids = select(train_rows, "train", TRAIN_VARIANTS_PER_FAMILY)
    heldout_ids = select(heldout_rows, "heldout", HELDOUT_VARIANTS_PER_FAMILY)
    return {
        "schema": "diverse-flow-pilot-plan-v1",
        "run_id": RUN_ID,
        "status": "pending_anchors",
        "rationale": "Materialize the complete balanced 96-train/32-heldout selection across four stages and 40 object-level families.",
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "counts": {"train": len(train_ids), "heldout": len(heldout_ids), "total": len(train_ids) + len(heldout_ids)},
        "stage_counts": {"train": {str(stage): TRAIN_PER_STAGE for stage in range(1, 5)}, "heldout": {str(stage): HELDOUT_PER_STAGE for stage in range(1, 5)}},
        "family_variant_counts": {"train": TRAIN_VARIANTS_PER_FAMILY, "heldout": HELDOUT_VARIANTS_PER_FAMILY},
        "anchor_requirement": "Rows remain pending until actual frozen SFT512 PNGs and their hashes are materialized.",
        "flow_guidance_note": "flow_guidance is a runtime-only optional hint; it is not appended to immutable reference briefs.",
    }


def _file_receipts(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(paths, key=lambda item: str(item.relative_to(root)))]


def build_proposal(output: Path, *, seed: int = 20261009) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(seed)
    prompt_receipts = _write_prompt_files(rows, output)
    train = [_compact_row(row) for row in rows if row["split"] == "train"]
    heldout = [_compact_row(row) for row in rows if row["split"] == "heldout"]
    _, family_assignment = _split_families()
    common = {
        "run_id": RUN_ID,
        "schema": SCHEMA,
        "status": "proposal_pending_anchors",
        "seed": seed,
        "anchor_policy": ANCHOR_POLICY,
        "anchor_model_alias": ANCHOR_MODEL_ALIAS,
        "anchor_requirement": "Every row needs a real same-source_group frozen SFT512 PNG and opponent_sha256 before native_painting_rl admission.",
        "native_rl_admission": "pending; proposal rows intentionally have no opponent path or hash",
    }
    train_doc = {**common, "split": "train", "counts": _count_receipt(train), "rows": train}
    heldout_doc = {**common, "split": "heldout", "counts": _count_receipt(heldout), "rows": heldout}
    _json_write(output / "train-proposal.json", train_doc)
    _json_write(output / "heldout-proposal.json", {**heldout_doc, "cases": heldout})
    sampling = build_sampling_plan(train)
    _json_write(output / "sampling-plan.json", sampling)
    pilot = build_pilot_plan(train, heldout)
    _json_write(output / "pilot-plan.json", pilot)
    manifest_paths = [output / name for name in ("train-proposal.json", "heldout-proposal.json", "sampling-plan.json", "pilot-plan.json")]
    receipt = {
        "run_id": RUN_ID,
        "schema": SCHEMA,
        "status": "prepared_unrendered",
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {"external_files": [], "seed": seed},
        "counts": {"all": _count_receipt(rows), "train": _count_receipt(train), "heldout": _count_receipt(heldout)},
        "family_split": family_assignment,
        "split_rule": "six train and four heldout families are assigned per stage before variants are expanded",
        "pilot": pilot,
        "prompt_files": prompt_receipts,
        "outputs": _file_receipts(output, [*manifest_paths, *(output / "prompts").rglob("*.txt")]),
        "limitations": [
            "No model inference, rendering, SFT, RL, visual judgment, GPU utilization, or cost was run.",
            "Prompt hashes identify text files only; they are not rendered image hashes.",
            "Rows are not runnable native RL cases until frozen SFT512 anchors are materialized.",
        ],
    }
    _json_write(output / "preparation-receipt.json", receipt)
    return receipt


def build_sampling_plan(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stage = {stage: sorted((dict(row) for row in train_rows if int(row["stage"]) == stage), key=lambda row: str(row["id"])) for stage in range(1, 5)}
    if any(len(by_stage[stage]) != TRAIN_PER_STAGE for stage in by_stage):
        raise ValueError(f"expected {TRAIN_PER_STAGE} training rows per stage")
    stages: list[dict[str, Any]] = []
    for stage in range(1, 5):
        replay: list[str] = []
        for prior_stage in range(1, stage):
            prior = by_stage[prior_stage]
            count = math.floor(len(prior) * REPLAY_FRACTION)
            replay.extend(str(row["id"]) for row in sorted(prior, key=lambda row: (_stable_number("replay", stage, row["id"]), str(row["id"])))[0:count])
        current = [str(row["id"]) for row in by_stage[stage]]
        stages.append({"stage": stage, "current_count": len(current), "replay_count": len(replay), "total_count": len(current) + len(replay), "replay_fraction_of_prior": REPLAY_FRACTION, "current_ids": current, "replay_ids": replay, "manifest": f"stage-{stage}-train-manifest.json"})
    return {"schema": "diverse-flow-staged-sampling-v1", "run_id": RUN_ID, "replay_fraction": REPLAY_FRACTION, "stages": stages, "note": "Replay is selected deterministically from earlier stage rows; each stage manifest keeps unique task IDs for native_painting_rl."}


def write_casebrief(row: dict[str, Any], proposal_root: Path, output_root: Path, *, enabled: bool) -> None:
    original = Path(str(row["reference"]))
    if not original.is_absolute():
        original = proposal_root / original
    original = original.resolve()
    if original.is_symlink() or not original.is_file():
        raise FileNotFoundError(f"prompt is missing: {original}")
    original_hash = sha256_file(original)
    if original_hash != row.get("reference_sha256"):
        raise ValueError(f"prompt hash mismatch for {row['id']}")
    hint = FLOW_HINT if enabled else "Structured-flow hint is disabled for this run."
    target = output_root / "casebriefs" / str(row["split"]) / f"{row['id']}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"[SYSTEM]\n{hint}\n\n[USER]\n{original.read_text(encoding='utf-8').strip()}\n", encoding="utf-8")
    row.update({"source_prompt": str(original), "source_prompt_sha256": original_hash, "casebrief": str(target.resolve()), "casebrief_sha256": sha256_file(target), "flow_policy_enabled": enabled, "system_hint_sha256": sha256_bytes((hint + "\n").encode())})


def _validate_checkpoint_hash(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("frozen_sft512_checkpoint_sha256 must be a lowercase 64-character SHA-256")
    return value


def _resolve_prompt(proposal: Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row["reference"]))
    if not path.is_absolute():
        path = proposal.parent / path
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"prompt is missing or symlinked: {path}")
    path = path.absolute()
    if sha256_file(path) != row.get("reference_sha256"):
        raise ValueError(f"prompt hash mismatch for {row.get('id')}")
    return path


def _load_anchor_provenance(anchor_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = anchor_root / ANCHOR_PROVENANCE_FILE
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"anchor provenance is missing or symlinked: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid anchor provenance JSON: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != ANCHOR_PROVENANCE_SCHEMA:
        raise ValueError("anchor provenance schema mismatch")
    anchors = raw.get("anchors")
    if not isinstance(anchors, dict) or not anchors:
        raise ValueError("anchor provenance must contain a non-empty anchors mapping")
    result: dict[str, dict[str, Any]] = {}
    for case_id, evidence in anchors.items():
        if not isinstance(case_id, str) or not _SAFE_ID_RE.fullmatch(case_id) or not isinstance(evidence, dict):
            raise ValueError(f"invalid anchor provenance entry: {case_id!r}")
        for key in ("checkpoint_sha256", "reference_sha256", "source_group", "png_sha256"):
            if not isinstance(evidence.get(key), str) or not evidence[key]:
                raise ValueError(f"anchor provenance {case_id} is missing {key}")
        _validate_checkpoint_hash(str(evidence["checkpoint_sha256"]))
        for key in ("reference_sha256", "png_sha256"):
            if not _SHA256_RE.fullmatch(str(evidence[key])):
                raise ValueError(f"anchor provenance {case_id} has invalid {key}")
        if evidence.get("valid") is not True:
            raise ValueError(f"anchor provenance {case_id} is not marked valid")
        receipt = evidence.get("generation_receipt") or evidence.get("trace") or evidence.get("trace_id")
        if not ((isinstance(receipt, str) and receipt.strip()) or (isinstance(receipt, dict) and any(isinstance(value, str) and value.strip() for value in receipt.values()))):
            raise ValueError(f"anchor provenance {case_id} lacks a generation receipt or trace")
        result[case_id] = dict(evidence)
    return result, sha256_file(path)


def _admitted_row(row: Mapping[str, Any], proposal: Path, anchors: Path, checkpoint: str, provenance: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    case_id = str(row["id"])
    prompt = _resolve_prompt(proposal, row)
    anchor = anchors / f"{case_id}.png"
    if anchor.is_symlink() or not anchor.is_file():
        raise FileNotFoundError(f"frozen SFT512 anchor is missing or symlinked: {anchor}")
    png_hash = sha256_file(anchor)
    evidence = provenance.get(case_id)
    if evidence is None:
        raise ValueError(f"anchor provenance is missing case: {case_id}")
    if evidence["checkpoint_sha256"] != checkpoint:
        raise ValueError(f"anchor checkpoint provenance mismatch: {case_id}")
    if evidence["reference_sha256"] != str(row["reference_sha256"]):
        raise ValueError(f"anchor reference provenance mismatch: {case_id}")
    if evidence["source_group"] != str(row["source_group"]):
        raise ValueError(f"anchor source_group provenance mismatch: {case_id}")
    if evidence["png_sha256"] != png_hash:
        raise ValueError(f"anchor PNG provenance mismatch: {case_id}")
    admitted = _compact_row(row)
    admitted.update({"reference": str(prompt), "opponent": str(anchor.absolute()), "opponent_sha256": png_hash, "opponent_source_group": str(row["source_group"]), "anchor_status": "verified_file", "admission_status": "ready_for_native_painting_rl", "frozen_sft512_checkpoint_sha256": checkpoint, "anchor_provenance_file": str((anchors / ANCHOR_PROVENANCE_FILE).absolute())})
    return admitted


def materialize_manifest(proposal: Path, output: Path, anchor_root: Path, *, split: str = "train", frozen_sft512_checkpoint_sha256: str, selected_ids: Sequence[str] | None = None) -> dict[str, Any]:
    if split not in {"train", "heldout"}:
        raise ValueError("split must be train or heldout")
    checkpoint = _validate_checkpoint_hash(frozen_sft512_checkpoint_sha256)
    proposal, output, anchor_root = proposal.expanduser().resolve(), output.expanduser().resolve(), anchor_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    provenance, provenance_hash = _load_anchor_provenance(anchor_root)
    raw = json.loads(proposal.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA or raw.get("split") != split:
        raise ValueError("proposal schema or split mismatch")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("proposal has no rows")
    if selected_ids is None:
        selected_rows = rows
    else:
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected_ids must be unique")
        by_id = {str(row["id"]): row for row in rows}
        missing = set(selected_ids) - set(by_id)
        if missing:
            raise ValueError(f"selected IDs are absent from proposal: {sorted(missing)[:3]}")
        selected_rows = [by_id[ident] for ident in selected_ids]
    admitted = [_admitted_row(row, proposal, anchor_root, checkpoint, provenance) for row in selected_rows]
    if any(row.get("split") != split for row in admitted):
        raise AssertionError("manifest admission included the wrong split")
    manifest: dict[str, Any] = {"schema": NATIVE_SCHEMA, "run_id": RUN_ID, "status": "admitted_with_verified_frozen_sft512_anchors", "split": split, "native_rl_compatible": split == "train", "frozen_opponent": {"policy": ANCHOR_POLICY, "model_alias": ANCHOR_MODEL_ALIAS, "checkpoint_sha256": checkpoint, "same_source_group_required": True, "provenance_file": ANCHOR_PROVENANCE_FILE, "provenance_sha256": provenance_hash}, "proposal_sha256": sha256_file(proposal), "rows": admitted}
    if split == "heldout":
        manifest["cases"] = admitted
    _json_write(output, manifest)
    return {"output": str(output), "output_sha256": sha256_file(output), "split": split, "rows": len(admitted), "anchor_count": len(admitted), "native_rl_compatible": split == "train", "proposal_sha256": manifest["proposal_sha256"], "anchor_provenance_sha256": provenance_hash}


def materialize_stage_manifests(proposal: Path, output_dir: Path, anchor_root: Path, *, frozen_sft512_checkpoint_sha256: str) -> dict[str, Any]:
    checkpoint = _validate_checkpoint_hash(frozen_sft512_checkpoint_sha256)
    proposal, output_dir, anchor_root = proposal.expanduser().resolve(), output_dir.expanduser().resolve(), anchor_root.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    provenance, provenance_hash = _load_anchor_provenance(anchor_root)
    raw = json.loads(proposal.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA or raw.get("split") != "train":
        raise ValueError("stage manifests require a train proposal")
    rows = [dict(row) for row in raw.get("rows", [])]
    plan = build_sampling_plan(rows)
    by_id = {str(row["id"]): row for row in rows}
    prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for stage_info in plan["stages"]:
        admitted: list[dict[str, Any]] = []
        for ident in [*stage_info["current_ids"], *stage_info["replay_ids"]]:
            if ident not in by_id:
                raise AssertionError(f"sampling plan references unknown row: {ident}")
            row = _admitted_row(by_id[ident], proposal, anchor_root, checkpoint, provenance)
            row.update({"sampling_stage": int(stage_info["stage"]), "sampling_role": "current" if int(row["stage"]) == int(stage_info["stage"]) else "replay", "order": len(admitted)})
            admitted.append(row)
        prepared.append((stage_info, admitted))
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for stage_info, admitted in prepared:
        stage = int(stage_info["stage"])
        path = output_dir / str(stage_info["manifest"])
        _json_write(path, {"schema": NATIVE_SCHEMA, "run_id": RUN_ID, "status": "admitted_with_verified_frozen_sft512_anchors", "split": "train", "native_rl_compatible": True, "sampling_stage": stage, "replay_fraction_of_prior": REPLAY_FRACTION, "frozen_opponent": {"policy": ANCHOR_POLICY, "model_alias": ANCHOR_MODEL_ALIAS, "checkpoint_sha256": checkpoint, "same_source_group_required": True, "provenance_file": ANCHOR_PROVENANCE_FILE, "provenance_sha256": provenance_hash}, "proposal_sha256": sha256_file(proposal), "rows": admitted})
        records.append({"stage": stage, "path": path.name, "sha256": sha256_file(path), "current_count": stage_info["current_count"], "replay_count": stage_info["replay_count"], "total_count": len(admitted)})
    receipt = {"schema": "diverse-flow-stage-manifest-receipt-v1", "run_id": RUN_ID, "status": "prepared_with_verified_anchors", "proposal_sha256": sha256_file(proposal), "frozen_sft512_checkpoint_sha256": checkpoint, "anchor_provenance_sha256": provenance_hash, "manifests": records, "counts": {"stages": len(records), "total_manifest_rows": sum(row["total_count"] for row in records)}}
    _json_write(output_dir / "stage-manifest-receipt.json", receipt)
    return receipt


def materialize_pilot_manifests(train_proposal: Path, heldout_proposal: Path, output_dir: Path, anchor_root: Path, *, frozen_sft512_checkpoint_sha256: str, admitted_train_ids: Sequence[str] | None = None) -> dict[str, Any]:
    checkpoint = _validate_checkpoint_hash(frozen_sft512_checkpoint_sha256)
    train_proposal, heldout_proposal, output_dir, anchor_root = train_proposal.expanduser().resolve(), heldout_proposal.expanduser().resolve(), output_dir.expanduser().resolve(), anchor_root.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    train_doc = json.loads(train_proposal.read_text(encoding="utf-8"))
    heldout_doc = json.loads(heldout_proposal.read_text(encoding="utf-8"))
    if train_doc.get("schema") != SCHEMA or heldout_doc.get("schema") != SCHEMA:
        raise ValueError("pilot requires diverse-flow proposals")
    plan = build_pilot_plan(train_doc.get("rows", []), heldout_doc.get("rows", []))
    selected_train = list(plan["train_ids"] if admitted_train_ids is None else [str(value) for value in admitted_train_ids])
    if len(selected_train) != len(set(selected_train)) or not set(selected_train) <= set(plan["train_ids"]):
        raise ValueError("admitted train IDs must be a unique subset of the balanced train selection")
    train_result = materialize_manifest(train_proposal, output_dir / "pilot-train-manifest.json", anchor_root, split="train", frozen_sft512_checkpoint_sha256=checkpoint, selected_ids=selected_train)
    heldout_result = materialize_manifest(heldout_proposal, output_dir / "pilot-heldout-manifest.json", anchor_root, split="heldout", frozen_sft512_checkpoint_sha256=checkpoint, selected_ids=plan["heldout_ids"])
    receipt = {"schema": "diverse-flow-pilot-manifest-receipt-v1", "run_id": RUN_ID, "status": "prepared_with_verified_anchors", "pilot_plan": plan, "registered_train_ids": list(plan["train_ids"]), "admitted_train_ids": selected_train, "frozen_sft512_checkpoint_sha256": checkpoint, "train": train_result, "heldout": heldout_result, "native_rl_train_manifest": "pilot-train-manifest.json", "heldout_manifest_is_native_rl_compatible": False}
    _json_write(output_dir / "pilot-manifest-receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write deterministic prompts and pending proposals")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=20261009)
    materialize = commands.add_parser("materialize", help="admit one split after anchor verification")
    materialize.add_argument("--proposal", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--anchor-root", type=Path, required=True)
    materialize.add_argument("--split", choices=("train", "heldout"), default="train")
    materialize.add_argument("--frozen-sft512-checkpoint-sha256", required=True)
    stages = commands.add_parser("stage-manifests", help="write current-plus-replay train manifests")
    stages.add_argument("--proposal", type=Path, required=True)
    stages.add_argument("--output-dir", type=Path, required=True)
    stages.add_argument("--anchor-root", type=Path, required=True)
    stages.add_argument("--frozen-sft512-checkpoint-sha256", required=True)
    pilot = commands.add_parser("pilot-manifests", help="write the complete balanced 96-train/32-heldout manifests")
    pilot.add_argument("--train-proposal", type=Path, required=True)
    pilot.add_argument("--heldout-proposal", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--anchor-root", type=Path, required=True)
    pilot.add_argument("--frozen-sft512-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_proposal(args.output, seed=args.seed)
    elif args.command == "materialize":
        result = materialize_manifest(args.proposal, args.output, args.anchor_root, split=args.split, frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256)
    elif args.command == "stage-manifests":
        result = materialize_stage_manifests(args.proposal, args.output_dir, args.anchor_root, frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256)
    else:
        result = materialize_pilot_manifests(args.train_proposal, args.heldout_proposal, args.output_dir, args.anchor_root, frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
