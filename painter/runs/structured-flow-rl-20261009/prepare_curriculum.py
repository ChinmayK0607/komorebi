"""Prepare the structured-flow text-to-paint curriculum.

This module is a controller-side preparation step.  It writes deterministic
briefs and metadata only; it never calls a model, starts a renderer, or
claims that a reference or frozen SFT512 anchor has been rendered.

The proposal contains 300 briefs: 240 training and 60 heldout, with 60/15
per stage across four progressively harder stages.  Every base template
family is assigned to exactly one split before its 15 variants are generated.
The ``materialize`` and ``stage-manifests`` commands are the node handoff:
they require real anchor PNGs and an immutable SFT512 checkpoint hash before
writing rows that :mod:`painter.native_painting_rl` can admit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_ID = "structured-flow-rl-20261009"
SCHEMA = "structured-flow-text-paint-proposal-v1"
NATIVE_SCHEMA = "native-painting-rl-structured-flow-v1"
VARIANTS_PER_FAMILY = 15
TRAIN_FAMILIES_PER_STAGE = 4
HELDOUT_FAMILIES_PER_STAGE = 1
TRAIN_PER_STAGE = TRAIN_FAMILIES_PER_STAGE * VARIANTS_PER_FAMILY
HELDOUT_PER_STAGE = HELDOUT_FAMILIES_PER_STAGE * VARIANTS_PER_FAMILY
REPLAY_FRACTION = 0.25
MAX_TURNS = 6
ANCHOR_POLICY = "frozen-sft512"
ANCHOR_MODEL_ALIAS = "sft-step512"
ANCHOR_PROVENANCE_FILE = "anchor-provenance.json"
PILOT_TRAIN_COUNT = 32
PILOT_HELDOUT_COUNT = 16
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


# Base families are intentionally semantic templates, not teacher programs.
# The split is made on this table before any palette/orientation/material
# variants are expanded, so no base family can leak across train and heldout.
BASE_FAMILIES: tuple[dict[str, Any], ...] = (
    {"id": "single-orb", "stage": 1, "subject": "smooth orb", "materials": ("matte ceramic", "polished stone", "painted wood")},
    {"id": "single-mug", "stage": 1, "subject": "handled mug", "materials": ("speckled clay", "glazed ceramic", "brushed metal")},
    {"id": "single-apple", "stage": 1, "subject": "apple", "materials": ("waxed fruit", "painted fruit", "natural fruit")},
    {"id": "single-leaf", "stage": 1, "subject": "broad leaf", "materials": ("waxy leaf", "velvety leaf", "crisp dry leaf")},
    {"id": "single-stone", "stage": 1, "subject": "faceted stone", "materials": ("granite", "slate", "river rock")},
    {"id": "connected-teapot", "stage": 2, "subject": "a teapot with its handle and spout", "materials": ("glazed earthenware", "painted enamel", "hammered copper")},
    {"id": "connected-bicycle", "stage": 2, "subject": "a bicycle with its frame, wheels, and handlebars", "materials": ("painted steel", "brushed aluminum", "dark carbon")},
    {"id": "connected-flower", "stage": 2, "subject": "a flower with petals, stem, and leaf", "materials": ("fresh petals", "paper cutout", "silk petals")},
    {"id": "connected-kite", "stage": 2, "subject": "a kite with its crossed spars and tail", "materials": ("crinkled paper", "bright nylon", "woven cloth")},
    {"id": "connected-key", "stage": 2, "subject": "a key with a bow, shaft, and teeth", "materials": ("tarnished brass", "forged iron", "silver metal")},
    {"id": "pair-cup-book", "stage": 3, "subject": "a cup beside a closed book", "relation": "Keep a small clear gap between the cup and book while showing both resting on the same surface; place their shared cast shadow beneath the pair.", "materials": ("ceramic and linen", "glass and leather", "metal and paper")},
    {"id": "pair-apple-cloth", "stage": 3, "subject": "an apple resting on a folded cloth", "relation": "Keep the apple in contact with the folded cloth; show the cloth supporting the apple and put the cast shadow beneath the cloth rather than beneath the apple alone.", "materials": ("waxed fruit and cotton", "painted fruit and wool", "natural fruit and linen")},
    {"id": "pair-lamp-table", "stage": 3, "subject": "a table lamp on a small table", "relation": "Place the lamp on top of the table; preserve contact at its base and show the tabletop supporting it, with the cast shadow beneath the table.", "materials": ("wood and linen", "brass and oak", "painted metal and glass")},
    {"id": "pair-boat-buoy", "stage": 3, "subject": "a small boat beside a floating buoy", "relation": "Keep the boat separate from the buoy with visible water between them; use a soft water reflection instead of a shared land shadow.", "materials": ("painted wood", "weathered fiberglass", "dark metal")},
    {"id": "pair-vase-branch", "stage": 3, "subject": "a vase holding a single branch", "relation": "Keep the branch entering the vase opening so the vase contains its base; place the cast shadow beneath the vase and give the branch no independent grounded shadow.", "materials": ("matte clay", "clear glass", "painted porcelain")},
    {"id": "scene-cabin-pine", "stage": 4, "subject": "a tiny cabin beneath one pine tree", "materials": ("rough timber", "painted boards", "stone and wood")},
    {"id": "scene-market-stall", "stage": 4, "subject": "a small market stall with one awning", "materials": ("striped canvas", "painted wood", "corrugated metal")},
    {"id": "scene-pond-reeds", "stage": 4, "subject": "a quiet pond with reeds and one stepping stone", "materials": ("muddy bank", "smooth stone", "wooden edge")},
    {"id": "scene-lighthouse-cliff", "stage": 4, "subject": "a lighthouse on a low cliff", "materials": ("white plaster", "weathered stone", "painted concrete")},
    {"id": "scene-tram-street", "stage": 4, "subject": "a tram passing one row of street trees", "materials": ("painted steel", "brushed metal", "aged red paint")},
)

PALETTES: tuple[tuple[str, str], ...] = (
    ("warm ochre", "ochre, umber, and a small vermilion accent"),
    ("cool harbor", "slate blue, sea green, and a pale sand accent"),
    ("spring garden", "leaf green, coral, and warm cream"),
    ("violet dusk", "violet, indigo, and muted rose"),
    ("graphite sun", "charcoal, warm gray, and a single amber highlight"),
)
ORIENTATIONS: tuple[tuple[str, str], ...] = (
    ("upright portrait", "tall portrait orientation"),
    ("wide landscape", "wide landscape orientation"),
    ("quiet square", "square orientation with comfortable margins"),
)

FLOW_GUIDANCE: dict[int, str] = {
    1: "Make one complete first pass and inspect the rendered canvas. Revise only if a concrete revision will improve the brief; otherwise finish.",
    2: "Paint the connected construction and inspect attachment and overlap. Revise only if a concrete revision will improve the brief; otherwise finish.",
    3: "Paint both objects and inspect their relation and silhouettes. Revise only if a concrete revision will improve the brief; otherwise finish.",
    4: "Paint the simplified scene and inspect its three depth bands. Revise only if a concrete revision will improve the brief; otherwise finish.",
}

# The casebrief is a separate, hash-bound runtime input.  Keep its canonical
# system text in the curriculum module so both the pre-anchor lazy handoff and
# the post-anchor materializer emit identical files.
FLOW_HINT = (
    "Plan before writing code: identify the subject hierarchy, placement, and "
    "largest shapes first; then make explicit canvas, palette, and brush "
    "choices. Revise by preserving correct structure and changing one control "
    "at a time. After observing the actual canvas, either return one meaningful "
    "complete painting revision or respond FINISHED when it satisfies the brief. "
    "For a paint action, return only executable painting code in the assistant turn."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_casebrief(
    row: dict[str, Any], proposal_root: Path, output_root: Path, *, enabled: bool
) -> None:
    """Write the flow casebrief for a lazy or admitted manifest row.

    Lazy manifests are loaded by the real evaluation taskset before frozen
    anchors are admitted, so they need the same casebrief contract as the
    final materialized manifests.  The immutable reference text remains
    separate and hash-checked.
    """

    original = Path(str(row["reference"]))
    if not original.is_absolute():
        original = proposal_root / original
    original = original.resolve()
    if original.is_symlink() or not original.is_file():
        raise FileNotFoundError(f"prompt is missing: {original}")
    original_hash = sha256_file(original)
    if original_hash != row.get("reference_sha256"):
        raise ValueError(f"prompt hash mismatch for {row['id']}")
    body = original.read_text(encoding="utf-8").strip()
    hint = FLOW_HINT if enabled else "Structured-flow hint is disabled for this run."
    text = f"[SYSTEM]\n{hint}\n\n[USER]\n{body}\n"
    relative = Path("casebriefs") / str(row["split"]) / f"{row['id']}.txt"
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    row["source_prompt"] = str(original)
    row["source_prompt_sha256"] = original_hash
    row["casebrief"] = str(target.resolve())
    row["casebrief_sha256"] = sha256_file(target)
    row["flow_policy_enabled"] = enabled
    row["system_hint_sha256"] = sha256_bytes((hint + "\n").encode("utf-8"))


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_number(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:16], 16)


def _safe_id(value: str) -> None:
    if not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"unsafe case identifier: {value!r}")


def _family_table() -> dict[str, dict[str, Any]]:
    result = {str(row["id"]): dict(row) for row in BASE_FAMILIES}
    if len(result) != len(BASE_FAMILIES):
        raise AssertionError("base family IDs must be unique")
    for row in result.values():
        if row["stage"] not in {1, 2, 3, 4}:
            raise AssertionError(f"unsupported stage: {row['stage']}")
    return result


def _split_families() -> tuple[dict[int, list[str]], dict[str, str]]:
    """Return stage-ordered families and the immutable family split map."""

    by_stage: dict[int, list[str]] = {stage: [] for stage in range(1, 5)}
    for row in BASE_FAMILIES:
        by_stage[int(row["stage"])].append(str(row["id"]))
    for stage, families in by_stage.items():
        if len(families) != TRAIN_FAMILIES_PER_STAGE + HELDOUT_FAMILIES_PER_STAGE:
            raise AssertionError(f"stage {stage} must contain five base families")
    assignment: dict[str, str] = {}
    for stage, families in by_stage.items():
        for family in families[:TRAIN_FAMILIES_PER_STAGE]:
            assignment[family] = "train"
        for family in families[TRAIN_FAMILIES_PER_STAGE:]:
            assignment[family] = "heldout"
    return by_stage, assignment


def _brief(family: Mapping[str, Any], variant: int, palette: tuple[str, str], orientation: tuple[str, str], material: str) -> str:
    stage = int(family["stage"])
    palette_name, palette_text = palette
    orientation_name, orientation_text = orientation
    subject = str(family["subject"])
    if stage == 1:
        body = (
            f"Paint exactly one {subject} as the clear subject. Place it centrally with a readable silhouette, "
            f"a grounded shadow, and one restrained highlight. Use {material}; keep the background simple and empty."
        )
    elif stage == 2:
        body = (
            f"Paint exactly one connected construction: {subject}. Make the attachment points and overlap readable, "
            f"with a grounded shadow and one restrained highlight. Use {material}; keep the background simple and empty."
        )
    elif stage == 3:
        body = (
            f"Paint exactly two objects: {subject}. {family['relation']} Preserve the stated contact, containment, "
            f"or separation instead of inventing another relationship. Use {material}; keep the background quiet with no extra objects."
        )
    else:
        body = (
            f"Paint a simplified scene containing only {subject}. Establish one clear foreground, one middle distance, "
            f"and a calm background shape; preserve the subject relationships without tiny clutter. Use {material}."
        )
    return (
        f"{body} The composition uses a {orientation_text}; the intended view is {orientation_name}. "
        f"Use the {palette_name} palette ({palette_text}) with clear value separation. "
        "Use painterly p5.brush marks with a few deliberate shapes and edges; do not add text, borders, or unrelated subjects."
    )


def generate_rows(seed: int = 20261009) -> list[dict[str, Any]]:
    """Generate all 300 deterministic brief records without writing files."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    families = _family_table()
    _, assignment = _split_families()
    rows: list[dict[str, Any]] = []
    for family_row in BASE_FAMILIES:
        family_id = str(family_row["id"])
        split = assignment[family_id]
        stage = int(family_row["stage"])
        for variant in range(VARIANTS_PER_FAMILY):
            palette = PALETTES[variant % len(PALETTES)]
            orientation = ORIENTATIONS[(variant // len(PALETTES)) % len(ORIENTATIONS)]
            material = tuple(family_row["materials"])[variant % len(tuple(family_row["materials"]))]
            case_id = f"sf-s{stage}-{split[:2]}-{family_id}-{variant:02d}"
            _safe_id(case_id)
            prompt = _brief(family_row, variant, palette, orientation, material)
            prompt_hash = sha256_bytes((prompt + "\n").encode("utf-8"))
            rows.append({
                "id": case_id,
                "task_id": case_id,
                "source_group": case_id,
                "family": family_id,
                "base_family_id": family_id,
                "stage": stage,
                "split": split,
                "variant": variant,
                "palette": palette[0],
                "orientation": orientation[0],
                "material": material,
                "mode": "fresh",
                "reference_kind": "text",
                "reference": f"prompts/{split}/stage-{stage}/{case_id}.txt",
                "reference_sha256": prompt_hash,
                "anchor_policy": ANCHOR_POLICY,
                "anchor_model_alias": ANCHOR_MODEL_ALIAS,
                "anchor_required": True,
                "anchor_status": "pending",
                "admission_status": "awaiting_frozen_sft512_anchor",
                # Runtime operators may use this as a separate system or
                # environment hint.  It is deliberately absent from the
                # reference text so judge input remains the original brief.
                "flow_guidance": FLOW_GUIDANCE[stage],
                "max_turns": MAX_TURNS,
                "order": _stable_number(seed, stage, family_id, variant),
                "prompt_signature": sha256_bytes(
                    "|".join((family_id, str(stage), palette[0], orientation[0], material, str(variant))).encode()
                ),
                "prompt_text": prompt,
            })
    rows.sort(key=lambda row: (int(row["stage"]), str(row["split"]), str(row["base_family_id"]), int(row["variant"])))
    if len(rows) != 300 or len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("curriculum must contain 300 unique rows")
    if len({row["prompt_signature"] for row in rows}) != len(rows):
        raise AssertionError("prompt variants must be unique")
    return rows


def _write_prompt_files(rows: Sequence[Mapping[str, Any]], root: Path) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    for row in rows:
        relative = Path(str(row["reference"]))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(row["prompt_text"]).strip() + "\n"
        target.write_text(content, encoding="utf-8")
        actual = sha256_file(target)
        if actual != row["reference_sha256"]:
            raise AssertionError(f"prompt hash mismatch after write: {row['id']}")
        written.append({"path": str(relative), "sha256": actual, "bytes": target.stat().st_size})
    return written


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove generation-only prompt text while keeping native metadata."""

    return {key: value for key, value in row.items() if key not in {"prompt_text"}}


def _count_receipt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_stage[str(row["stage"])] = by_stage.get(str(row["stage"]), 0) + 1
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + 1
        by_family[str(row["base_family_id"])] = by_family.get(str(row["base_family_id"]), 0) + 1
    return {"total": len(rows), "by_stage": by_stage, "by_split": by_split, "by_base_family": by_family}


def build_pilot_plan(train_rows: Sequence[Mapping[str, Any]], heldout_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the first small node slice without consuming the full pool."""

    def select(rows: Sequence[Mapping[str, Any]], stage: int, count: int) -> list[str]:
        candidates = [row for row in rows if int(row["stage"]) == stage]
        selected = sorted(candidates, key=lambda row: (_stable_number("pilot", row["split"], row["id"]), str(row["id"])))[:count]
        if len(selected) != count:
            raise ValueError(f"pilot needs {count} rows at stage {stage}, found {len(selected)}")
        return [str(row["id"]) for row in selected]

    train_ids = select(train_rows, 1, PILOT_TRAIN_COUNT // 2) + select(train_rows, 2, PILOT_TRAIN_COUNT // 2)
    heldout_ids = select(heldout_rows, 1, PILOT_HELDOUT_COUNT // 2) + select(heldout_rows, 2, PILOT_HELDOUT_COUNT // 2)
    return {
        "schema": "structured-flow-pilot-plan-v1",
        "run_id": RUN_ID,
        "status": "pending_anchors",
        "rationale": "Start with 32 early train briefs and 16 small heldout briefs; retain the full 240/60 pool for later stages.",
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "counts": {"train": len(train_ids), "heldout": len(heldout_ids), "total": len(train_ids) + len(heldout_ids)},
        "stage_counts": {"train": {"1": PILOT_TRAIN_COUNT // 2, "2": PILOT_TRAIN_COUNT // 2}, "heldout": {"1": PILOT_HELDOUT_COUNT // 2, "2": PILOT_HELDOUT_COUNT // 2}},
        "anchor_requirement": "Pilot rows remain pending until actual frozen SFT512 PNGs and their hashes are materialized.",
        "flow_guidance_note": "flow_guidance is a runtime-only optional hint; it is not appended to the immutable reference brief.",
    }


def _file_receipts(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        values.append({
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return values


def build_proposal(output: Path, *, seed: int = 20261009) -> dict[str, Any]:
    """Write prompt files and pending train/heldout proposal manifests."""

    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(seed)
    prompt_receipts = _write_prompt_files(rows, output)
    train = [_compact_row(row) for row in rows if row["split"] == "train"]
    heldout = [_compact_row(row) for row in rows if row["split"] == "heldout"]
    family_splits = {family: split for family, split in _split_families()[1].items()}
    common = {
        "run_id": RUN_ID,
        "schema": SCHEMA,
        "status": "proposal_pending_anchors",
        "seed": seed,
        "anchor_policy": ANCHOR_POLICY,
        "anchor_model_alias": ANCHOR_MODEL_ALIAS,
        "anchor_requirement": "Every train row needs a real same-source_group frozen SFT512 PNG and opponent_sha256 before native_painting_rl admission.",
        "native_rl_admission": "pending; these proposal rows intentionally have no opponent path or hash",
    }
    train_doc = {**common, "split": "train", "counts": _count_receipt(train), "rows": train}
    heldout_doc = {**common, "split": "heldout", "counts": _count_receipt(heldout), "rows": heldout}
    _json_write(output / "train-proposal.json", train_doc)
    # Keep the canonical ``rows`` key for the shared receipt shape and add the
    # native evaluation alias so a node can pass heldout-proposal.json to
    # NativePaintingTaskset without changing the original brief text.
    _json_write(output / "heldout-proposal.json", {**heldout_doc, "cases": heldout})
    sampling = build_sampling_plan(train)
    _json_write(output / "sampling-plan.json", sampling)
    pilot = build_pilot_plan(train, heldout)
    _json_write(output / "pilot-plan.json", pilot)
    manifest_paths = [output / "train-proposal.json", output / "heldout-proposal.json", output / "sampling-plan.json", output / "pilot-plan.json"]
    receipt = {
        "run_id": RUN_ID,
        "schema": SCHEMA,
        "status": "prepared_unrendered",
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {"external_files": [], "seed": seed},
        "counts": {"all": _count_receipt(rows), "train": _count_receipt(train), "heldout": _count_receipt(heldout)},
        "family_split": family_splits,
        "split_rule": "base families assigned first; all 15 variants inherit the base-family split",
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
    """Describe deterministic stage manifests with 25% earlier replay."""

    by_stage = {stage: sorted((dict(row) for row in train_rows if int(row["stage"]) == stage), key=lambda row: str(row["id"])) for stage in range(1, 5)}
    if any(len(by_stage[stage]) != TRAIN_PER_STAGE for stage in by_stage):
        raise ValueError("expected 60 training rows per stage")
    stages: list[dict[str, Any]] = []
    for stage in range(1, 5):
        replay: list[str] = []
        for prior_stage in range(1, stage):
            prior = by_stage[prior_stage]
            count = math.floor(len(prior) * REPLAY_FRACTION)
            chosen = sorted(prior, key=lambda row: (_stable_number("replay", stage, row["id"]), str(row["id"])))[:count]
            replay.extend(str(row["id"]) for row in chosen)
        current = [str(row["id"]) for row in by_stage[stage]]
        stages.append({
            "stage": stage,
            "current_count": len(current),
            "replay_count": len(replay),
            "total_count": len(current) + len(replay),
            "replay_fraction_of_prior": REPLAY_FRACTION,
            "current_ids": current,
            "replay_ids": replay,
            "manifest": f"stage-{stage}-train-manifest.json",
        })
    return {
        "schema": "structured-flow-staged-sampling-v1",
        "run_id": RUN_ID,
        "replay_fraction": REPLAY_FRACTION,
        "stages": stages,
        "note": "Replay is selected deterministically from earlier stage rows; each stage manifest keeps unique task IDs for native_painting_rl.",
    }


def _resolve_prompt(proposal_path: Path, row: Mapping[str, Any]) -> Path:
    prompt = Path(str(row["reference"]))
    if not prompt.is_absolute():
        prompt = proposal_path.parent / prompt
    if prompt.is_symlink() or not prompt.is_file():
        raise FileNotFoundError(f"prompt is missing or symlinked: {prompt}")
    prompt = prompt.absolute()
    actual = sha256_file(prompt)
    if actual != row.get("reference_sha256"):
        raise ValueError(f"prompt hash mismatch for {row.get('id')}: {actual}")
    return prompt


def _validate_checkpoint_hash(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("frozen_sft512_checkpoint_sha256 must be a lowercase 64-character SHA-256")
    return value


def _anchor_path(anchor_root: Path, case_id: str) -> Path:
    _safe_id(case_id)
    path = anchor_root / f"{case_id}.png"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"frozen SFT512 anchor is missing or symlinked: {path}")
    return path.absolute()


def _load_anchor_provenance(anchor_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Load the node-written provenance map required for anchor admission.

    The node launcher must write ``anchor-provenance.json`` with this shape::

        {"schema": "structured-flow-anchor-provenance-v1", "anchors": {
          "case-id": {"checkpoint_sha256": "...", "reference_sha256": "...",
                      "source_group": "case-id", "png_sha256": "...",
                      "valid": true,
                      "generation_receipt": "native-generation/<case-id>.json"}
        }}

    ``generation_receipt`` may be a non-empty trace/receipt ID or a mapping
    containing a path/trace ID.  This preparation step does not interpret the
    native trace; it requires its durable identity to be present and checks
    the hashes and source binding against the proposal and PNG on disk.
    """

    path = anchor_root / ANCHOR_PROVENANCE_FILE
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"anchor provenance is missing or symlinked: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid anchor provenance JSON: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != "structured-flow-anchor-provenance-v1":
        raise ValueError("anchor provenance schema mismatch")
    anchors = raw.get("anchors")
    if not isinstance(anchors, dict) or not anchors:
        raise ValueError("anchor provenance must contain a non-empty anchors mapping")
    result: dict[str, dict[str, Any]] = {}
    for case_id, value in anchors.items():
        if not isinstance(case_id, str) or not _SAFE_ID_RE.fullmatch(case_id) or not isinstance(value, dict):
            raise ValueError(f"invalid anchor provenance entry: {case_id!r}")
        for key in ("checkpoint_sha256", "reference_sha256", "source_group", "png_sha256"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError(f"anchor provenance {case_id} is missing {key}")
        _validate_checkpoint_hash(str(value["checkpoint_sha256"]))
        for key in ("reference_sha256", "png_sha256"):
            if not _SHA256_RE.fullmatch(str(value[key])):
                raise ValueError(f"anchor provenance {case_id} has invalid {key}")
        if value.get("valid") is not True:
            raise ValueError(f"anchor provenance {case_id} is not marked valid")
        receipt = value.get("generation_receipt")
        trace = value.get("trace") or value.get("trace_id")
        backing = receipt if receipt is not None else trace
        if isinstance(backing, str):
            backed = bool(backing.strip())
        elif isinstance(backing, dict):
            backed = any(isinstance(item, str) and item.strip() for item in backing.values())
        else:
            backed = False
        if not backed:
            raise ValueError(f"anchor provenance {case_id} lacks a generation receipt or trace")
        result[case_id] = dict(value)
    return result, sha256_file(path)


def _admitted_row(
    row: Mapping[str, Any],
    proposal_path: Path,
    anchor_root: Path,
    checkpoint_sha256: str,
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = str(row["id"])
    prompt = _resolve_prompt(proposal_path, row)
    anchor = _anchor_path(anchor_root, case_id)
    anchor_hash = sha256_file(anchor)
    evidence = provenance.get(case_id)
    if evidence is None:
        raise ValueError(f"anchor provenance is missing case: {case_id}")
    if evidence["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"anchor checkpoint provenance mismatch: {case_id}")
    if evidence["reference_sha256"] != str(row["reference_sha256"]):
        raise ValueError(f"anchor reference provenance mismatch: {case_id}")
    if evidence["source_group"] != str(row["source_group"]):
        raise ValueError(f"anchor source_group provenance mismatch: {case_id}")
    if evidence["png_sha256"] != anchor_hash:
        raise ValueError(f"anchor PNG provenance mismatch: {case_id}")
    admitted = _compact_row(row)
    admitted.update({
        "reference": str(prompt),
        "opponent": str(anchor),
        "opponent_sha256": anchor_hash,
        "opponent_source_group": str(row["source_group"]),
        "anchor_status": "verified_file",
        "admission_status": "ready_for_native_painting_rl",
        "frozen_sft512_checkpoint_sha256": checkpoint_sha256,
        "anchor_provenance_file": str(anchor_root / ANCHOR_PROVENANCE_FILE),
    })
    return admitted


def materialize_manifest(
    proposal: Path,
    output: Path,
    anchor_root: Path,
    *,
    split: str = "train",
    frozen_sft512_checkpoint_sha256: str,
    selected_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize a native-compatible manifest after verifying every anchor."""

    if split not in {"train", "heldout"}:
        raise ValueError("split must be train or heldout")
    checkpoint_sha256 = _validate_checkpoint_hash(frozen_sft512_checkpoint_sha256)
    proposal = proposal.expanduser().resolve()
    output = output.expanduser().resolve()
    anchor_root = anchor_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    provenance, provenance_sha256 = _load_anchor_provenance(anchor_root)
    raw = json.loads(proposal.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA or raw.get("split") != split:
        raise ValueError("proposal schema or split mismatch")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("proposal has no rows")
    selected = set(selected_ids) if selected_ids is not None else None
    if selected is not None and len(selected) != len(selected_ids or ()):
        raise ValueError("selected_ids must be unique")
    admitted: list[dict[str, Any]] = []
    if selected is None:
        selected_rows = rows
    else:
        by_id = {str(row["id"]): row for row in rows}
        if not selected <= set(by_id):
            missing = sorted(selected - set(by_id))
            raise ValueError(f"selected IDs are absent from proposal: {missing[:3]}")
        selected_rows = [by_id[ident] for ident in selected_ids or ()]
    for row in selected_rows:
        admitted.append(_admitted_row(row, proposal, anchor_root, checkpoint_sha256, provenance))
    if selected is not None and {str(row["id"]) for row in admitted} != selected:
        missing = sorted(selected - {str(row["id"]) for row in admitted})
        raise ValueError(f"selected IDs are absent from proposal: {missing[:3]}")
    if split == "train" and any(row.get("split") != "train" for row in admitted):
        raise AssertionError("train admission included non-train row")
    manifest = {
        "schema": NATIVE_SCHEMA,
        "run_id": RUN_ID,
        "status": "admitted_with_verified_frozen_sft512_anchors",
        "split": split,
        "native_rl_compatible": split == "train",
        "frozen_opponent": {
            "policy": ANCHOR_POLICY,
            "model_alias": ANCHOR_MODEL_ALIAS,
            "checkpoint_sha256": checkpoint_sha256,
            "same_source_group_required": True,
            "provenance_file": ANCHOR_PROVENANCE_FILE,
            "provenance_sha256": provenance_sha256,
        },
        "proposal_sha256": sha256_file(proposal),
        "rows": admitted,
    }
    if split == "heldout":
        # native_painting.py reads ``cases`` for evaluation manifests.  The
        # rows remain present for audit joins, while this alias keeps the
        # heldout handoff directly consumable by that taskset.
        manifest["cases"] = admitted
    _json_write(output, manifest)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "split": split,
        "rows": len(admitted),
        "anchor_count": len(admitted),
        "native_rl_compatible": split == "train",
        "proposal_sha256": manifest["proposal_sha256"],
        "anchor_provenance_sha256": provenance_sha256,
    }


def materialize_stage_manifests(
    proposal: Path,
    output_dir: Path,
    anchor_root: Path,
    *,
    frozen_sft512_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Write four current-plus-replay native manifests after anchor checks."""

    checkpoint_sha256 = _validate_checkpoint_hash(frozen_sft512_checkpoint_sha256)
    proposal = proposal.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    provenance, provenance_sha256 = _load_anchor_provenance(anchor_root)
    raw = json.loads(proposal.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA or raw.get("split") != "train":
        raise ValueError("stage manifests require a train proposal")
    rows = [dict(row) for row in raw.get("rows", [])]
    by_stage = {stage: sorted((row for row in rows if int(row["stage"]) == stage), key=lambda row: str(row["id"])) for stage in range(1, 5)}
    plan = build_sampling_plan(rows)
    by_id = {str(row["id"]): row for row in rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for stage_info in plan["stages"]:
        stage = int(stage_info["stage"])
        ids = list(stage_info["current_ids"]) + list(stage_info["replay_ids"])
        admitted = []
        for ident in ids:
            if ident not in by_id:
                raise AssertionError(f"sampling plan references unknown row: {ident}")
            row = _admitted_row(by_id[ident], proposal, anchor_root, checkpoint_sha256, provenance)
            row["sampling_stage"] = stage
            row["sampling_role"] = "current" if int(row["stage"]) == stage else "replay"
            row["order"] = len(admitted)
            admitted.append(row)
        prepared.append((stage_info, admitted))
    # Validate every requested anchor before writing even the first stage
    # manifest, so a late missing/mismatched provenance entry cannot leave a
    # partially admitted staged set.
    for stage_info, admitted in prepared:
        stage = int(stage_info["stage"])
        path = output_dir / str(stage_info["manifest"])
        manifest = {
            "schema": NATIVE_SCHEMA,
            "run_id": RUN_ID,
            "status": "admitted_with_verified_frozen_sft512_anchors",
            "split": "train",
            "native_rl_compatible": True,
            "sampling_stage": stage,
            "replay_fraction_of_prior": REPLAY_FRACTION,
            "frozen_opponent": {"policy": ANCHOR_POLICY, "model_alias": ANCHOR_MODEL_ALIAS, "checkpoint_sha256": checkpoint_sha256, "same_source_group_required": True, "provenance_file": ANCHOR_PROVENANCE_FILE, "provenance_sha256": provenance_sha256},
            "proposal_sha256": sha256_file(proposal),
            "rows": admitted,
        }
        _json_write(path, manifest)
        records.append({
            "stage": stage,
            "path": path.name,
            "sha256": sha256_file(path),
            "current_count": stage_info["current_count"],
            "replay_count": stage_info["replay_count"],
            "total_count": len(admitted),
        })
    receipt = {
        "schema": "structured-flow-stage-manifest-receipt-v1",
        "run_id": RUN_ID,
        "status": "prepared_with_verified_anchors",
        "proposal_sha256": sha256_file(proposal),
        "frozen_sft512_checkpoint_sha256": checkpoint_sha256,
        "anchor_provenance_sha256": provenance_sha256,
        "manifests": records,
        "counts": {"stages": len(records), "total_manifest_rows": sum(row["total_count"] for row in records)},
    }
    _json_write(output_dir / "stage-manifest-receipt.json", receipt)
    return receipt


def materialize_pilot_manifests(
    train_proposal: Path,
    heldout_proposal: Path,
    output_dir: Path,
    anchor_root: Path,
    *,
    frozen_sft512_checkpoint_sha256: str,
    admitted_train_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize the registered first-32/16 pilot after verifying anchors.

    ``admitted_train_ids`` is an optional train-only censoring result.  It must
    be a subset of the deterministic first-32 plan; heldout selection is never
    reduced or substituted.
    """

    checkpoint_sha256 = _validate_checkpoint_hash(frozen_sft512_checkpoint_sha256)
    train_proposal = train_proposal.expanduser().resolve()
    heldout_proposal = heldout_proposal.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    train_doc = json.loads(train_proposal.read_text(encoding="utf-8"))
    heldout_doc = json.loads(heldout_proposal.read_text(encoding="utf-8"))
    if train_doc.get("schema") != SCHEMA or heldout_doc.get("schema") != SCHEMA:
        raise ValueError("pilot requires structured-flow proposals")
    plan = build_pilot_plan(train_doc.get("rows", []), heldout_doc.get("rows", []))
    selected_train_ids = list(plan["train_ids"])
    if admitted_train_ids is not None:
        selected_train_ids = [str(ident) for ident in admitted_train_ids]
        if len(selected_train_ids) != len(set(selected_train_ids)):
            raise ValueError("admitted_train_ids must be unique")
        if not set(selected_train_ids) <= set(plan["train_ids"]):
            raise ValueError("admitted train IDs must be a subset of the registered first-32 pilot")
        if not 24 <= len(selected_train_ids) <= len(plan["train_ids"]):
            raise ValueError("admitted train count must be in [24, 32]")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_result = materialize_manifest(
        train_proposal,
        output_dir / "pilot-train-manifest.json",
        anchor_root,
        split="train",
        frozen_sft512_checkpoint_sha256=checkpoint_sha256,
        selected_ids=selected_train_ids,
    )
    heldout_result = materialize_manifest(
        heldout_proposal,
        output_dir / "pilot-heldout-manifest.json",
        anchor_root,
        split="heldout",
        frozen_sft512_checkpoint_sha256=checkpoint_sha256,
        selected_ids=plan["heldout_ids"],
    )
    receipt = {
        "schema": "structured-flow-pilot-manifest-receipt-v1",
        "run_id": RUN_ID,
        "status": "prepared_with_verified_anchors",
        "pilot_plan": plan,
        "registered_train_ids": list(plan["train_ids"]),
        "admitted_train_ids": selected_train_ids,
        "frozen_sft512_checkpoint_sha256": checkpoint_sha256,
        "train": train_result,
        "heldout": heldout_result,
        "native_rl_train_manifest": "pilot-train-manifest.json",
        "heldout_manifest_is_native_rl_compatible": False,
    }
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
    pilot = commands.add_parser("pilot-manifests", help="write only the 32-train/16-heldout pilot manifests")
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
        result = materialize_manifest(
            args.proposal,
            args.output,
            args.anchor_root,
            split=args.split,
            frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256,
        )
    elif args.command == "stage-manifests":
        result = materialize_stage_manifests(
            args.proposal,
            args.output_dir,
            args.anchor_root,
            frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256,
        )
    else:
        result = materialize_pilot_manifests(
            args.train_proposal,
            args.heldout_proposal,
            args.output_dir,
            args.anchor_root,
            frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
