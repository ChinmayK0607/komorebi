"""Prepare the scaled painter curriculum without rendering or model calls.

The builder deliberately separates three identities:

* a source program hash (available before rendering),
* an image hash (available for imported references, otherwise pending node
  rendering), and
* a training exposure (an admitted schedule entry).

Procedural programs use the existing ``expand_demonstrations`` recipes.  They
are already expressed in canvas coordinates after the contract's single
``translate(-300, -300)``; this module does not use the affine representation
from ``diverse_curriculum.py`` and does not invoke ``compile_teacher.cjs``.
The node can consume the generated manifests with the existing
``render_demos.py`` resumable renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:  # ``python painter/scaled_curriculum.py``
    from expand_demonstrations import FAMILIES, expanded
except ImportError:  # ``python -m painter.scaled_curriculum``
    from .expand_demonstrations import FAMILIES, expanded


RUN_ID = "scaled-curriculum-20260921"
TRAIN_PER_FAMILY = 200
DEV_PER_FAMILY = 20
TRAIN_TARGET = TRAIN_PER_FAMILY * len(FAMILIES)
DEV_TARGET = DEV_PER_FAMILY * len(FAMILIES)
COMPOSITIONS = ("single", "pair", "overlap", "offset", "scene")
DIFFICULTY_BY_FAMILY = {
    family: 0 if index < 4 else 1 if index < 6 else 2
    for index, family in enumerate(FAMILIES)
}
_DRAW_PREFIX = "function draw(){translate(-300,-300);\n"
_XY = re.compile(r"const x=(-?\d+),y=(-?\d+),s=(-?\d+)")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            yield value


def _read_inventory(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON/JSONL inventory in the few shapes used by inventory agents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return list(_jsonl(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("references", data.get("items", data.get("rows", data.get("records"))))
        if rows is None:
            raise ValueError(f"Inventory {path} must contain references/items/rows/records")
    else:
        raise ValueError(f"Inventory {path} must be a JSON list or object")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Inventory {path} contains a non-object row")
    return list(rows)


def _first(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _normal_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if _HASH.fullmatch(text) else None


def _stable_int(value: str, seed: int = 0) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16)


def _source_group(row: Mapping[str, Any]) -> str:
    value = _first(row, "source_group", "source_id", "id", "image_id", "path")
    if value is None:
        raise ValueError("Every inventory row needs source_group, source_id, id, image_id, or path")
    return str(value)


def _image_hash(row: Mapping[str, Any]) -> str | None:
    return _normal_hash(
        _first(
            row,
            "image_sha256",
            "png_sha256",
            "reference_sha256",
            "image_hash",
            "actual_sha256",
            "declared_sha256",
            "sha256",
        )
    )


def _local_image_hash(reference: Any) -> str | None:
    """Hash a local reference when available; remote URIs remain unverified."""
    if reference is None:
        return None
    try:
        path = Path(str(reference))
        if path.is_file():
            return sha256_bytes(path.read_bytes())
    except OSError:
        pass
    return None


def _split_for_inventory(row: Mapping[str, Any], source_group: str, seed: int) -> str:
    requested = str(_first(row, "split", "partition", default="")).lower()
    if requested in {"train", "dev", "validation", "test", "eval", "sealed", "heldout", "holdout"}:
        return requested
    # Keep the assignment stable when the inventory order changes.  The 10%
    # development allocation matches the procedural 200:20 ratio.
    return "dev" if _stable_int(source_group, seed) % 10 == 0 else "train"


def _body(program: str) -> tuple[str, str]:
    """Return the setup prefix and draw body without its wrapper."""
    if _DRAW_PREFIX not in program or not program.rstrip().endswith("}"):
        raise ValueError("Expected an expanded direct-canvas teacher program")
    prefix, remainder = program.split(_DRAW_PREFIX, 1)
    body = remainder.rsplit("\nnoLoop();}\n", 1)
    if len(body) != 2:
        raise ValueError("Expanded teacher program has an unexpected draw wrapper")
    return prefix, body[0]


def _shift_body(body: str, dx: int, dy: int) -> str:
    def shift(match: re.Match[str]) -> str:
        return f"const x={int(match.group(1)) + dx},y={int(match.group(2)) + dy},s={match.group(3)}"

    shifted, count = _XY.subn(shift, body, count=1)
    if count != 1:
        raise ValueError("Could not locate the direct-canvas x/y constants")
    return shifted


def _place_body(body: str, center: tuple[int, int], size: int) -> str:
    """Set direct-canvas recipe parameters to a conservative canvas slot."""
    def place(match: re.Match[str]) -> str:
        return f"const x={center[0]},y={center[1]},s={size}"

    placed, count = _XY.subn(place, body, count=1)
    if count != 1:
        raise ValueError("Could not locate the direct-canvas x/y/s constants")
    return placed


def _compose(programs: Sequence[str], offsets: Sequence[tuple[int, int]]) -> str:
    if len(programs) != len(offsets):
        raise ValueError("Composition programs and offsets must have equal length")
    prefixes = []
    bodies = []
    for program, (dx, dy) in zip(programs, offsets):
        prefix, body = _body(program)
        prefixes.append(prefix)
        # A block keeps each recipe's ``const x``/``const y`` declarations
        # independent while preserving the original direct-coordinate calls.
        bodies.append("{\n" + _shift_body(body, dx, dy) + "\n}")
    if len(set(prefixes)) != 1:
        raise ValueError("Composed recipes must share the same setup prefix")
    return prefixes[0] + _DRAW_PREFIX + "\n".join(bodies) + "\nnoLoop();}\n"


def _composition_families(primary_index: int, composition: str, rng: random.Random) -> list[str]:
    primary = FAMILIES[primary_index]
    if composition == "single":
        return [primary]
    choices = [family for family in FAMILIES if family != primary]
    rng.shuffle(choices)
    second = choices[0]
    if composition in {"pair", "overlap", "offset"}:
        return [primary, second]
    return [primary, second, choices[1]]


def _placements(composition: str, rng: random.Random, count: int) -> list[tuple[tuple[int, int], int]]:
    """Return conservative slots; 1.75*s is the largest existing extent."""
    if composition == "single":
        return [((rng.randint(255, 345), rng.randint(255, 345)), rng.randint(62, 100))]
    if composition == "pair":
        return [
            ((rng.randint(165, 205), rng.randint(270, 330)), rng.randint(42, 54)),
            ((rng.randint(395, 435), rng.randint(270, 330)), rng.randint(42, 54)),
        ]
    if composition == "overlap":
        return [
            ((rng.randint(225, 255), rng.randint(275, 325)), rng.randint(42, 52)),
            ((rng.randint(345, 375), rng.randint(280, 335)), rng.randint(42, 52)),
        ]
    if composition == "offset":
        return [
            ((rng.randint(165, 205), rng.randint(245, 285)), rng.randint(42, 52)),
            ((rng.randint(395, 435), rng.randint(315, 355)), rng.randint(42, 52)),
        ]
    if composition == "scene":
        return [
            ((rng.randint(130, 170), rng.randint(315, 355)), rng.randint(36, 46)),
            ((rng.randint(280, 320), rng.randint(245, 285)), rng.randint(36, 46)),
            ((rng.randint(430, 470), rng.randint(315, 355)), rng.randint(36, 46)),
        ]
    raise ValueError(f"Unknown composition {composition}")


def _variant_programs(
    family_index: int, variant: int, split: str, identifier: str
) -> tuple[list[str], list[str], str, list[int]]:
    seed = 9200000 + family_index * 100000 + variant + (10000000 if split == "dev" else 0)
    rng = random.Random(_stable_int(identifier, seed))
    composition = COMPOSITIONS[(variant + family_index) % len(COMPOSITIONS)]
    members = _composition_families(family_index, composition, rng)
    placements = _placements(composition, rng, len(members))
    stage_sources: list[list[str]] = []
    recipe_seeds: list[int] = []
    for member_index, member in enumerate(members):
        member_seed = seed + member_index * 3571
        recipe_seeds.append(member_seed)
        stage_sources.append(expanded(member, member_seed, split)[1])
    programs: list[str] = []
    for stage in range(3):
        sources = [
            _place_body(stages[stage], placement[0], placement[1])
            for stages, placement in zip(stage_sources, placements)
        ]
        source = _compose(sources, [(0, 0)] * len(sources))
        programs.append(source)
    return programs, members, composition, recipe_seeds


def _canonical_geometry(program: str) -> str:
    """Normalize non-geometric seed calls before deduplication."""
    canonical = re.sub(r"(?:randomSeed|noiseSeed)\([^;]+\);", "", program)
    canonical = re.sub(r"brush\.seed\([^;]+\);", "", canonical)
    canonical = re.sub(r"//[^\n]*\n", "", canonical)
    return canonical


def generate_procedural_records(
    train_per_family: int = TRAIN_PER_FAMILY,
    dev_per_family: int = DEV_PER_FAMILY,
    *,
    seed: int = 20260921,
) -> list[dict[str, Any]]:
    """Generate compact record metadata plus three source programs per row."""
    if train_per_family < 0 or dev_per_family < 0:
        raise ValueError("Per-family counts must be non-negative")
    rows: list[dict[str, Any]] = []
    seen_geometry: set[str] = set()
    for family_index, family in enumerate(FAMILIES):
        for split, count in (("train", train_per_family), ("dev", dev_per_family)):
            accepted = 0
            candidate = 0
            while accepted < count:
                identifier = f"sc-v1-{split}-{family}-{accepted:04d}"
                programs, members, composition, recipe_seeds = _variant_programs(
                    family_index, candidate + seed % 997, split, identifier
                )
                target = programs[-1]
                geometry_sha256 = sha256_text(_canonical_geometry(target))
                candidate += 1
                if geometry_sha256 in seen_geometry:
                    continue
                seen_geometry.add(geometry_sha256)
                difficulty = min(
                    4,
                    DIFFICULTY_BY_FAMILY[family]
                    + {"single": 0, "pair": 1, "overlap": 2, "offset": 2, "scene": 3}[composition],
                )
                rows.append(
                    {
                        "id": identifier,
                        "source_group": identifier,
                        "split": split,
                        "family": family,
                        "primary_family": family,
                        "member_families": members,
                        "difficulty": difficulty,
                        "composition": composition,
                        "reference_kind": "procedural_known_program",
                        "coordinate_representation": "canvas-pixels",
                        "candidate_index": candidate - 1,
                        "recipe_seeds": recipe_seeds,
                        "programs": programs,
                        "source_sha256": sha256_text(target),
                        "geometry_sha256": geometry_sha256,
                        "image_sha256": None,
                        "image_hash_status": "pending_node_render",
                        "distinct_image_claim": False,
                        "render_status": "pending_node_render",
                        "visual_review_status": "pending",
                        "generalization_scope": "parameter and composition variation within existing recipe families; requires render review",
                        "planned_training_exposure": 0,
                        "admitted_training_exposure": 0,
                    }
                )
                accepted += 1
    return rows


def import_inventory_rows(
    inventory: str | Path | Iterable[Mapping[str, Any]],
    *,
    seed: int = 20260921,
    seen_source_groups: Iterable[str] = (),
    seen_image_hashes: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize and deduplicate real-image inventory rows.

    Rows are metadata only.  The image is never copied or opened by this
    builder.  A supplied exact image hash is used for deduplication; missing
    hashes are retained with ``image_hash_status=unavailable``.
    """
    raw_rows = _read_inventory(inventory) if isinstance(inventory, (str, Path)) else list(inventory)
    seen_sources = set(map(str, seen_source_groups))
    seen_images = {str(x).lower() for x in seen_image_hashes if _normal_hash(x)}
    imported: list[dict[str, Any]] = []
    stats = Counter(total=len(raw_rows))
    for raw in raw_rows:
        try:
            source = _source_group(raw)
            reference = _first(raw, "reference", "image", "path", "image_path", "uri", "url")
            if reference is None:
                stats["skipped_missing_reference"] += 1
                continue
            source_group = f"inventory:{source}"
            if source_group in seen_sources:
                stats["skipped_duplicate_source_group"] += 1
                continue
            supplied_image_hash = _image_hash(raw)
            local_reference = _first(raw, "resolved_path", "local_path", default=reference)
            local_image_hash = _local_image_hash(local_reference)
            if supplied_image_hash and local_image_hash and supplied_image_hash != local_image_hash:
                stats["skipped_image_hash_mismatch"] += 1
                continue
            image_hash = supplied_image_hash or local_image_hash
            if image_hash and image_hash in seen_images:
                stats["skipped_duplicate_image_hash"] += 1
                continue
            split = _split_for_inventory(raw, source_group, seed)
            family_raw = str(_first(raw, "family", "primary_family", "object_family", default="real_reference"))
            # Preserve inventory taxonomy (animals, food, scenes, ...) for
            # later balancing.  Procedural coverage remains exactly the
            # existing twelve families above.
            family = family_raw or "real_reference"
            difficulty_raw = _first(raw, "difficulty", "level", default=2)
            try:
                difficulty = max(0, min(4, int(difficulty_raw)))
            except (TypeError, ValueError):
                difficulty = 2
            row = {
                "id": f"inv-v1-{source}",
                "source_group": source_group,
                "split": split,
                "family": family,
                "primary_family": family,
                "family_raw": family_raw,
                "member_families": [family],
                "difficulty": difficulty,
                "composition": str(_first(raw, "composition", "layout", default="real_reference")),
                "reference_kind": "real_reference_image",
                "reference": str(reference),
                "inventory_resolved_path": str(local_reference) if local_reference != reference else None,
                "source_id": source,
                "image_sha256": image_hash,
                "image_hash_status": (
                    "verified_local" if local_image_hash else
                    "supplied_unverified" if supplied_image_hash else
                    "unavailable"
                ),
                "distinct_image_claim": bool(local_image_hash),
                "programs": [],
                "source_sha256": None,
                "render_status": "inventory_reference",
                "visual_review_status": "pending",
                "generalization_scope": "real reference imported from inventory; provenance and visual review remain external",
                "admitted_training_exposure": 0,
            }
        except (TypeError, ValueError):
            stats["skipped_invalid"] += 1
            continue
        imported.append(row)
        seen_sources.add(source_group)
        if image_hash:
            seen_images.add(image_hash)
        stats["admitted"] += 1
        stats[f"admitted_{split}"] += 1
    return imported, dict(stats)


def _interleaved_rows(rows: Sequence[Mapping[str, Any]], *, seed: int, epoch: int) -> list[Mapping[str, Any]]:
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row.get("difficulty", 0)), str(row.get("family", "unknown")), str(row.get("composition", "unknown")))].append(row)
    rng_seed = seed + epoch * 1000003
    queues: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for key, values in groups.items():
        ordered = sorted(values, key=lambda row: str(row["id"]))
        # Offset each bucket by a different stable amount, preventing every
        # epoch from beginning with the same easy/reference rows.
        shift = _stable_int("|".join(map(str, key)), rng_seed) % len(ordered)
        queues[key] = ordered[shift:] + ordered[:shift]
    result: list[Mapping[str, Any]] = []
    keys = sorted(queues)
    while any(queues.values()):
        for key in keys:
            if queues[key]:
                result.append(queues[key].pop(0))
    return result


def build_exposure_schedule(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 1,
    seed: int = 20260921,
    admit_inventory: bool = True,
) -> list[dict[str, Any]]:
    """Create a deterministic interleaved *planned* exposure schedule.

    Each schedule entry is one explicit training exposure.  The schedule
    never implicitly repeats a row; callers must request ``epochs``.  The
    returned entries retain source identity and ``exposure_number``.  This is
    a plan before trainer admission; actual exposure must come from trainer
    receipts and is kept as a separate field by :func:`build`.
    """
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    train = [
        row
        for row in rows
        if row.get("split") == "train"
        and (admit_inventory or row.get("reference_kind") != "real_reference_image")
    ]
    schedule: list[dict[str, Any]] = []
    for epoch in range(epochs):
        for position, row in enumerate(_interleaved_rows(train, seed=seed, epoch=epoch)):
            schedule.append(
                {
                    "schedule_index": len(schedule),
                    "epoch": epoch,
                    "position_in_epoch": position,
                    "id": row["id"],
                    "source_group": row["source_group"],
                    "family": row.get("family", "unknown"),
                    "difficulty": row.get("difficulty", 0),
                    "composition": row.get("composition", "unknown"),
                    "reference_kind": row.get("reference_kind"),
                    "exposure_number": epoch + 1,
                }
            )
    return schedule


def _compact_hash_summary(rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_hashes = sorted(str(row["source_sha256"]) for row in rows if row.get("source_sha256"))
    image_hashes = sorted(str(row["image_sha256"]) for row in rows if row.get("image_sha256"))
    schedule_ids = [str(row["id"]) for row in schedule]
    return {
        "row_count": len(rows),
        "source_program_hashes": len(source_hashes),
        "source_program_hash_digest": sha256_text("\n".join(source_hashes)),
        "image_hashes_available": len(image_hashes),
        "image_hash_digest": sha256_text("\n".join(image_hashes)) if image_hashes else None,
        "schedule_exposures": len(schedule),
        "schedule_id_digest": sha256_text("\n".join(schedule_ids)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _absolute_preserving_symlink(path: str | Path) -> Path:
    """Make a path absolute without dereferencing a venv interpreter link."""
    return Path(path).expanduser().absolute()


def materialize_programs(
    rows: Sequence[Mapping[str, Any]],
    program_root: str | Path,
    *,
    shard_count: int = 32,
) -> dict[str, Any]:
    """Write procedural programs outside Git in reusable renderer shards."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    root = Path(program_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # The existing renderer is invoked as the unprivileged ``painter`` user.
    # Make only this generated node-data tree writable/traversable by it.
    os.chmod(root, 0o777)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    os.chmod(root / "shards", 0o777)
    procedural = [row for row in rows if row.get("programs")]
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    materialized_paths: dict[str, list[str]] = {}
    files_written = 0
    files_reused = 0
    for row in procedural:
        shard_index = _stable_int(str(row["id"])) % shard_count
        shard_dir = root / "shards" / f"shard-{shard_index:03d}"
        case_dir = shard_dir / str(row["id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(shard_dir, 0o777)
        os.chmod(case_dir, 0o777)
        program_paths: list[str] = []
        for stage, source in enumerate(row["programs"], 1):
            target = case_dir / f"turn-{stage}.js"
            digest = sha256_text(str(source))
            if target.exists() and sha256_bytes(target.read_bytes()) == digest:
                files_reused += 1
            else:
                target.write_text(str(source), encoding="utf-8")
                files_written += 1
            os.chmod(target, 0o644)
            program_paths.append(str(target.relative_to(shard_dir)))
        shard_row = {
            **{key: value for key, value in row.items() if key != "programs"},
            "programs": program_paths,
            "plans": [
                f"Establish the complete {row.get('composition', 'single')} composition with {', '.join(row.get('member_families', [row.get('family', 'reference')]))}.",
                "Develop connected parts, overlap and proportions while preserving the reference placement.",
                "Finish the complete reference faithfully with restrained detail and no extra objects.",
            ],
            "reference": str(Path(program_paths[-1]).with_suffix(".png")),
            "teacher": "existing expanded direct-canvas recipe; render and review on node",
        }
        shards[shard_index].append(shard_row)
        materialized_paths[str(row["id"])] = program_paths
    shard_records = []
    for index, shard_rows in enumerate(shards):
        shard_dir = root / "shards" / f"shard-{index:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(shard_dir, 0o777)
        manifest = shard_dir / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in sorted(shard_rows, key=lambda row: row["id"])),
            encoding="utf-8",
        )
        shard_records.append({"shard": index, "rows": len(shard_rows), "path": str(shard_dir)})
    # A top-level procedural manifest makes the same data consumable by
    # render_demos.py for a single-shard smoke test or a future node wrapper.
    top_manifest = root / "manifest.jsonl"
    top_manifest.write_text(
        "".join(
            json.dumps(
                {
                    **{key: value for key, value in row.items() if key != "programs"},
                    "programs": [
                        str(Path("shards") / f"shard-{_stable_int(str(row['id'])) % shard_count:03d}" / path)
                        for path in materialized_paths[str(row["id"])]
                    ],
                    "plans": [
                        f"Establish the complete {row.get('composition', 'single')} composition with {', '.join(row.get('member_families', [row.get('family', 'reference')]))}.",
                        "Develop connected parts, overlap and proportions while preserving the reference placement.",
                        "Finish the complete reference faithfully with restrained detail and no extra objects.",
                    ],
                    "reference": str(
                        Path("shards")
                        / f"shard-{_stable_int(str(row['id'])) % shard_count:03d}"
                        / Path(materialized_paths[str(row["id"])][-1]).with_suffix(".png")
                    ),
                    "teacher": "existing expanded direct-canvas recipe; render and review on node",
                },
                separators=(",", ":"),
            )
            + "\n"
            for row in sorted(procedural, key=lambda row: row["id"])
        ),
        encoding="utf-8",
    )
    # Keep a metadata-only manifest for downstream inventory joins.  Imported
    # real references are retained here even though they have no JS to render;
    # procedural source text stays in the external shard files.
    reference_manifest = root / "reference-manifest.jsonl"
    reference_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["id"])):
        compact = {key: value for key, value in row.items() if key != "programs"}
        if str(row["id"]) in materialized_paths:
            shard_name = f"shard-{_stable_int(str(row['id'])) % shard_count:03d}"
            compact["programs"] = [
                str(Path("shards") / shard_name / path)
                for path in materialized_paths[str(row["id"])]
            ]
            compact["reference_png"] = str(
                Path("shards") / shard_name / Path(materialized_paths[str(row["id"])][-1]).with_suffix(".png")
            )
        else:
            compact["programs"] = []
        reference_rows.append(compact)
    reference_manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in reference_rows),
        encoding="utf-8",
    )
    os.chmod(reference_manifest, 0o644)
    receipt = {
        "run_id": RUN_ID,
        "root": str(root),
        "procedural_rows": len(procedural),
        "shard_count": shard_count,
        "shards": shard_records,
        "files_written": files_written,
        "files_reused": files_reused,
        "renderer": "painter/render_demos.py",
        "reference_manifest": str(reference_manifest),
        "rendering_performed": False,
        "resume_policy": "reuse source files whose SHA-256 matches; render_demos.py reuses valid PNG/receipt pairs",
    }
    _write_json(root / "materialization-receipt.json", receipt)
    return receipt


def render_reference_shards(
    program_root: str | Path,
    renderer: str | Path,
    renderer_python: str | Path,
    *,
    workers: int = 4,
    browsers_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render only final references on Linux, reusing existing renderer rules.

    This is intentionally a bounded materialization helper, not a training or
    watcher framework.  It renders ``turn-3.js`` from each shard, preserves
    valid source/PNG receipts, and writes progress after completed batches so
    a later invocation can resume.  It does not create SFT examples.
    """
    if sys.platform != "linux":
        raise RuntimeError("Reference rendering runs on the Linux node, not the controller")
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(program_root).expanduser().resolve()
    # ``renderer-env/bin/python`` is commonly a symlink into the venv.  Keep
    # that spelling: resolve() would silently switch to system Python and lose
    # the node's installed Playwright runtime.
    renderer = _absolute_preserving_symlink(renderer)
    renderer_python = _absolute_preserving_symlink(renderer_python)
    shard_dirs = sorted(path for path in (root / "shards").glob("shard-*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No materialized shards under {root / 'shards'}")
    jobs: list[tuple[Path, dict[str, Any]]] = []
    for shard_dir in shard_dirs:
        manifest = shard_dir / "manifest.jsonl"
        for row in _jsonl(manifest):
            if not row.get("programs"):
                continue
            jobs.append((shard_dir, row))
    # Queue coverage is deliberate: the first 60 jobs cover all 12 families
    # and five composition labels when the standard corpus is used.  This lets
    # the parent spot-check a representative prefix while the finite job keeps
    # running to completion.
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for job in jobs:
        grouped[(str(job[1].get("family", "unknown")), str(job[1].get("composition", "unknown")))].append(job)
    for queue in grouped.values():
        queue.sort(key=lambda item: str(item[1]["id"]))
    ordered_jobs: list[tuple[Path, dict[str, Any]]] = []
    # The first pass takes one row from every family/composition bucket;
    # subsequent passes continue round-robin without exhausting a bucket.
    while any(grouped.values()):
        for key in sorted(grouped):
            if grouped[key]:
                ordered_jobs.append(grouped[key].pop(0))
    jobs = ordered_jobs
    result_path = root / "reference-results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if result_path.exists():
        try:
            for row in _jsonl(result_path):
                if row.get("source"):
                    existing[str(row["source"])] = row
        except ValueError:
            # Receipts beside each source remain authoritative if a process
            # stopped while the compact append file was being replaced.
            existing = {}
    result_map = dict(existing)

    def one(job: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
        shard_dir, row = job
        source = shard_dir / row["programs"][-1]
        png = source.with_suffix(".png")
        receipt = source.with_suffix(".json")
        source_digest = sha256_bytes(source.read_bytes())
        key = str(source.relative_to(root))
        if png.exists() and receipt.exists():
            try:
                previous = json.loads(receipt.read_text(encoding="utf-8"))
                png_digest = sha256_bytes(png.read_bytes())
                if (
                    previous.get("valid")
                    and previous.get("source_sha256") == source_digest
                    and previous.get("png_sha256") == png_digest
                ):
                    return {
                        "id": row["id"],
                        "source": key,
                        "reference_png": str(png.relative_to(root)),
                        "source_sha256": source_digest,
                        "image_sha256": png_digest,
                        "valid": True,
                        "resumed": True,
                    }
            except (OSError, json.JSONDecodeError):
                pass
        for path in (png, receipt):
            if path.exists():
                path.rename(path.with_name(path.name + f".previous-{time.time_ns()}"))
        try:
            environment = os.environ.copy()
            if browsers_path is not None:
                environment["PLAYWRIGHT_BROWSERS_PATH"] = str(Path(browsers_path).expanduser().resolve())
            job_result = subprocess.run(
                [
                    "/usr/sbin/runuser",
                    "-u",
                    "painter",
                    "--",
                    str(renderer_python),
                    str(renderer),
                    str(source),
                    "--output",
                    str(png),
                    "--backend",
                    "swiftshader",
                    "--timeout",
                    "120",
                ],
                capture_output=True,
                text=True,
                timeout=140,
                env=environment,
            )
            rendered = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
            image_digest = sha256_bytes(png.read_bytes()) if png.exists() else None
            valid = bool(
                job_result.returncode == 0
                and rendered.get("valid")
                and png.exists()
                and rendered.get("source_sha256") == source_digest
                and rendered.get("png_sha256") == image_digest
            )
            return {
                "id": row["id"],
                "source": key,
                "reference_png": str(png.relative_to(root)) if valid else None,
                "source_sha256": source_digest,
                "image_sha256": image_digest,
                "valid": valid,
                "resumed": False,
                "errors": rendered.get("errors", [rendered.get("error")] if rendered.get("error") else []) if not valid else [],
                "returncode": job_result.returncode,
                "stderr_tail": job_result.stderr[-1000:] if not valid else None,
            }
        except Exception as exc:  # keep neighboring jobs resumable
            return {
                "id": row["id"],
                "source": key,
                "reference_png": None,
                "source_sha256": source_digest,
                "image_sha256": None,
                "valid": False,
                "resumed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def save_progress() -> None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temp = result_path.with_suffix(result_path.suffix + ".tmp")
        temp.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in sorted(result_map.values(), key=lambda row: row["id"])),
            encoding="utf-8",
        )
        temp.replace(result_path)
        _write_json(
            root / "reference-progress.json",
            {
                "expected": len(jobs),
                "completed": len(result_map),
                "valid": sum(bool(row.get("valid")) for row in result_map.values()),
                "failed": sum(not bool(row.get("valid")) for row in result_map.values()),
                "renderer": str(renderer),
                "rendered_final_references_only": True,
            },
        )

    # Revisit every source: one() returns immediately for a valid receipt and
    # retries an earlier failure.  The on-disk result map prevents duplicates.
    pending = jobs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, job) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            result_map[str(result["source"])] = result
            if len(result_map) % 16 == 0:
                save_progress()
    save_progress()
    summary = {
        "expected": len(jobs),
        "completed": len(result_map),
        "valid": sum(bool(row.get("valid")) for row in result_map.values()),
        "failed": sum(not bool(row.get("valid")) for row in result_map.values()),
        "resumed": sum(bool(row.get("resumed")) for row in result_map.values()),
        "image_hashes": sum(bool(row.get("image_sha256")) for row in result_map.values()),
        "rendered_final_references_only": True,
        "training_or_inference_run": False,
    }
    _write_json(root / "reference-summary.json", summary)
    return summary


def build(
    output: str | Path,
    *,
    program_root: str | Path | None = None,
    inventory: str | Path | Iterable[Mapping[str, Any]] | None = None,
    train_per_family: int = TRAIN_PER_FAMILY,
    dev_per_family: int = DEV_PER_FAMILY,
    epochs: int = 1,
    shard_count: int = 32,
    seed: int = 20260921,
    admit_inventory: bool = True,
) -> dict[str, Any]:
    """Build the compact spec and optionally materialize external JS shards."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    procedural = generate_procedural_records(train_per_family, dev_per_family, seed=seed)
    all_rows = list(procedural)
    inventory_stats: dict[str, int] = {}
    if inventory is not None:
        imported, inventory_stats = import_inventory_rows(
            inventory,
            seed=seed,
            seen_source_groups=(row["source_group"] for row in all_rows),
        )
        all_rows.extend(imported)
    train_rows = [row for row in all_rows if row.get("split") == "train"]
    dev_rows = [row for row in all_rows if row.get("split") == "dev"]
    train_groups = {row["source_group"] for row in train_rows}
    dev_groups = {row["source_group"] for row in dev_rows}
    if train_groups & dev_groups:
        raise ValueError(f"Source group appears in both splits: {sorted(train_groups & dev_groups)[:3]}")
    source_hashes = [row["source_sha256"] for row in procedural]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("Procedural source program hashes are not unique")
    schedule = build_exposure_schedule(
        all_rows, epochs=epochs, seed=seed, admit_inventory=admit_inventory
    )
    for row in all_rows:
        row["planned_training_exposure"] = sum(item["id"] == row["id"] for item in schedule)
        # There is no trainer receipt at preparation time.  Do not turn a
        # planned schedule into a claim about admitted training exposure.
        row["admitted_training_exposure"] = 0
    receipt: dict[str, Any] | None = None
    if program_root is not None:
        receipt = materialize_programs(all_rows, program_root, shard_count=shard_count)
    schedule_summary = {
        "epochs": epochs,
        "admit_inventory": admit_inventory,
        "exposures": len(schedule),
        "source_groups_with_exposure": len({row["source_group"] for row in schedule}),
        "exposure_by_reference_kind": dict(Counter(row["reference_kind"] for row in schedule)),
        "exposure_by_split": {"train": len(schedule), "dev": 0},
        "exposure_by_difficulty": dict(Counter(str(row["difficulty"]) for row in schedule)),
        "exposure_id_digest": sha256_text("\n".join(row["id"] for row in schedule)),
        "schedule_path": "external/schedule.jsonl",
    }
    summary = {
        "run_id": RUN_ID,
        "status": "prepared_unrendered",
        "hypothesis": "A substantially larger, source-disjoint curriculum with interleaved families and compositions will support a measurable quality comparison after node rendering and review.",
        "matched_baseline": "The existing 12-family expanded direct-canvas curriculum and the current contract; no new model or renderer result is claimed here.",
        "targets": {"procedural_train": TRAIN_TARGET, "procedural_dev": DEV_TARGET},
        "procedural": {
            "train": sum(row["split"] == "train" for row in procedural),
            "dev": sum(row["split"] == "dev" for row in procedural),
            "families": len(FAMILIES),
            "family_counts": dict(Counter(row["family"] for row in procedural)),
            "difficulty_counts": dict(Counter(str(row["difficulty"]) for row in procedural)),
            "composition_counts": dict(Counter(row["composition"] for row in procedural)),
            "source_program_hashes": len(set(source_hashes)),
            "image_hashes_available": 0,
        },
        "inventory": inventory_stats,
        "planned_training_exposure": {
            "references": len(train_rows) if admit_inventory else sum(row["split"] == "train" for row in procedural),
            "exposures": len(schedule),
            "epochs": epochs,
            "inventory_included": admit_inventory,
        },
        "admitted_training_exposure": {
            "references": 0,
            "exposures": 0,
            "status": "awaiting trainer admission receipt",
        },
        "source_group_overlap": sorted(train_groups & dev_groups),
        "hash_summary": _compact_hash_summary(all_rows, schedule),
        "materialization": receipt,
        "limitations": [
            "No procedural image hash exists until the node renders the programs; source SHA-256 is not an image identity.",
            "Composition rows are direct-coordinate procedural programs, not evidence of distinct visual images until rendered and reviewed.",
            "Imported real references retain their inventory path/URI and provenance; this builder does not copy or inspect image bytes.",
            "No model inference, SFT, visual judging, GPU utilization, cost, or quality result was run by this builder.",
        ],
    }
    # The parent run may own EXPERIMENT.json at the selected output path;
    # preparation metadata must remain an isolated record.
    _write_json(output / "PREPARATION.json", summary)
    _write_json(output / "hash-summary.json", summary["hash_summary"])
    _write_json(output / "schedule-summary.json", schedule_summary)
    _write_json(output / "reference-counts.json", {
        "procedural_train": summary["procedural"]["train"],
        "procedural_dev": summary["procedural"]["dev"],
        "inventory_train": sum(row["split"] == "train" for row in all_rows if row.get("reference_kind") == "real_reference_image"),
        "inventory_dev": sum(row["split"] == "dev" for row in all_rows if row.get("reference_kind") == "real_reference_image"),
        "planned_training_exposures": len(schedule),
        "admitted_training_exposures": 0,
    })
    spec = (
        f"# Scaled curriculum: {RUN_ID}\n\n"
        f"This record prepares {summary['procedural']['train']} procedural training references and "
        f"{summary['procedural']['dev']} procedural development references across the existing "
        f"{len(FAMILIES)} families. It performs no rendering or model inference.\n\n"
        "Procedural teacher programs use the existing expanded direct canvas-coordinate recipes. "
        "`source_sha256` identifies source text; rendered image hashes are pending until the Linux node. "
        "Rows imported from the inventory agent use exact `image_sha256` values when supplied and are "
        "deduplicated on that hash.\n\n"
        "## Node commands\n\n"
        "```bash\n"
        "# On the controller, generate JS under an external writable directory:\n"
        "python3 painter/scaled_curriculum.py build \\\n"
        "  --output painter/runs/scaled-curriculum-20260921 \\\n"
        "  --program-root /data/painter/scaled-curriculum-20260921/programs \\\n"
        "  --epochs 1 --shards 32\n\n"
        "# On Linux, materialize final references only; the queue interleaves\n"
        "# all families/compositions and resumes valid source/PNG receipts:\n"
        "python3 painter/scaled_curriculum.py render-references \\\n"
        "  --program-root /data/painter/scaled-curriculum-20260921/programs \\\n"
        "  --renderer painter/vendor/integrations/watercolour/renderer.py \\\n"
        "  --python /data/painter/renderer-env/bin/python \\\n"
        "  --browsers-path /data/painter/browsers --workers 4\n"
        "```\n\n"
        "The helper reuses the renderer invocation and receipt checks from `render_demos.py`, "
        "but only renders final references and does not export SFT. The generated top-level manifest "
        "is for a smoke run; shard manifests are the resumable unit.\n"
    )
    (output / "DATASET-SPEC.md").write_text(spec, encoding="utf-8")
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="prepare the scaled curriculum")
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--program-root", default=None, help="external directory for generated JS")
    build_parser.add_argument("--inventory", default=None, help="inventory JSON/JSONL from the inventory agent")
    build_parser.add_argument("--train-per-family", type=int, default=TRAIN_PER_FAMILY)
    build_parser.add_argument("--dev-per-family", type=int, default=DEV_PER_FAMILY)
    build_parser.add_argument("--epochs", type=int, default=1)
    build_parser.add_argument("--shards", type=int, default=32)
    build_parser.add_argument("--seed", type=int, default=20260921)
    build_parser.add_argument("--exclude-inventory-from-schedule", action="store_true")
    render_parser = sub.add_parser("render-references", help="render final references on a Linux node")
    render_parser.add_argument("--program-root", required=True)
    render_parser.add_argument("--renderer", required=True)
    render_parser.add_argument("--python", dest="renderer_python", required=True)
    render_parser.add_argument("--workers", type=int, default=4)
    render_parser.add_argument("--browsers-path", default=None)
    args = parser.parse_args()
    if args.command == "build":
        result = build(
            args.output,
            program_root=args.program_root,
            inventory=args.inventory,
            train_per_family=args.train_per_family,
            dev_per_family=args.dev_per_family,
            epochs=args.epochs,
            shard_count=args.shards,
            seed=args.seed,
            admit_inventory=not args.exclude_inventory_from_schedule,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "render-references":
        result = render_reference_shards(
            args.program_root,
            args.renderer,
            args.renderer_python,
            workers=args.workers,
            browsers_path=args.browsers_path,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
