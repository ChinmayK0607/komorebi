"""Prepare a content-addressed, role-bound Luna identity diagnostic.

The earlier validation copied every role into every blinded view.  This
preparation path keeps one operational record for each unique PNG and binds
the opaque ``reference``, ``A`` and ``B`` aliases to those records.  A viewer
can therefore display each asset once and apply the aliases from ``rolemap``
afterwards.  The module only reads existing PNGs; it never renders or edits
them and it does not install rewards into the live queue.

The generated public files contain opaque view/ordinal aliases and asset IDs,
but no source pair IDs, orientations, or self-reported image paths.  The
private bindings file records hashes and orientation for audit and hash
verification.  Exact-identity controls are written to a separate manifest and
are never assigned to a visual judge shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any

ROLES = ("reference", "A", "B")
JUDGES = ("judge-01", "judge-02")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GENERATED = (
    "assets.json", "pairs.json", "deterministic-controls.json", "bindings.json",
    "judge-01.json", "judge-02.json",
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(value: str, base: Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if path.suffix.lower() != ".png" or not path.is_file():
        raise ValueError(f"expected an existing PNG: {value}")
    if path.read_bytes()[:8] != PNG_MAGIC:
        raise ValueError(f"expected a PNG signature: {value}")
    return path


def _selection(selection_path: Path, require_real_count: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if set(raw) != {"controls", "real_pairs"}:
        raise ValueError("selection must contain exactly controls and real_pairs")
    controls, real_pairs = [], []
    seen: set[str] = set()
    for key, out, is_control in (("controls", controls, True), ("real_pairs", real_pairs, False)):
        if not isinstance(raw[key], list):
            raise ValueError(f"{key} must be a list")
        for item in raw[key]:
            required = {"id", "reference", "A", "B"}
            if is_control:
                required.add("expected_winner")
            if set(item) != required:
                raise ValueError(f"{key} item has unexpected fields: {item.get('id')!r}")
            ident = item["id"]
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f"duplicate or invalid pair id: {ident!r}")
            if is_control and item["expected_winner"] not in {"A", "B", "tie"}:
                raise ValueError(f"invalid expected winner for {ident}")
            seen.add(ident)
            out.append({
                "id": ident,
                "sources": {role: _source(item[role], selection_path.parent) for role in ROLES},
                **({"expected_winner": item["expected_winner"]} if is_control else {}),
            })
    if require_real_count is not None and len(real_pairs) != require_real_count:
        raise ValueError(f"expected {require_real_count} real pairs, found {len(real_pairs)}")
    if not real_pairs:
        raise ValueError("selection has no real pairs")
    return controls, real_pairs


def _asset_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    """Return one asset record per byte hash and role-source lookup maps."""
    paths_by_hash: dict[str, set[str]] = {}
    for row in rows:
        for path in row["sources"].values():
            digest = sha(path)
            paths_by_hash.setdefault(digest, set()).add(str(path))
    hashes = sorted(paths_by_hash)
    hash_to_id = {digest: f"asset-{number:03d}" for number, digest in enumerate(hashes, 1)}
    assets = []
    for digest in hashes:
        paths = sorted(paths_by_hash[digest])
        canonical = Path(paths[0])
        assets.append({
            "asset_id": hash_to_id[digest],
            "bytes": canonical.stat().st_size,
            "path": str(canonical),
            "sha256": digest,
        })
    return assets, hash_to_id, {digest: paths for digest, paths in paths_by_hash.items()}


def _rolemap(row: dict[str, Any], hash_to_id: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    rolemap, hashes = {}, {}
    for role in ROLES:
        digest = sha(row["sources"][role])
        rolemap[role] = hash_to_id[digest]
        hashes[role] = digest
    return rolemap, hashes


def _deterministic_control(control: dict[str, Any], rolemap: dict[str, str], hashes: dict[str, str], ordinal: str) -> dict[str, Any]:
    a_equal_b = hashes["A"] == hashes["B"]
    reference_a = hashes["reference"] == hashes["A"]
    reference_b = hashes["reference"] == hashes["B"]
    if reference_a and reference_b:
        winner = "tie"
        classification = "both_candidates_exact_reference"
    elif a_equal_b:
        winner = "tie"
        classification = "identical_candidates_tie"
    elif reference_a and not reference_b:
        winner = "A"
        classification = "exact_reference_match_A"
    elif reference_b and not reference_a:
        winner = "B"
        classification = "exact_reference_match_B"
    else:
        raise ValueError(f"control {control['id']} has no deterministic identity result")
    if winner != control["expected_winner"]:
        raise ValueError(
            f"control {control['id']} expected {control['expected_winner']}, identity data resolves {winner}"
        )
    return {
        "id": ordinal,
        "ordinal": ordinal,
        "rolemap": rolemap,
        "asset_ids": list(dict.fromkeys(rolemap[role] for role in ROLES)),
        "deterministic": {
            "winner": winner,
            "classification": classification,
            "candidate_equal": a_equal_b,
            "reference_equal_A": reference_a,
            "reference_equal_B": reference_b,
        },
        "visual_judging": "excluded_deterministic_control",
    }


def _view_id(seed: int, ordinal: str, orientation: str) -> str:
    return "view-" + hashlib.sha256(f"{seed}:{ordinal}:{orientation}".encode()).hexdigest()[:16]


def prepare(selection: str | Path, output: str | Path | None = None, *, seed: int = 20260920,
            require_real_count: int | None = 6) -> dict[str, Any]:
    """Prepare assets, two orientation shards and deterministic controls."""
    selection_path = Path(selection).resolve()
    output_path = Path(output or Path(__file__).parent).resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    if any((output_path / name).exists() for name in GENERATED):
        raise FileExistsError("prepared diagnostic already exists; preserve it")
    controls, real_pairs = _selection(selection_path, require_real_count)
    all_rows = controls + real_pairs
    assets, hash_to_id, _ = _asset_table(all_rows)
    asset_lookup = {asset["asset_id"]: asset for asset in assets}

    rng = random.Random(seed)
    # Ordinals are stable and intentionally unrelated to source IDs.  The
    # order of each public shard is shuffled after both views are generated.
    real_order = sorted(real_pairs, key=lambda row: row["id"])
    views_by_judge = {judge: [] for judge in JUDGES}
    public_pairs: list[dict[str, Any]] = []
    private: dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "source_selection": str(selection_path),
        "selection_sha256": sha(selection_path),
        "real_pairs": {},
        "controls": {},
    }

    for number, row in enumerate(real_order, 1):
        ordinal = f"ordinal-{number:03d}"
        canonical_rolemap, source_hashes = _rolemap(row, hash_to_id)
        private_pair: dict[str, Any] = {
            "source_id": row["id"],
            "ordinal": ordinal,
            "source_hashes": source_hashes,
            "canonical_rolemap": canonical_rolemap,
            "views": {},
        }
        first_orientation = "ba" if rng.randrange(2) else "ab"
        orientations = (first_orientation, "ab" if first_orientation == "ba" else "ba")
        for index, (judge, orientation) in enumerate(zip(JUDGES, orientations), 1):
            rolemap = dict(canonical_rolemap)
            if orientation == "ba":
                rolemap["A"], rolemap["B"] = rolemap["B"], rolemap["A"]
            view_id = _view_id(seed, ordinal, orientation)
            asset_ids = list(dict.fromkeys(rolemap[role] for role in ROLES))
            item = {
                "id": view_id,
                "ordinal": ordinal,
                "rolemap": rolemap,
                "asset_ids": asset_ids,
                "display_order": list(ROLES),
            }
            views_by_judge[judge].append(item)
            private_pair["views"][view_id] = {
                "judge": judge,
                "orientation": orientation,
                "rolemap": rolemap,
                "asset_hashes": {role: asset_lookup[rolemap[role]]["sha256"] for role in ROLES},
            }
            public_pairs.append(item)
        private["real_pairs"][row["id"]] = private_pair

    deterministic_controls = []
    for number, control in enumerate(sorted(controls, key=lambda row: row["id"]), 1):
        ordinal = f"control-{number:03d}"
        rolemap, hashes = _rolemap(control, hash_to_id)
        result = _deterministic_control(control, rolemap, hashes, ordinal)
        deterministic_controls.append(result)
        private["controls"][control["id"]] = {
            "ordinal": ordinal,
            "source_hashes": hashes,
            "rolemap": rolemap,
            "expected_winner": control["expected_winner"],
            "deterministic": result["deterministic"],
        }

    for manifest in views_by_judge.values():
        rng.shuffle(manifest)
    output_path.mkdir(parents=True, exist_ok=True)
    _write(output_path / "assets.json", {
        "version": 1,
        "content_address": "sha256",
        "assets": assets,
    })
    _write(output_path / "pairs.json", public_pairs)
    _write(output_path / "deterministic-controls.json", {
        "version": 1,
        "visual_judging": "excluded",
        "controls": deterministic_controls,
    })
    _write(output_path / "bindings.json", private)
    for judge, manifest in views_by_judge.items():
        _write(output_path / f"{judge}.json", manifest)
    return {
        "output": str(output_path),
        "real_pairs": len(real_pairs),
        "views": len(public_pairs),
        "judges": list(JUDGES),
        "controls": len(controls),
        "unique_assets": len(assets),
        "seed": seed,
    }


def import_manifest(assets_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Validate an opaque shard against its unique asset table.

    This is intentionally an import/verification operation: it does not copy
    images and it never trusts a label's claimed paths as proof of inspection.
    """
    assets_doc = json.loads(Path(assets_path).read_text(encoding="utf-8"))
    assets = {row["asset_id"]: row for row in assets_doc.get("assets", [])}
    if len(assets) != len(assets_doc.get("assets", [])):
        raise ValueError("asset IDs are not unique")
    items = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("judge manifest must be a list")
    seen_ids: set[str] = set()
    for item in items:
        if set(item) != {"id", "ordinal", "rolemap", "asset_ids", "display_order"}:
            raise ValueError(f"unexpected public fields for {item.get('id')}")
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate view ID: {item['id']}")
        seen_ids.add(item["id"])
        if item["display_order"] != list(ROLES):
            raise ValueError(f"invalid display order for {item['id']}")
        rolemap = item["rolemap"]
        if set(rolemap) != set(ROLES):
            raise ValueError(f"role bijection missing a role for {item['id']}")
        expected_assets = list(dict.fromkeys(rolemap[role] for role in ROLES))
        if item["asset_ids"] != expected_assets:
            raise ValueError(f"asset IDs do not preserve first role order for {item['id']}")
        if any(asset_id not in assets for asset_id in expected_assets):
            raise ValueError(f"unknown asset ID for {item['id']}")
        for asset_id in expected_assets:
            asset = assets[asset_id]
            path = Path(asset["path"])
            if not path.is_file() or sha(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                raise ValueError(f"asset hash/size mismatch: {asset_id}")
    return {"manifest": str(Path(manifest_path).resolve()), "views": len(items), "assets": len(assets)}


def resolve_manifest(assets_path: str | Path, manifest_path: str | Path) -> list[dict[str, Any]]:
    """Resolve IDs for an operator's display loop, validating hashes first."""
    import_manifest(assets_path, manifest_path)
    assets_doc = json.loads(Path(assets_path).read_text(encoding="utf-8"))
    assets = {row["asset_id"]: row for row in assets_doc["assets"]}
    items = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    resolved = []
    for item in items:
        display = [
            {"asset_id": asset_id, "path": assets[asset_id]["path"],
             "sha256": assets[asset_id]["sha256"]}
            for asset_id in item["asset_ids"]
        ]
        resolved.append({"id": item["id"], "ordinal": item["ordinal"],
                         "rolemap": item["rolemap"], "display": display})
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare", help="prepare the two fresh-judge shards")
    make.add_argument("selection", type=Path)
    make.add_argument("--output", type=Path, default=Path(__file__).parent)
    make.add_argument("--seed", type=int, default=20260920)
    make.add_argument("--real-count", type=int, default=6)
    verify = sub.add_parser("import", help="verify an opaque shard and its content hashes")
    verify.add_argument("--assets", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    resolve = sub.add_parser("resolve", help="print one path per unique asset for an operator")
    resolve.add_argument("--assets", type=Path, required=True)
    resolve.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.selection, args.output, seed=args.seed, require_real_count=args.real_count)
    elif args.command == "import":
        result = import_manifest(args.assets, args.manifest)
    else:
        result = resolve_manifest(args.assets, args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
