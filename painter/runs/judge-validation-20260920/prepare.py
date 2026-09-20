"""Prepare opaque, exact PNG views for a three-shard judge validation run.

This script only copies bytes.  It never opens, rescales, concatenates, or
otherwise transforms an input image.  The private key is the sole place where
view orientation and control answers are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

JUDGES = ("judge-01", "judge-02", "judge-03")
PNG = b"\x89PNG\r\n\x1a\n"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: str, base: Path) -> Path:
    value = Path(path)
    value = value if value.is_absolute() else base / value
    value = value.resolve()
    if value.suffix.lower() != ".png" or not value.is_file() or value.read_bytes()[:8] != PNG:
        raise ValueError(f"expected an existing PNG: {path}")
    return value


def _swap(orientation: str) -> str:
    return "ba" if orientation == "ab" else "ab"


def prepare(selection_path: str | Path, output: str | Path | None = None,
            seed: int = 20260920) -> dict:
    selection_path = Path(selection_path).resolve()
    output = Path(output or Path(__file__).parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    if (output / "artifacts").exists() or any((output / f"{j}.json").exists() for j in JUDGES):
        raise FileExistsError("prepared artifacts or judge manifests already exist; preserve the run")
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if set(raw) != {"controls", "real_pairs"}:
        raise ValueError("selection must contain exactly controls and real_pairs")
    rows = []
    answers = {}
    seen = set()
    for kind, key in (("control", "controls"), ("real", "real_pairs")):
        if not isinstance(raw[key], list):
            raise ValueError(f"{key} must be a list")
        for item in raw[key]:
            required = {"id", "reference", "A", "B"} | ({"expected_winner"} if kind == "control" else set())
            if set(item) != required:
                raise ValueError(f"{kind} {item.get('id')} has unexpected selection fields")
            ident = item["id"]
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError(f"duplicate or invalid id: {ident!r}")
            if kind == "control" and item["expected_winner"] not in {"A", "B", "tie"}:
                raise ValueError(f"invalid expected winner for {ident}")
            seen.add(ident)
            source = {side: _source(item[side], selection_path.parent) for side in ("reference", "A", "B")}
            rows.append({"id": ident, "kind": kind, "source": source})
            if kind == "control":
                answers[ident] = {"expected_winner": item["expected_winner"]}
    if not rows:
        raise ValueError("selection has no pairs")

    order = list(rows)
    rng = random.Random(seed)
    rng.shuffle(order)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output))
    try:
        image_root = staging / "images"
        manifests = {judge: [] for judge in JUDGES}
        private = {"version": 1, "seed": seed, "controls": answers, "pairs": {}}
        # Each underlying pair contributes exactly two views.  The two views
        # go to different agents; with 12 pairs this gives 8 views per shard.
        for number, row in enumerate(order, 1):
            pair_dir = image_root / f"p{number:03d}"
            private_pair = {"kind": row["kind"], "source_hashes": {
                side: _sha(row["source"][side]) for side in ("reference", "A", "B")}, "views": {}}
            base = "ba" if rng.randrange(2) else "ab"
            first_judge = JUDGES[(number - 1) % len(JUDGES)]
            second_judge = JUDGES[number % len(JUDGES)]
            for view_number, (judge, orientation) in enumerate(
                    ((first_judge, base), (second_judge, _swap(base))), 1):
                # Opaque public IDs keep source IDs, control status, and
                # orientation out of the manifests and image paths.
                public_id = "view-" + hashlib.sha256(
                    f"{seed}:{number}:{view_number}".encode()).hexdigest()[:16]
                view_dir = pair_dir / public_id
                view_files, view_hashes = {}, {}
                for side in ("reference", "A", "B"):
                    canonical = side
                    if side in ("A", "B") and orientation == "ba":
                        canonical = "B" if side == "A" else "A"
                    destination = view_dir / f"{side}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(row["source"][canonical], destination)
                    # Record the final path, since the populated staging tree
                    # is atomically renamed to ``artifacts`` below.
                    view_files[side] = str((output / "artifacts" / "images" /
                                            f"p{number:03d}" / public_id /
                                            f"{side}.png").resolve())
                    view_hashes[side] = _sha(destination)
                private_pair["views"][public_id] = {
                    "judge": judge, "orientation": orientation,
                    "files": view_files, "hashes": view_hashes}
                # This is deliberately the complete public manifest item.  Do
                # not add kind, orientation, hashes, or source identities here.
                manifests[judge].append({"id": public_id, **view_files})
            private["pairs"][row["id"]] = private_pair
        for manifest in manifests.values():
            rng.shuffle(manifest)
        artifact_dir = output / "artifacts"
        staging_private = staging / "private.json"
        _write(staging_private, private)
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(artifact_dir)
        for judge, manifest in manifests.items():
            _write(output / f"{judge}.json", manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"pairs": len(rows), "views": len(rows) * 2, "controls": len(answers), "real_pairs": len(rows) - len(answers),
            "judges": list(JUDGES), "seed": seed, "artifacts": str((output / "artifacts").resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.output, args.seed), indent=2))
