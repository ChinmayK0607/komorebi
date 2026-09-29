"""Select a finite GPU-budget subset from the published 290-brief pool.

The deterministic ranking never looks at model outputs. All foundation,
validation, and reserve cases stay in; only complex training cases are sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SEED = "scaled-text-baseline-20260929"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(pool: Path, output: Path, complex_train: int = 112) -> dict:
    source = json.loads(pool.read_text())
    rows = source["cases"]
    if source.get("schema") != "painter-expanded-text-pool-v3" or len(rows) != 290:
        raise ValueError("expected the complete 290-brief v3 source pool")
    if output.exists():
        raise FileExistsError(output)
    candidates = [row for row in rows if row["split"] == "train" and row["family"] == "complex"]
    if not 64 <= complex_train <= len(candidates):
        raise ValueError("complex training count must be between 64 and inventory size")
    ranked = sorted(candidates, key=lambda row: hashlib.sha256(
        f"{SEED}:{row['id']}".encode()).hexdigest())
    selected = {row["id"] for row in ranked[:complex_train]}
    kept = [row for row in rows if row["family"] != "complex" or
            row["split"] != "train" or row["id"] in selected]
    output.mkdir(parents=True)
    for row in kept:
        src = pool.parent / row["reference"]
        if sha(src) != row["reference_sha256"]:
            raise ValueError(f"prompt mismatch: {row['id']}")
        dest = output / row["reference"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    manifest = output / "pool.json"
    manifest.write_text(json.dumps({"schema": "painter-scaled-text-run-pool-v1",
                                    "source_pool_sha256": sha(pool), "selection_seed": SEED,
                                    "complex_train_requested": complex_train,
                                    "cases": kept}, indent=2, sort_keys=True) + "\n")
    return {"count": len(kept), "pool_sha256": sha(manifest),
            "train": sum(row["split"] == "train" for row in kept),
            "validation": sum(row["split"] == "validation" for row in kept),
            "reserve": sum(row["split"] == "reserve" for row in kept)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complex-train", type=int, default=112)
    args = parser.parse_args()
    print(json.dumps(select(args.pool, args.output, args.complex_train), sort_keys=True))


if __name__ == "__main__":
    main()
