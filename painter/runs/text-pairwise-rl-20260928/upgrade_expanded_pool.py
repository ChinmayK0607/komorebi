"""Upgrade the immutable expanded text pool to include scoring families.

All task IDs, splits, prompts, and hashes must remain unchanged. Writes a new
directory rather than mutating the public v1 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upgrade(v1: Path, original: Path, output: Path) -> dict:
    source = json.loads(v1.read_text())
    base = json.loads(original.read_text())
    if source.get("schema") != "painter-expanded-text-pool-v1":
        raise ValueError("expected v1 expanded pool")
    old = {row["id"]: row for row in base["cases"]}
    cases = source["cases"]
    if len(cases) != 290 or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("expanded pool count or IDs changed")
    if output.exists():
        raise FileExistsError(output)
    for row in cases:
        ident = row["id"]
        prompt = v1.parent / row["reference"]
        if not prompt.is_file() or digest(prompt) != row["reference_sha256"]:
            raise ValueError(f"missing or changed prompt: {ident}")
        if ident in old:
            previous = old[ident]
            if row["split"] != previous["split"] or row["phase"] != "foundation":
                raise ValueError(f"changed original split: {ident}")
            if row["reference_sha256"] != previous["reference_sha256"]:
                raise ValueError(f"changed original prompt: {ident}")
            row["family"] = previous["family"]
        else:
            if row["phase"] != "complex" or row["split"] not in {"train", "validation"}:
                raise ValueError(f"invalid new split: {ident}")
            row["family"] = "complex"
    if len(old) != 72 or not set(old) <= {row["id"] for row in cases}:
        raise ValueError("original pool is not fully preserved")
    source["schema"] = "painter-expanded-text-pool-v2"
    source["parent_pool_sha256"] = digest(v1)
    output.mkdir(parents=True)
    for row in cases:
        target = output / row["reference"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v1.parent / row["reference"], target)
    manifest = output / "pool.json"
    manifest.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    return {"pool_sha256": digest(manifest), "total": len(cases),
            "family_counts": {name: sum(row["family"] == name for row in cases)
                              for name in ("simple", "intermediate", "complex")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.v1, args.original, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
