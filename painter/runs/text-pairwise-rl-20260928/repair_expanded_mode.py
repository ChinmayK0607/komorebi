"""Version the expanded pool after correcting its native task mode contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair(v2: Path, output: Path) -> dict:
    source = json.loads(v2.read_text())
    rows = source["cases"]
    if source.get("schema") != "painter-expanded-text-pool-v2" or len(rows) != 290:
        raise ValueError("expected the 290-brief v2 pool")
    if output.exists():
        raise FileExistsError(output)
    for row in rows:
        if row.get("mode") != "text_to_image" or row.get("family") not in {
                "simple", "intermediate", "complex"}:
            raise ValueError(f"unexpected task contract: {row.get('id')}")
        prompt = v2.parent / row["reference"]
        if not prompt.is_file() or sha(prompt) != row["reference_sha256"]:
            raise ValueError(f"prompt changed: {row['id']}")
        row["mode"] = "fresh"
    source["schema"] = "painter-expanded-text-pool-v3"
    source["parent_pool_sha256"] = sha(v2)
    output.mkdir(parents=True)
    for row in rows:
        target = output / row["reference"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v2.parent / row["reference"], target)
    manifest = output / "pool.json"
    manifest.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    return {"pool_sha256": sha(manifest), "count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.v2, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
