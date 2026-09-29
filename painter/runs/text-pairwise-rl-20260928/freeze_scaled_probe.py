"""Select a fixed within-training probe before the first optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SELECTION = "curriculum-first-four-per-family-20260929"


def freeze(train: Path, output: Path) -> dict:
    rows = json.loads(train.read_text())["rows"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate training IDs")
    chosen = []
    for family in ("simple", "complex"):
        eligible = [row for row in rows if row["family"] == family]
        if len(eligible) < 4:
            raise ValueError(f"too few {family} anchors")
        # The short run visits the manifest in curriculum order. Sampling
        # arbitrary later complex rows could label unseen prompts a train
        # probe, so freeze the first four from each observed family.
        chosen.extend(eligible[:4])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "painter-scaled-train-probe-v1",
                                  "selection": SELECTION, "cases": chosen}, indent=2,
                                 sort_keys=True) + "\n")
    return {"ids": [row["id"] for row in chosen],
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.train, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
