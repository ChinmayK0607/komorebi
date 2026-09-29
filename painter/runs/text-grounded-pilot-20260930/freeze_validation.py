"""Freeze the existing balanced 16 from the unchanged 47-case validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def freeze(source: Path, output: Path) -> dict:
    rows = json.loads(source.read_text())["cases"]
    if len(rows) != 47 or len({row["id"] for row in rows}) != 47:
        raise ValueError("expected immutable 47-case validation")
    foundation = [row for row in rows if row["family"] in {"simple", "intermediate"}][:8]
    complex_rows = [row for row in rows if row["family"] == "complex"][:8]
    selected = foundation + complex_rows
    if len(selected) != 16 or len({row["id"] for row in selected}) != 16:
        raise ValueError("balanced-16 subset is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "text-paint-grounded-balanced16-v1",
                                  "cases": selected}, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"cases": len(selected), "ids": [row["id"] for row in selected],
            "manifest_sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.output), sort_keys=True))
