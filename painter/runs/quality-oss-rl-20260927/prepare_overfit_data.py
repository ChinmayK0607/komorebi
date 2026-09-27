"""Select the fixed four-scene RL overfit set from the backed-up profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "58a5d1f06910d60d5b3e5726a1e73692a08d8739fc582847bf23e5fd7619c62a"
SELECT = {
    "sphere": "sc-v1-train-sphere-0099",
    "bus": "sc-v1-train-bus-0189",
    "flower": "sc-v1-train-flower-0199",
    "dog": None,
}


def build(profile: Path) -> bytes:
    manifest = json.loads(profile.read_text())
    rows = []
    for family, identifier in SELECT.items():
        if identifier:
            row = next(item for item in manifest["rows"] if item["id"] == identifier)
        else:
            row = next(item for item in manifest["rows"] if item["family"] == family)
        reference = profile.parent / row["reference"]
        if not reference.is_file():
            raise FileNotFoundError(reference)
        actual = hashlib.sha256(reference.read_bytes()).hexdigest()
        if actual != row["reference_sha256"]:
            raise ValueError(f"reference hash mismatch: {row['id']}")
        rows.append({**row, "order": len(rows)})
    manifest["rows"] = rows
    manifest["run_id"] = "gateway-overfit-four-20260928"
    manifest["training_scope"] = {
        "purpose": "fixed four-reference RL overfit canary",
        "families": list(SELECT),
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("fixed overfit manifest digest changed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.profile)
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise ValueError("existing overfit manifest differs")
    else:
        args.output.write_bytes(payload)
    print(json.dumps({"rows": len(SELECT), "sha256": EXPECTED_SHA256,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
