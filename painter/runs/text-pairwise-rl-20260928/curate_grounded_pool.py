"""Freeze a diverse, visually reviewed fixed-opponent text RL pilot pool.

Selection was made from the complete 119-anchor contact sheets. This is a
finite protocol choice, not a filter learned from the corrective run outputs.
The original public SFT canvases and brief hashes remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SIMPLE = ("510", "511", "521", "524", "527", "528", "530", "531",
          "535", "539", "544", "546")
INTERMEDIATE = ("427", "429", "430", "431", "435", "437", "442", "444",
                "446", "454")
COMPLEX = ("581", "368", "570", "592", "565", "402", "376", "460",
           "455", "324", "330", "393", "554", "321", "586", "407",
           "568", "359")
EXPECTED_CONTENT_SHA256 = "13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(source: Path, output: Path) -> dict:
    rows = json.loads(source.read_text())["rows"]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != 119:
        raise ValueError("expected the immutable 119-anchor source")
    groups = {"simple": SIMPLE, "intermediate": INTERMEDIATE, "complex": COMPLEX}
    chosen = []
    for index in range(max(map(len, groups.values()))):
        # Interleave families; avoid a training-time easy-to-hard distribution
        # shift that would make the online reward curve uninterpretable.
        for family, position in (("simple", index), ("complex", 2 * index),
                                 ("intermediate", index), ("complex", 2 * index + 1)):
            values = groups[family]
            if position >= len(values):
                continue
            ident = "t600-" + values[position]
            if any(row["id"] == ident for row in chosen):
                continue
            row = by_id[ident]
            if row["family"] != family:
                raise ValueError(f"family mismatch: {ident}")
            if sha(Path(row["reference"])) != row["reference_sha256"]:
                raise ValueError(f"brief hash mismatch: {ident}")
            if sha(Path(row["opponent"])) != row["opponent_sha256"]:
                raise ValueError(f"anchor hash mismatch: {ident}")
            chosen.append(row)
    ids = {row["id"] for row in chosen}
    expected = {"t600-" + value for values in groups.values() for value in values}
    if ids != expected or len(chosen) != 40:
        raise ValueError("curated pool coverage mismatch")
    semantic_rows = [{"id": row["id"], "family": row["family"],
                      "reference_sha256": row["reference_sha256"],
                      "opponent_sha256": row["opponent_sha256"]}
                     for row in chosen]
    content_sha = hashlib.sha256(json.dumps(
        semantic_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if content_sha != EXPECTED_CONTENT_SHA256:
        raise ValueError("curated pool content differs from reviewed 40")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": chosen}, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema": "text-paint-grounded-pilot-pool-v1",
        "source_manifest_sha256": sha(source), "manifest_sha256": sha(output),
        "content_sha256": content_sha,
        "source_count": len(rows), "selected_count": len(chosen),
        "family_counts": {family: len(values) for family, values in groups.items()},
        "selected_ids_in_order": [row["id"] for row in chosen],
        "selection_basis": "Manual visual review of complete SFT anchor contact sheets; recognizable subject and composition across simple, intermediate and complex briefs. Selection predates all corrective-run outputs.",
        "known_excluded_failure_anchors": ["t600-333", "t600-412", "t600-576", "t600-580"],
        "warning": "Visual review is unblinded and subjective; selected anchors are not gold labels.",
    }
    output.with_name(output.stem + "-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), sort_keys=True))
