"""Materialize the deterministic twelve-case photo evaluation subset.

The SFT photo manifest already records a reviewed ``selection_rank`` and a
``primary_family`` for each of its 28 development photographs.  This helper
uses the lowest ranks, while requiring distinct families, so the selected
subset is auditable and does not depend on Python's hash randomization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_COUNT = 12
EXPECTED_SOURCE_COUNT = 28


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    rows = raw if isinstance(raw, list) else raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("photo manifest must contain a non-empty cases list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("photo manifest cases must be objects")
    return rows


def select_cases(rows: list[dict[str, Any]], count: int = DEFAULT_COUNT) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"expected {EXPECTED_SOURCE_COUNT} source cases, found {len(rows)}")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")

    candidates: list[tuple[int, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for row in rows:
        case_id = row.get("id")
        family = row.get("primary_family") or row.get("family")
        rank = row.get("selection_rank")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every source case needs a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate source case id: {case_id}")
        if not isinstance(family, str) or not family:
            raise ValueError(f"case {case_id} needs primary_family/family")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError(f"case {case_id} needs a positive selection_rank")
        seen_ids.add(case_id)
        candidates.append((rank, case_id, row))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    for _rank, _case_id, row in candidates:
        family = row.get("primary_family") or row.get("family")
        if family in families:
            continue
        selected.append(row)
        families.add(family)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} distinct primary families available for {count} cases")
    return selected


def build_manifest(source: Path, *, count: int = DEFAULT_COUNT) -> dict[str, Any]:
    rows = _rows(source)
    selected = select_cases(rows, count=count)
    materialized: list[dict[str, Any]] = []
    for row in selected:
        copy = dict(row)
        reference = copy.get("reference_png") or copy.get("reference")
        if isinstance(reference, str) and not Path(reference).is_absolute():
            copy["reference"] = str((source.parent / reference).resolve())
        materialized.append(copy)
    return {
        "schema": "photo-curriculum-rl-matched-eval-v1",
        "selection": {
            "source_manifest": str(source),
            "source_manifest_sha256": _sha(source),
            "source_case_count": len(rows),
            "count": count,
            "rule": "lowest selection_rank with one case per primary_family",
            "selected_ids": [row["id"] for row in materialized],
            "selected_primary_families": [row.get("primary_family") or row.get("family") for row in materialized],
        },
        "cases": materialized,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args(argv)
    result = build_manifest(args.source.resolve(), count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
