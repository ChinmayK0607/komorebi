#!/usr/bin/env python3
"""Validate optional teacher code supervision as strict JSONL.

The file is intentionally separate from the RL manifest and reward.  A teacher
row is accepted only when it contains one next assistant token string; no
multi-turn transcript or style label can be smuggled into the warmup path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = {"case_id", "source_group", "system", "user", "next_assistant_token"}


def validate(path: Path, *, allowed_case_ids: set[str] | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("teacher supervision path must be a regular file")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line at {line_number}; JSONL rows must be contiguous")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        if not isinstance(value, dict) or set(value) != REQUIRED:
            raise ValueError(f"line {line_number} must have exactly {sorted(REQUIRED)}")
        if any(not isinstance(value[key], str) or not value[key].strip() for key in REQUIRED):
            raise ValueError(f"line {line_number} contains an empty or non-string field")
        case_id = value["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id at line {line_number}: {case_id}")
        if allowed_case_ids is not None and case_id not in allowed_case_ids:
            raise ValueError(f"teacher case is outside the admitted pilot: {case_id}")
        seen.add(case_id)
        rows.append(value)
    if not rows:
        raise ValueError("teacher supervision JSONL is empty")
    return {"schema": "Diverse-flow-code-supervision-v1", "rows": len(rows), "case_ids": sorted(seen), "rl_reward": "unchanged", "format": "strict-jsonl-nextassistanttokenonly"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
