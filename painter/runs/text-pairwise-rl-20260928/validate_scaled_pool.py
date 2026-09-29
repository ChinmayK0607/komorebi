"""Load every scaled brief through the actual native evaluation taskset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from painter.native_painting import NativePaintingConfig, NativePaintingTaskset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args()
    pool = args.pool.resolve(strict=True)
    rows = json.loads(pool.read_text())["cases"]
    tasks = list(NativePaintingTaskset(NativePaintingConfig(
        root=str(pool.parent), manifest=str(pool), expected_count=len(rows),
        max_turns=3)).load())
    if len(tasks) != 290:
        raise ValueError(f"expected 290 native tasks, got {len(tasks)}")
    print(json.dumps({"native_tasks_loaded": len(tasks)}))


if __name__ == "__main__":
    main()
