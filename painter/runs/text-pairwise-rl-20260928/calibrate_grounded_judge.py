"""One-off four-pair judge calibration; never changes a training receipt."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

try:
    from painter.text_pairwise_grounded_judge import judge_pair
except ImportError:
    from text_pairwise_grounded_judge import judge_pair


async def run(input_root: Path, key_file: Path, output: Path) -> None:
    rows = []
    for ident in ("t600-333", "t600-412", "t600-576", "t600-510"):
        case = input_root / ident
        result = await judge_pair(
            brief=case / "prompt.txt", anchor=case / "anchor.png",
            candidate=case / "candidate.png", key_file=key_file,
        )
        rows.append({"id": ident, "judge": result})
        print(json.dumps({"id": ident, "score": result["score"],
                          "raw_pairwise_score": result.get("raw_pairwise_score"),
                          "candidate_fidelity_views": result.get("candidate_fidelity_views")} ),
              flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.key_file, args.output))
