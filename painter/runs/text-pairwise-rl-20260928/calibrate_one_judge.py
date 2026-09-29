"""Replay a candidate judge on selected saved gallery pairs in two orders."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from painter import text_pairwise_beauty_v3_judge, text_pairwise_beauty_v4_judge

from calibrate_beauty import CASES, brief_for


JUDGES = {"v3": text_pairwise_beauty_v3_judge,
          "v4": text_pairwise_beauty_v4_judge}


async def run(*, gallery: Path, output: Path, key_file: Path,
              judge_name: str, ids: list[str]) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    page = (gallery / "index.html").read_text()
    judge = JUDGES[judge_name]
    limit = asyncio.Semaphore(3)

    async def one(ident: str) -> dict:
        direction, note, anchor_kind = CASES[ident]
        anchor = gallery / "images" / f"{ident}-{anchor_kind}.png"
        candidate = gallery / "images" / f"{ident}-RL-step-16.png"
        if not anchor.is_file() or not candidate.is_file():
            raise FileNotFoundError(ident)
        brief = output / f"{ident}-brief.txt"
        brief.write_text(brief_for(page, ident) + "\n")
        async with limit:
            receipt = await judge.judge_pair(brief=brief, anchor=anchor,
                                             candidate=candidate, key_file=key_file)
        (output / f"{ident}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return {"id": ident, "user_direction": direction, "note": note,
                "anchor_kind": anchor_kind, "score_rl": receipt["score"],
                "order_gap": receipt.get("order_gap"),
                "prompt_sha256": receipt["views"][0]["prompt_sha256"],
                "sft_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
                "rl_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}

    rows = await asyncio.gather(*(one(ident) for ident in ids))
    (output / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--judge", choices=JUDGES, required=True)
    parser.add_argument("--ids", nargs="+", choices=CASES, required=True)
    args = parser.parse_args()
    rows = asyncio.run(run(gallery=args.gallery, output=args.output,
                           key_file=args.key_file, judge_name=args.judge, ids=args.ids))
    print(json.dumps([{k: row[k] for k in ("id", "user_direction", "score_rl", "order_gap")}
                      for row in rows]))


if __name__ == "__main__":
    main()
