"""Compare saved text-paint pairs under frozen v1 and proposed beauty v2.

No model generation or painting render occurs. User-noted directions are
development signals, not independent blind labels or a validation score.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
from pathlib import Path

from painter import text_pairwise_beauty_judge, text_pairwise_margin_judge


CASES = {
    "t600-366": ("SFT", "muddy dark kiln/potter scene", "frozen SFT"),
    "t600-542": ("SFT", "yarn-ball boundaries", "SFT"),
    "t600-516": ("SFT", "hand-fan precision with warmth", "SFT"),
    "t600-478": ("SFT", "domed observatory composition", "SFT"),
    "t600-425": ("SFT", "seahorse appeal", "frozen SFT"),
    "t600-422": ("SFT", "table-lamp rounded form", "SFT"),
    "t600-543": ("SFT", "spinning-top margins", "SFT"),
    "t600-433": ("RL", "hand-crank drill recognizability", "SFT"),
    "t600-377": ("RL", "octopus shape and arms", "SFT"),
    "t600-439": (None, "clock: accurate RL versus soothing frozen SFT", "frozen SFT"),
}


def brief_for(page: str, ident: str) -> str:
    match = re.search(
        rf"<section><h2>{re.escape(ident)}\b.*?</h2><p class='brief'>(.*?)</p>",
        page, flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"brief missing: {ident}")
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))


async def calibrate(gallery: Path, output: Path, key_file: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    page = (gallery / "index.html").read_text()
    image_root = gallery / "images"
    limit = asyncio.Semaphore(3)

    async def one(ident: str, human: str | None, note: str, anchor_kind: str) -> dict:
        a, b = (image_root / f"{ident}-{anchor_kind}.png",
                image_root / f"{ident}-RL-step-16.png")
        if not a.is_file() or not b.is_file():
            raise FileNotFoundError((a, b))
        brief = output / f"{ident}-brief.txt"
        brief.write_text(brief_for(page, ident) + "\n")
        rows = {}
        for name, judge in (("v1", text_pairwise_margin_judge),
                            ("v2", text_pairwise_beauty_judge)):
            async with limit:
                receipt = await judge.judge_pair(
                    brief=brief, anchor=a, candidate=b, key_file=key_file,
                )
            (output / f"{ident}-{name}.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            rows[name] = {"score_rl": receipt["score"],
                          "order_gap": receipt.get("order_gap"),
                          "prompt_sha256": receipt["views"][0]["prompt_sha256"]}
        return {"id": ident, "user_direction": human, "note": note,
                "anchor_kind": anchor_kind,
                "sft_sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
                "rl_sha256": hashlib.sha256(b.read_bytes()).hexdigest(),
                "judges": rows}

    rows = await asyncio.gather(*(one(ident, *details) for ident, details in CASES.items()))
    labeled = [row for row in rows if row["user_direction"]]
    summary = {"schema": "text-beauty-v2-calibration-v1",
               "scope": "10 selected saved pairs; user feedback is development guidance, not blind gold labels; sailing boat RL had no valid canvas and is excluded from visual calibration",
               "rows": rows,
               "direction_agreement": {
                   judge: sum((row["judges"][judge]["score_rl"] > .5) ==
                              (row["user_direction"] == "RL") for row in labeled)
                   for judge in ("v1", "v2")
               },
               "labeled_count": len(labeled)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(calibrate(args.gallery, args.output, args.key_file))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
