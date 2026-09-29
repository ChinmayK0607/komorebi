"""Second holistic text-paint pairwise candidate; calibrate before RL use."""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    from painter import text_pairwise_margin_judge as margin
except ImportError:
    import text_pairwise_margin_judge as margin


PROTOCOL = "text-paint-beauty-pairwise-v3"
PROMPT_PATH = (Path(__file__).resolve().parent / "runs" / "text-pairwise-rl-20260928"
               / "JUDGE_BEAUTY_V3_DRAFT.md")
PROMPT_SHA256 = "a6150983695573de97a618755cd6ded87aed1abfb5affa98209ba0e8fa332075"


def load_prompt() -> str:
    document = PROMPT_PATH.read_text(encoding="utf-8")
    try:
        prompt = document.split("```text\n", 1)[1].split("\n```", 1)[0]
    except IndexError as exc:
        raise ValueError("beauty judge prompt code block missing") from exc
    if hashlib.sha256(prompt.encode()).hexdigest() != PROMPT_SHA256:
        raise ValueError("beauty judge prompt changed without protocol review")
    return prompt


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path,
                     key_file: Path) -> dict:
    return await margin.judge_pair(
        brief=brief, anchor=anchor, candidate=candidate, key_file=key_file,
        system_prompt=load_prompt(), protocol=PROTOCOL,
    )
