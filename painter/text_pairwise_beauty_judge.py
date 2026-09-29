"""Holistic text-paint pairwise candidate; calibrate before RL use."""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    from painter import text_pairwise_margin_judge as margin
except ImportError:
    import text_pairwise_margin_judge as margin


PROTOCOL = "text-paint-beauty-pairwise-v2"
PROMPT_PATH = (Path(__file__).resolve().parent / "runs" / "text-pairwise-rl-20260928"
               / "JUDGE_BEAUTY_V2_DRAFT.md")
PROMPT_SHA256 = "57d6be33542095e7a4298088478c22ed9ecda75c7d6996aee206c61f98a2e594"


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
