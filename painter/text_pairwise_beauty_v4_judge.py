"""Precision-with-warmth text-paint pairwise candidate; not an RL reward."""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    from painter import text_pairwise_margin_judge as margin
except ImportError:
    import text_pairwise_margin_judge as margin


PROTOCOL = "text-paint-beauty-pairwise-v4"
PROMPT_PATH = (Path(__file__).resolve().parent / "runs" / "text-pairwise-rl-20260928"
               / "JUDGE_BEAUTY_V4_DRAFT.md")
PROMPT_SHA256 = "90d498bfadbb15dce5944d6cbc57b1db7c56608e44010531a0d8538ea9dc5d31"


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
