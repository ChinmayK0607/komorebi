"""Isolated live v2 side judge for the finite judge-only RL ablation.

The production Pro adapter keeps its historical provider-default request.  RL
arms in ``rl-judge-ablation-20261002`` import this module instead, which binds
the same reviewed rubric and consensus transport to one explicit model and
``reasoning.effort=medium``.  Sampling parameters are omitted in this mode.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

try:
    from painter import text_pairwise_consistent_side_judge_v2_model as model_judge
except ImportError:  # pragma: no cover - direct node execution fallback
    import text_pairwise_consistent_side_judge_v2_model as model_judge


PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
FORMAT_VERSION = "identity-joint-side-schema-v2-generic-v1"
MODELS = (
    "xiaomi/mimo-v2.6-flash",
    "xiaomi/mimo-v2.6-pro",
    "openai/gpt-6-luna",
)
MODEL = MODELS[0]
REASONING = {"effort": "medium"}
REASONING_EFFORT = "medium"
PROMPT = model_judge.PROMPT
PROMPT_SHA256 = model_judge.PROMPT_SHA256
RUBRIC_SOURCE_FORMAT_SHA256 = model_judge.RUBRIC_SOURCE_FORMAT_SHA256
SOURCE_FORMAT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAGNITUDES = model_judge.MAGNITUDES


def _selected_model(model: str | None) -> str:
    selected = model or os.environ.get("PAINTER_JUDGE_MODEL") or MODEL
    if selected not in MODELS:
        raise ValueError(f"judge model is not registered for this ablation: {selected}")
    return selected


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path,
                     key_file: Path, model: str | None = None,
                     reasoning: dict[str, Any] | None = None) -> dict[str, Any]:
    """Judge one pair with the arm's model and the registered effort object."""

    selected = _selected_model(model)
    configured_effort = os.environ.get("PAINTER_JUDGE_EFFORT", REASONING_EFFORT)
    if configured_effort != REASONING_EFFORT:
        raise ValueError(f"the ablation requires PAINTER_JUDGE_EFFORT={REASONING_EFFORT}")
    if reasoning is not None and reasoning != REASONING:
        raise ValueError("the ablation requires reasoning.effort=medium")
    result = await model_judge.judge_pair(
        brief=brief,
        anchor=anchor,
        candidate=candidate,
        key_file=key_file,
        model=selected,
        reasoning=REASONING,
    )
    return {
        **result,
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "transport_protocol": model_judge.PROTOCOL,
        "transport_format_version": model_judge.FORMAT_VERSION,
        "transport_source_format_sha256": model_judge.SOURCE_FORMAT_SHA256,
        "judge_model": selected,
        "requested_model": selected,
        "reasoning": {"field": "reasoning", **REASONING},
        "reasoning_effort": REASONING_EFFORT,
        "sampling_parameters_omitted": True,
        "prompt_sha256": PROMPT_SHA256,
        "rubric_source_format_sha256": RUBRIC_SOURCE_FORMAT_SHA256,
        "source_format_sha256": SOURCE_FORMAT_SHA256,
    }


__all__ = [
    "FORMAT_VERSION", "MAGNITUDES", "MODELS", "MODEL", "PROMPT",
    "PROMPT_SHA256", "PROTOCOL", "REASONING", "REASONING_EFFORT",
    "RUBRIC_SOURCE_FORMAT_SHA256", "SOURCE_FORMAT_SHA256", "judge_pair",
]
