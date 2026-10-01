"""Production adapter for the reviewed side-v2 model-selectable judge.

The underlying prompt, strict side schema, balanced views, retries, and
three-of-four consensus are owned by
``text_pairwise_consistent_side_judge_v2_model``.  This small adapter binds
that reviewed transport to the existing RL consistent protocol so legacy
reward consumers remain compatible while receipts retain the requested and
provider-returned model identities.
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
FORMAT_VERSION = "identity-joint-side-schema-v2-pro-v1"
PROMPT = model_judge.PROMPT
PROMPT_SHA256 = model_judge.PROMPT_SHA256
MODEL = "xiaomi/mimo-v2.6-pro"
SOURCE_FORMAT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUBRIC_SOURCE_FORMAT_SHA256 = model_judge.RUBRIC_SOURCE_FORMAT_SHA256
MAGNITUDES = model_judge.MAGNITUDES


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path,
                     key_file: Path, model: str | None = None) -> dict[str, Any]:
    """Judge one pair with the reviewed v2 prompt and an explicit model."""

    selected = model or os.environ.get("PAINTER_JUDGE_MODEL") or MODEL
    if selected != MODEL:
        raise ValueError(f"production pro judge requires {MODEL}, got {selected}")
    result = await model_judge.judge_pair(
        brief=brief,
        anchor=anchor,
        candidate=candidate,
        key_file=key_file,
        model=selected,
    )
    # Keep the live RL protocol binding stable while preserving the reviewed
    # v2 transport identity for audit and model calibration consumers.
    return {
        **result,
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "transport_protocol": model_judge.PROTOCOL,
        "transport_format_version": model_judge.FORMAT_VERSION,
        "transport_source_format_sha256": model_judge.SOURCE_FORMAT_SHA256,
        "judge_model": selected,
        "prompt_sha256": model_judge.PROMPT_SHA256,
        "rubric_source_format_sha256": model_judge.RUBRIC_SOURCE_FORMAT_SHA256,
        "source_format_sha256": SOURCE_FORMAT_SHA256,
    }


__all__ = [
    "FORMAT_VERSION",
    "MAGNITUDES",
    "MODEL",
    "PROMPT",
    "PROMPT_SHA256",
    "PROTOCOL",
    "RUBRIC_SOURCE_FORMAT_SHA256",
    "SOURCE_FORMAT_SHA256",
    "judge_pair",
]
