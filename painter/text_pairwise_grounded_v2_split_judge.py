"""Presentation-only ablation for the grounded-v2 pairwise judge.

The v2 prompt, model, generation settings, two-order strength aggregation,
and grounded cap are unchanged.  Each view sends the original PNG bytes as
two separate image inputs with adjacent ``Painting A`` and ``Painting B``
text.  This module is intentionally separate from the frozen v2 module so a
presentation experiment cannot alter the training or pass-1 judge path.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from painter import text_pairwise_grounded_v2_judge as v2
    from painter.text_pairwise_margin_judge import ENDPOINT, MODEL, _label, digest, load_key
except ImportError:  # Running from a BRUSH_ROOT checkout.
    import text_pairwise_grounded_v2_judge as v2
    from text_pairwise_margin_judge import ENDPOINT, MODEL, _label, digest, load_key


PROTOCOL = v2.PROTOCOL
PROMPT = v2.PROMPT
FORMAT_VERSION = "separate-original-pngs-v1"


def _data_url(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if path.suffix.lower() != ".png" or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("presentation ablation requires PNG inputs")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _view(*, key: str, brief: str, a: Path, b: Path,
          system_prompt: str = PROMPT) -> dict:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "PAINTING BRIEF:\n" + brief + "\nPainting A (first image):"},
                {"type": "image_url", "image_url": {"url": _data_url(a)}},
                {"type": "text", "text": "Painting B (second image):"},
                {"type": "image_url", "image_url": {"url": _data_url(b)}},
            ]},
        ],
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"split presentation judge HTTP {exc.code}") from None
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("split presentation judge response truncated")
    raw = choice["message"]["content"]
    return {
        "label": _label(raw),
        "raw_response": raw,
        "image_sha256": {"A": digest(a), "B": digest(b)},
        "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "format_version": FORMAT_VERSION,
        "presentation": "separate_original_pngs",
        "elapsed_seconds": time.monotonic() - started,
        "usage": body.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "model": body.get("model", MODEL),
    }


async def _retry_view(*, key: str, brief: str, a: Path, b: Path,
                      system_prompt: str = PROMPT) -> dict:
    try:
        return await asyncio.to_thread(_view, key=key, brief=brief, a=a, b=b,
                                       system_prompt=system_prompt)
    except ValueError:
        return await asyncio.to_thread(_view, key=key, brief=brief, a=a, b=b,
                                       system_prompt=system_prompt)


async def _judge_once(*, brief_text: str, brief: Path, anchor: Path,
                      candidate: Path, key_file: Path,
                      anchor_sha: str, candidate_sha: str) -> dict:
    """Run both orders once; cap retry is owned by ``judge_pair``."""

    key = load_key(key_file)
    first, second = await asyncio.gather(
        _retry_view(key=key, brief=brief_text, a=anchor, b=candidate,
                    system_prompt=PROMPT),
        _retry_view(key=key, brief=brief_text, a=candidate, b=anchor,
                    system_prompt=PROMPT),
    )
    forward = (first["label"]["strength"] - 1) / 6
    reversed_score = (7 - second["label"]["strength"]) / 6
    return {
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "score": (forward + reversed_score) / 2,
        "status": "scored",
        "order_gap": abs(forward - reversed_score),
        "brief_sha256": digest(brief),
        "anchor_sha256": anchor_sha,
        "candidate_sha256": candidate_sha,
        "views": [first, second],
    }


async def judge_pair(*, brief: Path, anchor: Path, candidate: Path,
                     key_file: Path) -> dict:
    for path in (brief, anchor, candidate):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
    text = brief.read_text(encoding="utf-8").strip()
    if not text or len(text) > 4000:
        raise ValueError("invalid painting brief")
    anchor_sha = digest(anchor)
    candidate_sha = digest(candidate)
    if anchor_sha == candidate_sha:
        return {
            "protocol": PROTOCOL,
            "format_version": FORMAT_VERSION,
            "score": 0.5,
            "status": "identical_canvases",
            "views": [],
            "brief_sha256": digest(brief),
            "anchor_sha256": anchor_sha,
            "candidate_sha256": candidate_sha,
        }
    for attempt in range(2):
        result = await _judge_once(
            brief_text=text, brief=brief, anchor=anchor, candidate=candidate,
            key_file=key_file, anchor_sha=anchor_sha, candidate_sha=candidate_sha,
        )
        try:
            capped = v2.cap_candidate_margin(result)
            return {**capped, "protocol": PROTOCOL, "format_version": FORMAT_VERSION}
        except ValueError:
            if attempt:
                raise
    raise AssertionError("unreachable split grounded-v2 retry state")


__all__ = ["FORMAT_VERSION", "MODEL", "PROMPT", "PROTOCOL", "judge_pair"]
