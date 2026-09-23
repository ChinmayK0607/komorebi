#!/usr/bin/env python3
"""Resumable Vercel AI Gateway image-to-p5.brush teacher benchmark.

This runner is deliberately standard-library only.  It performs no network
request unless ``--run`` is supplied.  A run is five-model (or whatever the
checked-in config declares) multi-turn episode orchestration: each turn gets
the immutable reference image, the latest valid rendered canvas when one
exists, the previous full assistant response, and renderer feedback.

The benchmark directory is self-contained.  It expects ``config.json``,
``refs.json`` and ``prompt.txt`` beside this file; the parent experiment owns
those inputs.  Response and episode artifacts are written below ``episodes``
and can be resumed after interruption.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import getpass
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - Linux launch pins tqdm
    class _FallbackTqdm:
        """Small no-op tqdm substitute that still behaves like an iterator."""

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **kwargs):
        return _FallbackTqdm(iterable, **kwargs)


SCHEMA = "painter.ai-gateway-teachers.v1"
PROVIDER = "vercel-ai-gateway"
TRANSPORT_SCRIPT = "gateway_transport.ts"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_ERROR_CHARS = 4000
PROGRESS_HEARTBEAT_SECONDS = 5.0
EXPECTED_MODELS = (
    "zai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash",
    "stepfun/step-5-preview",
    "xiaomi/mimo-v2.6-flash",
    "xiaomi/mimo-v2.6-pro",
)
HEX = frozenset("0123456789abcdef")
_FENCE = re.compile(
    r"```[ \t]*(?:(?:javascript|js|p5(?:\.js)?|typescript|ts)[ \t]*)?\r?\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_SECRET_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


class BenchmarkError(RuntimeError):
    """A configuration or benchmark execution error."""


class APIError(BenchmarkError):
    """An AI Gateway transport or malformed response error."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 response_body: str | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.response_body = response_body


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes, *, mode: int | None = None) -> None:
    """Replace one file atomically, including when a process is interrupted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def redact(value: Any, secret: str | None = None) -> Any:
    """Redact bearer tokens and the configured API key from loggable values."""

    if isinstance(value, str):
        result = _SECRET_RE.sub(r"\1[REDACTED]", value)
        if secret:
            result = result.replace(secret, "[REDACTED]")
        return result
    if isinstance(value, dict):
        return {str(key): redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return value or "item"


def parse_program(reply: str) -> tuple[str | None, str]:
    """Extract the last complete fenced JS sketch and the surrounding plan.

    The model is free to use prose, JSON, or extra fenced non-JS material.  A
    complete JavaScript fence is the only thing that is sent to the renderer;
    the original response is always retained separately.
    """

    matches = list(_FENCE.finditer(reply or ""))
    if not matches:
        return None, (reply or "").strip()
    match = matches[-1]
    code = match.group("code").strip() + "\n"
    plan = _FENCE.sub("", reply).strip()
    return code, plan


def extract_content(message: Any) -> str:
    """Extract assistant text while tolerating AI SDK content parts."""

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for part in message:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    if message is None:
        return ""
    return str(message)


def image_mime(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        return guessed
    raise BenchmarkError(f"unsupported reference/canvas image type: {path}")


def image_data_uri(path: Path) -> str:
    return f"data:{image_mime(path)};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


@dataclass(frozen=True)
class Reference:
    id: str
    image: Path
    sha256: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BenchmarkInputs:
    root: Path
    config: Mapping[str, Any]
    config_sha256: str
    prompt: str
    prompt_sha256: str
    refs_sha256: str
    catalog_sha256: str
    catalog: Mapping[str, Any]
    references: tuple[Reference, ...]


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value.lower()):
        raise BenchmarkError(f"{label} must be a lowercase SHA-256 hex string")
    return value.lower()


def load_inputs(root: Path, *, model_ids: Sequence[str] | None = None, fetch_unknown: bool = False) -> BenchmarkInputs:
    root = root.resolve()
    config_path = root / "config.json"
    refs_path = root / "refs.json"
    prompt_path = root / "prompt.txt"
    for path in (config_path, refs_path, prompt_path):
        if not path.is_file():
            raise BenchmarkError(f"missing benchmark input: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        refs_doc = json.loads(refs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid benchmark JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise BenchmarkError("config.json must contain an object")
    models = list(model_ids) if model_ids is not None else config.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(model, str) and model for model in models):
        raise BenchmarkError("config.models must be a non-empty list of model IDs")
    if len(set(models)) != len(models):
        raise BenchmarkError("config.models contains duplicate model IDs")
    _positive_int(config, "concurrency", default=3)
    _positive_int(config, "render_concurrency", default=1)
    samples = _positive_int(config, "samples_per_image", default=1)
    if samples != 1:
        raise BenchmarkError("samples_per_image must be 1 for the first benchmark")
    tracks = config.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        raise BenchmarkError("config.tracks must contain at least one track")
    for track_name, track in tracks.items():
        if not isinstance(track, dict):
            raise BenchmarkError(f"config.tracks.{track_name} must be an object")
        _positive_int(track, "max_turns", default=6)
        max_tokens = track.get("max_tokens")
        if max_tokens != "native" and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0):
            raise BenchmarkError(f"config.tracks.{track_name}.max_tokens must be positive or 'native'")
        timeout = track.get("episode_timeout_seconds")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise BenchmarkError(f"config.tracks.{track_name}.episode_timeout_seconds must be positive or null")
    temperature = config.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
        raise BenchmarkError("config.temperature must be between 0 and 2")
    references = refs_doc.get("references") if isinstance(refs_doc, dict) else None
    if not isinstance(references, list) or not references:
        raise BenchmarkError("refs.json must contain a non-empty references list")
    parsed: list[Reference] = []
    seen: set[str] = set()
    for item in references:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise BenchmarkError("each reference needs a non-empty id")
        ref_id = item["id"]
        if ref_id in seen:
            raise BenchmarkError(f"duplicate reference id: {ref_id}")
        seen.add(ref_id)
        image_value = item.get("image")
        if not isinstance(image_value, str) or not image_value:
            raise BenchmarkError(f"reference {ref_id} needs an image path")
        image = Path(image_value)
        if image.is_absolute():
            raise BenchmarkError(f"reference image must be relative: {ref_id}")
        image = (root / image).resolve()
        if root not in image.parents and image != root:
            raise BenchmarkError(f"reference image escapes benchmark directory: {ref_id}")
        if image.is_symlink() or not image.is_file():
            raise BenchmarkError(f"reference image is missing or symlinked: {image}")
        actual = sha_file(image)
        declared = _validate_hash(item.get("sha256"), f"reference {ref_id}.sha256")
        if actual != declared:
            raise BenchmarkError(f"reference hash mismatch for {ref_id}: expected {declared}, found {actual}")
        image_mime(image)
        parsed.append(Reference(ref_id, image, declared, dict(item)))
    if "screen" in tracks:
        screen_ids = config.get("screen_reference_ids")
        if not isinstance(screen_ids, list) or not screen_ids or not all(isinstance(item, str) and item for item in screen_ids):
            raise BenchmarkError("config.screen_reference_ids must be a non-empty list when the screen track is configured")
        if len(set(screen_ids)) != len(screen_ids):
            raise BenchmarkError("config.screen_reference_ids contains duplicate reference IDs")
        by_id = {reference.id: reference for reference in parsed}
        missing_screen = [item for item in screen_ids if item not in by_id]
        if missing_screen:
            raise BenchmarkError(f"config.screen_reference_ids contains unknown references: {', '.join(missing_screen)}")
        screen_categories = [str(by_id[item].metadata.get("category", "")) for item in screen_ids]
        if any(not category for category in screen_categories):
            raise BenchmarkError("screen references must have non-empty categories in refs.json")
        if len(set(screen_categories)) != len(screen_categories):
            raise BenchmarkError("config.screen_reference_ids must select at most one reference per category")
    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise BenchmarkError("prompt.txt is empty")
    catalog_value = config.get("catalog", "model-catalog.json")
    if not isinstance(catalog_value, str) or Path(catalog_value).is_absolute():
        raise BenchmarkError("config.catalog must be a relative JSON path")
    catalog_path = (root / catalog_value).resolve()
    if root not in catalog_path.parents or not catalog_path.is_file():
        raise BenchmarkError(f"model catalog is missing or escapes benchmark directory: {catalog_path}")
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid model catalog JSON: {exc}") from exc
    catalog_models = catalog.get("models") if isinstance(catalog, dict) else None
    catalog_by_id = {row.get("id"): row for row in catalog_models or [] if isinstance(row, dict) and isinstance(row.get("id"), str)}
    # Gateway accepts arbitrary ``provider/model`` IDs.  The checked-in
    # catalog is useful provenance when it has a profile, but it must not be a
    # capability allowlist or trigger a provider catalog fetch.  Unknown IDs
    # receive an explicitly unverified profile; the Gateway request itself is
    # the capability probe and its redacted error is retained in the episode.
    missing = [model for model in models if model not in catalog_by_id]
    for model in missing:
        catalog_by_id[model] = {"id": model, "gateway_profile": "unverified"}
    if missing:
        catalog = dict(catalog)
        catalog["models"] = list(catalog_by_id.values())
    catalog_sha = sha_file(catalog_path)
    for model in models:
        architecture = catalog_by_id[model].get("architecture") or {}
        modalities = architecture.get("input_modalities") or []
        # Gateway model availability and multimodal support are probed by the
        # SDK request; the offline profile is advisory and never an allowlist.
    if model_ids is not None:
        config = dict(config)
        config["models"] = list(models)
    return BenchmarkInputs(
        root=root,
        config=config,
        config_sha256=sha_file(config_path),
        prompt=prompt,
        prompt_sha256=sha_bytes(prompt.encode("utf-8")),
        refs_sha256=sha_file(refs_path),
        catalog_sha256=catalog_sha,
        catalog=catalog,
        references=tuple(parsed),
    )


def _positive_int(config: Mapping[str, Any], name: str, *, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkError(f"config.{name} must be a positive integer")
    return value


def settings_from_inputs(inputs: BenchmarkInputs, track: str) -> dict[str, Any]:
    config = inputs.config
    tracks = config.get("tracks", {})
    if track not in tracks:
        raise BenchmarkError(f"unknown track {track!r}; choose from {', '.join(sorted(tracks))}")
    track_config = tracks[track]
    return {
        "track": track,
        "max_turns": _positive_int(track_config, "max_turns", default=6),
        "max_tokens": track_config.get("max_tokens", "native"),
        "episode_timeout_seconds": track_config.get("episode_timeout_seconds"),
        "temperature": config.get("temperature", 0.7),
        "samples_per_image": _positive_int(config, "samples_per_image", default=1),
        "provider": PROVIDER,
        "transport": "ai-sdk",
        "transport_script": str(config.get("transport_script", TRANSPORT_SCRIPT)),
        "api_timeout_seconds": float(config.get("request_timeout_seconds", config.get("api_timeout_seconds", DEFAULT_TIMEOUT))),
        "max_retries": _positive_int(config, "max_retries", default=DEFAULT_MAX_RETRIES),
        "reasoning": config.get("reasoning", {}),
    }


def catalog_model(inputs: BenchmarkInputs, model: str) -> Mapping[str, Any]:
    for row in inputs.catalog.get("models", []):
        if isinstance(row, dict) and row.get("id") == model:
            return row
    raise BenchmarkError(f"model is missing from catalog: {model}")


def resolve_reasoning(inputs: BenchmarkInputs, model: str, track: str) -> dict[str, Any]:
    row = catalog_model(inputs, model)
    if inputs.catalog.get("profile_kind") == "gateway-unverified" or row.get("gateway_profile") == "unverified":
        return {"_omit": True, "source": "gateway_model_default"}
    supported_parameters = set(row.get("supported_parameters") or [])
    if supported_parameters and not ({"reasoning", "reasoning_effort", "include_reasoning"} & supported_parameters):
        return {"_omit": True, "source": "catalog_reasoning_unsupported"}
    reasoning = row.get("reasoning") or {}
    supported = reasoning.get("supported_efforts") or []
    requested = (inputs.config.get("tracks", {}).get(track) or {}).get("reasoning_effort")
    selection = requested
    if selection is None:
        selection = "max" if track in {"quality", "screen"} else "low"
    if supported:
        order = {"low": 0, "medium": 1, "high": 2, "max": 3}
        if selection == "highest_supported":
            requested = max(supported, key=lambda value: order.get(value, -1))
        elif selection == "lowest_supported":
            requested = min(supported, key=lambda value: order.get(value, 99))
        elif selection in supported:
            requested = selection
        else:
            requested = max(supported, key=lambda value: order.get(value, -1)) if track in {"quality", "screen"} else min(supported, key=lambda value: order.get(value, 99))
        return {"enabled": True, "effort": requested, "source": "catalog_supported_effort"}
    enabled = bool(reasoning.get("mandatory") or reasoning.get("default_enabled") or True)
    return {"enabled": enabled, "source": "catalog_enabled_without_exposed_effort"}


def model_limits(inputs: BenchmarkInputs, model: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    row = catalog_model(inputs, model)
    if inputs.catalog.get("profile_kind") == "gateway-unverified" or row.get("gateway_profile") == "unverified":
        return {
            "native_ceiling": None,
            "provider_context_length": None,
            "requested_max_tokens": None if settings["max_tokens"] == "native" else int(settings["max_tokens"]),
        }
    provider = row.get("top_provider") or {}
    ceiling = provider.get("max_completion_tokens")
    context = provider.get("context_length") or row.get("context_length")
    # Gateway model profiles are intentionally optional.  When a user passes a
    # model absent from the offline catalog, omit the native ceiling and let
    # the provider decide it; only the explicit speed-track ceiling is known.
    if not isinstance(ceiling, int) or ceiling <= 0:
        ceiling = None
    if not isinstance(context, int) or context <= 0:
        context = None
    requested = None if settings["max_tokens"] == "native" else int(settings["max_tokens"])
    if requested is not None and ceiling is not None:
        requested = min(requested, ceiling)
    return {"native_ceiling": ceiling, "provider_context_length": context, "requested_max_tokens": requested}


def context_accounting(messages: Sequence[Mapping[str, Any]], model_info: Mapping[str, Any], requested_max: int | None) -> dict[str, Any]:
    text_bytes = 0
    image_count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_bytes += len(content.encode("utf-8"))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_bytes += len(str(part.get("text", "")).encode("utf-8"))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    image_count += 1
    # Image tokenization varies by provider.  A fixed allowance is an explicit
    # accounting bound, not a claim about the provider's actual image tokens.
    image_allowance = 4096
    # Treat each UTF-8 byte as one possible token.  This is intentionally a
    # conservative upper bound for reserving full conversation history; the
    # provider's tokenizer may be much more efficient.
    estimated_text_tokens = text_bytes
    estimated_input_tokens = estimated_text_tokens + image_count * image_allowance
    context_limit = model_info.get("provider_context_length")
    if isinstance(context_limit, int) and context_limit > 0:
        physical_remaining: int | None = context_limit - estimated_input_tokens - 128
        effective: int | None = min(int(requested_max), max(0, physical_remaining)) if requested_max is not None else max(0, physical_remaining)
    else:
        physical_remaining = None
        effective = requested_max
    return {
        "text_utf8_bytes": text_bytes,
        "estimated_text_tokens_upper_bound": estimated_text_tokens,
        "image_count": image_count,
        "image_token_allowance_each": image_allowance,
        "estimated_input_tokens": estimated_input_tokens,
        "provider_context_length": context_limit,
        "native_completion_ceiling": model_info.get("native_ceiling"),
        "requested_max_tokens": int(requested_max) if requested_max is not None else None,
        "physical_remaining_context": max(0, physical_remaining) if physical_remaining is not None else None,
        "effective_max_tokens": effective,
        "reduction_reason": "physical_context_remaining" if physical_remaining is not None and requested_max is not None and effective < int(requested_max) else None,
        "context_exhausted": physical_remaining is not None and physical_remaining <= 0,
    }


def job_id(model: str, reference_id: str, sample: int = 1, track: str = "quality") -> str:
    model_tag = f"{safe_slug(model)}-{sha_bytes(model.encode('utf-8'))[:8]}"
    # Provider prefix keeps yesterday's OpenRouter episode directories
    # incompatible even when a model/reference/track name is unchanged.
    return f"gateway--{safe_slug(track)}--{model_tag}--{safe_slug(reference_id)}--s{sample:02d}"


def request_binding(
    *, inputs: BenchmarkInputs, model: str, reference: Reference, turn: int,
    settings: Mapping[str, Any], current_sha256: str | None, prior_response_sha256: str | None,
    render_feedback: str | None, reasoning: Mapping[str, Any], context: Mapping[str, Any],
    history_sha256: str,
) -> dict[str, Any]:
    """The immutable fields that decide whether a completed API turn is reusable."""

    return {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "transport": "ai-sdk-generateText-jsonl",
        "model": model,
        "reference_id": reference.id,
        "reference_sha256": reference.sha256,
        "prompt_sha256": inputs.prompt_sha256,
        "model_profile_sha256": sha_bytes(canonical(catalog_model(inputs, model))),
        "turn": turn,
        "track": settings["track"],
        "settings": {
            "max_tokens": context["effective_max_tokens"],
            "temperature": settings["temperature"],
            "provider": settings["provider"],
            "transport": settings["transport"],
            "transport_script": settings["transport_script"],
        },
        "reasoning": dict(reasoning),
        "context_accounting": dict(context),
        "render_settings": {
            "renderer_timeout": inputs.config.get("renderer_timeout", 180),
            "render_concurrency": inputs.config.get("render_concurrency", 1),
            "renderer_identity": settings.get("renderer_identity"),
        },
        "current_canvas_sha256": current_sha256,
        "prior_response_sha256": prior_response_sha256,
        "history_sha256": history_sha256,
        "render_feedback": render_feedback or "",
    }


def fingerprint(binding: Mapping[str, Any]) -> str:
    return sha_bytes(canonical(binding))


def sanitized_image_part(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = resolved.name
    return {
        "type": "image_url",
        "image": {"path": relative, "sha256": sha_file(resolved), "mime": image_mime(resolved)},
    }


def build_messages(
    *, inputs: BenchmarkInputs, reference: Reference, current_canvas: Path | None,
    previous_response: str | None, render_feedback: str | None, track: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact API messages and a no-base64 artifact-safe equivalent."""

    user_message, safe_user_message = build_user_message(
        inputs=inputs, reference=reference, current_canvas=current_canvas,
        previous_response=previous_response, render_feedback=render_feedback, track=track,
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": inputs.prompt}]
    if previous_response is not None:
        messages.append({"role": "assistant", "content": previous_response})
    messages.append(user_message)
    safe_messages: list[dict[str, Any]] = [{"role": "system", "content": inputs.prompt}]
    if previous_response is not None:
        safe_messages.append({"role": "assistant", "content": previous_response})
    safe_messages.append(safe_user_message)
    return messages, safe_messages


def build_user_message(
    *, inputs: BenchmarkInputs, reference: Reference, current_canvas: Path | None,
    previous_response: str | None, render_feedback: str | None, track: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = (
        "Inspect the attached reference image. Return a brief visual plan followed by a complete "
        "p5.brush JavaScript sketch in a fenced javascript code block."
        if previous_response is None
        else (
            "Inspect the attached REFERENCE and CURRENT CANVAS. Revise the complete sketch to fix "
            "the remaining visual mismatch. Return a brief plan followed by the complete sketch "
            "in a fenced javascript code block. If the current canvas faithfully matches the "
            "reference, respond with a brief reason and FINISHED without a code block."
        )
    )
    if render_feedback:
        text += "\nRenderer feedback from the submitted program:\n" + render_feedback
    if track == "speed":
        text += "\nSpeed track: prioritize a usable first valid painting within the turn budget and keep the plan concise."
    elif track == "screen":
        text += "\nScreen track: aim for a strong result in roughly 4–6 turns, with six as the cap. This is a target range, not a minimum; after observing a faithful current canvas, finish immediately with FINISHED."
    user_parts: list[dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_data_uri(reference.image)}},
    ]
    if current_canvas is not None:
        user_parts.append({"type": "image_url", "image_url": {"url": image_data_uri(current_canvas)}})
    safe_parts: list[dict[str, Any]] = [{"type": "text", "text": text}, sanitized_image_part(reference.image, root=inputs.root)]
    if current_canvas is not None:
        safe_parts.append(sanitized_image_part(current_canvas, root=inputs.root))
    return {"role": "user", "content": user_parts}, {"role": "user", "content": safe_parts}


def build_conversation_messages(
    *, inputs: BenchmarkInputs, reference: Reference, prior_turns: Sequence[Mapping[str, Any]],
    current_canvas: Path | None, render_feedback: str | None, track: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct the full assistant/user history from immutable turn records."""

    messages: list[dict[str, Any]] = [{"role": "system", "content": inputs.prompt}]
    safe_messages: list[dict[str, Any]] = [{"role": "system", "content": inputs.prompt}]
    for index, old in enumerate(prior_turns):
        request_canvas_value = old.get("request_current_canvas")
        request_canvas = inputs.root / request_canvas_value if isinstance(request_canvas_value, str) else None
        if request_canvas is not None and not request_canvas.is_file():
            request_canvas = None
        user, safe_user = build_user_message(
            inputs=inputs, reference=reference, current_canvas=request_canvas,
            previous_response=None if index == 0 else str((prior_turns[index - 1].get("response") or {}).get("raw_text") or ""),
            render_feedback=old.get("request_render_feedback"), track=track,
        )
        messages.append(user)
        safe_messages.append(safe_user)
        response = old.get("response") or {}
        raw_text = response.get("raw_text")
        if isinstance(raw_text, str):
            messages.append({"role": "assistant", "content": raw_text})
            safe_messages.append({"role": "assistant", "content": raw_text})
    user, safe_user = build_user_message(
        inputs=inputs, reference=reference, current_canvas=current_canvas,
        previous_response=None if not prior_turns else str((prior_turns[-1].get("response") or {}).get("raw_text") or ""),
        render_feedback=render_feedback, track=track,
    )
    messages.append(user)
    safe_messages.append(safe_user)
    return messages, safe_messages


def build_payload(messages: Sequence[Mapping[str, Any]], settings: Mapping[str, Any], model: str,
                  *, max_tokens: int | None = None, reasoning: Mapping[str, Any] | None = None) -> dict[str, Any]:
    requested_max_tokens = max_tokens if max_tokens is not None else settings.get("max_tokens")
    reasoning = dict(reasoning or {})
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": settings["temperature"],
    }
    if requested_max_tokens is not None and requested_max_tokens != "native":
        payload["max_tokens"] = int(requested_max_tokens)
    if not reasoning.get("_omit"):
        # ``source`` is provenance for the receipt/fingerprint only; the
        # provider schema accepts the reasoning controls, not our metadata.
        payload["reasoning"] = {
            key: value for key, value in reasoning.items()
            if key in {"effort", "enabled", "exclude", "max_tokens"}
        }
    return payload


def format_elapsed(seconds: float) -> str:
    """Format a monotonic duration without implying a clock timestamp."""

    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remainder:02d}s"
    return f"{minutes:02d}m{remainder:02d}s"


class ProgressReporter:
    """Print bounded, stderr-only live benchmark status.

    The Gateway transport currently uses ``generateText`` and returns only
    after a provider response is complete.  While that call is pending this
    reporter deliberately says that the provider response is still pending;
    it does not estimate or imply streamed token counts.
    """

    def __init__(self, *, episodes_total: int, stream: Any = None):
        self.episodes_total = episodes_total
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._finished_tokens: dict[str, int] = {}
        self._finished_turns = 0
        self._sequence = 0

    def _write(self, line: str) -> None:
        # tqdm.write preserves an active tqdm line when tqdm is installed.  A
        # plain stderr write keeps the fallback and non-TTY paths dependency
        # free.  Both paths remain outside stdout/JSONL transport.
        writer = getattr(tqdm, "write", None)
        try:
            if callable(writer):
                writer(line, file=self.stream)
            else:
                print(line, file=self.stream, flush=True)
        except (OSError, ValueError):
            # A closed terminal stream must never turn a benchmark result into
            # an API or renderer failure.
            return

    @staticmethod
    def _token_value(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def emit(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Record one phase and print it; return a safe progress snapshot."""

        job_id_value = str(event.get("job_id") or "unknown")
        with self._lock:
            active = self._active.get(job_id_value)
            if active is None:
                active = {
                    "started_monotonic": time.monotonic(),
                    "episode_number": event.get("episode_number"),
                    "episodes_total": self.episodes_total,
                    "model": str(event.get("model") or "unknown"),
                    "reference_id": str(event.get("reference_id") or "unknown"),
                    "turn": 0,
                    "max_turns": event.get("max_turns"),
                    "turns_completed": 0,
                    "total_tokens": 0,
                    "phase": "starting",
                    "status": "pending",
                }
                self._active[job_id_value] = active
            previous_phase = active.get("phase")
            previous_turn = active.get("turn")
            phase = str(event.get("phase") or previous_phase or "working")
            turn = event.get("turn")
            if phase != previous_phase or (turn is not None and turn != previous_turn):
                # These values describe only the phase/turn that emitted them;
                # retaining them would make a later line claim stale timing,
                # usage, or response reuse.
                for key in (
                    "phase_elapsed_seconds", "prompt_tokens", "completion_tokens",
                    "response_total_tokens", "api_reused",
                ):
                    active.pop(key, None)
            active.update({
                key: event[key]
                for key in (
                    "episode_number", "model", "reference_id", "turn", "max_turns",
                    "turns_completed", "status", "phase_elapsed_seconds",
                    "prompt_tokens", "completion_tokens", "response_total_tokens",
                    "api_reused",
                )
                if key in event and event[key] is not None
            })
            episode_total = self._token_value(event.get("total_tokens"))
            if episode_total is not None:
                active["total_tokens"] = max(0, episode_total)
            active["phase"] = phase
            elapsed = time.monotonic() - float(active["started_monotonic"])
            if phase == "turn_complete":
                self._finished_turns += 1
            if phase == "episode_complete":
                self._finished_tokens[job_id_value] = int(active.get("total_tokens") or 0)
                self._active.pop(job_id_value, None)
            known_tokens = sum(int(item.get("total_tokens") or 0) for item in self._active.values())
            known_tokens += sum(self._finished_tokens.values())
            self._sequence += 1
            snapshot = {
                "job_id": job_id_value,
                "sequence": self._sequence,
                "phase": phase,
                "model": active["model"],
                "reference_id": active["reference_id"],
                "turn": active.get("turn", 0),
                "turns_completed": active.get("turns_completed", 0),
                "status": active.get("status", "pending"),
                "elapsed_seconds": round(elapsed, 3),
                "total_tokens": known_tokens,
                "episode_total_tokens": int(active.get("total_tokens") or 0),
            }
            line = self._line(active, phase=phase, elapsed=elapsed, known_tokens=known_tokens)
        self._write(line)
        return snapshot

    def _line(self, active: Mapping[str, Any], *, phase: str, elapsed: float, known_tokens: int) -> str:
        number = active.get("episode_number") or "?"
        total = active.get("episodes_total") or self.episodes_total
        model = safe_slug(str(active.get("model") or "unknown"))
        reference = safe_slug(str(active.get("reference_id") or "unknown"))
        turn = active.get("turn") or 0
        max_turns = active.get("max_turns") or "?"
        turns_completed = active.get("turns_completed") or 0
        line = (
            f"[teacher] episode {number}/{total} model={model} ref={reference} "
            f"turn={turn}/{max_turns} phase={phase} elapsed={format_elapsed(elapsed)} "
            f"turns_completed={turns_completed} known_tokens={known_tokens}"
        )
        phase_elapsed = active.get("phase_elapsed_seconds")
        if isinstance(phase_elapsed, (int, float)):
            line += f" phase_elapsed={format_elapsed(float(phase_elapsed))}"
        if phase == "waiting_for_provider":
            line += " current_response_tokens=unknown (provider response pending; no live token stream)"
        elif phase == "response_received":
            prompt = active.get("prompt_tokens")
            completion = active.get("completion_tokens")
            response_total = active.get("response_total_tokens")
            line += f" response_tokens=prompt:{prompt if prompt is not None else '?'} completion:{completion if completion is not None else '?'} total:{response_total if response_total is not None else '?'}"
        if active.get("api_reused"):
            line += " api=reused"
        return line


class GatewayClient:
    """Persistent AI SDK JSONL transport with bounded retry.

    A process is shared by all episodes for one API-key bundle.  The Node
    transport can answer requests out of order, so concurrent Python workers
    do not create one Node process per turn or per episode.
    """

    def __init__(
        self, *, api_key: str, root: Path, timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES, transport_script: str = TRANSPORT_SCRIPT,
        command: Sequence[str] | None = None, sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise APIError("AI_GATEWAY_API_KEY is not set")
        self.api_key = api_key
        self.root = root.resolve()
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep = sleep
        script = (self.root / transport_script).resolve()
        if self.root not in script.parents or not script.is_file():
            raise APIError(f"AI SDK transport script is missing: {script}")
        self.command = list(command or ("pnpm", "exec", "tsx", script.name))
        child_env = os.environ.copy()
        child_env.pop("OPENROUTER_API_KEY", None)
        child_env["AI_GATEWAY_API_KEY"] = api_key
        try:
            self.process = subprocess.Popen(
                self.command, cwd=str(self.root), env=child_env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except (OSError, ValueError) as exc:
            raise APIError(f"AI SDK transport could not start: {type(exc).__name__}", retryable=True) from exc
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._stderr_tail = ""
        self.last_attempts = 0
        self._reader = threading.Thread(target=self._read_responses, name="ai-gateway-jsonl", daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, name="ai-gateway-stderr", daemon=True)
        self._stderr_reader.start()

    def _read_responses(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        for line in stream:
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    continue
            except json.JSONDecodeError:
                continue
            request_id = value.get("id")
            if not isinstance(request_id, str):
                continue
            with self._pending_lock:
                item = self._pending.get(request_id)
                if item is not None:
                    item[1]["value"] = value
                    item[0].set()
        with self._pending_lock:
            for event, result in self._pending.values():
                result["value"] = {"ok": False, "error": {"message": "AI SDK transport exited", "retryable": True}}
                event.set()

    def _read_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        for line in stream:
            self._stderr_tail = redact((self._stderr_tail + line)[-DEFAULT_MAX_ERROR_CHARS:], self.api_key)

    def complete(self, payload: Mapping[str, Any], *, deadline: float | None = None) -> dict[str, Any]:
        last: APIError | None = None
        self.last_attempts = 0
        for attempt in range(self.max_retries + 1):
            self.last_attempts = attempt + 1
            if deadline is not None and time.monotonic() >= deadline:
                raise APIError("episode deadline reached", retryable=False)
            request_id = uuid.uuid4().hex
            event = threading.Event()
            result_box: dict[str, Any] = {}
            with self._pending_lock:
                self._pending[request_id] = (event, result_box)
            started = time.monotonic()
            try:
                line = json.dumps({"id": request_id, "payload": payload}, ensure_ascii=False, separators=(",", ":")) + "\n"
                with self._write_lock:
                    if self.process.poll() is not None or self.process.stdin is None:
                        raise APIError("AI SDK transport exited", retryable=True)
                    self.process.stdin.write(line)
                    self.process.stdin.flush()
                wait_seconds = float(self.timeout)
                if deadline is not None:
                    wait_seconds = min(wait_seconds, max(0.0, deadline - time.monotonic()))
                if not event.wait(wait_seconds):
                    raise APIError("AI Gateway request timed out", retryable=True)
                value = result_box.get("value")
                if not isinstance(value, dict):
                    raise APIError("AI SDK transport returned no response", retryable=True)
                if not value.get("ok"):
                    error = value.get("error") if isinstance(value.get("error"), dict) else {}
                    message = redact(str(error.get("message") or "AI Gateway request failed"), self.api_key)
                    raise APIError(
                        "AI Gateway request failed", status=error.get("status") if isinstance(error.get("status"), int) else None,
                        retryable=bool(error.get("retryable")), response_body=message,
                    )
                response = value.get("response")
                if not isinstance(response, dict):
                    raise APIError("AI SDK transport returned malformed response", retryable=True)
                response["_latency_seconds"] = round(time.monotonic() - started, 3)
                response["_attempts"] = self.last_attempts
                return response
            except APIError as exc:
                last = exc
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = min(30.0, 2.0 ** attempt)
                if deadline is not None and time.monotonic() + delay >= deadline:
                    raise APIError("episode deadline reached during retry backoff", retryable=False)
                self.sleep(delay)
            except (BrokenPipeError, OSError, ValueError) as exc:
                last = APIError(f"AI SDK transport error: {type(exc).__name__}", retryable=True)
                if attempt >= self.max_retries:
                    raise last from exc
                self.sleep(min(30.0, 2.0 ** attempt))
            finally:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
        raise last or APIError("AI Gateway request failed")

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


class GatewayClientPool:
    """Lazily create one persistent transport per distinct key."""

    def __init__(self, *, root: Path, timeout: float, max_retries: int, transport_script: str):
        self.root = root
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport_script = transport_script
        self._lock = threading.Lock()
        self._clients: dict[str, GatewayClient] = {}

    def get(self, api_key: str) -> GatewayClient:
        with self._lock:
            client = self._clients.get(api_key)
            if client is None:
                client = GatewayClient(
                    api_key=api_key, root=self.root, timeout=self.timeout,
                    max_retries=self.max_retries, transport_script=self.transport_script,
                )
                self._clients[api_key] = client
            return client

    def close(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()


def normalize_usage(response: Mapping[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    result: dict[str, Any] = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens", completion_details.get("reasoning_tokens")),
        "cost": usage.get("cost", response.get("cost")),
        "cost_details": usage.get("cost_details"),
        "raw": usage,
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        if isinstance(result[key], str):
            try:
                result[key] = int(result[key])
            except ValueError:
                pass
    return result


def response_record(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    content = extract_content(message.get("content"))
    reasoning = message.get("reasoning", message.get("reasoning_content"))
    return {
        "raw_text": content,
        "reasoning": reasoning if isinstance(reasoning, (str, list, dict)) else None,
        "usage": normalize_usage(response),
        "finish_reason": choice.get("finish_reason"),
        "provider": choice.get("provider") or response.get("provider"),
        "id": response.get("id"),
        "model": response.get("model"),
        "raw_response": redact(dict(response)),
        "latency_seconds": response.get("_latency_seconds"),
        "attempts": response.get("_attempts"),
        "http_status": response.get("_http_status"),
        "errors": response.get("error") if response.get("error") else None,
    }


def token_total(usage: Mapping[str, Any]) -> int:
    value = usage.get("total_tokens")
    if isinstance(value, (int, float)):
        return int(value)
    return sum(int(usage.get(key) or 0) for key in ("prompt_tokens", "completion_tokens"))


def cost_value(usage: Mapping[str, Any]) -> float | None:
    value = usage.get("cost")
    return float(value) if isinstance(value, (int, float)) else None


def is_finished_text(text: str) -> bool:
    """Recognize the two finish forms allowed by the teacher prompt.

    A model may put its brief reason in a Markdown ``**Reason**:`` paragraph
    and append the finish marker to that paragraph.  Keep the broader marker
    out of ordinary prose and code by requiring either the historical
    standalone line or that explicit reason label, with ``FINISHED`` as the
    final token.  Fenced responses are never finish-only responses; callers
    still parse and render their program first.
    """

    value = str(text or "")
    if "```" in value:
        return False
    if re.fullmatch(r"\s*FINISHED\s*[.!]?\s*", value, flags=re.IGNORECASE):
        return True
    if re.search(r"(?im)^\s*FINISHED\s*[.!]?\s*$", value):
        return True
    return bool(re.fullmatch(
        r"\s*\*\*Reason\*\*\s*:\s*.+?\s+FINISHED\s*[.!]?\s*",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    ))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def local_dotenv_api_key(root: Path) -> str | None:
    """Read only ``AI_GATEWAY_API_KEY`` from the ignored local dotenv file.

    The local launcher uses this instead of sourcing the file in a shell.  A
    dotenv file is user-controlled input, so executing it would both be
    surprising and needlessly expose unrelated variables to the benchmark.
    The key stays in memory and is never included in a progress or result
    record.
    """

    path = root / ".env.local"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkError(f"could not read local dotenv file: {path}") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if separator and name.strip() == "AI_GATEWAY_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not value or "\x00" in value:
                raise BenchmarkError("AI_GATEWAY_API_KEY in .env.local is empty or invalid")
            return value
    return None


def _render_result_is_service_failure(result: Mapping[str, Any]) -> bool:
    return bool(result.get("service_failure") or result.get("error_code") in {"renderer_error", "render_timeout", "worker_failure"})


def render_program(
    *, root: Path, source: Path, output: Path, renderer: Path, renderer_python: Path,
    timeout: int = 180, browser_path: Path | None = None, run_as_user: str = "painter",
    local: bool = False, renderer_backend: str = "metal",
) -> dict[str, Any]:
    """Invoke the pinned renderer with a secret-free child environment.

    Linux keeps the existing unprivileged ``runuser`` boundary.  The explicit
    local mode is for macOS, where the same renderer can use Chromium's Metal
    backend under the current user.  It is opt-in so a normal node launch
    cannot accidentally run with a weaker local process boundary.
    """

    if local:
        if sys.platform != "darwin":
            raise BenchmarkError("--local rendering requires macOS")
        if renderer_backend not in {"metal", "swiftshader"}:
            raise BenchmarkError("local renderer backend must be metal or swiftshader")
    elif sys.platform != "linux":
        raise BenchmarkError("rendering generated JavaScript is Linux-only; use --local on macOS")
    for path in (source, renderer, renderer_python):
        if not path.is_file():
            raise BenchmarkError(f"renderer input is missing: {path}")
    if not 1 <= int(timeout) <= 180:
        raise BenchmarkError("renderer timeout must be between 1 and 180 seconds")
    source = source.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700 if local else 0o777)
    if local:
        command = [
            str(renderer_python.absolute()), str(renderer.absolute()), str(source), "--output", str(output),
            "--timeout", str(int(timeout)), "--backend", renderer_backend,
        ]
    else:
        command = [
            "/usr/sbin/runuser", "-u", run_as_user, "--", str(renderer_python.absolute()), str(renderer.absolute()),
            str(source), "--output", str(output), "--timeout", str(int(timeout)), "--backend", "swiftshader",
        ]
    temp_root = Path(tempfile.mkdtemp(prefix="brush-render-env-"))
    # ``runuser -u painter`` must be able to enter the temporary HOME/TMPDIR;
    # the renderer itself still runs with a fresh profile and no credentials.
    temp_root.chmod(0o777 if not local else 0o700)
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(temp_root),
        "TMPDIR": str(temp_root),
        "LANG": "en_US.UTF-8",
    }
    if browser_path:
        child_env["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_path.resolve())
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=int(timeout) + 20, check=False, env=child_env,
        )
    except Exception as exc:
        return {"valid": False, "service_failure": True, "error_code": "renderer_error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    receipt_path = output.with_suffix(".json")
    if not receipt_path.is_file():
        return {
            "valid": False, "service_failure": True, "error_code": "renderer_error",
            "error": redact((completed.stderr or "")[-DEFAULT_MAX_ERROR_CHARS:]),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    receipt = _load_json(receipt_path) or {}
    if not receipt.get("valid"):
        service = bool(receipt.get("error_code") or receipt.get("timed_out"))
        result = {
            "valid": False,
            "service_failure": service,
            "error_code": receipt.get("error_code"),
            "errors": receipt.get("errors") or receipt.get("error") or (completed.stderr or "")[-DEFAULT_MAX_ERROR_CHARS:],
            "receipt": receipt,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        return redact(result)
    if completed.returncode != 0 or not output.is_file():
        return {
            "valid": False, "service_failure": True, "error_code": "renderer_error",
            "error": f"renderer exited {completed.returncode}; {(completed.stderr or '')[-1000:]}",
            "receipt": receipt,
        }
    if receipt.get("source_sha256") != sha_file(source) or receipt.get("png_sha256") != sha_file(output):
        return {"valid": False, "service_failure": True, "error_code": "renderer_receipt_mismatch", "receipt": receipt}
    return {
        "valid": True,
        "service_failure": False,
        "canvas_sha256": sha_file(output),
        "render_seconds": receipt.get("painting_seconds"),
        "receipt": receipt,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def make_episode_record(inputs: BenchmarkInputs, model: str, reference: Reference, sample: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "job_id": job_id(model, reference.id, sample, str(settings["track"])),
        "model": model,
        "reference_id": reference.id,
        "reference_image": _relative(reference.image, inputs.root),
        "reference_sha256": reference.sha256,
        "sample": sample,
        "inputs": {
            "config_sha256": inputs.config_sha256,
            "prompt_sha256": inputs.prompt_sha256,
            "refs_sha256": inputs.refs_sha256,
            "catalog_sha256": inputs.catalog_sha256,
        },
        "settings": dict(settings),
        "resolved_reasoning": resolve_reasoning(inputs, model, str(settings["track"])),
        "resolved_model_profile": catalog_model(inputs, model),
        "status": "pending",
        "turns": [],
        "first_valid_canvas": None,
        "final_valid_canvas": None,
        "last_turn_invalid": False,
        "total_tokens": 0,
        "total_cost": 0.0,
        "known_cost": 0.0,
        "cost_missing_turns": 0,
        "cost_complete": True,
    }


class EpisodeRunner:
    def __init__(
        self, inputs: BenchmarkInputs, model: str, reference: Reference, sample: int,
        *, episode_dir: Path, client: GatewayClient, render_fn: Callable[..., dict[str, Any]],
        render_lock: threading.Semaphore, renderer_options: Mapping[str, Any], settings: Mapping[str, Any],
        on_turn: Callable[[Mapping[str, Any], int], None] | None = None,
        on_progress: Callable[[Mapping[str, Any]], None] | None = None,
        progress_heartbeat_seconds: float = PROGRESS_HEARTBEAT_SECONDS,
    ):
        self.inputs = inputs
        self.model = model
        self.reference = reference
        self.sample = sample
        self.episode_dir = episode_dir
        self.episode_path = episode_dir / "episode.json"
        self.client = client
        self.render_fn = render_fn
        self.render_lock = render_lock
        self.renderer_options = renderer_options
        self.settings = settings
        self.on_turn = on_turn
        self.on_progress = on_progress
        self.progress_heartbeat_seconds = max(0.1, float(progress_heartbeat_seconds))
        self.model_info = model_limits(inputs, model, settings)
        self.reasoning = resolve_reasoning(inputs, model, str(settings["track"]))

    def _load_or_init(self) -> dict[str, Any]:
        existing = _load_json(self.episode_path)
        expected = make_episode_record(self.inputs, self.model, self.reference, self.sample, self.settings)
        if existing is None:
            self.episode_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(self.episode_path, expected)
            return expected
        immutable = ("job_id", "model", "reference_id", "reference_sha256")
        if any(existing.get(key) != expected.get(key) for key in immutable):
            raise BenchmarkError(f"episode identity mismatch: {self.episode_path}")
        expected_settings = dict(self.settings)
        if existing.get("settings") != expected_settings:
            raise BenchmarkError(f"episode settings changed; use a new benchmark directory: {self.episode_path}")
        expected_inputs = expected.get("inputs", {})
        existing_inputs = existing.get("inputs", {})
        if any(existing_inputs.get(key) != expected_inputs.get(key) for key in ("prompt_sha256",)):
            raise BenchmarkError(f"episode prompt identity changed; use a new benchmark directory: {self.episode_path}")
        if existing.get("resolved_model_profile") != expected.get("resolved_model_profile"):
            raise BenchmarkError(f"model catalog profile changed; use a new benchmark directory: {self.episode_path}")
        return existing

    def _save(self, state: Mapping[str, Any]) -> None:
        atomic_json(self.episode_path, state)

    def _notify_turn(self, state: Mapping[str, Any], turn: int) -> None:
        if self.on_turn:
            self.on_turn(state, turn)

    def _notify_progress(self, phase: str, turn: int, state: Mapping[str, Any] | None = None, **details: Any) -> None:
        if not self.on_progress:
            return
        event: dict[str, Any] = {
            "job_id": self.episode_path.parent.name,
            "model": self.model,
            "reference_id": self.reference.id,
            "sample": self.sample,
            "track": self.settings.get("track"),
            "turn": turn,
            "max_turns": self.settings.get("max_turns"),
            "phase": phase,
        }
        if state is not None:
            event.update({
                "status": state.get("status"),
                "turns_completed": len(state.get("turns", [])),
                "total_tokens": state.get("total_tokens"),
            })
        event.update(details)
        try:
            self.on_progress(event)
        except Exception:
            # Observability must never change benchmark semantics.
            return

    def _with_progress_heartbeat(self, *, turn: int, phase: str, operation: Callable[[], Any]) -> Any:
        """Run a blocking operation while reporting truthful elapsed heartbeats."""

        started = time.monotonic()
        stopped = threading.Event()
        self._notify_progress(phase, turn, phase_elapsed_seconds=0.0)

        def heartbeat() -> None:
            while not stopped.wait(self.progress_heartbeat_seconds):
                self._notify_progress(
                    phase, turn,
                    phase_elapsed_seconds=round(time.monotonic() - started, 3),
                )

        thread = threading.Thread(target=heartbeat, name=f"teacher-progress-{phase}", daemon=True)
        thread.start()
        try:
            return operation()
        finally:
            stopped.set()
            thread.join(timeout=max(1.0, self.progress_heartbeat_seconds + 0.5))

    def _turn_path(self, turn: int) -> Path:
        return self.episode_dir / f"turn-{turn:02d}.json"

    def _program_path(self, turn: int) -> Path:
        return self.episode_dir / f"turn-{turn:02d}.program.js"

    def _load_reusable_response(self, turn: int, expected_fp: str) -> dict[str, Any] | None:
        value = _load_json(self._turn_path(turn))
        if value and value.get("request_fingerprint") == expected_fp and value.get("api_status") == "completed":
            return value
        return None

    def run(self) -> dict[str, Any]:
        state = self._load_or_init()
        if state.get("status") in {"complete", "turn_limit", "invalid", "context_exhausted", "deadline_censored"}:
            return state
        if state.get("status") == "renderer_error" and state.get("turns"):
            # Keep the completed API response on disk; remove only the
            # state's failed render turn so the exact response is reused.
            state["turns"] = state["turns"][:-1]
            state["status"] = "pending"
            self._save(state)
        if state.get("turns") and state["turns"][-1].get("api_status") == "error":
            state["turns"] = state["turns"][:-1]
            state["status"] = "pending"
            self._save(state)
        current_canvas: Path | None = None
        previous_response: str | None = None
        render_feedback: str | None = None
        last_turn_invalid = False
        started = time.monotonic()
        prior_active = float(state.get("active_seconds") or 0.0)
        render_queue_seconds = float(state.get("render_queue_seconds") or 0.0)

        def active_seconds() -> float:
            return max(0.0, prior_active + (time.monotonic() - started) - render_queue_seconds)
        for old_turn in state.get("turns", []):
            retained = old_turn.get("current_canvas_after") or (old_turn.get("canvas") if old_turn.get("render", {}).get("valid") else None)
            if retained:
                candidate = self.inputs.root / retained
                if candidate.is_file():
                    current_canvas = candidate
            if old_turn.get("response", {}).get("raw_text") is not None:
                previous_response = old_turn["response"]["raw_text"]
            render_feedback = old_turn.get("render_feedback")
        start_turn = len(state.get("turns", [])) + 1
        self._notify_progress("episode_started", 0, state=state)
        # Turn artifacts can outlive a process after the episode state was
        # written.  Reconstruct from them only when the state contains the turn.
        for turn in range(start_turn, int(self.settings["max_turns"]) + 1):
            self._notify_progress("preparing_turn", turn, state=state)
            episode_timeout = self.settings.get("episode_timeout_seconds")
            if episode_timeout is not None and active_seconds() >= float(episode_timeout):
                state["status"] = "deadline_censored"
                state["deadline_censored"] = True
                state["deadline_seconds"] = float(episode_timeout)
                state["active_seconds"] = round(active_seconds(), 3)
                state["render_queue_seconds"] = round(render_queue_seconds, 3)
                state["last_turn_invalid"] = bool(last_turn_invalid)
                self._save(state)
                return state
            safe_feedback = render_feedback or ""
            current_sha = sha_file(current_canvas) if current_canvas and current_canvas.is_file() else None
            previous_sha = sha_bytes(previous_response.encode("utf-8")) if previous_response is not None else None
            messages, safe_messages = build_conversation_messages(
                inputs=self.inputs, reference=self.reference, prior_turns=state.get("turns", []),
                current_canvas=current_canvas, render_feedback=render_feedback, track=str(self.settings["track"]),
            )
            context = context_accounting(messages, self.model_info, self.model_info["requested_max_tokens"] if self.settings["max_tokens"] == "native" else int(self.settings["max_tokens"]))
            history_sha = sha_bytes(canonical(safe_messages))
            binding = request_binding(
                inputs=self.inputs, model=self.model, reference=self.reference, turn=turn,
                settings=self.settings, current_sha256=current_sha, prior_response_sha256=previous_sha,
                render_feedback=safe_feedback, reasoning=self.reasoning, context=context, history_sha256=history_sha,
            )
            request_fp = fingerprint(binding)
            api_path = self._turn_path(turn)
            turn_record = self._load_reusable_response(turn, request_fp)
            reused_response = turn_record is not None
            if context["effective_max_tokens"] is not None and context["effective_max_tokens"] <= 0:
                turn_record = {
                    "schema": SCHEMA,
                    "turn": turn,
                    "request_fingerprint": request_fp,
                    "request_binding": binding,
                    "request_messages": safe_messages,
                    "api_status": "context_exhausted",
                    "api_reused": False,
                    "response": None,
                    "api_error": {"message": "physical context remaining is zero", "status": None, "retryable": False},
                }
                atomic_json(api_path, turn_record)
                state["turns"].append(turn_record)
                state["status"] = "context_exhausted"
                state["last_turn_invalid"] = bool(last_turn_invalid)
                self._save(state)
                return state
            if turn_record is None:
                payload = build_payload(messages, self.settings, self.model, max_tokens=context["effective_max_tokens"], reasoning=self.reasoning)
                try:
                    old_timeout = self.client.timeout
                    episode_timeout = self.settings.get("episode_timeout_seconds")
                    if episode_timeout is not None:
                        remaining = float(episode_timeout) - active_seconds()
                        if remaining <= 0:
                            raise APIError("episode deadline reached", retryable=False)
                        self.client.timeout = min(old_timeout, max(1.0, remaining))
                        deadline_at = time.monotonic() + remaining
                    else:
                        deadline_at = None
                    raw = self._with_progress_heartbeat(
                        turn=turn,
                        phase="waiting_for_provider",
                        operation=lambda: self.client.complete(payload, deadline=deadline_at),
                    )
                    self.client.timeout = old_timeout
                    response = response_record(raw)
                    usage = response.get("usage") or {}
                    self._notify_progress(
                        "response_received", turn, state=state,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        response_total_tokens=token_total(usage),
                        total_tokens=int(state.get("total_tokens") or 0) + token_total(usage),
                    )
                    turn_record = {
                        "schema": SCHEMA,
                        "turn": turn,
                        "request_fingerprint": request_fp,
                        "request_binding": binding,
                        "request_messages": safe_messages,
                        "request_current_canvas": _relative(current_canvas, self.inputs.root) if current_canvas else None,
                        "request_render_feedback": render_feedback,
                        "api_status": "completed",
                        "api_reused": False,
                        "response": response,
                        "api_error": None,
                    }
                except APIError as exc:
                    if "old_timeout" in locals():
                        self.client.timeout = old_timeout
                    self._notify_progress("provider_error", turn, state=state, status="api_error")
                    turn_record = {
                        "schema": SCHEMA,
                        "turn": turn,
                        "request_fingerprint": request_fp,
                        "request_binding": binding,
                        "request_messages": safe_messages,
                        "request_current_canvas": _relative(current_canvas, self.inputs.root) if current_canvas else None,
                        "request_render_feedback": render_feedback,
                        "api_status": "error",
                        "api_reused": False,
                        "response": None,
                        "api_error": {
                            "message": redact(str(exc), self.client.api_key),
                            "status": exc.status,
                            "retryable": exc.retryable,
                            "response_body": redact(exc.response_body or "", self.client.api_key),
                        },
                        "attempts": getattr(self.client, "last_attempts", None),
                    }
                    atomic_json(api_path, turn_record)
                    state["turns"].append(turn_record)
                    state["status"] = "deadline_censored" if "deadline" in str(exc).lower() and self.settings.get("episode_timeout_seconds") is not None else "api_error"
                    state["deadline_censored"] = state["status"] == "deadline_censored"
                    state["active_seconds"] = round(active_seconds(), 3)
                    state["render_queue_seconds"] = round(render_queue_seconds, 3)
                    state["last_turn_invalid"] = False
                    self._save(state)
                    self._notify_turn(state, turn)
                    return state
            if reused_response:
                turn_record["api_reused"] = True
                atomic_json(api_path, turn_record)
                reused_usage = (turn_record.get("response") or {}).get("usage") or {}
                self._notify_progress(
                    "response_reused", turn, state=state, api_reused=True,
                    prompt_tokens=reused_usage.get("prompt_tokens"),
                    completion_tokens=reused_usage.get("completion_tokens"),
                    response_total_tokens=token_total(reused_usage),
                    total_tokens=int(state.get("total_tokens") or 0) + token_total(reused_usage),
                )
            response = turn_record.get("response") or {}
            reply = response.get("raw_text") or ""
            program, plan = parse_program(reply)
            turn_record["plan"] = plan
            turn_record["program_found"] = program is not None
            if program is not None:
                program_path = self._program_path(turn)
                atomic_bytes(program_path, program.encode("utf-8"))
                turn_record["program"] = _relative(program_path, self.inputs.root)
            render = {"valid": False, "skipped": True}
            canvas_path: Path | None = None
            feedback = None
            if program is not None:
                canvas_path = self.episode_dir / f"turn-{turn:02d}.png"
                if canvas_path.exists():
                    receipt = canvas_path.with_suffix(".json")
                    if receipt.exists():
                        receipt_value = _load_json(receipt) or {}
                        source_hash_ok = receipt_value.get("source_sha256") == sha_file(self.inputs.root / turn_record["program"])
                        canvas_hash_ok = receipt_value.get("png_sha256") == sha_file(canvas_path)
                        render = receipt_value if receipt_value.get("valid") and source_hash_ok and canvas_hash_ok else {
                            "valid": False, "service_failure": True, "error_code": "renderer_receipt_mismatch",
                            "errors": "existing render receipt or source/canvas hash does not match",
                        }
                    else:
                        canvas_path.unlink(missing_ok=True)
                if not render.get("valid"):
                    canvas_path.unlink(missing_ok=True)
                    canvas_path.with_suffix(".json").unlink(missing_ok=True)
                if not render.get("valid"):
                    try:
                        remaining = None
                        episode_timeout = self.settings.get("episode_timeout_seconds")
                        if episode_timeout is not None:
                            remaining = float(episode_timeout) - active_seconds()
                            if remaining <= 0:
                                state["status"] = "deadline_censored"
                                state["deadline_censored"] = True
                                state["deadline_seconds"] = float(episode_timeout)
                                state["active_seconds"] = round(active_seconds(), 3)
                                state["render_queue_seconds"] = round(render_queue_seconds, 3)
                                state["last_turn_invalid"] = bool(last_turn_invalid)
                                self._save(state)
                                return state
                        queue_started = time.monotonic()
                        self.render_lock.acquire()
                        render_queue_seconds += time.monotonic() - queue_started
                        try:
                            if episode_timeout is not None:
                                remaining = float(episode_timeout) - active_seconds()
                                if remaining <= 0:
                                    render = {"valid": False, "service_failure": False, "deadline_censored": True, "error_code": "episode_deadline"}
                            if remaining is None or remaining > 0:
                                render = self._with_progress_heartbeat(
                                    turn=turn,
                                    phase="rendering",
                                    operation=lambda: self.render_fn(
                                        root=self.inputs.root, source=self.inputs.root / turn_record["program"],
                                        output=canvas_path, timeout=max(1, min(int(self.inputs.config.get("renderer_timeout", 180)), int(remaining) if remaining is not None else int(self.inputs.config.get("renderer_timeout", 180)))),
                                        **self.renderer_options,
                                    ),
                                )
                        finally:
                            self.render_lock.release()
                    except BenchmarkError as exc:
                        render = {"valid": False, "service_failure": True, "error_code": "renderer_error", "error": str(exc)}
                if render.get("valid") and canvas_path.is_file():
                    current_canvas = canvas_path
                    if state.get("first_valid_canvas") is None:
                        state["first_valid_canvas"] = _relative(canvas_path, self.inputs.root)
                    state["final_valid_canvas"] = _relative(canvas_path, self.inputs.root)
                    last_turn_invalid = False
                    feedback = "The submitted program rendered successfully. Inspect the CURRENT CANVAS and continue revising or finish after observing it."
                else:
                    last_turn_invalid = True
                    feedback = "The submitted program did not produce a valid canvas. " + redact(str(render.get("errors") or render.get("error") or "renderer failure"))
            elif is_finished_text(reply) and current_canvas is not None:
                turn_record["finished_observed"] = True
                turn_record["render"] = {"valid": True, "skipped": True, "finished_without_code": True}
                turn_record["canvas"] = _relative(current_canvas, self.inputs.root)
                turn_record["current_canvas_after"] = _relative(current_canvas, self.inputs.root)
                turn_record["current_canvas_sha256"] = sha_file(current_canvas)
                turn_record["render_feedback"] = None
                state["turns"].append(turn_record)
                state["status"] = "complete"
                state["last_turn_invalid"] = False
                state["active_seconds"] = round(active_seconds(), 3)
                state["render_queue_seconds"] = round(render_queue_seconds, 3)
                self._update_totals(state)
                atomic_json(api_path, turn_record)
                self._save(state)
                self._notify_progress("turn_complete", turn, state=state)
                self._notify_turn(state, turn)
                return state
            else:
                last_turn_invalid = True
                feedback = "No complete JavaScript sketch was returned. Return a complete sketch in a javascript code block; FINISHED is only allowed after an observed valid canvas."
            turn_record["render"] = render
            turn_record["canvas"] = _relative(canvas_path, self.inputs.root) if render.get("valid") and canvas_path else None
            turn_record["current_canvas_after"] = _relative(current_canvas, self.inputs.root) if current_canvas else None
            turn_record["current_canvas_sha256"] = sha_file(current_canvas) if current_canvas and current_canvas.is_file() else None
            turn_record["render_feedback"] = feedback
            turn_record["finished_observed"] = False
            turn_record["render_error"] = None if render.get("valid") else redact(str(render.get("errors") or render.get("error") or ""))
            turn_record["active_seconds_after"] = round(active_seconds(), 3)
            turn_record["render_queue_seconds_after"] = round(render_queue_seconds, 3)
            state["turns"].append(turn_record)
            state["active_seconds"] = round(active_seconds(), 3)
            state["render_queue_seconds"] = round(render_queue_seconds, 3)
            self._update_totals(state)
            atomic_json(api_path, turn_record)
            self._save(state)
            self._notify_progress("turn_complete", turn, state=state)
            self._notify_turn(state, turn)
            previous_response = reply
            render_feedback = feedback
            if render.get("deadline_censored"):
                state["status"] = "deadline_censored"
                state["deadline_censored"] = True
                state["deadline_seconds"] = float(self.settings.get("episode_timeout_seconds"))
                state["last_turn_invalid"] = True
                self._save(state)
                return state
            if _render_result_is_service_failure(render):
                state["status"] = "renderer_error"
                state["last_turn_invalid"] = True
                self._save(state)
                return state
        state["status"] = "invalid" if last_turn_invalid else "turn_limit"
        state["last_turn_invalid"] = bool(last_turn_invalid)
        if state["status"] == "turn_limit" and state.get("final_valid_canvas"):
            state["status"] = "turn_limit"
        self._save(state)
        return state

    @staticmethod
    def _update_totals(state: dict[str, Any]) -> None:
        total_tokens = 0
        total_cost = 0.0
        missing_cost = 0
        for turn in state.get("turns", []):
            usage = (turn.get("response") or {}).get("usage") or {}
            total_tokens += token_total(usage)
            cost = cost_value(usage)
            if cost is not None:
                total_cost += cost
            else:
                missing_cost += 1
        state["total_tokens"] = total_tokens
        state["total_cost"] = total_cost
        state["known_cost"] = total_cost
        state["cost_missing_turns"] = missing_cost
        state["cost_complete"] = missing_cost == 0


def references_for_track(inputs: BenchmarkInputs, track: str) -> tuple[Reference, ...]:
    """Return the immutable reference set for a track.

    Quality and speed retain all 40 references. Screening uses the explicit
    one-per-category IDs in config so a small screen cannot be dominated by
    the first category's contiguous rows in refs.json.
    """
    if track != "screen":
        return inputs.references
    screen_ids = inputs.config.get("screen_reference_ids")
    if isinstance(screen_ids, list) and screen_ids:
        by_id = {reference.id: reference for reference in inputs.references}
        return tuple(by_id[item] for item in screen_ids if item in by_id)
    # Keep hand-built test fixtures useful even when they omit the optional
    # screen manifest: choose the first row in each category deterministically.
    selected: list[Reference] = []
    seen_categories: set[str] = set()
    for reference in inputs.references:
        category = str(reference.metadata.get("category", reference.id))
        if category not in seen_categories:
            selected.append(reference)
            seen_categories.add(category)
    return tuple(selected)


def jobs(inputs: BenchmarkInputs, *, track: str = "all", limit: int | None = None) -> list[tuple[str, str, Reference, int]]:
    # Preserve the existing quality+speed meaning of `all`. Screening is an
    # explicit first pass, not an extra charge added to established launches.
    track_names = ["quality", "speed"] if track == "all" else [track]
    values = [
        (track_name, model, reference, sample)
        for track_name in track_names
        for reference in references_for_track(inputs, track_name)
        for model in inputs.config["models"]
        for sample in range(1, int(inputs.config.get("samples_per_image", 1)) + 1)
    ]
    return values[:limit] if limit is not None else values


def dry_run_report(inputs: BenchmarkInputs, *, track: str = "all", limit: int | None = None) -> dict[str, Any]:
    selected = jobs(inputs, track=track, limit=limit)
    all_jobs = jobs(inputs, track=track)
    per_track = {
        name: {
            "episodes": len(jobs(inputs, track=name)),
            "reference_ids": [reference.id for reference in references_for_track(inputs, name)],
            "max_turns": settings_from_inputs(inputs, name)["max_turns"],
            "max_tokens": settings_from_inputs(inputs, name)["max_tokens"],
            "max_requests": len(jobs(inputs, track=name)) * settings_from_inputs(inputs, name)["max_turns"],
        }
        for name in (["quality", "speed"] if track == "all" else [track])
    }
    image_bytes = sum(ref.image.stat().st_size for _, _, ref, _ in selected)
    max_context = max((len(inputs.prompt) + len(image_data_uri(ref.image)) for _, _, ref, _ in selected), default=len(inputs.prompt))
    return {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "transport": "ai-sdk-generateText-jsonl",
        "mode": "dry-run",
        "track": track,
        "models": list(inputs.config["models"]),
        "references": len(inputs.references),
        "reference_ids_selected": list(dict.fromkeys(reference.id for _, _, reference, _ in selected)),
        "episodes_total": len(all_jobs),
        "episodes_selected": len(selected),
        "requests_max_selected": sum(settings_from_inputs(inputs, item[0])["max_turns"] for item in selected),
        "requests_max_total": sum(settings_from_inputs(inputs, item[0])["max_turns"] for item in all_jobs),
        "samples_per_image": int(inputs.config.get("samples_per_image", 1)),
        "tracks": per_track,
        "temperature": inputs.config.get("temperature", 0.7),
        "api_concurrency": int(inputs.config.get("concurrency", 3)),
        "render_concurrency": int(inputs.config.get("render_concurrency", 1)),
        "reference_bytes_selected": image_bytes,
        "max_reference_context_chars": max_context,
        "paid_calls_made": 0,
        "api_key_read": False,
    }


def preflight_renderer(inputs: BenchmarkInputs, overrides: Mapping[str, Any]) -> dict[str, Any]:
    local = bool(overrides.get("local"))
    if local:
        if sys.platform != "darwin":
            raise BenchmarkError("--local requires macOS; use the existing node launcher on Linux")
        renderer = Path(
            overrides.get("renderer")
            or inputs.config.get("local_renderer")
            or inputs.root / "../../vendor/integrations/watercolour/renderer.py"
        )
        renderer_python = Path(overrides.get("renderer_python") or sys.executable)
        default_browser = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        if not default_browser:
            workspace_browsers = inputs.root.parents[2] / "work/playwright-browsers"
            default_browser = str(workspace_browsers) if workspace_browsers.is_dir() else None
    else:
        if sys.platform != "linux":
            raise BenchmarkError("--run requires Linux; use --local on macOS")
        renderer = Path(overrides.get("renderer") or inputs.config.get("renderer") or inputs.root / "../../native_painting_renderer.py")
        renderer_python = Path(overrides.get("renderer_python") or inputs.config.get("renderer_python") or inputs.root / "../../renderer-env/bin/python")
        default_browser = None
    if not renderer.is_file():
        raise BenchmarkError(f"renderer is missing: {renderer}")
    if not renderer_python.is_file():
        raise BenchmarkError(f"renderer Python is missing: {renderer_python}")
    return {
        "renderer": renderer.absolute(),
        "renderer_python": renderer_python.absolute(),
        "browser_path": Path(overrides["browser_path"]).resolve() if overrides.get("browser_path") else (Path(default_browser).resolve() if default_browser else None),
        "run_as_user": None if local else (overrides.get("run_as_user") or inputs.config.get("run_as_user", "painter")),
        "local": local,
        "renderer_backend": overrides.get("renderer_backend") or ("metal" if local else "swiftshader"),
    }


def run_benchmark(inputs: BenchmarkInputs, *, track: str = "all", limit: int | None = None, renderer_options: Mapping[str, Any], api_key: str | Mapping[str, str] | None = None) -> dict[str, Any]:
    lock_handle = inputs.root.joinpath(".benchmark.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise BenchmarkError("another benchmark run already holds the output lock") from exc
    configured_keys: Mapping[str, str] | None = api_key if isinstance(api_key, Mapping) else None
    default_key = None if configured_keys is not None else (api_key or os.environ.get("AI_GATEWAY_API_KEY"))
    if not default_key and configured_keys is None and renderer_options.get("local"):
        default_key = local_dotenv_api_key(root)
    if not default_key and configured_keys is None and sys.stdin.isatty():
        default_key = getpass.getpass("Vercel AI Gateway API key (input hidden): ").strip()
    if not default_key and not configured_keys:
        raise APIError("AI_GATEWAY_API_KEY is not set; dry-run does not need a key")
    selected = jobs(inputs, track=track, limit=limit)
    render_lock = threading.Semaphore(int(inputs.config.get("render_concurrency", 1)))
    root = inputs.root
    transport_script = str(inputs.config.get("transport_script", TRANSPORT_SCRIPT))
    client_pool = GatewayClientPool(
        root=root, timeout=float(inputs.config.get("request_timeout_seconds", DEFAULT_TIMEOUT)),
        max_retries=int(inputs.config.get("max_retries", DEFAULT_MAX_RETRIES)),
        transport_script=transport_script,
    )
    progress_path = root / "progress.json"
    events_path = root / "events.jsonl"
    progress_lock = threading.Lock()
    progress: dict[str, Any] = {
        "schema": SCHEMA + ".progress",
        "track": track,
        "episodes_requested": len(selected),
        "episodes_completed": 0,
        "turns_completed": 0,
        "total_tokens": 0,
        "status_counts": {},
        "current": None,
        "live": None,
        "live_sequence": 0,
        "started_at": time.time(),
    }
    atomic_json(progress_path, progress)
    progress_reporter = ProgressReporter(episodes_total=len(selected))

    def on_progress(event: Mapping[str, Any]) -> None:
        snapshot = progress_reporter.emit(event)
        with progress_lock:
            if int(snapshot.get("sequence") or 0) < int(progress.get("live_sequence") or 0):
                return
            progress["live_sequence"] = snapshot["sequence"]
            progress["live"] = snapshot
            progress["total_tokens"] = snapshot["total_tokens"]
            progress["current"] = {
                key: event.get(key)
                for key in ("job_id", "track", "model", "reference_id", "turn", "phase", "status")
                if event.get(key) is not None
            }
            atomic_json(progress_path, progress)

    def on_turn(state: Mapping[str, Any], turn: int) -> None:
        with progress_lock:
            progress["turns_completed"] += 1
            progress["current"] = {
                "job_id": state.get("job_id"), "track": state.get("settings", {}).get("track"),
                "model": state.get("model"), "reference_id": state.get("reference_id"), "turn": turn,
                "status": state.get("status"),
            }
            atomic_json(progress_path, progress)
            event = {
                "event": "turn",
                "at": time.time(),
                "job_id": state.get("job_id"), "track": state.get("settings", {}).get("track"),
                "model": state.get("model"), "reference_id": state.get("reference_id"), "turn": turn,
                "status": state.get("status"), "total_tokens": state.get("total_tokens"),
            }
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

    def run_one(item: tuple[str, str, Reference, int]) -> dict[str, Any]:
        track_name, model, reference, sample = item
        settings = settings_from_inputs(inputs, track_name)
        settings = dict(settings)
        renderer_path = Path(renderer_options["renderer"])
        renderer_python_path = Path(renderer_options["renderer_python"])
        asset_dir = renderer_path.parent / "assets"
        renderer_assets = {
            asset.name: sha_file(asset)
            for asset in sorted(asset_dir.glob("*.js"))
            if asset.is_file()
        }
        settings["renderer_identity"] = {
            "renderer": str(renderer_path.absolute()),
            "renderer_sha256": sha_file(renderer_path),
            "renderer_python": str(renderer_python_path.absolute()),
            "renderer_python_sha256": sha_file(renderer_python_path),
            "renderer_assets_sha256": renderer_assets,
            "browser_path": str(renderer_options.get("browser_path")) if renderer_options.get("browser_path") else None,
            "run_as_user": renderer_options.get("run_as_user"),
        }
        model_key = (configured_keys or {}).get(model) or (configured_keys or {}).get("default") or default_key
        if not model_key:
            raise APIError(f"no API key configured for model {model}")
        client = client_pool.get(model_key)
        client.timeout = float(settings["api_timeout_seconds"])
        client.max_retries = int(settings["max_retries"])
        directory = root / "episodes" / job_id(model, reference.id, sample, track_name)
        episode_job_id = job_id(model, reference.id, sample, track_name)
        episode_number = selected.index(item) + 1
        on_progress({
            "job_id": episode_job_id,
            "episode_number": episode_number,
            "episodes_total": len(selected),
            "model": model,
            "reference_id": reference.id,
            "track": track_name,
            "turn": 0,
            "max_turns": settings["max_turns"],
            "phase": "starting",
            "status": "pending",
            "total_tokens": 0,
        })
        runner = EpisodeRunner(
            inputs, model, reference, sample, episode_dir=directory, client=client,
            render_fn=render_program, render_lock=render_lock, renderer_options=renderer_options, settings=settings,
            on_turn=on_turn, on_progress=lambda event: on_progress({**event, "episode_number": episode_number, "episodes_total": len(selected)}),
        )
        try:
            result = runner.run()
            on_progress({
                "job_id": episode_job_id,
                "episode_number": episode_number,
                "episodes_total": len(selected),
                "model": model,
                "reference_id": reference.id,
                "track": track_name,
                "turn": len(result.get("turns", [])),
                "max_turns": settings["max_turns"],
                "phase": "episode_complete",
                "status": result.get("status"),
                "turns_completed": len(result.get("turns", [])),
                "total_tokens": result.get("total_tokens"),
            })
            return result
        except Exception as exc:
            state = runner._load_or_init()
            state["status"] = "api_error"
            state["fatal_error"] = redact(f"{type(exc).__name__}: {exc}")
            state["traceback"] = traceback.format_exc(limit=3)
            runner._save(state)
            on_progress({
                "job_id": episode_job_id,
                "episode_number": episode_number,
                "episodes_total": len(selected),
                "model": model,
                "reference_id": reference.id,
                "track": track_name,
                "turn": len(state.get("turns", [])),
                "max_turns": settings["max_turns"],
                "phase": "episode_complete",
                "status": "api_error",
                "turns_completed": len(state.get("turns", [])),
                "total_tokens": state.get("total_tokens"),
            })
            return state

    workers = int(inputs.config.get("concurrency", 3))
    results: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(run_one, item): item for item in selected}
            progress_bar = tqdm(concurrent.futures.as_completed(future_map), total=len(future_map), desc="teacher episodes", unit="episode")
            for future in progress_bar:
                result = future.result()
                results.append(result)
                with progress_lock:
                    progress["episodes_completed"] += 1
                    status = str(result.get("status", "unknown"))
                    progress["status_counts"][status] = progress["status_counts"].get(status, 0) + 1
                    progress["current"] = {
                        "job_id": result.get("job_id"), "track": result.get("settings", {}).get("track"),
                        "model": result.get("model"), "reference_id": result.get("reference_id"),
                        "turn": len(result.get("turns", [])), "status": status,
                    }
                    atomic_json(progress_path, progress)
                    set_postfix = getattr(progress_bar, "set_postfix", None)
                    if callable(set_postfix):
                        set_postfix(done=f"{progress['episodes_completed']}/{progress['episodes_requested']}", status=status)
    finally:
        client_pool.close()
    progress["finished_at"] = time.time()
    atomic_json(progress_path, progress)
    counts: dict[str, int] = {}
    for row in results:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
    summary = {
        "schema": SCHEMA,
        "mode": "run",
        "track": track,
        "episodes_selected": len(selected),
        "reference_ids_selected": list(dict.fromkeys(reference.id for _, _, reference, _ in selected)),
        "status_counts": counts,
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in results),
        "total_cost": sum(float(row.get("total_cost") or 0) for row in results),
        "cost_missing_turns": sum(int(row.get("cost_missing_turns") or 0) for row in results),
        "cost_complete": all(bool(row.get("cost_complete", False)) for row in results),
        "response_turns_total": sum(len(row.get("turns", [])) for row in results),
        "network_requests_made_this_invocation": sum(
            sum(int((turn.get("response") or {}).get("attempts") or turn.get("attempts") or 0) for turn in row.get("turns", []) if not turn.get("api_reused"))
            for row in results
        ),
        "resumed_response_turns": sum(
            sum(1 for turn in row.get("turns", []) if turn.get("api_reused")) for row in results
        ),
        "completed_at": time.time(),
    }
    atomic_json(root / "run-summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="validate inputs and print the request budget; no key or network")
    modes.add_argument("--run", action="store_true", help="run/resume paid API episodes and render each submitted sketch")
    parser.add_argument("--local", action="store_true", help="use the macOS-local Watercolour renderer (with --run)")
    parser.add_argument("--limit-episodes", type=int, help="select only the first N deterministic episodes")
    parser.add_argument("--track", choices=["quality", "speed", "screen", "all"], default="all")
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--renderer-python", type=Path)
    parser.add_argument("--browser-path", type=Path)
    parser.add_argument("--run-as-user", default=None)
    parser.add_argument("--renderer-backend", choices=["metal", "swiftshader"], help="local Chromium graphics backend")
    parser.add_argument("--api-key-file", type=Path, help="read a key from a protected file without recording it")
    parser.add_argument("--models", action="append", help="override models for this invocation; repeat or comma-separate IDs")
    args = parser.parse_args(argv)
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error("--limit-episodes must be positive")
    try:
        requested_models = None
        if args.models:
            requested_models = []
            for value in args.models:
                requested_models.extend(item.strip() for item in value.split(",") if item.strip())
            if not requested_models:
                raise BenchmarkError("--models needs at least one model ID")
        inputs = load_inputs(args.root, model_ids=requested_models, fetch_unknown=args.run)
        if args.dry_run:
            print(json.dumps(dry_run_report(inputs, track=args.track, limit=args.limit_episodes), indent=2, sort_keys=True))
            return 0
        if args.local and not args.run:
            raise BenchmarkError("--local is only valid with --run")
        renderer = preflight_renderer(inputs, vars(args))
        key = None
        if args.api_key_file:
            key_text = args.api_key_file.read_text(encoding="utf-8").strip()
            if key_text.startswith("{"):
                try:
                    key_document = json.loads(key_text)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(f"invalid --api-key-file JSON: {exc}") from exc
                key = {}
                if isinstance(key_document, dict):
                    if isinstance(key_document.get("default"), str):
                        key["default"] = key_document["default"]
                    if isinstance(key_document.get("models"), dict):
                        key.update({str(model): value for model, value in key_document["models"].items() if isinstance(value, str)})
                if not key:
                    raise BenchmarkError("--api-key-file JSON needs default or models keys")
            else:
                key = key_text
            if not key:
                raise BenchmarkError("--api-key-file is empty")
        summary = run_benchmark(inputs, track=args.track, limit=args.limit_episodes, renderer_options=renderer, api_key=key)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BenchmarkError as exc:
        print(f"benchmark error: {redact(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
