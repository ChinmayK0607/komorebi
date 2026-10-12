"""Node-free checks for memory-saving hooks that affect staged training.

The pinned Prime checkout is not vendored in this run surface.  These checks
therefore recognize only an explicit, source-visible hook contract.  A
missing or partial contract is reported as unsupported and must block a
launch; silently guessing a private Prime attribute would make the OOM repair
unverifiable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HIGH_MEMORY_TRAINER_GIB = 80
TOKEN_HOOK_MARKERS = (
    "def prepare_batch(",
    "def balanced_partition(",
    "sequence_lengths",
    "num_train_workers",
)
VISION_HOOK_MARKERS = (
    "omit_frozen_vision_encoder",
    "vision_encoder_offload",
)


def _source_text(root: Path, names: tuple[str, ...]) -> tuple[str, list[str]]:
    chunks: list[str] = []
    checked: list[str] = []
    for relative in names:
        path = root / relative
        checked.append(relative)
        try:
            if path.is_file() and not path.is_symlink():
                chunks.append(path.read_text(encoding="utf-8", errors="strict"))
        except (OSError, UnicodeError):
            continue
    return "\n".join(chunks), checked


def inspect_resource_hooks(
    prime_root: Path,
    *,
    expected_world_size: int = 4,
    trainer_memory_gib: int | None = None,
) -> dict[str, Any]:
    """Return a fail-closed report without importing Prime or touching CUDA."""

    prime_root = prime_root.expanduser().resolve()
    token_sources = (
        "src/prime_rl/trainer/batch.py",
        "src/prime_rl/trainer/rl/data.py",
        "src/prime_rl/trainer/rl/train.py",
    )
    vision_sources = (
        "src/prime_rl/trainer/model.py",
        "src/prime_rl/utils/vlm.py",
        "src/prime_rl/trainer/rl/train.py",
    )
    token_text, token_checked = _source_text(prime_root, token_sources)
    vision_text, vision_checked = _source_text(prime_root, vision_sources)
    token_supported = prime_root.is_dir() and all(marker in token_text for marker in TOKEN_HOOK_MARKERS)
    vision_supported = prime_root.is_dir() and all(marker in vision_text for marker in VISION_HOOK_MARKERS)
    memory_supported = isinstance(trainer_memory_gib, int) and not isinstance(trainer_memory_gib, bool) and trainer_memory_gib >= HIGH_MEMORY_TRAINER_GIB
    token = {
        "status": "supported" if token_supported else "unsupported_no_safe_hook",
        "launch_permitted": bool(token_supported),
        "expected_world_size": expected_world_size,
        "measurement": "prepare_batch.sequence_lengths" if token_supported else "not_run_fail_closed",
        "rank_balancing": "balanced_partition_by_bin_cost" if token_supported else "not_run_fail_closed",
        "checked_sources": token_checked,
        "reason": (
            "Prime prepare_batch exposes selected sequence_lengths and balanced_partition across num_train_workers"
            if token_supported else
            "Prime exposes no verified public hook for selecting and balancing the admitted trace token lengths across ranks"
        ),
    }
    vision = {
        "status": "supported" if vision_supported else "unsupported_no_safe_hook",
        "launch_permitted": bool(vision_supported),
        "checked_sources": vision_checked,
        "reason": (
            "explicit frozen vision omission/offload hooks are source-visible"
            if vision_supported else
            "Prime/model integration exposes freeze_vision_encoder but no verified hook to omit or offload it for text-only stages"
        ),
    }
    memory_contract = {
        "status": "supported" if memory_supported else "unknown_or_insufficient",
        "launch_permitted": bool(memory_supported),
        "required_trainer_memory_gib": HIGH_MEMORY_TRAINER_GIB,
        "observed_trainer_memory_gib": trainer_memory_gib,
        "reason": (
            f"explicit {trainer_memory_gib} GiB trainer contract covers the unsupported vision hook"
            if memory_supported else
            f"a {HIGH_MEMORY_TRAINER_GIB} GiB-or-larger trainer contract is required when vision omission/offload is unavailable"
        ),
    }
    return {
        "schema": "staged-family-balanced-resource-preflight-v1",
        "prime_root": str(prime_root),
        "prime_root_present": prime_root.is_dir(),
        "prime_source_digest": hashlib.sha256(token_text.encode() + vision_text.encode()).hexdigest(),
        "token_length_balance": token,
        "text_only_vision_memory": vision,
        "trainer_memory_contract": memory_contract,
        "launch_permitted": bool(token_supported and (vision_supported or memory_supported)),
    }


def read_resource_preflight(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resource preflight is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != "staged-family-balanced-resource-preflight-v1":
        raise RuntimeError("resource preflight schema is unsupported")
    if value.get("launch_permitted") is not True:
        raise RuntimeError("resource preflight is fail-closed: no safe token-balance and vision-memory hook")
    return value
