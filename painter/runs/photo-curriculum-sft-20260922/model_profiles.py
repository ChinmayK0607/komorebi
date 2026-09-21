"""Immutable model identity for the photo curriculum run."""

import hashlib
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
PRIME_REVISION = "26b3131d2716a4f8210b165df584b83b4bc54f61"
ADAPTER_TENSORS = 992
ADAPTER_PARAMETERS = 116727808


def profile() -> dict[str, int | str]:
    """Compatibility shape consumed by the reviewed export/init hooks."""
    return {
        "name": "qwen38_27b",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "tensors": ADAPTER_TENSORS,
        "parameters": ADAPTER_PARAMETERS,
        "config_sha256": CONFIG_SHA256,
    }


def verify_snapshot(path: str | Path) -> dict[str, str]:
    root = Path(path).resolve(strict=True)
    if root.name != REVISION:
        raise ValueError(f"model path must end in pinned revision {REVISION}")
    config = root / "config.json"
    if hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Qwen config hash does not match the pinned revision")
    return {"model_id": MODEL_ID, "revision": REVISION, "config_sha256": CONFIG_SHA256}
