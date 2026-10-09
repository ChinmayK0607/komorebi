"""Reviewed hardware profiles for the Diverse-flow run.

The profile is deliberately small and run-local.  Every caller resolves the
same name, role counts, visible device ordinals, and inference parallelism;
there is no implicit hardware auto-detection.
"""

from __future__ import annotations

import copy
import os


DEFAULT_PROFILE = "l40s_4"
PROFILE_ENV = "PAINTER_HARDWARE_PROFILE"

PROFILES = {
    "l40s_4": {
        "gpu_type": "L40S",
        "gpu_count": 4,
        "min_gpu_memory_mib": 49_140,
        "max_offer_usd_hour_exclusive": 1.60,
        "gpus_per_node": 4,
        "num_train_gpus": 2,
        "num_infer_gpus": 2,
        "train_gpu_ids": ("0", "1"),
        "infer_gpu_ids": ("2", "3"),
        "inference_tensor_parallel": 2,
        "warmup_gpu_ids": ("0", "1", "2"),
        "warmup_world_size": 3,
        "sft_resume_required": False,
    },
    "h100_2": {
        "gpu_type": "H100",
        "gpu_count": 2,
        # Match the reviewed H100 allocator's binary MiB guard.  An offer
        # with fewer than two cards, or either card below this floor, fails.
        "min_gpu_memory_mib": 80_000,
        "max_offer_usd_hour_exclusive": 4.00,
        "gpus_per_node": 2,
        "num_train_gpus": 1,
        "num_infer_gpus": 1,
        "train_gpu_ids": ("0",),
        "infer_gpu_ids": ("1",),
        "inference_tensor_parallel": 1,
        # H100 retries restore the four-update public SFT checkpoint.  A
        # fresh three-rank warmup is invalid on a two-GPU node.
        "warmup_gpu_ids": (),
        "warmup_world_size": 0,
        "sft_resume_required": True,
    },
}


def selected_name(name: str | None = None, *, environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = name if name is not None else source.get(PROFILE_ENV, DEFAULT_PROFILE)
    value = str(value).strip().lower()
    if value not in PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of: {', '.join(sorted(PROFILES))}")
    return value


def get(name: str | None = None, *, environ: dict[str, str] | None = None) -> dict:
    return copy.deepcopy(PROFILES[selected_name(name, environ=environ)])


def require_resume(name: str | None = None, *, environ: dict[str, str] | None = None) -> None:
    profile = get(name, environ=environ)
    if profile["sft_resume_required"]:
        source = os.environ if environ is None else environ
        if source.get("PAINTER_RESUME_SAME_RUN_ARCHIVE", "0") != "1":
            raise ValueError("h100_2 requires PAINTER_RESUME_SAME_RUN_ARCHIVE=1; fresh three-rank SFT is disabled")


def validate_bindings(name: str | None, values: dict[str, object], *, evaluation_only: bool = False, environ: dict[str, str] | None = None) -> dict:
    """Validate exact visible-device bindings and return the selected profile."""
    profile = get(name, environ=environ)
    if evaluation_only:
        expected = {
            "gpus_per_node": 1,
            "num_train_gpus": 0,
            "num_infer_gpus": 1,
            "train_gpu_ids": "",
            "infer_gpu_ids": "0",
            "inference_tensor_parallel": 1,
        }
    else:
        expected = {
            "gpus_per_node": profile["gpus_per_node"],
            "num_train_gpus": profile["num_train_gpus"],
            "num_infer_gpus": profile["num_infer_gpus"],
            "train_gpu_ids": ",".join(profile["train_gpu_ids"]),
            "infer_gpu_ids": ",".join(profile["infer_gpu_ids"]),
            "inference_tensor_parallel": profile["inference_tensor_parallel"],
        }
    for key, wanted in expected.items():
        if values.get(key) != wanted:
            raise ValueError(f"{PROFILE_ENV}={selected_name(name, environ=environ)} requires {key}={wanted!r}")
    if not evaluation_only:
        require_resume(selected_name(name, environ=environ), environ=environ)
    return profile
