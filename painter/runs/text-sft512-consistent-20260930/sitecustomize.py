"""Fail-closed activation of the consistent group admission guard.

Prime starts the orchestrator in a separate Python process.  Importing the
rank module through an environment variable does not install its patches, so
this role-gated hook invokes the explicit installer in that child only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid


def _worker_cache_identity() -> str | None:
    """Return a safe vLLM worker rank, or a process fallback.

    vLLM's data-parallel children normally expose one of these rank names.
    A PID fallback still prevents two unlabelled spawned workers from racing
    on the same Triton JSON artifact; the parent process does not import this
    hook with the worker marker unless Prime has launched the inference role.
    """

    for name in ("VLLM_DP_RANK", "VLLM_WORKER_ID", "RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value and all(char.isalnum() or char in "._-" for char in value):
            return f"rank-{value}-pid-{os.getpid()}-{uuid.uuid4().hex}"
    if os.environ.get("PAINTER_COMPILE_CACHE_PROCESS_ISOLATION") == "1":
        # Include a process token so a later process that receives the same
        # recycled PID cannot reuse a dead worker's compiler artifacts.
        return f"pid-{os.getpid()}-{uuid.uuid4().hex}"
    return None


def _isolate_inference_compile_cache() -> None:
    """Give each spawned inference worker a private Triton/Inductor cache."""

    if os.environ.get("PAINTER_PRIME_ROLE") != "inference":
        return
    root = os.environ.get("PAINTER_COMPILE_CACHE_ROOT")
    identity = _worker_cache_identity()
    if not root or not identity:
        return
    worker = Path(root) / "workers" / identity
    (worker / "vllm").mkdir(parents=True, exist_ok=True)
    (worker / "triton").mkdir(parents=True, exist_ok=True)
    (worker / "torchinductor").mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(worker / "vllm")
    os.environ["TRITON_CACHE_DIR"] = str(worker / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(worker / "torchinductor")


def _activate() -> None:
    if os.environ.get("PAINTER_PRIME_ROLE") != "orchestrator":
        return
    try:
        from painter.consistent_group_admission import install_from_environment
    except ModuleNotFoundError as exc:
        # The reviewed Prime launcher normally places the repository root on
        # PYTHONPATH.  Keep the hook usable with its legacy ``painter/``
        # source-root layout too, where the module is top-level.
        if exc.name != "painter":
            raise
        from consistent_group_admission import install_from_environment

    result = install_from_environment()
    if not result.get("installed") and not result.get("already_installed"):
        raise RuntimeError(f"consistent group admission did not install: {result}")
    receipt = os.environ.get("PAINTER_CONSISTENT_GUARD_RECEIPT")
    if receipt:
        path = Path(receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "consistent-group-admission-activation-v1",
                    "role": "orchestrator",
                    "installed": bool(result.get("installed")),
                    "already_installed": bool(result.get("already_installed")),
                    "protocol": os.environ.get("PAINTER_JUDGE_PROTOCOL"),
                    "group_size": int(os.environ["PAINTER_WHOLE_GROUP_SIZE"]),
                },
                sort_keys=True,
            )
            + "\n"
        )


try:
    _isolate_inference_compile_cache()
    _activate()
except BaseException as exc:
    # CPython's site importer reports ordinary Exception subclasses and then
    # continues.  SystemExit is deliberate here: a Prime child that cannot
    # install the guard must fail before it creates scorer/trainer classes.
    if isinstance(exc, SystemExit):
        raise
    raise SystemExit(f"consistent group admission activation failed: {exc}") from exc
