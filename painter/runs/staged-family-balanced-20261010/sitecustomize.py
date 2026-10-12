"""Install the staged binding and isolate per-rank compiler caches.

Prime starts vLLM workers and trainer ranks independently. Each process must get a private
Triton/Inductor namespace: concurrent TP ranks can otherwise overwrite or
partially observe the same autotuning ``.cubin``. TorchInductor's own
compile-worker sidecar is the exception; it is a child of one vLLM worker and
must retain that worker's namespace so compiled artifacts can be handed back
to its parent.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import uuid


def _is_torchinductor_compile_worker() -> bool:
    """Identify TorchInductor's sidecar before changing its cache paths."""

    argv = [str(value) for value in getattr(sys, "argv", ())]
    if argv and "/torch/_inductor/compile_worker/__main__.py" in Path(argv[0]).as_posix():
        return True
    return (
        any(value.startswith("--pickler=torch._inductor.compile_worker.") for value in argv)
        and any(value.startswith("--kind=") for value in argv)
        and any(value.startswith("--parent=") for value in argv)
    )


def _worker_cache_identity() -> str | None:
    """Return a rank-scoped identity, falling back to a unique process token."""

    for name in ("VLLM_DP_RANK", "VLLM_WORKER_ID", "RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value and all(char.isalnum() or char in "._-" for char in value):
            return f"rank-{value}-pid-{os.getpid()}-{uuid.uuid4().hex}"
    if os.environ.get("PAINTER_COMPILE_CACHE_PROCESS_ISOLATION") == "1":
        return f"pid-{os.getpid()}-{uuid.uuid4().hex}"
    return None


def _isolate_compile_cache() -> None:
    """Give every independent inference or trainer rank private cache roots."""

    role = os.environ.get("PAINTER_PRIME_ROLE")
    if role not in {"inference", "trainer"} or _is_torchinductor_compile_worker():
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


_isolate_compile_cache()


if os.environ.get("PAINTER_PRIME_ROLE") == "orchestrator":
    from runtime_binding import install_from_environment

    # With no exact run-owned enable/protocol variables this is a no-op.  Once
    # enabled, any mismatch is fatal before Prime constructs its orchestrator.
    result = install_from_environment()
    if os.environ.get("PAINTER_STAGED_CURRICULUM_ENABLE") == "1" and not (
        result.get("installed") or result.get("already_installed")
    ):
        raise RuntimeError(f"staged curriculum binding did not install: {result}")
