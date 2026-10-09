#!/usr/bin/env python3
"""Load the reviewed runtime guard without editing shared native code."""
from pathlib import Path
import os
import runpy

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "sitecustomize.py"


def _install_prime_runtime_harness_environment() -> None:
    """Mark verifier runtime subprocesses as isolated harnesses.

    Prime's orchestrator environment is inherited by ``Runtime.run`` and by
    its uv bootstrap/interpreter probes.  The pinned verifier runtime exposes
    one process boundary, ``Runtime.process_env``; patch that boundary in the
    orchestrator so every remote task process receives the harness role before
    Python starts.  This avoids argv-based guesses and leaves the orchestrator
    itself under the shared fail-closed group guard.
    """

    if os.environ.get("PAINTER_PRIME_ROLE") != "orchestrator":
        return
    try:
        from verifiers.v1.runtimes.base import Runtime
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "structured-flow Prime runtime environment isolation requires the pinned verifiers runtime"
        ) from exc
    sentinel = "_painter_prime_harness_process_env"
    if getattr(Runtime, sentinel, False):
        return
    original = Runtime.process_env

    def process_env(self, env):
        child_env = original(self, env)
        child_env["PAINTER_PRIME_ROLE"] = "harness"
        return child_env

    Runtime.process_env = process_env
    setattr(Runtime, sentinel, True)


runpy.run_path(str(SOURCE), run_name="structured_flow_sitecustomize")
_install_prime_runtime_harness_environment()

# The shared sitecustomize installs the process-group guard only for the RL
# orchestrator.  Baseline/evaluation Prime processes are intentionally
# ungrouped, but still need the run-local flow hook so their prompts match the
# policy being evaluated.  Keep this role split here rather than weakening the
# shared fail-closed group guard.
_ROLE = os.environ.get("PAINTER_PRIME_ROLE")
if _ROLE in {"orchestrator", "evaluation"}:
    from flow_prompt_hook import install
    result = install()
    if os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1" and not result.get("installed"):
        raise RuntimeError(f"structured-flow prompt hook did not install: {result}")

if os.environ.get("PAINTER_ANCHOR_REQUIRE_NO_LOOP") == "1":
    from anchor_no_loop_guard import install
    result = install()
    if not result.get("installed"):
        raise RuntimeError(f"anchor noLoop guard did not install: {result}")
