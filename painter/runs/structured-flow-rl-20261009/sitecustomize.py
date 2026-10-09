#!/usr/bin/env python3
"""Load the reviewed runtime guard without editing shared native code."""
from pathlib import Path
import os
import runpy
import sys

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "sitecustomize.py"


def _is_prime_runtime_harness() -> bool:
    """Recognize the verifier's isolated PEP 723 rollout process.

    Prime gives each verifier runtime program an argv script below the pinned
    ``/tmp/vf-scripts`` directory.  The RL launcher, by contrast, enters the
    orchestrator as a module/console process.  The rollout inherits the
    orchestrator's environment, including the role and group-admission
    settings, but it does not contain Prime's ``prime_rl`` package.  Narrowing
    the exemption to this published runtime path keeps a real orchestrator
    launch fail-closed on an incomplete environment.
    """

    if os.environ.get("PAINTER_PRIME_ROLE") != "orchestrator":
        return False
    argv = [str(value) for value in getattr(sys, "argv", ())]
    if not argv:
        return False
    script = Path(argv[0])
    return script.parent == Path("/tmp/vf-scripts") and script.suffix == ".py"


_PRIME_RUNTIME_HARNESS = _is_prime_runtime_harness()
_previous_role = os.environ.get("PAINTER_PRIME_ROLE")
if _PRIME_RUNTIME_HARNESS:
    # The shared hook is role-gated.  Temporarily mark only this child so its
    # import-time group guard does not try to import Prime in the lightweight
    # verifier interpreter; restore the inherited value immediately after.
    os.environ["PAINTER_PRIME_ROLE"] = "harness"
try:
    runpy.run_path(str(SOURCE), run_name="structured_flow_sitecustomize")
finally:
    if _previous_role is None:
        os.environ.pop("PAINTER_PRIME_ROLE", None)
    else:
        os.environ["PAINTER_PRIME_ROLE"] = _previous_role

# The shared sitecustomize installs the process-group guard only for the RL
# orchestrator.  Baseline/evaluation Prime processes are intentionally
# ungrouped, but still need the run-local flow hook so their prompts match the
# policy being evaluated.  Keep this role split here rather than weakening the
# shared fail-closed group guard.
_ROLE = "harness" if _PRIME_RUNTIME_HARNESS else os.environ.get("PAINTER_PRIME_ROLE")
if _ROLE in {"orchestrator", "evaluation"}:
    from flow_prompt_hook import install
    result = install()
    if os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1" and not result.get("installed"):
        raise RuntimeError(f"structured-flow prompt hook did not install: {result}")

if not _PRIME_RUNTIME_HARNESS and os.environ.get("PAINTER_ANCHOR_REQUIRE_NO_LOOP") == "1":
    from anchor_no_loop_guard import install
    result = install()
    if not result.get("installed"):
        raise RuntimeError(f"anchor noLoop guard did not install: {result}")
