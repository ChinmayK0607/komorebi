#!/usr/bin/env python3
"""Load the reviewed runtime guard without editing shared native code."""
from pathlib import Path
import runpy

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "sitecustomize.py"
runpy.run_path(str(SOURCE), run_name="structured_flow_sitecustomize")

# The shared sitecustomize installs the process-group guard only for the RL
# orchestrator.  Baseline/evaluation Prime processes are intentionally
# ungrouped, but still need the run-local flow hook so their prompts match the
# policy being evaluated.  Keep this role split here rather than weakening the
# shared fail-closed group guard.
_ROLE = __import__("os").environ.get("PAINTER_PRIME_ROLE")
if _ROLE in {"orchestrator", "evaluation"}:
    from flow_prompt_hook import install
    result = install()
    if __import__("os").environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1" and not result.get("installed"):
        raise RuntimeError(f"structured-flow prompt hook did not install: {result}")

if __import__("os").environ.get("PAINTER_ANCHOR_REQUIRE_NO_LOOP") == "1":
    from anchor_no_loop_guard import install
    result = install()
    if not result.get("installed"):
        raise RuntimeError(f"anchor noLoop guard did not install: {result}")
