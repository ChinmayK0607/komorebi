#!/usr/bin/env python3
"""Load the reviewed runtime guard without editing shared native code."""
from pathlib import Path
import runpy

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "sitecustomize.py"
runpy.run_path(str(SOURCE), run_name="structured_flow_sitecustomize")

# The shared sitecustomize only installs the process-group guard.  The
# structured-flow prompt hook is run-local and opt-in; it patches the native
# RL taskset before Prime serializes tasks so the casebrief hint is real input.
if __import__("os").environ.get("PAINTER_PRIME_ROLE") == "orchestrator":
    from flow_prompt_hook import install
    result = install()
    if __import__("os").environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1" and not result.get("installed"):
        raise RuntimeError(f"structured-flow prompt hook did not install: {result}")
