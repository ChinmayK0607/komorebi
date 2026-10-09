"""Explicit orchestrator-only activation for the staged Prime binding."""

from __future__ import annotations

import os


if os.environ.get("PAINTER_PRIME_ROLE") == "orchestrator":
    from runtime_binding import install_from_environment

    # With no exact run-owned enable/protocol variables this is a no-op.  Once
    # enabled, any mismatch is fatal before Prime constructs its orchestrator.
    result = install_from_environment()
    if os.environ.get("PAINTER_STAGED_CURRICULUM_ENABLE") == "1" and not (
        result.get("installed") or result.get("already_installed")
    ):
        raise RuntimeError(f"staged curriculum binding did not install: {result}")
