#!/usr/bin/env python3
"""Use the reviewed archive screen while keeping this run's path local."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "safe_archive.py"
spec = importlib.util.spec_from_file_location("structured_safe_archive_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"archive helper unavailable: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
globals().update({name: value for name, value in vars(base).items() if not name.startswith("__")})

# The shared pipeline can pass its historical preferred step (8) even after
# this run's registered target has been amended to 20. Keep the archive
# selector run-owned so the requested value, registered target, actual
# selected checkpoint, and any censored fallback remain explicit.
REGISTERED_FINAL_STEP = 20
_base_verified_checkpoint = base._verified_public_checkpoint


def select_preferred_public_checkpoint(
    run_root: Path, preferred_step: int | None, *, run_name: str = RUN_NAME
) -> tuple[dict[str, object] | None, dict[str, object]]:
    requested_step = preferred_step
    preferred = base.select_public_checkpoint(run_root, REGISTERED_FINAL_STEP, run_name=run_name)
    candidates: list[dict[str, object]] = []
    root = run_root / "async-output" / run_name / "artifacts" / "adapters"
    if root.is_dir():
        for receipt_path in root.glob("step_*/hf-upload.json"):
            candidate = _base_verified_checkpoint(receipt_path)
            if candidate is not None and int(candidate["step"]) <= REGISTERED_FINAL_STEP:
                candidates.append(candidate)
    latest = max(candidates, key=lambda row: int(row["step"])) if candidates else None
    if preferred is not None:
        selected = preferred
        quality = "primary"
        fallback_reason = None
    elif latest is not None:
        selected = latest
        quality = "exploratory_censored"
        fallback_reason = "registered_final_checkpoint_not_verified_or_unavailable"
    else:
        selected = None
        quality = "no_verified_checkpoint"
        fallback_reason = "no_verified_checkpoint_at_or_below_registered_final"
    selection = {
        "policy": "registered_final_checkpoint_with_bounded_fallback",
        "requested_preferred_step": requested_step,
        "preferred_step": REGISTERED_FINAL_STEP,
        "registered_final_step": REGISTERED_FINAL_STEP,
        "selected_step": selected["step"] if selected is not None else None,
        "primary_quality_status": quality,
        "fallback_reason": fallback_reason,
    }
    return selected, selection

if __name__ == "__main__":
    raise SystemExit(base.main())
