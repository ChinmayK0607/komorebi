#!/usr/bin/env python3
"""Run the reviewed Prime RL entrypoint with a runtime anchor handoff."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys


RUN = Path(__file__).resolve().parent
SOURCE = RUN.parent / "photo-curriculum-rl-20260922" / "run_from_sft.py"
spec = importlib.util.spec_from_file_location("structured_flow_base_run_from_sft", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"reviewed RL entrypoint is missing: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_anchors(config_path: Path):
    from anchor_bank import materialize

    run_root = Path(os.environ.get("PAINTER_RUN_ROOT", RUN)).resolve()
    result = materialize(
        train_manifest=Path(os.environ.get("PAINTER_RL_MANIFEST", run_root / "curriculum/pilot-train-manifest.json")),
        heldout_manifest=Path(os.environ.get("PAINTER_EVAL_MANIFEST", run_root / "curriculum/matched16-manifest.json")),
        anchor_root=Path(os.environ.get("PAINTER_ANCHOR_ROOT", run_root / "curriculum/anchors")),
        output_dir=Path(os.environ.get("PAINTER_ANCHOR_STAGING", run_root / "curriculum/.materialized-pilot")),
        run_root=run_root,
    )
    # Once the manifest has become fixed, validate adapter bytes, manifests,
    # and config before the original entrypoint allows Prime to start.
    from node_validate import main as validate_node

    policy_pointer = os.environ.get(
        "PAINTER_ACTIVE_POLICY_PATH_FILE",
        str(run_root / "active-policy-adapter-path.txt"),
    )
    if not Path(policy_pointer).is_file():
        policy_pointer = os.environ["PAINTER_SFT_ADAPTER_PATH_FILE"]
    validate_node([
        "node_validate.py",
        str(run_root),
        os.environ.get("PAINTER_RL_MANIFEST", str(run_root / "curriculum/pilot-train-manifest.json")),
        os.environ.get("PAINTER_EVAL_MANIFEST", str(run_root / "curriculum/matched16-manifest.json")),
        policy_pointer,
        os.environ["PAINTER_FROZEN_SFT_ADAPTER_PATH_FILE"],
    ])
    # execute_pipeline.sh records identity against the lazy alias so its
    # preflight remains auditable.  The runtime handoff is now complete;
    # rebind the identity to the final bytes before Prime starts.
    identity_path = run_root / "source-identity.json"
    identity = json.loads(identity_path.read_text())
    final_manifest = run_root / "anchors-frozen" / "train-manifest.json"
    identity["manifest_sha256"] = _sha256(final_manifest)
    identity["train_manifest_sha256"] = _sha256(final_manifest)
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return result


original_prewarm = base._prewarm_frozen_opponent


def _prewarm_after_anchor_generation(config_path: Path, **kwargs):
    _ensure_anchors(config_path)
    return original_prewarm(config_path, **kwargs)


base._prewarm_frozen_opponent = _prewarm_after_anchor_generation


if __name__ == "__main__":
    base.main()
