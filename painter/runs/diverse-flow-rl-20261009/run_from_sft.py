#!/usr/bin/env python3
"""Run the reviewed Prime RL entrypoint with a runtime anchor handoff."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib


RUN = Path(__file__).resolve().parent
if str(RUN) not in sys.path:
    sys.path.insert(0, str(RUN))
import hardware_profile
SOURCE = RUN.parent / "photo-curriculum-rl-20260922" / "run_from_sft.py"
spec = importlib.util.spec_from_file_location("diverse_flow_base_run_from_sft", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"reviewed RL entrypoint is missing: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


# The current Diverse-flow amendment caps external judge pressure at one
# batch-sized decode window.  Do not inherit the historical 32/32 values from
# the public SFT receipt when the post-anchor config is regenerated.
AMENDED_INITIAL_INFLIGHT = 16
AMENDED_MAX_INFLIGHT = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refresh_final_config(config_path: Path, manifest_path: Path, admitted_count: int) -> None:
    """Regenerate the Prime config/receipt after train-only censoring."""
    receipt_path = config_path.with_name("rl-config-receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    profile_name = hardware_profile.selected_name(receipt.get("hardware_profile"))
    if profile_name != hardware_profile.selected_name():
        raise ValueError("prepared RL receipt hardware profile does not match PAINTER_HARDWARE_PROFILE")
    hardware_profile.require_resume(profile_name)
    if (
        receipt.get("curriculum", {}).get("active_rows") == admitted_count
        and receipt.get("manifest_sha256") == _sha256(manifest_path)
        and receipt.get("updates") == 20
        and receipt.get("effective_updates") == 20
        and receipt.get("checkpoint_steps") == [4, 8, 12, 16, 20]
    ):
        return
    if not 88 <= admitted_count <= 96:
        raise ValueError(f"Diverse-flow requires 88..96 admitted train rows after bounded censoring: {admitted_count}")
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    taskset = raw_config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
    spec = importlib.util.spec_from_file_location("diverse_flow_final_config", RUN / "prepare_structured_config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Diverse-flow config builder is unavailable")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    temporary = config_path.with_name(config_path.name + f".final-{os.getpid()}.tmp")
    builder.prepare(
        output=temporary,
        model=Path(str(raw_config["model"]["name"])),
        run_root=config_path.parent,
        manifest=manifest_path,
        prime_root=Path(os.environ["PRIME_ROOT"]) if os.environ.get("PRIME_ROOT") else None,
        expected_count=admitted_count,
        max_steps=int(receipt.get("updates", 20)),
        checkpoint_interval=int(receipt.get("checkpoint_interval", 4)),
        output_root=str(taskset.get("output_root", "rl-rollouts")),
        max_turns=int(receipt.get("max_turns", 3)),
        judge_protocol=str(receipt.get("reward", "text-paint-sft-pairwise-consistent-v1")),
        adapter_revision=str(receipt["policy_initializer"]["revision"]),
        adapter_prefix=str(receipt["policy_initializer"]["prefix"]),
        adapter_sha256=str(receipt["policy_initializer"]["sha256"]),
        mode=str(receipt.get("mode", "recovered_policy")),
        learning_rate=float(receipt.get("learning_rate", 1e-6)),
        hardware_profile_name=profile_name,
        gpus_per_node=int(receipt.get("deployment", {}).get("gpus_per_node", 4)),
        num_train_gpus=int(receipt.get("deployment", {}).get("num_train_gpus", 2)),
        num_infer_gpus=int(receipt.get("deployment", {}).get("num_infer_gpus", 2)),
        batch_size=int(receipt.get("batch_size", 16)),
        group_size=int(receipt.get("group_size", 4)),
        initial_inflight=AMENDED_INITIAL_INFLIGHT,
        max_inflight=AMENDED_MAX_INFLIGHT,
        renderer=taskset.get("renderer"),
        renderer_python=taskset.get("renderer_python"),
        activation_cpu_offload=bool(receipt.get("activation_cpu_offload", True)),
        flow_policy_enabled=bool(receipt.get("structured_flow_policy", {}).get("enabled", False)),
        code_supervision_enabled=bool(receipt.get("code_supervision", {}).get("enabled", False)),
        allow_pending_anchors=False,
    )
    temporary.replace(config_path)


def refresh_final_config_entry(config_path: Path, manifest_path: Path) -> None:
    """Refresh the config after matched-baseline anchor admission, without RL."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("final train manifest has no rows")
    _refresh_final_config(config_path.resolve(), manifest_path.resolve(), len(rows))


def _ensure_anchors(config_path: Path):
    from anchor_bank import materialize

    run_root = Path(os.environ.get("PAINTER_RUN_ROOT", RUN)).resolve()
    result = materialize(
        train_manifest=Path(os.environ.get("PAINTER_RL_MANIFEST", run_root / "curriculum/pilot-train-manifest.json")),
        heldout_manifest=Path(os.environ.get("PAINTER_EVAL_MANIFEST", run_root / "curriculum/matched32-manifest.json")),
        anchor_root=Path(os.environ.get("PAINTER_ANCHOR_ROOT", run_root / "curriculum/anchors")),
        output_dir=Path(os.environ.get("PAINTER_ANCHOR_STAGING", run_root / "curriculum/.materialized-pilot")),
        run_root=run_root,
    )
    final_manifest_path = Path(os.environ.get("PAINTER_RL_MANIFEST", run_root / "curriculum/pilot-train-manifest.json"))
    _refresh_final_config(config_path, final_manifest_path, int(result["train_rows"]))
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
        os.environ.get("PAINTER_EVAL_MANIFEST", str(run_root / "curriculum/matched32-manifest.json")),
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
    if len(sys.argv) == 4 and sys.argv[1] == "--refresh-final-config":
        refresh_final_config_entry(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        base.main()
