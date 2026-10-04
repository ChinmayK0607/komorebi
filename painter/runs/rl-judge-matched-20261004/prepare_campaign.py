#!/usr/bin/env python3
"""Validate and materialize the finite matched three-judge campaign.

The plan path uses only the Python standard library.  Materialization is
called after the pinned Prime environment and the shared 40/47 source have
been prepared; it creates isolated arm roots and never starts training,
rendering, a judge, or a provider operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from typing import Any


RUN = Path(__file__).resolve().parent
PROTOCOL_PATH = RUN / "PROTOCOL.json"
ARM_IDS = ("flash", "pro", "luna")
MODELS = {
    "flash": "xiaomi/mimo-v2.6-flash",
    "pro": "xiaomi/mimo-v2.6-pro",
    "luna": "openai/gpt-6-luna",
}
INITIALIZER = {
    "revision": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
    "prefix": "text-pairwise-scaled-20260929/step_16",
    "sha256": "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
}
SFT = {
    "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
    "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    "sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
}
SOURCE_ARCHIVE_SHA256 = "ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3"
TRAIN_SEMANTIC_SHA256 = "13ac0eebb3595fbd931ee7864e69101abdc8b7c9c1faccf11dfcd9bd495c6501"
BASE_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PRIME_REVISION = "26b3131d2716a4f8210b165df584b83b4bc54f61"
RUBRIC_SHA256 = "d21e4644e7a3874a0694b2ad406420e733b7638222eb7717e7f5f91cbdf1613c"
COMMON_FIELDS = (
    "base_model_revision", "prime_revision", "initializer", "frozen_sft",
    "train_cases", "heldout_cases", "train_semantic_sha256", "protocol",
    "training", "evaluation", "gpu_plan",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    value = _load(path)
    if value.get("schema") != "matched-three-judge-protocol-v1":
        raise ValueError("unexpected matched campaign protocol schema")
    arms = value.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != ARM_IDS:
        raise ValueError("protocol arms must be flash, pro, luna in order")
    for arm_id in ARM_IDS:
        if arms[arm_id] != MODELS[arm_id]:
            raise ValueError(f"registered model mismatch for {arm_id}")
    common = value.get("judge_common", {})
    if common != {
        "requested_effort": "medium",
        "max_output_tokens": 4096,
        "format": "identity-joint-side-schema-v2-generic-v1",
        "rubric_sha256": RUBRIC_SHA256,
        "sampling_controls": "omitted for explicit reasoning",
        "internal_compute_equivalence": False,
    }:
        raise ValueError("judge common bindings changed")
    init = value.get("initializer", {})
    if init.get("revision") != INITIALIZER["revision"] or init.get("prefix") != INITIALIZER["prefix"] \
            or init.get("sha256") != INITIALIZER["sha256"] or init.get("fresh_optimizer_every_arm") is not True:
        raise ValueError("fresh RL16 initializer binding changed")
    frozen = value.get("frozen_reward_opponent", {})
    if frozen.get("revision") != SFT["revision"] or frozen.get("prefix") != SFT["prefix"] \
            or frozen.get("sha256") != SFT["sha256"]:
        raise ValueError("frozen SFT512 opponent binding changed")
    training = value.get("training_common", {})
    expected_training = {
        "actual_updates": 12, "batch_size": 16, "group_size": 4,
        "initial_and_max_inflight": 32, "max_policy_age": 1,
        "learning_rate": 1e-6, "cpu_activation_offload": True,
        "strict_final_action": True, "brush_api_reference": "v2",
        "orchestrator_seed": 42, "inference_seed": 0,
    }
    if training != expected_training:
        raise ValueError("matched training bindings changed")
    evaluation = value.get("evaluation_common", {})
    if evaluation != {
        "cases": 16, "turns": 3, "tokens_per_turn": 8192, "seed": 918210,
        "missing_or_invalid_terminal_cases": "Reported separately; no retained earlier image counted as a successful terminal action.",
    }:
        raise ValueError("matched evaluation bindings changed")
    gpu = value.get("gpu_plan", {})
    if gpu.get("topology") != "twoH10080: inference1 trainer1, same node reused sequentially":
        raise ValueError("campaign GPU topology changed")
    if gpu.get("hard_cap_per_arm_usd") != 19.5 or gpu.get("hard_total_cap_usd") != 58.5:
        raise ValueError("campaign cost cap changed")
    return value


def common_fingerprint(value: dict[str, Any]) -> dict[str, Any]:
    """Return the equality surface shared by all arms.

    Judge identity and output locations are intentionally excluded.  This is
    used by tests and receipts to prevent accidental per-arm recipe drift.
    """
    return {
        "base_model_revision": BASE_MODEL_REVISION,
        "prime_revision": PRIME_REVISION,
        "initializer": dict(INITIALIZER),
        "frozen_sft": dict(SFT),
        "train_cases": 40,
        "heldout_cases": 47,
        "train_semantic_sha256": TRAIN_SEMANTIC_SHA256,
        "protocol": {
            "protocol": "text-paint-sft-pairwise-consistent-v1",
            "format": "identity-joint-side-schema-v2-generic-v1",
            "rubric_sha256": RUBRIC_SHA256,
            "effort": "medium",
            "max_output_tokens": 4096,
            "rechecks": "adaptive initial agreement and three-of-four recheck",
            "strict_final_action": True,
            "renderer": "watercolour-p5-brush-2.2.1-centroid-fix-v1 + cancellable derivative",
        },
        "training": {
            "updates": 12, "checkpoint_steps": [4, 8, 12],
            "batch_size": 16, "group_size": 4, "initial_inflight": 32,
            "max_inflight": 32, "max_policy_age": 1, "learning_rate": 1e-6,
            "cpu_activation_offload": True, "train_gpus": 1, "infer_gpus": 1,
            "optimizer_resume": False, "orchestrator_seed": 42, "inference_seed": 0,
        },
        "evaluation": {
            "heldout_cases": 16, "turns": 3, "tokens_per_turn": 8192,
            "seed": 918210, "scope": "balanced16", "same_case_order": True,
        },
        "gpu_plan": {"gpus_per_node": 2, "inference_gpus": 1, "trainer_gpus": 1},
    }


def arm_identity(value: dict[str, Any], arm_id: str, *, run_root: str = "") -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown arm: {arm_id}")
    return {
        "arm": arm_id,
        "judge_model": MODELS[arm_id],
        "run_root": run_root,
        "run_name": f"rl-judge-matched-20261004-{arm_id}",
        "output_root": "rl-rollouts",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _link_or_verify(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise ValueError(f"refusing to replace prepared input {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def matched16_manifest(source: Path, target: Path, *, count: int = 16) -> None:
    value = _load(source)
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) < count:
        raise ValueError("validation manifest does not contain the matched panel")
    simple = [row for row in cases if row.get("family") == "simple"]
    intermediate = [row for row in cases if row.get("family") == "intermediate"]
    complex_rows = [row for row in cases if row.get("family") == "complex"]
    selected = simple[:4] + intermediate[:4] + complex_rows[:8]
    if len(selected) != count or len({row.get("id") for row in selected}) != count:
        raise ValueError("validation manifest cannot form balanced16")
    _write_atomic(target, {"cases": selected, "schema": "matched-balanced16-v1", "seed": 918210})


def normalized_native_config(path: Path) -> dict[str, Any]:
    """Normalize whole TOMLs, removing only genuine arm identity paths."""
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    value.pop("output_dir", None)
    value.get("run", {}).pop("name", None)
    trainer = value.get("trainer", {})
    model = trainer.get("model", {})
    orchestrator = value.get("orchestrator", {})
    concurrency = orchestrator.get("concurrency", {})
    source = (orchestrator.get("train", {}).get("source") or [{}])[0]
    env = source.get("env", {})
    taskset = env.get("taskset", {})
    sampling = orchestrator.get("train", {}).get("sampling", {})
    # These are the only fields expected to differ by arm.  A full recursive
    # comparison below catches an accidental drift in any unlisted Prime key.
    taskset.pop("root", None)
    taskset.pop("renderer", None)
    taskset.pop("renderer_python", None)
    return value


def assert_common_native_configs(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("no arm configs to compare")
    baseline = normalized_native_config(paths[0])
    for path in paths[1:]:
        if normalized_native_config(path) != baseline:
            raise ValueError(f"arm config differs from the common recipe: {path}")
    if baseline.get("max_steps") != 12 or baseline.get("ckpt", {}).get("interval") != 4 \
            or baseline.get("ckpt", {}).get("keep_last") != 1:
        raise ValueError("common native config update/checkpoint binding mismatch")
    trainer = baseline.get("trainer", {})
    if trainer.get("max_steps") != 12 or trainer.get("optim", {}).get("lr") != 1e-6:
        raise ValueError("common native config trainer binding mismatch")
    model = trainer.get("model", {})
    if model.get("ac_offloading") != {"pin_memory": True, "max_inflight_activations": 1}:
        raise ValueError("common native config CPU offload binding mismatch")
    orchestrator = baseline.get("orchestrator", {})
    if (orchestrator.get("batch_size"), orchestrator.get("group_size"), orchestrator.get("max_steps"),
        orchestrator.get("max_off_policy_steps")) != (16, 4, 12, 1):
        raise ValueError("common native config optimizer binding mismatch")
    if orchestrator.get("concurrency", {}).get("initial_inflight") != 32 \
            or orchestrator.get("concurrency", {}).get("max_inflight") != 32:
        raise ValueError("common native config in-flight binding mismatch")
    deployment = baseline.get("deployment") or {}
    if deployment != {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1}:
        raise ValueError("common native config topology mismatch")
    return baseline


def materialize(*, shared_root: Path, prime_root: Path, model_dir: Path,
                prepare_config: Path, source_root: Path = RUN) -> dict[str, Any]:
    protocol = load_protocol()
    shared_root = shared_root.resolve()
    if not (shared_root / "preparation-receipt.json").is_file():
        raise ValueError("shared preparation receipt is missing")
    manifest = shared_root / "anchors-frozen/train-manifest.json"
    validation = shared_root / "anchors-frozen/validation-manifest.json"
    for required in (manifest, validation, shared_root / "policy-adapter-path.txt",
                     shared_root / "frozen-sft-adapter-path.txt"):
        if not required.is_file():
            raise ValueError(f"shared prepared input is missing: {required}")
    # The reviewed base pipeline derives REPO from RUN/../../.., so each arm
    # is a flat sibling under painter/runs.  Keeping this layout also makes the
    # inherited archive/evaluation paths resolve to the intended repository.
    arms_root = source_root.parent
    matched16_manifest(validation, shared_root / "anchors-frozen/validation-matched16.json")
    materialized: list[dict[str, Any]] = []
    config_paths: list[Path] = []
    for arm_id in ARM_IDS:
        arm_root = arms_root / f"rl-judge-matched-20261004-{arm_id}"
        arm_root.mkdir(parents=True, exist_ok=True)
        for artifact in (arm_root / "async-output", arm_root / "supervisor-state.json", arm_root / "terminal-receipt.json"):
            if artifact.exists() or artifact.is_symlink():
                raise ValueError(f"refusing to rematerialize an arm with training output: {artifact}")
        # All policy/data/runtime inputs are shared and immutable; output roots
        # and native configs stay isolated so optimizer state cannot cross arms.
        for name in ("anchors-all", "vendor", "renderer-env", "browsers", "runtime",
                     "sitecustomize.py", "safe_archive.py",
                     "completion_relay.py", "launch_rl.sh", "run_from_sft.py", "train_from_sft.py"):
            if (shared_root / name).exists() or (shared_root / name).is_symlink():
                _link_or_verify(shared_root / name, arm_root / name)
        for name in ("policy-adapter-path.txt", "frozen-sft-adapter-path.txt"):
            _link_or_verify(shared_root / name, arm_root / name)
        _link_or_verify(shared_root / "preparation-receipt.json", arm_root / "preparation-receipt.json")
        arm_anchors = arm_root / "anchors-frozen"
        if arm_anchors.is_symlink():
            raise ValueError(f"refusing to replace arm anchor directory: {arm_anchors}")
        arm_anchors.mkdir(exist_ok=True)
        for item in (shared_root / "anchors-frozen").iterdir():
            if item.is_file():
                destination = arm_anchors / item.name
                if destination.exists():
                    if _sha256(destination) != _sha256(item):
                        raise ValueError(f"arm anchor differs from shared source: {destination}")
                else:
                    shutil.copy2(item, destination)
        for name in ("prepare_arm.sh", "prepare_node.sh", "run.sh", "evaluate.sh", "finalize.sh"):
            _link_or_verify(source_root / name, arm_root / name)
        _link_or_verify(source_root / "runtime_env.sh", arm_root / "runtime_env.sh")
        _link_or_verify(source_root / "finite_supervisor.py", arm_root / "finite_supervisor.py")
        matched = arm_root / "anchors-frozen/validation-matched16.json"
        matched16_manifest(validation, matched)
        renderer = arm_root / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"
        renderer_python = arm_root / "renderer-env/bin/python"
        output = arm_root / "native.toml"
        command = [
            sys.executable, os.fspath(prepare_config), "--output", os.fspath(output),
            "--model", os.fspath(model_dir), "--run-root", os.fspath(arm_root),
            "--manifest", os.fspath(arm_root / "anchors-frozen/train-manifest.json"), "--prime-root", os.fspath(prime_root),
            "--max-steps", "12", "--checkpoint-interval", "4",
            "--run-name", f"rl-judge-matched-20261004-{arm_id}",
            "--output-root", "rl-rollouts", "--judge-protocol",
            "text-paint-sft-pairwise-consistent-v1", "--expected-count", "40",
            "--adapter-revision", INITIALIZER["revision"], "--adapter-prefix", INITIALIZER["prefix"],
            "--adapter-sha256", INITIALIZER["sha256"], "--mode", "recovered_policy",
            "--learning-rate", "1e-6", "--gpus-per-node", "2", "--num-train-gpus", "1",
            "--num-infer-gpus", "1", "--batch-size", "16", "--group-size", "4",
            "--initial-inflight", "32", "--max-inflight", "32", "--activation-cpu-offload",
            "--renderer", os.fspath(renderer), "--renderer-python", os.fspath(renderer_python),
        ]
        if output.exists():
            receipt = arm_root / "rl-config-receipt.json"
            if not receipt.is_file():
                raise ValueError(f"arm config exists without receipt: {output}")
        else:
            subprocess.run(command, check=True, cwd=source_root)
        config_paths.append(output)
        _write_atomic(arm_root / "source-identity.json", {
            "schema": "matched-three-judge-source-identity-v1",
            "run_name": f"rl-judge-matched-20261004-{arm_id}",
            "base_model_revision": BASE_MODEL_REVISION,
            "model_revision": BASE_MODEL_REVISION,
            "initializer_revision": INITIALIZER["revision"],
            "initializer_prefix": INITIALIZER["prefix"],
            "initializer_sha256": INITIALIZER["sha256"],
            "frozen_sft_model_revision": SFT["revision"],
            "frozen_sft_initializer": dict(SFT),
            "prime_revision": PRIME_REVISION,
            "protocol": "text-paint-sft-pairwise-consistent-v1",
            "effective_seed_defaults": {"orchestrator": 42, "inference": 0},
        })
        identity = arm_identity(protocol, arm_id, run_root=str(arm_root))
        identity["common"] = common_fingerprint(protocol)
        identity["judge_config_sha256"] = hashlib.sha256(
            json.dumps({"model": MODELS[arm_id], "effort": "medium", "format": "identity-joint-side-schema-v2-generic-v1",
                        "rubric_sha256": RUBRIC_SHA256}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _write_atomic(arm_root / "arm-identity.json", identity)
        materialized.append(identity)
    common_config = assert_common_native_configs(config_paths)
    receipt = {
        "schema": "matched-three-judge-materialization-v1",
        "run_name": "rl-judge-matched-20261004",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "shared_root": str(shared_root),
        "base_model_revision": BASE_MODEL_REVISION,
        "prime_revision": PRIME_REVISION,
        "initializer": dict(INITIALIZER),
        "frozen_sft": dict(SFT),
        "effective_seed_defaults": {"orchestrator": 42, "inference": 0},
        "train_manifest_sha256": _sha256(manifest),
        "validation_manifest_sha256": _sha256(validation),
        "arms": materialized,
        "normalized_common_native_config": common_config,
        "baseline": {
            "required": True,
            "mode": "fresh_rl16_initializer",
            "optimizer_resume": False,
            "case_count": 16,
        },
    }
    _write_atomic(source_root / "materialization-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prepare-config", type=Path)
    args = parser.parse_args(argv)
    if args.plan == args.materialize:
        parser.error("choose exactly one of --plan or --materialize")
    value = load_protocol()
    if args.plan:
        print(json.dumps({"schema": "matched-three-judge-plan-v1", "common": common_fingerprint(value),
                          "arms": [arm_identity(value, arm) for arm in ARM_IDS]}, sort_keys=True))
        return 0
    for name, item in (("shared-root", args.shared_root), ("prime-root", args.prime_root),
                       ("model-dir", args.model_dir), ("prepare-config", args.prepare_config)):
        if item is None:
            parser.error(f"--{name} is required with --materialize")
    receipt = materialize(shared_root=args.shared_root, prime_root=args.prime_root,
                          model_dir=args.model_dir, prepare_config=args.prepare_config)
    print(json.dumps({"schema": receipt["schema"], "arms": len(receipt["arms"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
