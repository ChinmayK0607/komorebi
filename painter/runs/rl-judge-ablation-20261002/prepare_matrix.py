"""Validate and materialize the finite three-arm RL judge ablation.

This module only handles small plans, manifests, symlinks, and receipts.  It
does not contact a provider, start Prime, rent a GPU, or download weights.
The node-facing ``prepare.sh`` performs one shared source preparation and then
uses this module to give each arm an isolated run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


RUN = Path(__file__).resolve().parent
MATRIX_PATH = RUN / "matrix.json"
ARM_IDS = ("flash", "pro", "luna")
MATCHED_CASE_IDS = (
    "t600-512", "t600-516", "t600-542", "t600-543",
    "t600-425", "t600-433", "t600-439", "t600-445",
    "t600-309", "t600-366", "t600-375", "t600-377",
    "t600-413", "t600-466", "t600-478", "t600-560",
)
MODELS = {
    "flash": "xiaomi/mimo-v2.6-flash",
    "pro": "xiaomi/mimo-v2.6-pro",
    "luna": "openai/gpt-6-luna",
}
COMMON_LINKS = (
    "anchors-all",
    "anchors-frozen",
    "vendor",
    "renderer-env",
    "browsers",
    "sitecustomize.py",
    "finite_supervisor.py",
    "safe_archive.py",
    "completion_relay.py",
    "launch_rl.sh",
    "run_from_sft.py",
    "train_from_sft.py",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "text-paint-rl-judge-ablation-matrix-v1":
        raise ValueError("unexpected judge ablation matrix schema")
    arms = value.get("arms")
    if (not isinstance(arms, list) or any(not isinstance(arm, dict) for arm in arms)
            or tuple(arm.get("id") for arm in arms) != ARM_IDS):
        raise ValueError("matrix must contain flash, pro, and luna arms in order")
    seen_models: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict) or set(("id", "model", "directory", "run_name", "output_root", "public_checkpoint_prefix")) - set(arm):
            raise ValueError("each matrix arm is missing an identity field")
        arm_id = arm["id"]
        if arm["model"] != MODELS[arm_id] or arm["model"] in seen_models:
            raise ValueError("matrix arm models must be the exact unique registered IDs")
        registered_hash = arm.get("judge_config_sha256")
        if not isinstance(registered_hash, str) or len(registered_hash) != 64:
            raise ValueError("each matrix arm must carry a judge config SHA-256")
        if registered_hash != judge_config_sha256(value, arm):
            raise ValueError(f"judge config hash mismatch for arm {arm_id}")
        seen_models.add(arm["model"])
        if "/" in arm["run_name"] or "/" in arm["output_root"]:
            raise ValueError("run and output names must be single path components")
    protocol = value.get("protocol", {})
    if protocol.get("judge_format") != "identity-joint-side-schema-v2-generic-v1":
        raise ValueError("matrix is not bound to the isolated generic v2 judge")
    if protocol.get("reasoning") != {"field": "reasoning", "effort": "medium"}:
        raise ValueError("matrix reasoning configuration is not explicit medium")
    if (protocol.get("batch_size") != 16 or protocol.get("group_size") != 4
            or protocol.get("initial_inflight") != 96 or protocol.get("max_inflight") != 96
            or protocol.get("gpus_per_node") != 8
            or protocol.get("num_train_gpus") != 1
            or protocol.get("num_infer_gpus") != 7
            or protocol.get("activation_cpu_offload") is not True
            or protocol.get("render_workers") != 8):
        raise ValueError("matrix optimizer shape changed")
    if protocol.get("training_seeds") != {"orchestrator": 42, "inference": 0}:
        raise ValueError("matrix must pin Prime's actual training seed defaults")
    if value.get("evaluation", {}).get("seed") != 918210:
        raise ValueError("matrix must pin the matched evaluation seed")
    evaluation = value.get("evaluation", {})
    if evaluation.get("common_checkpoint_policy") != "parent_postpass_only_when_all_arms_reach_and_reserve_fits":
        raise ValueError("matrix must declare deferred common-checkpoint collection")
    if evaluation.get("preferred_primary_checkpoint") != 12 or evaluation.get("preferred_primary_fallback") != "latest_verified_public_checkpoint_is_exploratory_censored":
        raise ValueError("matrix must declare the fixed step-12 primary comparison policy")
    if evaluation.get("operational_timeouts_seconds") != {
        "evaluation": 1200,
        "archive": 360,
        "parent_release": 240,
        "completion_margin": 120,
        "total_reserve": 1920,
    }:
        raise ValueError("matrix operational reserve changed")
    source_data = value.get("source", {}).get("training_data", {})
    if source_data.get("train_briefs") != 40 or source_data.get("heldout_briefs") != 47:
        raise ValueError("matrix must use the matched 40/47 source split")
    return value


def source_identity(matrix: dict[str, Any], *, repo: Path = RUN.parents[2]) -> dict[str, Any]:
    """Return immutable source/config identity without node-specific secrets."""

    files = {
        "matrix_sha256": MATRIX_PATH,
        "foundation_lock_sha256": repo / "painter/foundation-lock.json",
        "native_rl_source_sha256": repo / "painter/native_painting_rl.py",
        "model_transport_source_sha256": repo / "painter/text_pairwise_consistent_side_judge_v2_model.py",
        "generic_judge_source_sha256": repo / "painter/text_pairwise_consistent_side_judge_v2_generic.py",
        "matrix_prepare_source_sha256": RUN / "prepare_matrix.py",
        "preparation_source_sha256": RUN / "prepare.sh",
        "campaign_source_sha256": RUN / "campaign.sh",
        "execute_source_sha256": RUN / "execute_pipeline.sh",
        "evaluate_source_sha256": RUN / "evaluate.sh",
        "finalize_source_sha256": RUN / "finalize.sh",
        "arm_env_source_sha256": RUN / "arm_env.sh",
        "base_execute_source_sha256": repo / "painter/runs/text-sft512-consistent-20260930/execute_pipeline.sh",
        "base_evaluate_source_sha256": repo / "painter/runs/text-sft512-consistent-20260930/evaluate.sh",
        "base_finalize_source_sha256": repo / "painter/runs/text-sft512-consistent-20260930/finalize.sh",
        "base_prepare_source_sha256": repo / "painter/runs/text-sft512-consistent-20260930/prepare_node.sh",
        "text_config_source_sha256": repo / "painter/runs/text-pairwise-rl-20260928/prepare_config.py",
        "photo_config_source_sha256": repo / "painter/runs/photo-curriculum-rl-20260922/prepare_rl_config.py",
        "run_from_sft_source_sha256": repo / "painter/runs/photo-curriculum-rl-20260922/run_from_sft.py",
        "native_config_writer_source_sha256": repo / "painter/prepare_scaled_native_rl.py",
        "runtime_sitecustomize_source_sha256": repo / "painter/runs/text-sft512-consistent-20260930/sitecustomize.py",
        "safe_archive_source_sha256": repo / "painter/runs/text-sft512-consistent-20260930/safe_archive.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    source = matrix["source"]
    base_model = source["base_model"]
    policy_initializer = source["policy_initializer"]
    frozen_sft = source["frozen_sft_opponent"]
    # The base pipeline's write_identity contract reads these revision keys at
    # the top level.  Keep the richer source object below as well so a receipt
    # remains self-describing when inspected without the matrix.
    return {
        "schema": "text-paint-rl-judge-ablation-source-identity-v1",
        "run_name": matrix["run_name"],
        "base_model_revision": base_model["revision"],
        "model_revision": base_model["revision"],
        "initializer_revision": policy_initializer["revision"],
        "initializer_prefix": policy_initializer["prefix"],
        "initializer_sha256": policy_initializer["adapter_sha256"],
        "policy_initializer": {
            "revision": policy_initializer["revision"],
            "prefix": policy_initializer["prefix"],
            "adapter_sha256": policy_initializer["adapter_sha256"],
        },
        "frozen_sft_model_revision": frozen_sft["revision"],
        "sft_model_revision": frozen_sft["revision"],
        "frozen_sft_initializer": {
            "revision": frozen_sft["revision"],
            "prefix": frozen_sft["prefix"],
            "adapter_sha256": frozen_sft["adapter_sha256"],
        },
        "prime_revision": source["prime_revision"],
        "source": source,
        "protocol": matrix["protocol"],
        "files": {key: sha256(path) for key, path in files.items()},
        "arms": [
            {
                "id": arm["id"],
                "model": arm["model"],
                "directory": arm["directory"],
                "run_name": arm["run_name"],
                "output_root": arm["output_root"],
                "public_checkpoint_prefix": arm["public_checkpoint_prefix"],
                "judge_config_sha256": judge_config_sha256(matrix, arm),
            }
            for arm in matrix["arms"]
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def judge_config_sha256(matrix: dict[str, Any], arm: dict[str, Any]) -> str:
    value = {
        "protocol": matrix["protocol"]["judge_protocol"],
        "format": matrix["protocol"]["judge_format"],
        "model": arm["model"],
        "reasoning": matrix["protocol"]["reasoning"],
        "sampling_parameters": matrix["protocol"]["sampling_parameters"],
        "prompt_sha256": matrix["protocol"]["prompt_sha256"],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _link_once(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise ValueError(f"refusing to replace existing arm input: {link}")
        return
    link.symlink_to(target, target_is_directory=target.is_dir())


def _write_wrapper_once(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to replace existing arm wrapper: {path}")
        return
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _arm_wrappers(arm_id: str) -> dict[str, str]:
    """Return the fixed-arm entrypoints materialized in each flat run root."""

    wrapper_root = '$(cd "$(dirname "${BASH_SOURCE[0]}")/../rl-judge-ablation-20261002" && pwd -P)'
    return {
        "prepare_node.sh": (
            "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            f"ROOT={wrapper_root}\n"
            f"source \"$ROOT/arm_env.sh\" \"{arm_id}\"\n"
            "exec bash \"$ROOT/../text-sft512-consistent-20260930/prepare_node.sh\" \"$@\"\n"
        ),
        "run.sh": (
            "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            f"ROOT={wrapper_root}\n"
            f"exec bash \"$ROOT/run.sh\" \"{arm_id}\" \"$@\"\n"
        ),
        "evaluate.sh": (
            "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            f"ROOT={wrapper_root}\n"
            f"exec bash \"$ROOT/evaluate.sh\" \"{arm_id}\" \"$@\"\n"
        ),
        "finalize.sh": (
            "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            f"ROOT={wrapper_root}\n"
            f"exec bash \"$ROOT/finalize.sh\" \"{arm_id}\" \"$@\"\n"
        ),
    }


def matched_manifest(source: Path, *, seed: int = 918210, count: int = 16) -> dict[str, Any]:
    value = json.loads(source.read_text(encoding="utf-8"))
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 47:
        raise ValueError("matched evaluation source must contain exactly 47 cases")
    if len({case.get("id") for case in cases}) != 47:
        raise ValueError("matched evaluation source has duplicate case IDs")
    by_id = {case["id"]: case for case in cases}
    if count != len(MATCHED_CASE_IDS) or any(case_id not in by_id for case_id in MATCHED_CASE_IDS):
        raise ValueError("predeclared matched16 case IDs are not present in the 47-case source")
    selected = [by_id[case_id] for case_id in MATCHED_CASE_IDS]
    return {
        "schema": "text-paint-rl-judge-ablation-matched-eval-v2",
        "selection": {"seed": seed, "count": count, "source_count": 47},
        "cases": selected,
    }


def materialize(*, shared_root: Path, matrix: dict[str, Any] | None = None) -> dict[str, Any]:
    matrix = matrix or load_matrix()
    shared_root = shared_root.resolve()
    required = [shared_root / item for item in COMMON_LINKS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("shared preparation is incomplete: " + ", ".join(missing))
    shared_anchor = shared_root / "anchors-frozen"
    for filename in ("train-manifest.json", "validation-manifest.json", "reserve-manifest.json"):
        if not (shared_anchor / filename).is_file():
            raise FileNotFoundError(shared_anchor / filename)
    identity = source_identity(matrix)
    identity["shared_root_relpath"] = str(shared_root.relative_to(RUN)) if shared_root.is_relative_to(RUN) else None
    for arm in matrix["arms"]:
        arm_root = RUN / arm["directory"]
        arm_root.mkdir(parents=True, exist_ok=True)
        for item in COMMON_LINKS:
            _link_once(shared_root / item, arm_root / item)
        # Keep per-arm manifests at distinct paths while all rows continue to
        # reference the one hash-verified frozen evidence materialization.
        arm_anchor = arm_root / "anchors-frozen"
        if arm_anchor.is_symlink():
            arm_anchor.unlink()
        arm_anchor.mkdir(exist_ok=True)
        for filename in ("train-manifest.json", "validation-manifest.json", "reserve-manifest.json"):
            source = shared_anchor / filename
            target = arm_anchor / filename
            if target.exists():
                if sha256(target) != sha256(source):
                    raise ValueError(f"existing arm manifest differs: {target}")
            else:
                shutil.copy2(source, target)
        for name, text in _arm_wrappers(arm["id"]).items():
            _write_wrapper_once(arm_root / name, text)
        evaluation = arm_root / "evaluation"
        evaluation.mkdir(exist_ok=True)
        matched = evaluation / "matched16-manifest.json"
        value = matched_manifest(arm_anchor / "validation-manifest.json")
        if matched.exists() and json.loads(matched.read_text(encoding="utf-8")) != value:
            raise ValueError(f"existing matched evaluation manifest differs: {matched}")
        _write_json(matched, value)
        arm_identity = dict(identity)
        arm_identity["arm"] = {
            **arm,
            "judge_config_sha256": judge_config_sha256(matrix, arm),
            "run_root": str(arm_root.resolve()),
            "policy_download_path_file": str((arm_root / "policy-adapter-path.txt").resolve()),
            "output_path": str((arm_root / arm["output_root"]).resolve()),
        }
        arm_identity["manifest_sha256"] = sha256(arm_anchor / "train-manifest.json")
        arm_identity["heldout_manifest_sha256"] = sha256(arm_anchor / "validation-manifest.json")
        arm_identity["matched16_manifest_sha256"] = sha256(matched)
        _write_json(arm_root / "source-identity.json", arm_identity)
        _write_json(arm_root / "arm-plan.json", {
            "schema": "text-paint-rl-judge-ablation-arm-plan-v1",
            "arm": arm,
            "judge": {
                "protocol": matrix["protocol"]["judge_protocol"],
                "format": matrix["protocol"]["judge_format"],
                "model": arm["model"],
                "reasoning": matrix["protocol"]["reasoning"],
                "judge_config_sha256": judge_config_sha256(matrix, arm),
            },
            "manifests": {
                "train": str((arm_anchor / "train-manifest.json").resolve()),
                "heldout": str((arm_anchor / "validation-manifest.json").resolve()),
                "matched16": str(matched.resolve()),
            },
        })
    _write_json(RUN / "source-identity.json", identity)
    _write_json(RUN / "matrix-preparation-receipt.json", {
        "schema": "text-paint-rl-judge-ablation-preparation-v1",
        "matrix": matrix["run_name"],
        "shared_root": str(shared_root),
        "arms": [arm["id"] for arm in matrix["arms"]],
        "train_count": 40,
        "heldout_count": 47,
        "matched_eval_count": 16,
        "evaluation_seed": 918210,
        "source_identity_sha256": sha256(RUN / "source-identity.json"),
    })
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--plan", action="store_true", help="validate and print the static plan")
    parser.add_argument("--materialize", type=Path, metavar="SHARED_ROOT")
    args = parser.parse_args(argv)
    matrix = load_matrix(args.matrix)
    if args.materialize is not None:
        materialize(shared_root=args.materialize, matrix=matrix)
        print(f"prepared {len(matrix['arms'])} isolated arms from one shared source")
    else:
        print(json.dumps({"run_name": matrix["run_name"], "arms": matrix["arms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
