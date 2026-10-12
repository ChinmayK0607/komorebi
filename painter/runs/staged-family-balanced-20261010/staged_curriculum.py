"""Prepare and control the next staged, family-balanced painting RL arm.

This module is deliberately a preparation and receipt layer.  It does not
render, call a model or judge, allocate a node, or download an artifact.  The
prompt rows come from the completed diverse-flow proposal, but optimizer
exposure is scheduled explicitly by stage and family.  A normal cyclic mixed
sampler is therefore never described as a curriculum here.

The controller expects a normalized runtime receipt for every rollout. A
model-owned invalid code, canvas, current-canvas, or terminal action is a
scored trainable zero; judge or service uncertainty is excluded without a
fabricated score. Missing ownership, status, evidence, or validity fields
stops the controller before its state advances. This keeps code/canvas
validity fail-closed at the action-admission boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ID = "staged-family-balanced-20261010"
PRIOR_RUN_ID = "diverse-flow-rl-20261009"
SEED = 20261009
PRIOR_GENERATOR_SHA256 = "b86c1729a83a73ba1fc7bdf934146178c19abd21a0876c1e12ec404e4e5ffbe5"
PRIOR_PROPOSAL_SHA256 = "49262afcb3138fe4e21f2c98cff84a6bf3acb513c23d8762306c7ffb0a390f95"
STAGE_NAMES = {
    1: "single_forms",
    2: "connected_objects",
    3: "spatial_pairs",
    4: "scenes",
}
STAGE_CYCLES = {1: 4, 2: 4, 3: 8, 4: 4}
REFILL_CYCLE_MULTIPLIER = 2
TRAIN_FAMILIES_PER_STAGE = 6
HELDOUT_FAMILIES_PER_STAGE = 4
ROWS_PER_TRAIN_FAMILY = 4
ROWS_PER_HELDOUT_FAMILY = 2
ROLLOUTS_PER_PROMPT = 4
GROUP_SIZE = 4
TASK_GROUPS_PER_CYCLE = TRAIN_FAMILIES_PER_STAGE
TRACE_CAPACITY_PER_CYCLE = TASK_GROUPS_PER_CYCLE * ROLLOUTS_PER_PROMPT
BATCH_SIZE = TRACE_CAPACITY_PER_CYCLE
MIN_SCORED_ROLLOUTS_PER_GROUP = 3
OPTIMIZER_UPDATES_TARGET = 20
PRIME_REVISION = "26b3131d2716a4f8210b165df584b83b4bc54f61"
VALIDITY_FIELDS = (
    "code_valid",
    "canvas_valid",
    "current_canvas_valid",
    "terminal_action_valid",
)

STARTING_POLICY = {
    "repo": "CK0607/qwen3.8-27b-brush-painting",
    "revision": "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c",
    "prefix": "rl-judge-matched-20261004-flash/step_12",
    "sha256": "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79",
}
BASE_MODEL = {
    "id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
REWARD_OPPONENT = {
    "alias": "sft-step512",
    "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
    "sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
}
RENDERER = {
    "sha256": "dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794",
    "p5_brush_sha256": "db5b0094cb223529a8c5fc3bbba7bf8f2d5b5f5a5c0a847062194feaf917d16b",
}
JUDGE = {
    "model": "xiaomi/mimo-v2.6-flash",
    "effort": "medium",
    "rubric": "existing-generic-rubric",
    "protocol": "text-paint-sft-pairwise-consistent-v1",
    "format": "identity-joint-side-schema-v2-generic-v1",
    "rubric_sha256": "d21e4644e7a3874a0694b2ad406420e733b7638222eb7717e7f5f91cbdf1613c",
}


class CurriculumError(ValueError):
    """Raised when preparation or a runtime receipt is unsafe to continue."""


class ActionAdmissionError(CurriculumError):
    """Raised when a rollout lacks strict code/canvas/action evidence."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_surface() -> dict[str, Any]:
    """Return hash-bound preparation, canary, and launch surface metadata."""

    run_root = Path(__file__).resolve().parent
    local_files = {
        "bootstrap": "bootstrap.py",
        "bootstrap_shell": "bootstrap.sh",
        "materializer": "materialize.py",
        "prepare_config": "prepare_config.py",
        "prepare_node": "prepare_node.py",
        "canary": "canary.py",
        "launch": "launch.sh",
        "launch_guard": "launch_guard.py",
        "launch_prepared": "launch_prepared.py",
        "real_backward_canary": "real_backward_canary.py",
        "resource_preflight": "resource_preflight.py",
        "run_from_sft": "run_from_sft.py",
        "staged_trainer": "train_from_sft.py",
        "staged_init_adapter": "prime_runtime/integrations/prime/watercolour_sft/init_adapter.py",
        "runtime_binding": "runtime_binding.py",
        "sitecustomize": "sitecustomize.py",
        "prime_tail_purge": "prime_tail_purge.py",
    }
    files: dict[str, Any] = {}
    for name, relative in local_files.items():
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise CurriculumError(f"run surface file is missing or symlinked: {relative}")
        files[name] = {"path": relative, "sha256": sha256_file(path)}
    reviewed_root = run_root.parent / "photo-curriculum-rl-20260922"
    reviewed_files = {"photo_runner": "run_from_sft.py"}
    reviewed: dict[str, Any] = {}
    for name, relative in reviewed_files.items():
        path = reviewed_root / relative
        if not path.is_file() or path.is_symlink():
            raise CurriculumError(f"reviewed photo wrapper is missing or symlinked: {relative}")
        reviewed[name] = {
            "path": f"../photo-curriculum-rl-20260922/{relative}",
            "sha256": sha256_file(path),
            "status": "reviewed_external_wrapper",
        }
    files.update(reviewed)
    return {
        "schema": "staged-family-balanced-run-surface-v1",
        "status": "prepared_offline_not_run",
        "files": files,
        "bootstrap_scope": {
            "status": "prepared_not_run",
            "receipt_status": "ready_no_launch",
            "public_pinned_downloads": "prepared_only_not_executed_in_this_preparation",
            "topologies": [
                {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
                {"gpus_per_node": 4, "num_train_gpus": 2, "num_infer_gpus": 2},
                {"gpus_per_node": 8, "num_train_gpus": 4, "num_infer_gpus": 4},
            ],
            "renderer_sandbox_helper": "reviewed prepare_renderer_runtime.py/patch_brush_centroid.py with a root-to-runuser painter boundary",
            "steps": [
                "public pinned Prime/model/adapter inputs",
                "reviewed renderer sandbox helper",
                "materialize.py",
                "prepare_config.py",
                "prepare_node.py",
                "canary.py",
            ],
            "launch_called": False,
            "training_started": False,
            "model_generation": False,
            "judge_calls": False,
            "network_or_hardware_used": False,
        },
        "canary_scope": {
            "status": "prepared_not_run",
            "scope": "pinned Prime config, trainer dependency imports, import hook, console scripts, and one bounded real renderer probe",
            "orchestrator_setup_called": False,
            "gpu_or_cuda": False,
            "network": False,
            "model": False,
            "judge": False,
        },
        "launch_scope": {
            "status": "prepared_guarded_foreground_not_run",
            "background_processes_started_by_shell": False,
            "bounded_timeout": True,
            "gpu_model_judge_execution": "deferred_until_explicit_launch",
            "receipt_derived_launcher": "launch_prepared.py",
            "requires_explicit_timeout_seconds": True,
            "requires_owner_only_gateway_mode": "0600",
            "gateway_contents_read_or_logged": False,
            "scheduling": False,
        },
        "no_gpu_or_external_calls": True,
        "brush_root": "painter/rl_helpers",
    }


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _prior_generator_path() -> Path:
    return Path(__file__).resolve().parents[1] / PRIOR_RUN_ID / "prepare_curriculum.py"


def _load_prior_generator() -> Any:
    path = _prior_generator_path()
    if not path.is_file() or sha256_file(path) != PRIOR_GENERATOR_SHA256:
        raise CurriculumError("prior curriculum generator is missing or its immutable hash changed")
    spec = importlib.util.spec_from_file_location("diverse_flow_prepare_curriculum", path)
    if spec is None or spec.loader is None:
        raise CurriculumError("could not load prior curriculum generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only immutable source fields needed by the schedule."""

    return {
        "id": str(row["id"]),
        "source_group": str(row["source_group"]),
        "family": str(row["base_family_id"]),
        "family_kind": str(row["family_kind"]),
        "stage": int(row["stage"]),
        "split": str(row["split"]),
        "variant": int(row["variant"]),
        "reference_sha256": str(row["reference_sha256"]),
        "prompt_signature": str(row["prompt_signature"]),
    }


def source_rows(seed: int = SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load exact prior rows and split them without changing prompt bytes."""

    if seed != SEED:
        raise CurriculumError(f"only the prior immutable seed {SEED} is allowed")
    module = _load_prior_generator()
    rows = module.generate_rows(seed)
    train = [_row_identity(row) for row in rows if row["split"] == "train"]
    heldout = [_row_identity(row) for row in rows if row["split"] == "heldout"]
    if len(train) != 96 or len(heldout) != 32:
        raise CurriculumError("prior source rows do not have the recorded 96/32 split")
    return train, heldout


def _families(rows: Sequence[Mapping[str, Any]], stage: int, split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["stage"]) == stage and row["split"] == split:
            grouped.setdefault(str(row["family"]), []).append(dict(row))
    for values in grouped.values():
        values.sort(key=lambda item: (int(item["variant"]), str(item["id"])))
    return dict(sorted(grouped.items()))


def _schedule_for_stage(stage: int, families: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    cycles = STAGE_CYCLES[stage] * REFILL_CYCLE_MULTIPLIER
    if len(families) != TRAIN_FAMILIES_PER_STAGE:
        raise CurriculumError(f"stage {stage} needs {TRAIN_FAMILIES_PER_STAGE} train families")
    for family, rows in families.items():
        if len(rows) != ROWS_PER_TRAIN_FAMILY:
            raise CurriculumError(f"{family} needs {ROWS_PER_TRAIN_FAMILY} source rows")
    slots: list[dict[str, Any]] = []
    family_names = tuple(families)
    for cycle in range(1, cycles + 1):
        for group_index, family in enumerate(family_names):
            # One task dispatch creates one Prime group. Its four rollouts
            # share this prompt row, so each family contributes one group and
            # a capacity of four traces per sampling cycle.
            variant = (cycle - 1) % ROWS_PER_TRAIN_FAMILY
            row = families[family][variant]
            slots.append({
                "stage": stage,
                "cycle": cycle,
                "cycle_id": f"s{stage}-c{cycle:02d}",
                "group_index": group_index,
                "group_id": f"s{stage}-c{cycle:02d}-g{group_index:02d}",
                "family": family,
                "prompt_row_id": str(row["id"]),
                "variant": int(row["variant"]),
                "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
                "trace_capacity": ROLLOUTS_PER_PROMPT,
            })
        if len([slot for slot in slots if slot["cycle"] == cycle]) != TASK_GROUPS_PER_CYCLE:
            raise AssertionError("family-balanced cycle has the wrong slot count")
    return slots


def ordered_cycle_slots(stage: Mapping[str, Any], cycle: int) -> list[dict[str, Any]]:
    """Return one cycle in a stable rotation of its family slots."""

    slots = [slot for slot in stage["schedule"] if slot["cycle"] == cycle]
    if not slots:
        return []
    offset = (cycle - 1) % len(slots)
    return slots[offset:] + slots[:offset]


def build_curriculum(seed: int = SEED) -> dict[str, Any]:
    """Return the deterministic staged train schedule and fixed heldout panel."""

    train, heldout = source_rows(seed)
    stages: list[dict[str, Any]] = []
    all_train_ids: list[str] = []
    for stage in range(1, 5):
        train_families = _families(train, stage, "train")
        heldout_families = _families(heldout, stage, "heldout")
        if len(heldout_families) != HELDOUT_FAMILIES_PER_STAGE:
            raise CurriculumError(f"stage {stage} needs {HELDOUT_FAMILIES_PER_STAGE} heldout families")
        for family, rows in heldout_families.items():
            if len(rows) != ROWS_PER_HELDOUT_FAMILY:
                raise CurriculumError(f"{family} needs {ROWS_PER_HELDOUT_FAMILY} heldout rows")
        train_rows = [row for values in train_families.values() for row in values]
        heldout_rows = [row for values in heldout_families.values() for row in values]
        schedule = _schedule_for_stage(stage, train_families)
        all_train_ids.extend(row["id"] for row in train_rows)
        stages.append({
            "stage": stage,
            "name": STAGE_NAMES[stage],
            "cycles": STAGE_CYCLES[stage],
            "max_sampling_cycles": STAGE_CYCLES[stage] * REFILL_CYCLE_MULTIPLIER,
            "optimizer_updates_target": STAGE_CYCLES[stage],
            "task_groups_per_cycle": TASK_GROUPS_PER_CYCLE,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "trace_capacity_per_cycle": TRACE_CAPACITY_PER_CYCLE,
            "trace_exposure": "measured_after_admission",
            "minimum_scored_rollouts_per_group": MIN_SCORED_ROLLOUTS_PER_GROUP,
            "train_families": list(train_families),
            "heldout_families": list(heldout_families),
            "train_rows": train_rows,
            "heldout_rows": heldout_rows,
            "schedule": schedule,
            "schedule_sha256": sha256_bytes(_canonical(schedule)),
            "family_group_exposure_per_cycle": {family: 1 for family in train_families},
            "family_trace_capacity_per_cycle": {family: ROLLOUTS_PER_PROMPT for family in train_families},
            "family_trace_capacity_total": {
                family: STAGE_CYCLES[stage] * ROLLOUTS_PER_PROMPT for family in train_families
            },
        })
    heldout_panel = [row for stage in stages for row in stage["heldout_rows"]]
    train_unique = [row for stage in stages for row in stage["train_rows"]]
    if len(train_unique) != 96 or len({row["id"] for row in train_unique}) != 96:
        raise CurriculumError("staged train rows are not the exact 96 unique prior rows")
    if len(heldout_panel) != 32 or len({row["id"] for row in heldout_panel}) != 32:
        raise CurriculumError("matched heldout panel is not exactly 32 unique rows")
    panel_hash = sha256_bytes(_canonical(heldout_panel))
    return {
        "schema": "staged-family-balanced-curriculum-v1",
        "run_id": RUN_ID,
        "prior_run_id": PRIOR_RUN_ID,
        "seed": seed,
        "source": {
            "generator": f"painter/runs/{PRIOR_RUN_ID}/prepare_curriculum.py",
            "generator_sha256": PRIOR_GENERATOR_SHA256,
            "prior_canonical_proposal_sha256": PRIOR_PROPOSAL_SHA256,
            "row_bytes_unchanged": True,
        },
        "sampling": {
            "kind": "sequential_stage_family_quota",
            "cyclic_mixed_sampling_claim": False,
            "stage_advancement": "optimizer_target_receipt_only",
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "task_groups_per_cycle": TASK_GROUPS_PER_CYCLE,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "trace_capacity_per_cycle": TRACE_CAPACITY_PER_CYCLE,
            "trace_exposure": "measured_after_admission; each family has capacity four and may contribute three under judge uncertainty",
            "minimum_scored_rollouts_per_group": MIN_SCORED_ROLLOUTS_PER_GROUP,
            "complete_groups_per_cycle": TASK_GROUPS_PER_CYCLE,
            "stage_cycles": {str(stage): STAGE_CYCLES[stage] for stage in range(1, 5)},
            "sampling_cycles": sum(STAGE_CYCLES.values()),
            "sampling_cycle_cap": sum(STAGE_CYCLES.values()) * REFILL_CYCLE_MULTIPLIER,
            "refill_cycle_multiplier": REFILL_CYCLE_MULTIPLIER,
            "optimizer_updates_target": OPTIMIZER_UPDATES_TARGET,
        },
        "stages": stages,
        "matched_heldout_panel": {
            "cases": len(heldout_panel),
            "panel_sha256": panel_hash,
            "same_panel_for_start_and_final": True,
            "rows": heldout_panel,
        },
        "train_row_ids": all_train_ids,
    }


def build_config(curriculum: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create the launch config while preserving prior identities."""

    plan = build_curriculum() if curriculum is None else curriculum
    panel = plan["matched_heldout_panel"]
    return {
        "schema": "staged-family-balanced-config-v1",
        "run_id": RUN_ID,
        "status": "preparation_only_pending_runtime_review",
        "base_model": BASE_MODEL,
        "starting_policy": STARTING_POLICY,
        "reward_opponent": REWARD_OPPONENT,
        "judge": JUDGE,
        "renderer": RENDERER,
        "curriculum": {
            "manifest": "curriculum-manifest.json",
            "manifest_sha256": sha256_bytes(_canonical(plan)),
            "sampling_kind": plan["sampling"]["kind"],
            "stage_advancement": plan["sampling"]["stage_advancement"],
            "stage_cycles": plan["sampling"]["stage_cycles"],
            "sampling_cycles": sum(STAGE_CYCLES.values()),
            "sampling_cycle_cap": sum(STAGE_CYCLES.values()) * REFILL_CYCLE_MULTIPLIER,
            "refill_cycle_multiplier": REFILL_CYCLE_MULTIPLIER,
            "optimizer_updates_target": OPTIMIZER_UPDATES_TARGET,
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "task_groups_per_cycle": TASK_GROUPS_PER_CYCLE,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "trace_capacity_per_cycle": TRACE_CAPACITY_PER_CYCLE,
        },
        "rl": {
            "learning_rate": 1e-6,
            "max_policy_age": 1,
            "task_groups_per_cycle": TASK_GROUPS_PER_CYCLE,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "trace_capacity_per_cycle": TRACE_CAPACITY_PER_CYCLE,
            "new_code_style_reward": False,
            "require_valid_final_action": True,
            "validity_admission": {
                "required_boolean_fields": list(VALIDITY_FIELDS),
                "model_owned_invalid": "score_zero_and_train",
                "judge_uncertain_or_service_failure": "exclude_without_zero",
                "malformed_or_missing_ownership_status_evidence": "fail_closed_stop",
                "minimum_scored_rollouts_per_group": MIN_SCORED_ROLLOUTS_PER_GROUP,
            },
        },
        "supervision": {
            "enabled": False,
            "updates": 0,
            "reason": "same-run four-step code/canvas SFT was mixed and heldout validity was 32/32 both before and after; retain exact Flash12 initializer",
        },
        "evaluation": {
            "cases": panel["cases"],
            "panel_sha256": panel["panel_sha256"],
            "policies": ["exact starting Flash12", "final staged-family-balanced adapter"],
            "seed": 918210,
            "temperature": 1.0,
            "enable_thinking": False,
            "max_turns": 3,
            "max_tokens_per_turn": 8192,
            "strict_final_action_and_current_canvas": True,
            "same_renderer_and_prompt_panel": True,
        },
        "pending_launch_choices": {
            "batch_size_24_runtime_compatibility": "must pass a node-free Prime adapter check before allocation; six task groups times four rollouts give 24 trace capacity per sampling cycle, while admitted traces may span cycles to fill an optimizer batch",
            "prime_tail_purge_hook": "required runtime hook must physically consume and purge the recorded stage-tail trace IDs before an optimizer-target receipt can advance stages",
            "renderer_canary": "must measure the repaired renderer tail before any scaled rental",
            "stage_quality_threshold": "no unregistered quality threshold is applied; optimizer target completion advances the schedule, while receipts require valid admitted groups and explicit family quotas",
            "sampling_attempt_accounting": "each admitted attempt, including censored retries and bounded partial refills, increments persisted per-stage attempts and is bounded by the two-times refill cap",
            "batch_boundary_guard": "native TrainSink._maybe_batch is blocked during group assembly and released only after a run-owned complete-group-prefix pass receipt",
            "stale_group_cancellation": "stale GroupCancellation voids every arrived member as pipeline evidence and never enters the trainer candidate inventory",
            "bootstrap": "bootstrap.py/bootstrap.sh are prepared but unrun; on an explicit Linux execution they accept only exact 1+1, 2+2, or 4+4 train/inference GPU topologies, with 4+4 reserved for an 8-GPU whole-node offer split into four trainer and four inference GPUs; verify public pinned downloads, run the reviewed renderer sandbox helper plus materialization/preflight/canary, and write ready_no_launch without starting Prime",
            "real_backward_canary": "optional explicit --real-backward-canary mode; exactly one optimizer update, one native resumable checkpoint, one verified public portable adapter, declared elapsed <= 2400 seconds, declared attempt cost <= 9 USD, and zero leftover processes are required",
        },
        "runtime_binding": {
            "module": "runtime_binding.py",
            "sitecustomize": "sitecustomize.py",
            "status": "node_free_fake_prime_validated_pending_linux_canary",
            "prime_revision": PRIME_REVISION,
            "activation": {
                "PAINTER_REQUIRE_VALID_FINAL_ACTION": "1",
                "PAINTER_STAGED_CURRICULUM_ENABLE": "1",
                "PAINTER_STAGED_CURRICULUM_RUN_ID": RUN_ID,
                "PAINTER_STAGED_CURRICULUM_PROTOCOL": "staged-family-balanced-protocol-v1",
                "PAINTER_STAGED_CURRICULUM_ADMISSION": "1",
                "PAINTER_STAGED_CURRICULUM_ROOT": "<run-root>",
                "PAINTER_STAGED_CURRICULUM_STATE": "<run-root>/runtime-state.json",
            },
            "atomic_state": True,
            "stage_tail_purge_required_before_advance": True,
            "optimizer_update_confirmation": "TrainBatch selection is journaled first; only WeightWatcher.on_update for the expected policy version records the optimizer update",
            "dispatch_gate": "binding-owned complete-group-prefix gate is layered with Prime lead permission; active groups drain while fresh dispatch is paused",
            "trainer_candidate_inventory": "controller inventory contains only exact trace IDs present in native TrainSink.pending_batch after process_group",
            "batch_boundary": "_maybe_batch is allowed only after the active complete-group-prefix pass; in-group calls are fail-closed and cannot ship carry-over traces",
            "resume_scope": "only clean optimizer/stage boundaries or a persisted exact selected-batch/update/tail-purge reconciliation may resume; partial native buffers fail closed",
            "full_prime_compatibility": False,
        },
        "run_surface": build_run_surface(),
    }


def _git_revision(root: Path) -> str | None:
    """Read a local Prime checkout revision without network or mutation."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root.expanduser().resolve()), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _load_prime_config_helper() -> Any:
    """Load the repository's no-GPU Prime config validator."""

    path = Path(__file__).resolve().parents[2] / "prepare_scaled_native_rl.py"
    if not path.is_file():
        raise CurriculumError(f"shared Prime config helper is missing: {path}")
    spec = importlib.util.spec_from_file_location("prepare_scaled_native_rl", path)
    if spec is None or spec.loader is None:
        raise CurriculumError("could not load shared Prime config helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prime_compatibility_config() -> dict[str, Any]:
    """Build the pinned helper's native config shape for batch-24 checking."""

    helper = _load_prime_config_helper()
    config = helper.build_config(
        model="/staged/model/pending",
        run_root=Path("/staged/run/pending"),
        checkpoint_interval=4,
        run_name=RUN_ID,
    )
    config["max_steps"] = OPTIMIZER_UPDATES_TARGET
    config["trainer"]["max_steps"] = OPTIMIZER_UPDATES_TARGET
    config["orchestrator"]["max_steps"] = OPTIMIZER_UPDATES_TARGET
    config["orchestrator"]["batch_size"] = BATCH_SIZE
    config["orchestrator"]["group_size"] = GROUP_SIZE
    config["orchestrator"]["concurrency"].update({
        "min_inflight": BATCH_SIZE,
        "initial_inflight": BATCH_SIZE,
        "max_inflight": BATCH_SIZE,
    })
    config["inference"]["vllm"]["max_num_seqs"] = BATCH_SIZE
    return config


def validate_prime_compatibility(prime_root: Path | None = None) -> dict[str, Any]:
    """Validate batch/group semantics through pinned Prime when available.

    The result is intentionally explicit when Prime is unavailable.  It never
    allocates hardware, imports CUDA, starts a trainer, or treats an unknown
    local package as evidence for the pinned revision.
    """

    result: dict[str, Any] = {
        "schema": "staged-family-balanced-prime-compatibility-v1",
        "prime_revision": PRIME_REVISION,
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "task_groups_per_cycle": TASK_GROUPS_PER_CYCLE,
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "trace_capacity_per_cycle": TRACE_CAPACITY_PER_CYCLE,
        "status": "unavailable",
        "validated": False,
        "no_gpu_or_external_calls": True,
    }
    if prime_root is None:
        result["message"] = "Prime root not supplied; pinned revision cannot be verified offline on this host"
        return result
    root = prime_root.expanduser().resolve()
    actual = _git_revision(root)
    result["prime_root"] = str(root)
    result["observed_revision"] = actual
    if actual != PRIME_REVISION:
        result["status"] = "unavailable" if actual is None else "failed"
        result["message"] = "Prime checkout is missing or does not match the pinned revision"
        return result
    try:
        helper = _load_prime_config_helper()
        for dependency in (
            root / "verifiers",
            root.parent / "verifiers-upstream",
            root / "deps/renderers",
            root / "deps/renderers/renderers",
        ):
            if dependency.is_dir() and str(dependency) not in sys.path:
                sys.path.insert(0, str(dependency))
        valid, message = helper.validate_with_prime(build_prime_compatibility_config(), root)
    except Exception as exc:  # optional dependency availability is host-specific
        valid, message = False, f"Prime validation helper failed: {type(exc).__name__}: {exc}"
    result["message"] = message
    result["validated"] = bool(valid)
    result["status"] = "passed" if valid else "failed"
    return result


def _require_exact_bool(record: Mapping[str, Any], key: str) -> None:
    if key not in record or type(record.get(key)) is not bool:
        raise ActionAdmissionError(f"{key} must be an explicit boolean for action admission")


def _require_evidence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ActionAdmissionError("rollout evidence is missing or malformed")
    source = evidence.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ActionAdmissionError("rollout evidence needs an explicit source")
    return evidence


def admit_action(
    record: Mapping[str, Any],
    *,
    expected_row_id: str,
    expected_family: str,
    expected_stage: int,
) -> dict[str, Any]:
    """Classify one rollout while preserving model-invalid learning signal.

    A model-owned invalid action is a completed rollout with an explicit zero
    reward and remains trainable.  Judge uncertainty and service failure are
    excluded with no score.  Every branch requires explicit ownership, status,
    validity fields, and evidence so missing metadata cannot become a zero.
    """

    if not isinstance(record, Mapping):
        raise ActionAdmissionError("rollout receipt must be a mapping")
    if record.get("row_id") != expected_row_id:
        raise ActionAdmissionError("rollout row does not match its scheduled slot")
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise ActionAdmissionError("rollout trace_id is missing or malformed")
    if record.get("family") != expected_family or type(record.get("stage")) is not int or record.get("stage") != expected_stage:
        raise ActionAdmissionError("rollout family or stage does not match its scheduled slot")
    ownership = record.get("ownership")
    status = record.get("status")
    if ownership not in {"model", "judge", "service"}:
        raise ActionAdmissionError("rollout ownership is missing or malformed")
    if status not in {"scored", "model_invalid", "judge_uncertain", "service_failure"}:
        raise ActionAdmissionError("rollout status is missing or malformed")
    evidence = _require_evidence(record)
    for key in VALIDITY_FIELDS:
        if key not in record or (record[key] is not None and type(record[key]) is not bool):
            raise ActionAdmissionError(f"{key} must be an explicit boolean or null")
    validity = [record[key] for key in VALIDITY_FIELDS]
    if status == "scored":
        if ownership != "model" or validity != [True, True, True, True]:
            raise ActionAdmissionError("scored rollout must be an explicitly valid model outcome")
        if record.get("reward_status") != "scored":
            raise ActionAdmissionError("scored rollout has inconsistent reward status")
        reward = record.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ActionAdmissionError("scored rollout needs one finite numeric reward")
        return {
            "row_id": expected_row_id,
            "trace_id": trace_id,
            "family": expected_family,
            "stage": expected_stage,
            "ownership": ownership,
            "status": status,
            "evidence": dict(evidence),
            "reward": float(reward),
            "outcome": "scored",
            "trainable": True,
            "model_invalid": False,
            "admission": "scored",
        }
    if status == "model_invalid":
        if ownership != "model" or not all(type(value) is bool for value in validity):
            raise ActionAdmissionError("model-invalid rollout needs model-owned boolean validity evidence")
        if all(value is True for value in validity):
            raise ActionAdmissionError("model-invalid rollout must identify at least one invalid action/canvas field")
        reward_status = record.get("reward_status")
        if not (reward_status == "scored" or (isinstance(reward_status, str) and reward_status.startswith("model_invalid_"))) or "reward" not in record:
            raise ActionAdmissionError("model-invalid rollout needs an explicit model-invalid or scored zero")
        reward = record.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)) or float(reward) != 0.0:
            raise ActionAdmissionError("model-invalid rollout must carry finite reward zero")
        if not isinstance(evidence.get("reason"), str) or not evidence["reason"].strip():
            raise ActionAdmissionError("model-invalid rollout needs an evidence reason")
        return {
            "row_id": expected_row_id,
            "trace_id": trace_id,
            "family": expected_family,
            "stage": expected_stage,
            "ownership": ownership,
            "status": status,
            "evidence": dict(evidence),
            "reward": 0.0,
            "outcome": "model_invalid",
            "trainable": True,
            "model_invalid": True,
            "admission": "scored_model_invalid",
        }
    if status == "judge_uncertain":
        expected_owner = "judge"
    else:
        expected_owner = "service"
    if ownership != expected_owner:
        raise ActionAdmissionError(f"{status} needs {expected_owner}-owned validity evidence")
    if record.get("reward_status") != "unscored" or "reward" not in record or record.get("reward") is not None:
        raise ActionAdmissionError(f"{status} must remain explicitly unscored without a reward")
    if not isinstance(evidence.get("reason"), str) or not evidence["reason"].strip():
        raise ActionAdmissionError(f"{status} needs an evidence reason")
    return {
        "row_id": expected_row_id,
        "trace_id": trace_id,
        "family": expected_family,
        "stage": expected_stage,
        "ownership": ownership,
        "status": status,
        "evidence": dict(evidence),
        "reward": None,
        "outcome": status,
        "trainable": False,
        "model_invalid": False,
        "admission": "excluded",
    }


def admit_group(
    record: Mapping[str, Any],
    *,
    expected_group_id: str,
    expected_row_id: str,
    expected_family: str,
    expected_stage: int,
) -> dict[str, Any]:
    """Admit one Prime task group and all four of its trainable traces."""

    if not isinstance(record, Mapping):
        raise ActionAdmissionError("task group receipt must be a mapping")
    if record.get("group_id") != expected_group_id:
        raise ActionAdmissionError("task group does not match its scheduled group")
    if record.get("row_id") != expected_row_id or record.get("family") != expected_family:
        raise ActionAdmissionError("task group prompt does not match its scheduled row")
    if type(record.get("stage")) is not int or record.get("stage") != expected_stage:
        raise ActionAdmissionError("task group stage does not match its scheduled stage")
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != ROLLOUTS_PER_PROMPT:
        raise ActionAdmissionError("task group must contain exactly four rollouts")
    admitted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()
    for rollout_index, rollout in enumerate(rollouts):
        if not isinstance(rollout, Mapping):
            raise ActionAdmissionError("task group contains a malformed rollout")
        if rollout.get("group_id") != expected_group_id or type(rollout.get("rollout_index")) is not int or rollout.get("rollout_index") != rollout_index:
            raise ActionAdmissionError("rollout identity does not match its task group")
        trace_id = rollout.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id.strip() or trace_id in seen_trace_ids:
            raise ActionAdmissionError("task group contains a missing or duplicate trace_id")
        seen_trace_ids.add(trace_id)
        result = admit_action(
            rollout,
            expected_row_id=expected_row_id,
            expected_family=expected_family,
            expected_stage=expected_stage,
        )
        (admitted if result["trainable"] else excluded).append(result)
    if len(admitted) < MIN_SCORED_ROLLOUTS_PER_GROUP:
        return {
            "group_id": expected_group_id,
            "row_id": expected_row_id,
            "family": expected_family,
            "stage": expected_stage,
            "rollouts": admitted,
            "excluded": excluded,
            "scored_rollouts": len(admitted),
            "excluded_rollouts": len(excluded),
            "scored_trace_count": len(admitted),
            "trainer_candidate_trace_ids": record.get("trainer_candidate_trace_ids"),
            "admission": "censored",
            "censor_reason": "fewer_than_minimum_scored_rollouts",
        }
    return {
        "group_id": expected_group_id,
        "row_id": expected_row_id,
        "family": expected_family,
        "stage": expected_stage,
        "rollouts": admitted,
        "excluded": excluded,
        "scored_rollouts": len(admitted),
        "excluded_rollouts": len(excluded),
        "scored_trace_count": len(admitted),
        "trainer_candidate_trace_ids": record.get("trainer_candidate_trace_ids"),
        "admission": "admitted",
    }


@dataclass
class StageController:
    """State machine for sampling-cycle receipts and schedule progression.

    The first pass through a sampling cycle dispatches six task groups.
    Prime may combine several cycles into one optimizer update when
    uncertainty leaves fewer than 24 scored traces; later refills dispatch
    only the deterministic complete-group prefix needed to reach 24, so
    optimizer updates are recorded independently.
    """

    plan: Mapping[str, Any]
    active_stage: int = 1
    next_cycle: int = 1
    optimizer_updates_observed: int = 0
    stage_optimizer_updates_observed: int = 0
    completed_cycle_ids: list[str] = field(default_factory=list)
    trace_ids_by_cycle: dict[str, list[str]] = field(default_factory=dict)
    sampling_attempts_by_stage: dict[str, int] = field(default_factory=dict)
    halted: bool = False

    def _available_trace_total(self) -> int:
        return sum(len(trace_ids) for trace_ids in self.trace_ids_by_cycle.values())

    @property
    def trace_inventory_by_cycle(self) -> dict[str, int]:
        """Count view retained for receipt compatibility; IDs are authoritative."""

        return {cycle_id: len(trace_ids) for cycle_id, trace_ids in self.trace_ids_by_cycle.items()}

    def _stage(self) -> Mapping[str, Any]:
        if self.active_stage > 4:
            raise CurriculumError("curriculum is already complete")
        return self.plan["stages"][self.active_stage - 1]

    def observe_cycle(self, records: Sequence[Mapping[str, Any]], *, partial: bool = False) -> dict[str, Any]:
        """Admit one deterministic cycle or a bounded group refill.

        The first pass through a stage remains the exact six-family cycle.  A
        later optimizer refill may use only the prefix of the next scheduled
        cycle needed to reach 24 traces.  Every refill still admits complete
        groups of four and records its exact family prefix, so the controller
        never launches a speculative six-group cycle just to fill a small
        deficit.
        """

        if self.halted:
            raise CurriculumError("controller is halted; a new run is required")
        stage = self._stage()
        attempt_key = str(self.active_stage)
        attempts = self.sampling_attempts_by_stage.get(attempt_key, 0)
        if attempts >= stage["max_sampling_cycles"]:
            return self._stop(stage, "sampling_cycle_cap_exhausted", {
                "max_sampling_cycles": stage["max_sampling_cycles"],
                "sampling_attempts": attempts,
                "stage_optimizer_updates_observed": self.stage_optimizer_updates_observed,
                "stage_optimizer_updates_target": stage["optimizer_updates_target"],
            })
        attempts += 1
        self.sampling_attempts_by_stage[attempt_key] = attempts
        cycle_slots = ordered_cycle_slots(stage, self.next_cycle)
        if partial:
            if not records or len(records) > len(cycle_slots):
                return self._stop(stage, "invalid_partial_refill_cardinality", {
                    "cycle": self.next_cycle, "observed": len(records), "available": len(cycle_slots),
                })
            expected = cycle_slots[:len(records)]
        else:
            expected = cycle_slots
        if len(records) != len(expected):
            return self._stop(stage, "wrong_cycle_cardinality", {"expected": len(expected), "observed": len(records)})
        by_id: dict[str, Mapping[str, Any]] = {}
        for record in records:
            group_id = record.get("group_id") if isinstance(record, Mapping) else None
            if not isinstance(group_id, str) or group_id in by_id:
                return self._stop(stage, "duplicate_or_missing_group_id", {})
            by_id[group_id] = record
        admitted_groups: list[dict[str, Any]] = []
        try:
            for slot in expected:
                admitted_groups.append(admit_group(
                    by_id[slot["group_id"]],
                    expected_group_id=slot["group_id"],
                    expected_row_id=slot["prompt_row_id"],
                    expected_family=slot["family"],
                    expected_stage=self.active_stage,
                ))
        except (KeyError, ActionAdmissionError) as exc:
            return self._stop(stage, "action_admission_failed", {"error": str(exc)})
        censored_groups = [group for group in admitted_groups if group["admission"] == "censored"]
        if censored_groups:
            return self._censor_cycle(stage, admitted_groups)
        group_counts: dict[str, int] = {}
        scored_counts: dict[str, int] = {}
        excluded_counts: dict[str, int] = {}
        trainer_candidate_counts: dict[str, int] = {}
        model_invalid_counts: dict[str, int] = {}
        admitted_rollouts = [rollout for group in admitted_groups for rollout in group["rollouts"]]
        excluded_rollouts = [rollout for group in admitted_groups for rollout in group["excluded"]]
        trainer_candidate_ids_by_group: dict[str, list[str]] = {}
        for group in admitted_groups:
            group_counts[group["family"]] = group_counts.get(group["family"], 0) + 1
            scored_counts[group["family"]] = scored_counts.get(group["family"], 0) + group["scored_rollouts"]
            excluded_counts[group["family"]] = excluded_counts.get(group["family"], 0) + group["excluded_rollouts"]
            candidates = group.get("trainer_candidate_trace_ids")
            if candidates is None:
                candidates = [rollout["trace_id"] for rollout in group["rollouts"]]
            if not isinstance(candidates, list) or any(not isinstance(value, str) or not value.strip() for value in candidates):
                return self._stop(stage, "trainer_candidate_identity_malformed", {"group_id": group["group_id"]})
            scored_ids = {rollout["trace_id"] for rollout in group["rollouts"]}
            if len(set(candidates)) != len(candidates) or not set(candidates).issubset(scored_ids):
                return self._stop(stage, "trainer_candidate_not_scored_subset", {"group_id": group["group_id"]})
            trainer_candidate_ids_by_group[group["group_id"]] = list(candidates)
            trainer_candidate_counts[group["family"]] = trainer_candidate_counts.get(group["family"], 0) + len(candidates)
            model_invalid_counts[group["family"]] = model_invalid_counts.get(group["family"], 0) + sum(
                1 for rollout in group["rollouts"] if rollout["model_invalid"]
            )
        expected_group_counts: dict[str, int] = {}
        for slot in expected:
            expected_group_counts[slot["family"]] = expected_group_counts.get(slot["family"], 0) + 1
        if group_counts != expected_group_counts:
            return self._stop(stage, "family_group_quota_mismatch", {"expected_groups": expected_group_counts, "observed_groups": group_counts})
        mean_reward = sum(row["reward"] for row in admitted_rollouts) / len(admitted_rollouts)
        receipt: dict[str, Any] = {
            "schema": "staged-family-balanced-controller-receipt-v1",
            "run_id": RUN_ID,
            "stage": self.active_stage,
            "stage_name": stage["name"],
            "cycle": self.next_cycle,
            "cycle_id": f"s{self.active_stage}-c{self.next_cycle:02d}",
            "status": "partial_refill_complete" if partial else "cycle_complete",
            "decision": "pass",
            "progression_kind": "schedule_only_not_quality_gate",
            "task_groups": len(admitted_groups),
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "rollouts_observed": len(admitted_rollouts) + len(excluded_rollouts),
            "scored_rollouts": len(admitted_rollouts),
            "excluded_rollouts": len(excluded_rollouts),
            "trainer_traces": sum(len(values) for values in trainer_candidate_ids_by_group.values()),
            "scored_trace_exposure": len(admitted_rollouts),
            "family_scheduled_trace_counts": {
                family: expected_group_counts.get(family, 0) * ROLLOUTS_PER_PROMPT
                for family in expected_group_counts
            },
            "family_group_counts": group_counts,
            "family_scored_trace_counts": scored_counts,
            "family_trainer_candidate_trace_counts": trainer_candidate_counts,
            "family_excluded_rollout_counts": excluded_counts,
            "family_model_invalid_counts": model_invalid_counts,
            "excluded_evidence": [
                {"family": row["family"], "status": row["status"], "evidence": row["evidence"]}
                for row in excluded_rollouts
            ],
            "model_invalid_evidence": [
                {"family": row["family"], "status": row["status"], "evidence": row["evidence"]}
                for row in admitted_rollouts if row["model_invalid"]
            ],
            "trace_exposure": "measured_after_admission",
            "optimizer_updates_observed": self.optimizer_updates_observed,
            "stage_optimizer_updates_observed": self.stage_optimizer_updates_observed,
            "stage_optimizer_updates_target": stage["optimizer_updates_target"],
            "mean_reward": mean_reward,
            "sampling_attempt": attempts,
            "sampling_attempt_cap": stage["max_sampling_cycles"],
            "partial_refill": partial,
            "scheduled_group_count": len(expected),
            "scheduled_group_ids": [slot["group_id"] for slot in expected],
        }
        self.completed_cycle_ids.append(receipt["cycle_id"])
        cycle_trace_ids = [trace_id for group in admitted_groups for trace_id in trainer_candidate_ids_by_group[group["group_id"]]]
        if len(set(cycle_trace_ids)) != len(cycle_trace_ids):
            return self._stop(stage, "duplicate_trace_id", {"cycle": receipt["cycle_id"]})
        prior_trace_ids = {trace_id for trace_ids in self.trace_ids_by_cycle.values() for trace_id in trace_ids}
        if prior_trace_ids.intersection(cycle_trace_ids):
            return self._stop(stage, "duplicate_trace_id_across_cycles", {"cycle": receipt["cycle_id"]})
        self.trace_ids_by_cycle[receipt["cycle_id"]] = cycle_trace_ids
        receipt["trace_inventory_by_cycle"] = dict(self.trace_inventory_by_cycle)
        receipt["trace_ids_by_cycle"] = {cycle_id: list(trace_ids) for cycle_id, trace_ids in self.trace_ids_by_cycle.items()}
        receipt["available_scored_traces"] = self._available_trace_total()
        receipt["trainer_candidate_trace_ids_by_group"] = trainer_candidate_ids_by_group
        if self.next_cycle == stage["max_sampling_cycles"]:
            receipt["cycle_cap_reached"] = True
        self.next_cycle += 1
        return receipt

    def record_optimizer_update(self, *, trace_ids_by_cycle: Mapping[str, Sequence[str]]) -> dict[str, Any]:
        """Consume exactly one Prime batch, attributed to active-stage cycles."""

        if self.halted:
            raise CurriculumError("controller is halted; a new run is required")
        if not isinstance(trace_ids_by_cycle, Mapping) or not trace_ids_by_cycle:
            raise CurriculumError("optimizer update needs a non-empty trace_ids_by_cycle mapping")
        if self.stage_optimizer_updates_observed >= self._stage()["optimizer_updates_target"]:
            raise CurriculumError("stage optimizer target reached; drain the stage tail before advancing")
        active_prefix = f"s{self.active_stage}-c"
        if any(not isinstance(cycle_id, str) for cycle_id in trace_ids_by_cycle) or any(
            not cycle_id.startswith(active_prefix)
            or cycle_id not in self.trace_ids_by_cycle
            or re.fullmatch(rf"s{self.active_stage}-c[0-9]{{2}}", cycle_id) is None
            for cycle_id in trace_ids_by_cycle
        ):
            raise CurriculumError("optimizer update must bind only active-stage completed cycles")
        if any(not isinstance(trace_ids, Sequence) or isinstance(trace_ids, (str, bytes)) or not trace_ids for trace_ids in trace_ids_by_cycle.values()):
            raise CurriculumError("optimizer trace IDs must be non-empty sequences")
        all_trace_ids = [trace_id for trace_ids in trace_ids_by_cycle.values() for trace_id in trace_ids]
        if any(not isinstance(trace_id, str) or not trace_id.strip() for trace_id in all_trace_ids):
            raise CurriculumError("optimizer trace IDs must be non-empty strings")
        if len(set(all_trace_ids)) != len(all_trace_ids) or len(all_trace_ids) != BATCH_SIZE:
            raise CurriculumError("an optimizer update must attribute exactly 24 unique scored trace IDs")
        if any(
            not set(trace_ids).issubset(set(self.trace_ids_by_cycle[cycle_id]))
            for cycle_id, trace_ids in trace_ids_by_cycle.items()
        ):
            raise CurriculumError("optimizer update attributes a trace ID outside controller inventory")
        stage = self._stage()
        inventory_before = dict(self.trace_inventory_by_cycle)
        trace_ids_before = {cycle_id: list(trace_ids) for cycle_id, trace_ids in self.trace_ids_by_cycle.items()}
        for cycle_id, trace_ids in trace_ids_by_cycle.items():
            consumed = set(trace_ids)
            self.trace_ids_by_cycle[cycle_id] = [trace_id for trace_id in self.trace_ids_by_cycle[cycle_id] if trace_id not in consumed]
            if not self.trace_ids_by_cycle[cycle_id]:
                del self.trace_ids_by_cycle[cycle_id]
        self.stage_optimizer_updates_observed += 1
        self.optimizer_updates_observed += 1
        receipt: dict[str, Any] = {
            "schema": "staged-family-balanced-optimizer-receipt-v1",
            "run_id": RUN_ID,
            "stage": self.active_stage,
            "stage_name": stage["name"],
            "status": "optimizer_update_recorded",
            "optimizer_update": self.optimizer_updates_observed,
            "stage_optimizer_update": self.stage_optimizer_updates_observed,
            "stage_optimizer_updates_target": stage["optimizer_updates_target"],
            "scored_traces": len(all_trace_ids),
            "trace_ids_by_cycle": {cycle_id: list(trace_ids) for cycle_id, trace_ids in trace_ids_by_cycle.items()},
            "consumed_trace_ids": list(all_trace_ids),
            "trace_inventory_before": inventory_before,
            "trace_ids_before": trace_ids_before,
            "trace_inventory_after": dict(self.trace_inventory_by_cycle),
            "trace_ids_after": {cycle_id: list(trace_ids) for cycle_id, trace_ids in self.trace_ids_by_cycle.items()},
            "sampling_cycles": list(trace_ids_by_cycle),
            "progression_kind": "optimizer_count_independent_of_sampling_cycle",
        }
        if self.stage_optimizer_updates_observed == stage["optimizer_updates_target"]:
            remaining = self._available_trace_total()
            if remaining:
                receipt["decision"] = "drain_stage_tail"
                receipt["remaining_stage_tail_traces"] = remaining
            elif self.active_stage == 4:
                receipt["decision"] = "stop"
                receipt["reason"] = "curriculum_optimizer_schedule_complete"
                self.halted = True
            else:
                receipt["decision"] = "advance"
                receipt["next_stage"] = self.active_stage + 1
                self._advance_stage()
        else:
            receipt["decision"] = "continue_stage"
        return receipt

    def record_stage_tail_drop(
        self,
        *,
        trace_ids_by_cycle: Mapping[str, Sequence[str]],
        reason: str,
        purge_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume the exact active-stage tail before schedule progression."""

        if self.halted:
            raise CurriculumError("controller is halted; a new run is required")
        stage = self._stage()
        if self.stage_optimizer_updates_observed != stage["optimizer_updates_target"]:
            raise CurriculumError("stage tail may be dropped only after the optimizer target")
        if not isinstance(reason, str) or not reason.strip():
            raise CurriculumError("stage tail drop needs an explicit reason")
        if not isinstance(purge_receipt, Mapping):
            raise CurriculumError("stage tail drop needs a physical Prime purge receipt")
        if purge_receipt.get("status") != "purged_pending_controller_commit":
            raise CurriculumError("stage tail purge receipt is missing or not pending controller commit")
        if purge_receipt.get("prime_revision") != PRIME_REVISION:
            raise CurriculumError("stage tail purge receipt is for an unsupported Prime revision")
        if purge_receipt.get("physical_removal_verified") is not True:
            raise CurriculumError("stage tail purge receipt does not verify physical removal")
        if purge_receipt.get("controller_stage") != self.active_stage:
            raise CurriculumError("stage tail purge receipt is stale or from another stage")
        if purge_receipt.get("controller_inventory_before") != self.trace_inventory_by_cycle:
            raise CurriculumError("stage tail purge receipt does not match current controller inventory")
        if purge_receipt.get("controller_trace_ids_before") != self.trace_ids_by_cycle:
            raise CurriculumError("stage tail purge receipt IDs do not match current controller inventory")
        if not isinstance(trace_ids_by_cycle, Mapping) or not trace_ids_by_cycle:
            raise CurriculumError("stage tail drop needs a non-empty trace_ids_by_cycle mapping")
        if any(not isinstance(cycle_id, str) for cycle_id in trace_ids_by_cycle) or any(
            not isinstance(trace_ids, Sequence) or isinstance(trace_ids, (str, bytes)) or not trace_ids
            for trace_ids in trace_ids_by_cycle.values()
        ):
            raise CurriculumError("stage tail trace IDs must be non-empty sequences")
        if set(trace_ids_by_cycle) != set(self.trace_ids_by_cycle):
            raise CurriculumError("stage tail drop must cover exactly the remaining cycle inventory")
        if any(set(trace_ids_by_cycle[cycle_id]) != set(trace_ids) for cycle_id, trace_ids in self.trace_ids_by_cycle.items()):
            raise CurriculumError("stage tail drop IDs must equal the remaining cycle inventory")
        all_trace_ids = [trace_id for trace_ids in trace_ids_by_cycle.values() for trace_id in trace_ids]
        if len(set(all_trace_ids)) != len(all_trace_ids):
            raise CurriculumError("stage tail drop trace IDs must be unique")
        dropped = len(all_trace_ids)
        if purge_receipt.get("trace_ids_by_cycle") != {cycle_id: list(trace_ids) for cycle_id, trace_ids in trace_ids_by_cycle.items()}:
            raise CurriculumError("stage tail purge receipt IDs do not match the controller commit")
        if purge_receipt.get("removed_trace_count") != dropped:
            raise CurriculumError("stage tail purge receipt removed the wrong number of traces")
        inventory_before = dict(self.trace_inventory_by_cycle)
        self.trace_ids_by_cycle.clear()
        receipt: dict[str, Any] = {
            "schema": "staged-family-balanced-tail-drop-receipt-v1",
            "run_id": RUN_ID,
            "stage": self.active_stage,
            "stage_name": stage["name"],
            "status": "stage_tail_dropped",
            "trace_ids_by_cycle": {cycle_id: list(trace_ids) for cycle_id, trace_ids in trace_ids_by_cycle.items()},
            "dropped_trace_ids": list(all_trace_ids),
            "trace_inventory_before": inventory_before,
            "trace_inventory_after": {},
            "dropped_traces": dropped,
            "reason": reason,
            "purge_receipt_id": purge_receipt.get("receipt_id"),
            "progression_kind": "optimizer_target_then_explicit_tail_drain",
        }
        if self.active_stage == 4:
            receipt["decision"] = "stop"
            receipt["reason"] = "curriculum_optimizer_schedule_complete"
            self.halted = True
        else:
            receipt["decision"] = "advance"
            receipt["next_stage"] = self.active_stage + 1
            self._advance_stage()
        return receipt

    def _advance_stage(self) -> None:
        self.active_stage += 1
        self.next_cycle = 1
        self.stage_optimizer_updates_observed = 0
        self.completed_cycle_ids.clear()
        self.trace_ids_by_cycle.clear()

    def _censor_cycle(self, stage: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return a retry receipt without advancing schedule or training."""

        scheduled_counts = {family: 0 for family in stage["train_families"]}
        scored_counts = {family: 0 for family in stage["train_families"]}
        excluded_counts = {family: 0 for family in stage["train_families"]}
        model_invalid_counts = {family: 0 for family in stage["train_families"]}
        excluded_evidence: list[dict[str, Any]] = []
        model_invalid_evidence: list[dict[str, Any]] = []
        for group in groups:
            family = group["family"]
            scheduled_counts[family] += len(group["rollouts"]) + len(group["excluded"])
            scored_counts[family] += group["scored_rollouts"]
            excluded_counts[family] += group["excluded_rollouts"]
            model_invalid_counts[family] += sum(1 for rollout in group["rollouts"] if rollout["model_invalid"])
            excluded_evidence.extend(
                {"family": rollout["family"], "status": rollout["status"], "evidence": rollout["evidence"]}
                for rollout in group["excluded"]
            )
            model_invalid_evidence.extend(
                {"family": rollout["family"], "status": rollout["status"], "evidence": rollout["evidence"]}
                for rollout in group["rollouts"] if rollout["model_invalid"]
            )

        receipt = {
            "schema": "staged-family-balanced-controller-receipt-v1",
            "run_id": RUN_ID,
            "stage": self.active_stage,
            "stage_name": stage["name"],
            "cycle": self.next_cycle,
            "status": "censored",
            "decision": "retry_cycle",
            "admission": "group_censored",
            "reason": "fewer_than_minimum_scored_rollouts",
            "minimum_scored_rollouts_per_group": MIN_SCORED_ROLLOUTS_PER_GROUP,
            "family_scheduled_trace_counts": scheduled_counts,
            "family_scored_trace_counts": scored_counts,
            "family_excluded_rollout_counts": excluded_counts,
            "family_model_invalid_counts": model_invalid_counts,
            "excluded_evidence": excluded_evidence,
            "model_invalid_evidence": model_invalid_evidence,
            "trace_exposure": "measured_after_admission",
            "censored_groups": [
                {
                    "group_id": group["group_id"],
                    "family": group["family"],
                    "scored_rollouts": group["scored_rollouts"],
                    "excluded_rollouts": group["excluded_rollouts"],
                }
                for group in groups
                if group["admission"] == "censored"
            ],
            "progression_kind": "schedule_only_not_quality_gate",
            "optimizer_updates_observed": self.optimizer_updates_observed,
            "stage_optimizer_updates_observed": self.stage_optimizer_updates_observed,
            "stage_optimizer_updates_target": stage["optimizer_updates_target"],
            "sampling_attempt": self.sampling_attempts_by_stage.get(str(self.active_stage), 0),
            "sampling_attempt_cap": stage["max_sampling_cycles"],
        }
        if self.sampling_attempts_by_stage.get(str(self.active_stage), 0) >= stage["max_sampling_cycles"]:
            receipt["decision"] = "stop"
            receipt["reason"] = "sampling_cycle_cap_exhausted_before_optimizer_target"
            self.halted = True
        return receipt

    def _stop(self, stage: Mapping[str, Any], reason: str, details: Mapping[str, Any]) -> dict[str, Any]:
        self.halted = True
        return {
            "schema": "staged-family-balanced-controller-receipt-v1",
            "run_id": RUN_ID,
            "stage": self.active_stage,
            "stage_name": stage["name"],
            "cycle": self.next_cycle,
            "status": "stop",
            "decision": "stop",
            "reason": reason,
            "details": dict(details),
            "admission": "fail_closed",
        }


def write_preparation(output: Path, *, prime_root: Path | None = None) -> dict[str, Any]:
    """Write compact manifests, config and an auditable preparation receipt."""

    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    curriculum = build_curriculum()
    config = build_config(curriculum)
    (output / "curriculum-manifest.json").write_bytes(_canonical(curriculum))
    (output / "config.json").write_bytes(_canonical(config))
    receipt = {
        "schema": "staged-family-balanced-preparation-receipt-v1",
        "run_id": RUN_ID,
        "status": "prepared_offline_pending_runtime_review",
        "outputs": {
            "curriculum_manifest_sha256": sha256_file(output / "curriculum-manifest.json"),
            "config_sha256": sha256_file(output / "config.json"),
        },
        "prime_compatibility": validate_prime_compatibility(prime_root),
        "tail_purge_hook": {
            "schema": "staged-family-balanced-prime-tail-purge-v1",
            "status": "node_free_validated_pending_prime_runtime_integration",
            "validated": True,
            "validation_scope": "fake TrainSink batch and token regression tests in test_staged_curriculum.py",
            "full_prime_compatibility": False,
            "required_before_stage_advance": True,
            "physically_purges_train_sink": True,
            "no_gpu_or_external_calls": True,
        },
        "runtime_binding": {
            "schema": "staged-family-balanced-runtime-binding-v1",
            "module": "runtime_binding.py",
            "sitecustomize": "sitecustomize.py",
            "status": "node_free_fake_prime_validated_pending_linux_canary",
            "validated": True,
            "validation_scope": "fake Prime async/order and pinned-source-shape tests in test_runtime_binding.py",
            "atomic_state": True,
            "stage_tail_purge_required_before_advance": True,
            "optimizer_update_confirmation": "TrainBatch selection is journaled first; only WeightWatcher.on_update for the expected policy version records the optimizer update",
            "dispatch_gate": "binding-owned complete-group-prefix gate is layered with Prime lead permission; active groups drain while fresh dispatch is paused",
            "trainer_candidate_inventory": "controller inventory contains only exact trace IDs present in native TrainSink.pending_batch after process_group",
            "full_prime_compatibility": False,
            "no_gpu_or_external_calls": True,
        },
        "no_gpu_or_external_calls": True,
        "limitations": [
            "Source rows and heldout panel are inherited by hash from the completed diverse-flow proposal.",
            "Batch size 24 is the optimizer trace capacity; each sampling cycle schedules six exact per-family groups with four rollout capacity, while uncertainty can reduce a group to three scored traces and make an optimizer batch span cycles.",
            "Stages advance only on their optimizer target receipts; each stage has a finite two-times sampling-cycle refill cap and cannot mix leftover traces into the next stage.",
            "Censored retries count toward a persisted per-stage sampling-attempt cap; repeated censorship stops before the optimizer target instead of retrying indefinitely.",
            "The node-free Prime tail-purge adapter and run-owned runtime binding are tested against fake TrainSink/TrainSource/Dispatcher surfaces, but the Linux canary and full Prime compatibility remain pending; no launch-ready claim is made.",
            "Anchors, renderer canary, model execution, rewards and visual judgments remain pending.",
        ],
    }
    (output / "preparation-receipt.json").write_bytes(_canonical(receipt))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", nargs="?", choices=("build",), default="build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path, help="optional local checkout of the pinned Prime revision")
    args = parser.parse_args(argv)
    print(json.dumps(write_preparation(args.output, prime_root=args.prime_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
