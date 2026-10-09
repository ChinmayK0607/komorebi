"""Run-owned binding between the staged controller and pinned Prime.

The module is inert unless :func:`install_from_environment` sees every
run-owned activation variable.  On activation it patches the current Prime
interpreter only; it never edits the Prime checkout, reward code, renderer,
or model configuration.  The binding keeps native task objects and group UUIDs
as the provenance bridge, then records the exact trace IDs that Prime places in
each shipped ``TrainBatch``.

This file intentionally has no Prime imports at module import time.  That
keeps the preparation and its fake-Prime tests node-free while the canary on a
pinned Linux checkout remains an explicit launch prerequisite.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prime_tail_purge import commit_purged_stage_tail, purge_pending_stage_tail
from staged_curriculum import (
    BATCH_SIZE,
    GROUP_SIZE,
    MIN_SCORED_ROLLOUTS_PER_GROUP,
    PRIME_REVISION,
    RUN_ID,
    StageController,
    CurriculumError,
    build_curriculum,
)


ENABLE_ENV = "PAINTER_STAGED_CURRICULUM_ENABLE"
RUN_ENV = "PAINTER_STAGED_CURRICULUM_RUN_ID"
PROTOCOL_ENV = "PAINTER_STAGED_CURRICULUM_PROTOCOL"
ROOT_ENV = "PAINTER_STAGED_CURRICULUM_ROOT"
STATE_ENV = "PAINTER_STAGED_CURRICULUM_STATE"
ADMISSION_ENV = "PAINTER_STAGED_CURRICULUM_ADMISSION"
EXPECTED_PROTOCOL = "staged-family-balanced-protocol-v1"
STATE_SCHEMA = "staged-family-balanced-runtime-state-v1"
INSTALL_SENTINEL = "_painter_staged_family_binding"


class RuntimeBindingError(CurriculumError):
    """Raised when native Prime provenance or state cannot be proven."""


def _fail(message: str) -> None:
    raise RuntimeBindingError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"runtime binding cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"runtime binding JSON must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _protocol_root(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _json(root / "PROTOCOL.json")
    config_path = root / "config.json"
    manifest_path = root / "curriculum-manifest.json"
    config = _json(config_path)
    manifest = _json(manifest_path)
    if protocol.get("schema") != EXPECTED_PROTOCOL or protocol.get("run_id") != RUN_ID:
        _fail("runtime protocol schema or run ID does not match the staged run")
    source_ids = protocol.get("source_ids")
    if not isinstance(source_ids, Mapping) or source_ids.get("prime_revision") != PRIME_REVISION:
        _fail("runtime protocol is not pinned to the reviewed Prime revision")
    if config.get("run_id") != RUN_ID or manifest.get("run_id") != RUN_ID:
        _fail("runtime config/manifest run IDs do not match the staged run")
    declared_config = (protocol.get("config") or {}).get("sha256")
    if declared_config != _sha256(config_path):
        _fail("runtime config hash does not match PROTOCOL.json")
    declared_manifest = (protocol.get("curriculum") or {}).get("manifest_sha256")
    if declared_manifest != _sha256(manifest_path):
        _fail("runtime curriculum manifest hash does not match PROTOCOL.json")
    curriculum = config.get("curriculum")
    if not isinstance(curriculum, Mapping) or curriculum.get("batch_size") != BATCH_SIZE:
        _fail("runtime config does not declare batch_size=24")
    if curriculum.get("group_size") != GROUP_SIZE or curriculum.get("rollouts_per_prompt") != GROUP_SIZE:
        _fail("runtime config has incompatible Prime group/rollout shape")
    return manifest, protocol


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _task_data(task: Any) -> Any:
    data = _get(task, "data")
    if data is None:
        _fail("native Prime task is missing data")
    return data


def _task_row_id(task: Any) -> str:
    data = _task_data(task)
    value = _get(data, "source_row_id") or _get(data, "task_id") or _get(data, "id")
    if not isinstance(value, str) or not value:
        _fail("native task is missing source_row_id/task_id identity")
    return value


def _task_family(task: Any) -> str:
    value = _get(_task_data(task), "family")
    if not isinstance(value, str) or not value:
        _fail("native task is missing family identity")
    return value


def _task_key(task: Any) -> str:
    value = _get(task, "key")
    if not isinstance(value, str) or not value:
        _fail("native task is missing a stable key")
    return value


def _trace_list(episode: Any) -> list[Any]:
    traces = _get(episode, "traces")
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        _fail("native episode traces are missing or malformed")
    return list(traces)


def _trace_id(trace: Any) -> str:
    value = _get(trace, "id")
    if not isinstance(value, str) or not value.strip():
        _fail("native rollout has no unique non-empty trace ID")
    return value


def _native_validity(info: Mapping[str, Any], key: str) -> bool | None:
    reward_status = info.get("reward_status")
    if isinstance(reward_status, str) and reward_status.startswith("model_invalid_"):
        invalid_field = {
            "model_invalid_code": "code_valid",
            "model_invalid_canvas": "canvas_valid",
            "model_invalid_current_canvas": "current_canvas_valid",
            "model_invalid_final_action": "terminal_action_valid",
        }.get(reward_status)
        if invalid_field == key:
            return False
    value = info.get(key)
    if type(value) is bool:
        return value
    if key == "canvas_valid" and type(info.get("final_canvas_valid")) is bool:
        return info["final_canvas_valid"]
    if key == "current_canvas_valid" and type(info.get("final_canvas_valid")) is bool and "final_canvas" in info:
        return info["final_canvas_valid"]
    turns = info.get("turns")
    if isinstance(turns, list) and turns:
        values = [turn.get("valid") for turn in turns if isinstance(turn, Mapping)]
        if len(values) == len(turns) and all(type(item) is bool for item in values):
            if key == "code_valid":
                paint_turns = [turn for turn in turns if turn.get("action") in {"paint", "repair"}]
                if paint_turns:
                    return bool(paint_turns[-1]["valid"])
                return None
            if key == "terminal_action_valid":
                return values[-1]
    final_action = info.get("final_action_validation")
    if key == "terminal_action_valid" and isinstance(final_action, Mapping) and type(final_action.get("valid")) is bool:
        return final_action["valid"]
    return None


def _intermediate_invalid_turns(info: Mapping[str, Any]) -> int:
    turns = info.get("turns")
    if not isinstance(turns, list):
        return 0
    paint_turns = [turn for turn in turns if isinstance(turn, Mapping) and turn.get("action") in {"paint", "repair"}]
    return sum(1 for turn in paint_turns[:-1] if turn.get("valid") is False)


def _trace_reward(trace: Any, info: Mapping[str, Any]) -> float | None:
    candidates = (_get(trace, "reward"), info.get("reward"), info.get("final_canvas_score"))
    for value in candidates:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _trace_status(episode: Any, trace: Any, info: Mapping[str, Any]) -> tuple[str, str, str | None]:
    service_failure = info.get("service_failure")
    if isinstance(service_failure, str) and service_failure.strip():
        return "service_failure", "service", service_failure
    if service_failure is True:
        return "service_failure", "service", "explicit native service_failure"
    reward_status = info.get("reward_status")
    evidence = info.get("score_evidence")
    evidence_status = evidence.get("status") if isinstance(evidence, Mapping) else None
    if reward_status in {"judge_uncertain", "uncertain", "censored", "missing"} or evidence_status in {
        "uncertain", "censored", "missing",
    } or ("final_canvas_score" in info and info.get("final_canvas_score") is None):
        return "judge_uncertain", "judge", str(reward_status or evidence_status or "unscored native judgment")
    if _get(episode, "ok", True) is False or _get(episode, "error") or _get(trace, "error") or _get(trace, "has_error", False):
        _fail("native failure has no explicit service_failure ownership")
    if isinstance(reward_status, str) and reward_status.startswith("model_invalid_"):
        return "model_invalid", "model", reward_status
    return "scored", "model", None


def _native_rollout(episode: Any, *, slot: Mapping[str, Any], index: int) -> dict[str, Any]:
    task = _get(episode, "task")
    if task is None or _task_row_id(task) != slot["prompt_row_id"] or _task_family(task) != slot["family"]:
        _fail(f"native episode task identity does not match scheduled slot {slot['group_id']}")
    traces = _trace_list(episode)
    if len(traces) != 1:
        _fail(f"native episode {slot['group_id']} must contain exactly one rollout trace")
    trace = traces[0]
    trace_id = _trace_id(trace)
    info = _get(trace, "info")
    if not isinstance(info, Mapping):
        _fail(f"trace {trace_id} is missing native info evidence")
    validity = {key: _native_validity(info, key) for key in (
        "code_valid", "canvas_valid", "current_canvas_valid", "terminal_action_valid",
    )}
    status, ownership, reason = _trace_status(episode, trace, info)
    evidence: dict[str, Any] = {
        "source": "native_trace_info",
        "trace_id": trace_id,
        "native_reward_status": str(info.get("reward_status") or "scored"),
        "intermediate_invalid_paint_turns": _intermediate_invalid_turns(info),
    }
    if reason:
        evidence["reason"] = reason
    reward = _trace_reward(trace, info)
    if status == "scored" and all(type(value) is bool for value in validity.values()) and not all(validity.values()):
        _fail(f"native trace {trace_id} has failed validity without reward_status model_invalid_ ownership")
    if status == "model_invalid":
        evidence["reason"] = reason
        if reward is None:
            reward = 0.0
        if reward != 0.0:
            _fail(f"model-invalid native trace {trace_id} does not carry reward zero")
    if status in {"scored", "model_invalid"} and not all(type(value) is bool for value in validity.values()):
        _fail(f"native scored trace {trace_id} lacks fail-closed validity evidence")
    if status in {"scored", "model_invalid"} and reward is None:
        _fail(f"native scored trace {trace_id} lacks a finite reward")
    if status in {"judge_uncertain", "service_failure"}:
        reward = None
    return {
        "group_id": slot["group_id"],
        "row_id": slot["prompt_row_id"],
        "family": slot["family"],
        "stage": slot["stage"],
        "rollout_index": index,
        "trace_id": trace_id,
        "ownership": ownership,
        "status": status,
        **validity,
        "reward_status": (
            str(info.get("reward_status")) if status == "model_invalid" and isinstance(info.get("reward_status"), str)
            else "scored" if status == "scored" else "unscored"
        ),
        "reward": reward,
        "evidence": evidence,
    }


def _dispatch_failure_rollout(failure: Any, *, slot: Mapping[str, Any], index: int) -> dict[str, Any]:
    error = _get(failure, "error")
    return {
        "group_id": slot["group_id"],
        "row_id": slot["prompt_row_id"],
        "family": slot["family"],
        "stage": slot["stage"],
        "rollout_index": index,
        "trace_id": f"{slot['group_id']}:dispatch_failure:{index}",
        "ownership": "service",
        "status": "service_failure",
        "code_valid": None,
        "canvas_valid": None,
        "current_canvas_valid": None,
        "terminal_action_valid": None,
        "reward_status": "unscored",
        "reward": None,
        "evidence": {
            "source": "native_dispatch_failure",
            "trace_id": f"{slot['group_id']}:dispatch_failure:{index}",
            "reason": str(error or "Prime dispatch failure"),
            "service_failure": True,
        },
    }


def _cancellation_rollout(cancellation: Any, *, slot: Mapping[str, Any], index: int) -> dict[str, Any]:
    reason = str(_get(cancellation, "reason") or "pipeline cancellation")
    trace_id = f"{slot['group_id']}:cancellation:{index}"
    return {
        "group_id": slot["group_id"],
        "row_id": slot["prompt_row_id"],
        "family": slot["family"],
        "stage": slot["stage"],
        "rollout_index": index,
        "trace_id": trace_id,
        "ownership": "service",
        "status": "service_failure",
        "code_valid": None,
        "canvas_valid": None,
        "current_canvas_valid": None,
        "terminal_action_valid": None,
        "reward_status": "unscored",
        "reward": None,
        "evidence": {
            "source": "native_group_cancellation",
            "trace_id": trace_id,
            "reason": reason,
            "service_failure": True,
            "pipeline_cancellation": True,
            "stale": reason == "stale",
        },
    }


def _pipeline_censor_rollout(rollout: Mapping[str, Any], *, cancellation: Any) -> dict[str, Any]:
    """Turn an arrived rollout in a stale group into an explicit exclusion.

    Prime treats a stale GroupCancellation as a pipeline decision that voids
    every arrived member of that dispatch group, even when ``count`` only
    accounts for the not-yet-delivered members.  Preserve the native trace ID
    and task evidence for audit, but never let the arrived episode become a
    scored result or a fabricated zero.
    """

    evidence = dict(rollout.get("evidence") or {})
    evidence.update({
        "source": "native_group_cancellation",
        "pipeline_cancellation": True,
        "stale": True,
        "reason": str(_get(cancellation, "reason") or "stale"),
    })
    return {
        **rollout,
        "ownership": "service",
        "status": "service_failure",
        "code_valid": None,
        "canvas_valid": None,
        "current_canvas_valid": None,
        "terminal_action_valid": None,
        "reward_status": "unscored",
        "reward": None,
        "evidence": evidence,
    }


def _group_receipt(
    group: Sequence[Any], *, slot: Mapping[str, Any], failures: Sequence[Any] = (), cancellation: Any = None,
) -> dict[str, Any]:
    if cancellation is None:
        cancellation_count = 0
        cancellation_reason = None
    else:
        raw_count = _get(cancellation, "count", None)
        if type(raw_count) is not int or raw_count < 0:
            _fail(f"native task group {slot['group_id']} has malformed cancellation count")
        cancellation_count = raw_count
        cancellation_reason = _get(cancellation, "reason")
        if cancellation_reason not in {"stale", "overload", "superseded"}:
            _fail(f"native task group {slot['group_id']} has unknown cancellation reason")
    total = len(group) + len(failures) + cancellation_count
    if total != GROUP_SIZE:
        _fail(f"native task group {slot['group_id']} accounted for {total} attempts; expected {GROUP_SIZE}")
    rollouts = [_native_rollout(episode, slot=slot, index=index) for index, episode in enumerate(group)]
    if cancellation_reason == "stale":
        rollouts = [_pipeline_censor_rollout(rollout, cancellation=cancellation) for rollout in rollouts]
    next_index = len(rollouts)
    rollouts.extend(_dispatch_failure_rollout(failure, slot=slot, index=next_index + index) for index, failure in enumerate(failures))
    next_index = len(rollouts)
    rollouts.extend(_cancellation_rollout(cancellation, slot=slot, index=next_index + index) for index in range(cancellation_count))
    trace_ids = [row["trace_id"] for row in rollouts]
    if len(set(trace_ids)) != len(trace_ids):
        _fail(f"native task group {slot['group_id']} has duplicate trace IDs")
    return {
        "group_id": slot["group_id"],
        "row_id": slot["prompt_row_id"],
        "family": slot["family"],
        "stage": slot["stage"],
        "rollouts": rollouts,
        "pipeline_censored": cancellation_reason == "stale",
        "cancellation_reason": cancellation_reason,
    }


def _batch_trace_ids(batch: Any) -> list[str]:
    cohort = _get(batch, "cohort")
    cohort_ids = _get(cohort, "sampled_trace_ids") if cohort is not None else None
    ids: list[str] = []
    if isinstance(cohort_ids, (set, frozenset, list, tuple)):
        ids.extend(cohort_ids)
    samples = _get(batch, "samples")
    if isinstance(samples, list):
        sample_ids = [_get(sample, "trace_id") for sample in samples]
        sample_ids = [value for value in sample_ids if isinstance(value, str) and value]
        if sample_ids and (not ids or set(sample_ids) != set(ids)):
            ids = sample_ids
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        _fail("native TrainBatch has no exact shipped trace IDs")
    if len(set(ids)) != len(ids):
        _fail("native TrainBatch contains duplicate trace IDs")
    return ids


class AtomicRuntimeState:
    """Small atomic JSON journal suitable for crash resume and audit."""

    def __init__(self, path: Path, *, protocol_sha256: str) -> None:
        self.path = path.expanduser().resolve()
        self.protocol_sha256 = protocol_sha256
        self.revision = 0
        self.state: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "run_id": RUN_ID,
            "prime_revision": PRIME_REVISION,
            "protocol_sha256": protocol_sha256,
            "controller": {},
            "cycle_records": {},
            "trace_provenance": {},
            "group_receipts": [],
            "cycle_receipts": [],
            "batch_receipts": [],
            "update_confirmations": [],
            "purge_receipts": [],
            "events": [],
        }
        if self.path.exists():
            loaded = _json(self.path)
            if loaded.get("schema") != STATE_SCHEMA or loaded.get("run_id") != RUN_ID:
                _fail("runtime state schema or run ID does not match")
            if loaded.get("prime_revision") != PRIME_REVISION or loaded.get("protocol_sha256") != protocol_sha256:
                _fail("runtime state is bound to another Prime revision or protocol")
            self.state = loaded
            # Older preparation snapshots used a separate optimizer-update
            # map.  The shipped-batch receipt is now the sole pending update
            # authority; discard that obsolete field on resume.
            self.state.pop("pending_optimizer_update", None)
            self.revision = int(loaded.get("revision", 0))

    def write(self, *, controller: Mapping[str, Any], event: Mapping[str, Any], **updates: Any) -> None:
        self.revision += 1
        self.state["revision"] = self.revision
        self.state["controller"] = dict(controller)
        append_fields = {"group_receipts", "cycle_receipts", "batch_receipts", "update_confirmations", "purge_receipts"}
        for key, value in updates.items():
            if key in append_fields:
                existing = list(self.state.get(key) or [])
                if isinstance(value, list):
                    existing.extend(value)
                else:
                    existing.append(value)
                self.state[key] = existing
            else:
                self.state[key] = value
        events = list(self.state.get("events", []))
        events.append(dict(event))
        self.state["events"] = events
        _atomic_json(self.path, self.state)


class RuntimeBinding:
    """Attach exact staged scheduling and accounting to one Prime instance."""

    def __init__(
        self,
        *,
        orchestrator: Any,
        root: Path,
        state_path: Path,
        manifest: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        self.orchestrator = orchestrator
        self.root = root.expanduser().resolve()
        self.manifest = manifest
        self.protocol = protocol
        self.controller = StageController(manifest)
        protocol_path = self.root / "PROTOCOL.json"
        self.journal = AtomicRuntimeState(state_path, protocol_sha256=_sha256(protocol_path))
        self.task_by_row: dict[str, tuple[str, Any]] = {}
        self.env_name: str | None = None
        self.pending_task_slots: dict[int, dict[str, Any]] = {}
        self.group_slots: dict[str, dict[str, Any]] = {}
        self.active_group_ids: set[str] = set()
        self.cycle_records: dict[str, dict[str, dict[str, Any]]] = {}
        self.trace_provenance: dict[str, dict[str, Any]] = {}
        self.completed_group_ids: set[str] = set()
        self.dispatch_slots_sent = 0
        self.dispatch_paused = False
        self.pending_batch_receipt: dict[str, Any] | None = None
        self.confirmed_policy_update: dict[str, Any] | None = None
        self.pending_tail_purge: dict[str, Any] | None = None
        self.receipts: list[dict[str, Any]] = []
        self.startup_policy_version: int | None = None
        self.binding_gate_open = True
        self.prime_lead_permit = True
        self.update_hook_seen = False
        self.batch_cut_allowed = False
        self.batch_release_cycle_id: str | None = None
        self._restore()
        self._index_native_tasks()
        self._attach()

    def _controller_state(self) -> dict[str, Any]:
        return {
            "active_stage": self.controller.active_stage,
            "next_cycle": self.controller.next_cycle,
            "optimizer_updates_observed": self.controller.optimizer_updates_observed,
            "stage_optimizer_updates_observed": self.controller.stage_optimizer_updates_observed,
            "completed_cycle_ids": list(self.controller.completed_cycle_ids),
            "sampling_attempts_by_stage": dict(self.controller.sampling_attempts_by_stage),
            "trace_ids_by_cycle": {key: list(value) for key, value in self.controller.trace_ids_by_cycle.items()},
            "halted": self.controller.halted,
        }

    def _restore(self) -> None:
        saved = self.journal.state.get("controller") or {}
        if not saved:
            return
        expected = {
            "active_stage", "next_cycle", "optimizer_updates_observed", "stage_optimizer_updates_observed",
            "completed_cycle_ids", "trace_ids_by_cycle", "halted",
            "sampling_attempts_by_stage",
        }
        if set(saved) != expected:
            _fail("runtime controller state fields are incomplete or unknown")
        for key in ("active_stage", "next_cycle", "optimizer_updates_observed", "stage_optimizer_updates_observed"):
            minimum = 1 if key in {"active_stage", "next_cycle"} else 0
            if type(saved[key]) is not int or saved[key] < minimum:
                _fail(f"runtime controller state field {key} is malformed")
        if not isinstance(saved["completed_cycle_ids"], list) or not isinstance(saved["trace_ids_by_cycle"], Mapping) or not isinstance(saved["sampling_attempts_by_stage"], Mapping):
            _fail("runtime controller cycle state is malformed")
        self.controller.active_stage = saved["active_stage"]
        self.controller.next_cycle = saved["next_cycle"]
        self.controller.optimizer_updates_observed = saved["optimizer_updates_observed"]
        self.controller.stage_optimizer_updates_observed = saved["stage_optimizer_updates_observed"]
        self.controller.completed_cycle_ids = list(saved["completed_cycle_ids"])
        self.controller.trace_ids_by_cycle = {str(key): list(value) for key, value in saved["trace_ids_by_cycle"].items()}
        if any(not isinstance(key, str) or type(value) is not int or value < 0 for key, value in saved["sampling_attempts_by_stage"].items()):
            _fail("runtime sampling attempt counters are malformed")
        self.controller.sampling_attempts_by_stage = {str(key): value for key, value in saved["sampling_attempts_by_stage"].items()}
        if type(saved["halted"]) is not bool:
            _fail("runtime controller halted state is malformed")
        self.controller.halted = saved["halted"]
        saved_cycles = self.journal.state.get("cycle_records") or {}
        if not isinstance(saved_cycles, Mapping):
            _fail("runtime cycle records are malformed")
        self.cycle_records = {str(key): dict(value) for key, value in saved_cycles.items()}
        saved_provenance = self.journal.state.get("trace_provenance") or {}
        if not isinstance(saved_provenance, Mapping):
            _fail("runtime trace provenance is malformed")
        self.trace_provenance = {str(key): dict(value) for key, value in saved_provenance.items()}
        saved_group_receipts = self.journal.state.get("group_receipts") or []
        if not isinstance(saved_group_receipts, list):
            _fail("runtime group receipt history is malformed")
        self.completed_group_ids = {
            str(value.get("native_group_id")) for value in saved_group_receipts
            if isinstance(value, Mapping) and isinstance(value.get("native_group_id"), str)
        }
        self.receipts = list(self.journal.state.get("batch_receipts") or [])
        saved_dispatch_count = self.journal.state.get("dispatch_slots_sent", 0)
        if type(saved_dispatch_count) is not int or saved_dispatch_count < 0:
            _fail("runtime dispatch cursor is malformed")
        self.dispatch_slots_sent = saved_dispatch_count
        pending_batch = self.journal.state.get("pending_batch")
        if pending_batch is not None:
            if not isinstance(pending_batch, Mapping):
                _fail("pending train batch receipt is malformed")
            self.pending_batch_receipt = dict(pending_batch)
        pending_tail = self.journal.state.get("pending_stage_tail_commit")
        if pending_tail is not None:
            if not isinstance(pending_tail, Mapping):
                _fail("pending stage-tail purge receipt is malformed")
            self.pending_tail_purge = dict(pending_tail)
        confirmed = self.journal.state.get("confirmed_policy_update")
        if confirmed is not None:
            if not isinstance(confirmed, Mapping):
                _fail("confirmed policy update receipt is malformed")
            self.confirmed_policy_update = dict(confirmed)
        self.batch_cut_allowed = bool(self.journal.state.get("batch_cut_allowed", False))
        release_cycle = self.journal.state.get("batch_release_cycle_id")
        if release_cycle is not None and not isinstance(release_cycle, str):
            _fail("runtime batch release cycle is malformed")
        self.batch_release_cycle_id = release_cycle
        self.dispatch_paused = bool(self.controller.halted or self.pending_batch_receipt or self.confirmed_policy_update or self.pending_tail_purge)
        reconciliation_state = (
            self.pending_batch_receipt is not None
            or self.confirmed_policy_update is not None
            or self.pending_tail_purge is not None
        )
        if self.cycle_records:
            _fail("runtime state contains an unfinished cycle; native Prime buffers are not durable")
        if self.dispatch_slots_sent > 0:
            _fail("runtime state contains a partial dispatch cursor; native Prime buffers are not durable")
        if self.controller.trace_ids_by_cycle and not reconciliation_state:
            _fail(
                "runtime state contains admitted trace inventory without an exact selected-batch, "
                "confirmed-update, or tail-purge reconciliation state"
            )

    def _index_native_tasks(self) -> None:
        source = getattr(self.orchestrator, "train_source", None)
        if source is None:
            _fail("Prime orchestrator has no TrainSource after setup")
        envs = getattr(source, "envs", None)
        if not isinstance(envs, Sequence) or len(envs) != 1:
            _fail("staged binding requires exactly one native train environment")
        env = envs[0]
        self.env_name = str(getattr(env, "name", ""))
        if not self.env_name:
            _fail("native train environment has no name")
        env_config = getattr(env, "config", None)
        if getattr(env_config, "group_size", None) != GROUP_SIZE:
            _fail("native train environment group_size is not four")
        curricula = getattr(source, "curricula", None)
        curriculum = curricula.get(self.env_name) if isinstance(curricula, Mapping) else None
        sampler = getattr(curriculum, "sampler", None)
        tasks = getattr(sampler, "tasks", None)
        if not isinstance(tasks, Sequence) or not tasks:
            _fail("staged binding requires a finite native task sampler with task objects")
        for task in tasks:
            row_id = _task_row_id(task)
            if row_id in self.task_by_row:
                _fail(f"native task rows are duplicated: {row_id}")
            _task_key(task)
            self.task_by_row[row_id] = (self.env_name, task)
        schedule_rows = {
            slot["prompt_row_id"]
            for stage in self.manifest["stages"]
            for slot in stage["schedule"]
        }
        missing = sorted(schedule_rows - set(self.task_by_row))
        if missing:
            _fail(f"native task sampler is missing scheduled rows: {missing[:3]}")

    def _attach(self) -> None:
        source = self.orchestrator.train_source
        dispatcher = self.orchestrator.dispatcher
        sink = self.orchestrator.train_sink
        for obj, name in ((source, "next_task"), (dispatcher, "schedule_group_episode"), (sink, "process_group"), (sink, "process_batch"), (sink, "_maybe_batch"), (dispatcher, "fill_inflight"), (self.orchestrator, "update_dispatch_gate"), (self.orchestrator, "on_policy_update")):
            if not callable(getattr(obj, name, None)):
                _fail(f"Prime runtime hook {name} is unavailable")
        watcher = getattr(self.orchestrator, "watcher", None)
        if watcher is None or not callable(getattr(watcher, "on_update", None)):
            _fail("Prime WeightWatcher.on_update is unavailable")
        dispatch_allowed = getattr(dispatcher, "dispatch_allowed", None)
        if dispatch_allowed is None or not callable(getattr(dispatch_allowed, "is_set", None)):
            _fail("Prime dispatcher.dispatch_allowed is unavailable")
        self._original_source_next_task = source.next_task
        self._original_schedule_group_episode = dispatcher.schedule_group_episode
        self._original_process_group = sink.process_group
        self._original_process_batch = sink.process_batch
        self._original_maybe_batch = sink._maybe_batch
        self._original_fill_inflight = dispatcher.fill_inflight
        self._original_update_dispatch_gate = self.orchestrator.update_dispatch_gate
        self._original_on_policy_update = self.orchestrator.on_policy_update
        policy = getattr(self.orchestrator, "policy", None)
        if policy is None or type(getattr(policy, "version", None)) is not int:
            _fail("Prime policy version is unavailable for update binding")
        self.startup_policy_version = int(policy.version)
        self.prime_lead_permit = bool(dispatch_allowed.is_set())
        source.next_task = types.MethodType(self._source_next_task, source)
        dispatcher.schedule_group_episode = types.MethodType(self._schedule_group_episode, dispatcher)
        dispatcher.fill_inflight = types.MethodType(self._fill_inflight, dispatcher)
        sink.process_group = types.MethodType(self._process_group, sink)
        sink.process_batch = types.MethodType(self._process_batch, sink)
        sink._maybe_batch = types.MethodType(self._maybe_batch, sink)
        self.orchestrator.update_dispatch_gate = types.MethodType(self._update_dispatch_gate, self.orchestrator)
        self.orchestrator.on_policy_update = types.MethodType(self._on_policy_update, self.orchestrator)
        watcher.on_update(self._on_watcher_update)
        self._apply_binding_gate()
        setattr(self.orchestrator, INSTALL_SENTINEL, self)
        self._persist("binding_attached", {"stage": self.controller.active_stage, "next_cycle": self.controller.next_cycle})
        self._reconcile_pending_startup()

    def _apply_binding_gate(self) -> None:
        gate = self.orchestrator.dispatcher.dispatch_allowed
        if self.binding_gate_open and self.prime_lead_permit and not self.dispatch_paused and not self.controller.halted:
            gate.set()
        else:
            gate.clear()

    def _close_sampling_gate(self, reason: str) -> None:
        gate = self.orchestrator.dispatcher.dispatch_allowed
        self.prime_lead_permit = bool(gate.is_set())
        self.binding_gate_open = False
        gate.clear()
        self._persist("sampling_gate_closed", {"reason": reason, "cycle": self.controller.next_cycle})

    def _open_sampling_gate(self, reason: str) -> None:
        self.binding_gate_open = True
        self._apply_binding_gate()
        self._persist("sampling_gate_reopened", {"reason": reason, "prime_lead_permit": self.prime_lead_permit})

    def _update_dispatch_gate(self, orchestrator: Any) -> Any:
        result = self._original_update_dispatch_gate()
        self.prime_lead_permit = bool(orchestrator.dispatcher.dispatch_allowed.is_set())
        self._apply_binding_gate()
        return result

    async def _on_policy_update(self, orchestrator: Any, step: int) -> Any:
        result = await self._original_on_policy_update(step)
        self.prime_lead_permit = bool(orchestrator.dispatcher.dispatch_allowed.is_set())
        self._apply_binding_gate()
        return result

    async def _fill_inflight(self, dispatcher: Any) -> None:
        """Prime fill loop with a binding gate that still drains active groups."""
        # The pinned loop is intentionally mirrored here: the only changed
        # branch permits existing train groups to finish while the binding gate
        # is closed after slot six.  Prime's lead permit remains authoritative.
        while True:
            if getattr(dispatcher, "policy_update_pending", False):
                return
            if dispatcher.available_permits <= 0 or dispatcher.admission_budget() <= 0:
                return
            async with dispatcher.scheduling_lock:
                if getattr(dispatcher, "policy_update_pending", False):
                    return
                mode = getattr(dispatcher, "mode", None)
                if getattr(mode, "name", str(mode)) == "PREFER_EVAL":
                    if not getattr(dispatcher, "eval_has_work", False):
                        dispatcher.switch_mode(type(mode).PREFER_TRAIN, reason="the eval queue drained")
                        continue
                    scheduled = await dispatcher.try_schedule("eval")
                else:
                    active_train = any(
                        getattr(group, "kind", None) == "train" and getattr(group, "episodes_to_schedule", 0) > 0
                        for group in getattr(dispatcher, "groups", {}).values()
                    )
                    prime_permit = bool(dispatcher.dispatch_allowed.is_set()) or (
                        not self.binding_gate_open and self.prime_lead_permit
                    )
                    if not prime_permit or (not self.binding_gate_open and not active_train):
                        return
                    scheduled = await dispatcher.try_schedule("train")
                if not scheduled:
                    return

    def _stage_slots(self) -> list[dict[str, Any]]:
        stage = self.manifest["stages"][self.controller.active_stage - 1]
        return [slot for slot in stage["schedule"] if slot["cycle"] == self.controller.next_cycle]

    def _source_next_task(self, source: Any, *, step: int) -> Any:
        if self.dispatch_paused or self.controller.halted:
            return None
        slots = self._stage_slots()
        if self.dispatch_slots_sent >= len(slots):
            return None
        slot = slots[self.dispatch_slots_sent]
        if self.batch_release_cycle_id != slot["cycle_id"]:
            # A release is valid only for the cycle whose pass receipt opened
            # it.  Starting the next cycle closes the batch boundary again so
            # carry-over cannot ship from an early group.
            self.batch_cut_allowed = False
        env_name, task = self.task_by_row.get(slot["prompt_row_id"], (None, None))
        if env_name != self.env_name or task is None:
            _fail(f"scheduled row {slot['prompt_row_id']} is not bound to the native task")
        self.dispatch_slots_sent += 1
        self.pending_task_slots[id(task)] = dict(slot)
        self._persist("task_dispatched", {
            "cycle_id": slot["cycle_id"], "group_id": slot["group_id"], "prompt_row_id": slot["prompt_row_id"],
            "family": slot["family"], "stage": slot["stage"], "step": step,
        })
        try:
            from prime_rl.orchestrator.types import TaskRequest
        except ModuleNotFoundError as exc:
            _fail(f"pinned Prime TaskRequest is unavailable: {exc}")
        return TaskRequest(env_name=self.env_name, task=task, step=step, source_index=None)

    async def _schedule_group_episode(self, dispatcher: Any, group_id: Any, group: Any) -> Any:
        native_group_id = str(group_id)
        slot = self.pending_task_slots.pop(id(_get(group, "task")), None)
        if slot is not None:
            if native_group_id in self.group_slots:
                _fail(f"native group {native_group_id} was scheduled twice")
            self.group_slots[native_group_id] = slot
            self.active_group_ids.add(native_group_id)
            self._persist("group_bound", {"native_group_id": native_group_id, "group_id": slot["group_id"], "stage": slot["stage"]})
        result = await self._original_schedule_group_episode(group_id, group)
        if native_group_id in self.group_slots and self.dispatch_slots_sent == len(self._stage_slots()):
            remaining = getattr(group, "episodes_to_schedule", None)
            if type(remaining) is not int:
                _fail("native Prime group is missing episodes_to_schedule at cycle boundary")
            if remaining == 0:
                self._close_sampling_gate("sixth_group_fully_issued")
        return result

    async def _process_group(self, sink: Any, group_id: Any) -> Any:
        native_group_id = str(group_id)
        slot = self.group_slots.get(native_group_id)
        if slot is None:
            _fail(f"native group {native_group_id} has no staged provenance")
        if native_group_id in self.completed_group_ids:
            # A replayed completion is already represented in the atomic
            # journal.  Prime's callback is idempotent at this boundary.
            return None
        groups = getattr(sink, "pending_groups", None)
        if not isinstance(groups, Mapping):
            _fail("native TrainSink pending_groups is unavailable")
        group = groups.get(group_id, groups.get(native_group_id, []))
        failures_by_group = getattr(sink, "pending_group_failures", None)
        cancellations_by_group = getattr(sink, "pending_group_cancellations", None)
        if not isinstance(failures_by_group, Mapping) or not isinstance(cancellations_by_group, Mapping):
            _fail("native TrainSink failure/cancellation maps are unavailable")
        failures = failures_by_group.get(group_id, failures_by_group.get(native_group_id, []))
        cancellation = cancellations_by_group.get(group_id, cancellations_by_group.get(native_group_id))
        if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
            _fail("native TrainSink pending_group_failures shape is malformed")
        record = _group_receipt(list(group), slot=slot, failures=list(failures), cancellation=cancellation)
        result = await self._original_process_group(group_id)
        pending_batch = getattr(sink, "pending_batch", None)
        if not isinstance(pending_batch, Mapping):
            _fail("native TrainSink pending_batch is unavailable after group admission")
        pending_ids = [row["trace_id"] for row in record["rollouts"] if row["trace_id"] in pending_batch]
        candidate_ids = [
            row["trace_id"] for row in record["rollouts"]
            if row["trace_id"] in pending_batch and row["status"] in {"scored", "model_invalid"}
        ]
        if record.get("pipeline_censored") and pending_ids:
            _fail(f"stale-cancelled group {native_group_id} left native pending trace IDs")
        if len(set(candidate_ids)) != len(candidate_ids):
            _fail(f"native group {native_group_id} has duplicate trainer candidate IDs")
        record["pending_trace_ids"] = pending_ids
        record["trainer_candidate_trace_ids"] = candidate_ids
        record["trainer_candidate_count"] = len(candidate_ids)
        if native_group_id in self.completed_group_ids:
            self.active_group_ids.discard(native_group_id)
            return result
        self.completed_group_ids.add(native_group_id)
        self.active_group_ids.discard(native_group_id)
        self.cycle_records.setdefault(slot["cycle_id"], {})[slot["group_id"]] = record
        for trace_id in pending_ids:
            self.trace_provenance[trace_id] = {
                "trace_id": trace_id,
                "cycle_id": slot["cycle_id"],
                "group_id": slot["group_id"],
                "stage": slot["stage"],
            }
        self._persist(
            "group_completed",
            {"native_group_id": native_group_id, "cycle_id": slot["cycle_id"], "group_id": slot["group_id"], "trainer_candidate_count": len(candidate_ids)},
            cycle_records=self.cycle_records,
            trace_provenance=self.trace_provenance,
            group_receipts=[{"native_group_id": native_group_id, "cycle_id": slot["cycle_id"], "group_id": slot["group_id"], "receipt": record}],
        )
        self._maybe_observe_cycle(slot["cycle_id"])
        await self._maybe_finish_stage()
        return result

    def _purge_censored_cycle(self, sink: Any, cycle_id: str, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Remove every native pending trace attributable to a rejected cycle."""
        pending_batch = getattr(sink, "pending_batch", None)
        episode_by_trace = getattr(sink, "episode_by_trace", None)
        if not isinstance(pending_batch, Mapping) or not isinstance(episode_by_trace, Mapping):
            _fail("censored cycle purge requires native TrainSink pending maps")
        if set(pending_batch) != set(episode_by_trace):
            _fail("TrainSink pending maps diverged before censored cycle purge")
        ids_by_group = {
            group_id: list(record.get("pending_trace_ids") or [])
            for group_id, record in records.items()
        }
        all_ids = [trace_id for values in ids_by_group.values() for trace_id in values]
        if len(set(all_ids)) != len(all_ids):
            _fail(f"censored cycle {cycle_id} has duplicate pending trace IDs")
        missing = [trace_id for trace_id in all_ids if trace_id not in pending_batch]
        if missing:
            self._halt_dispatch("censored_cycle_pending_trace_missing")
            _fail(f"censored cycle {cycle_id} trace was already shipped or missing: {missing[:3]}")
        token_mode = type(getattr(sink, "token_batch_size", None)) is int and getattr(sink, "token_batch_size") > 0 and getattr(sink, "batch_size", None) is None
        batch_mode = type(getattr(sink, "batch_size", None)) is int and getattr(sink, "batch_size") > 0 and getattr(sink, "token_batch_size", None) is None
        if not (batch_mode or token_mode):
            _fail("unsupported TrainSink mode during censored cycle purge")
        pending_tokens = getattr(sink, "pending_tokens", None)
        if type(pending_tokens) is not int or pending_tokens < 0:
            _fail("unsupported pending_tokens shape during censored cycle purge")
        removed_tokens = 0
        if token_mode:
            from prime_tail_purge import _payload_tokens
            for trace_id in all_ids:
                episode = episode_by_trace[trace_id]
                trace = _get(episode, "traces")
                if not isinstance(trace, Sequence) or len(trace) != 1:
                    _fail("censored cycle trace episode shape is unsupported")
                removed_tokens += _payload_tokens(pending_batch[trace_id], trace[0])
            if removed_tokens > pending_tokens:
                _fail("censored cycle purge would make pending token accounting negative")
        before = list(pending_batch)
        for trace_id in all_ids:
            del pending_batch[trace_id]
            del episode_by_trace[trace_id]
        if token_mode:
            sink.pending_tokens = pending_tokens - removed_tokens
        if set(pending_batch) != set(episode_by_trace):
            _fail("TrainSink pending maps diverged after censored cycle purge")
        receipt = {
            "schema": "staged-family-balanced-censored-cycle-purge-v1",
            "status": "censored_cycle_pending_purged",
            "cycle_id": cycle_id,
            "reason": "cycle_censored_before retry; purge native pending samples before reopening",
            "trace_ids_by_group": ids_by_group,
            "removed_trace_count": len(all_ids),
            "removed_token_count": removed_tokens,
            "pending_trace_ids_before": before,
            "pending_trace_count_after": len(pending_batch),
            "pending_tokens_after": getattr(sink, "pending_tokens"),
            "physical_removal_verified": True,
        }
        self._persist("censored_cycle_pending_purged", receipt, purge_receipts=[receipt])
        return receipt

    def _maybe_observe_cycle(self, cycle_id: str) -> None:
        slots = self._stage_slots()
        if not slots or cycle_id != slots[0]["cycle_id"]:
            return
        records = self.cycle_records.get(cycle_id, {})
        if set(records) != {slot["group_id"] for slot in slots}:
            return
        receipt = self.controller.observe_cycle([records[slot["group_id"]] for slot in slots])
        self._persist("sampling_cycle_receipt", {"cycle_id": cycle_id, "status": receipt.get("status"), "decision": receipt.get("decision")}, cycle_receipts=[receipt])
        if receipt.get("status") == "stop":
            self.batch_cut_allowed = False
            self._halt_dispatch("sampling_cycle_stop")
            return
        self.cycle_records.pop(cycle_id, None)
        self.dispatch_slots_sent = 0
        self.batch_release_cycle_id = cycle_id
        self.batch_cut_allowed = receipt.get("status") == "cycle_complete" and receipt.get("decision") == "pass"
        self._persist("sampling_cursor_reset", {"cycle_id": cycle_id, "next_cycle": self.controller.next_cycle})
        if receipt.get("status") == "censored":
            self.batch_cut_allowed = False
            self.batch_release_cycle_id = None
            sink = getattr(self.orchestrator, "train_sink", None)
            if sink is None:
                _fail("censored cycle requires the native TrainSink for pending purge")
            self._purge_censored_cycle(sink, cycle_id, records)
            if receipt.get("decision") == "stop":
                self._halt_dispatch("sampling_cycle_cap_exhausted")
                return
            self._persist("sampling_cycle_retry", {"cycle_id": cycle_id, "reason": receipt.get("reason")})
        if self.pending_batch_receipt is None and self.confirmed_policy_update is None and not self.dispatch_paused:
            self._open_sampling_gate("sampling_cycle_admitted")
        elif self.controller.next_cycle > self.manifest["stages"][self.controller.active_stage - 1]["max_sampling_cycles"]:
            stop = self.controller.observe_cycle([])
            self._persist("sampling_cycle_cap_stop", {"cycle_id": cycle_id, "reason": stop.get("reason")}, cycle_receipts=[stop])
            self._halt_dispatch("sampling_cycle_cap_exhausted")

    def _predict_batch_ids(self, sink: Any) -> list[str]:
        pending = getattr(sink, "pending_batch", None)
        if not isinstance(pending, Mapping):
            _fail("native TrainSink pending_batch is unavailable")
        ids = list(pending)
        batch_size = getattr(sink, "batch_size", None)
        token_batch_size = getattr(sink, "token_batch_size", None)
        if type(batch_size) is int and batch_size > 0 and token_batch_size is None:
            return ids[:batch_size]
        if type(token_batch_size) is int and token_batch_size > 0 and batch_size is None:
            running = 0
            selected: list[str] = []
            payload = getattr(sink, "_trace", None)
            if not callable(payload):
                _fail("token-mode TrainSink lacks _trace for deterministic selection")
            from prime_tail_purge import _payload_tokens
            for trace_id in ids:
                trace = payload(trace_id)
                running += _payload_tokens(pending[trace_id], trace)
                selected.append(trace_id)
                if running >= token_batch_size:
                    break
            return selected
        _fail("native TrainSink has neither supported batch nor token mode")

    def _current_cycle_has_unobserved_groups(self) -> bool:
        slots = self._stage_slots()
        if not slots:
            return True
        cycle_id = slots[0]["cycle_id"]
        return bool(self.cycle_records.get(cycle_id))

    def _maybe_batch(self, sink: Any) -> Any:
        """Allow one native batch cut only after a complete cycle pass.

        This guard is unconditional while a cycle is being assembled, so a
        Prime ``process_group`` implementation that calls ``_maybe_batch``
        internally cannot ship a carry-over batch before this binding has
        recorded the sixth group and admitted the cycle.
        """
        if not self.batch_cut_allowed:
            return None
        return self._original_maybe_batch()

    def _trace_ids_by_cycle(self, trace_ids: Sequence[str]) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        inventory = self.controller.trace_ids_by_cycle
        for trace_id in trace_ids:
            matches = [cycle_id for cycle_id, values in inventory.items() if trace_id in values]
            if len(matches) != 1:
                _fail(f"shipped trace {trace_id!r} is absent or ambiguous in controller inventory")
            mapping.setdefault(matches[0], []).append(trace_id)
        return mapping

    def _runtime_trace_ids_by_cycle(self, trace_ids: Sequence[str]) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for trace_id in trace_ids:
            provenance = self.trace_provenance.get(trace_id)
            if not isinstance(provenance, Mapping):
                _fail(f"shipped trace {trace_id!r} is absent from runtime group provenance")
            cycle_id = provenance.get("cycle_id")
            stage = provenance.get("stage")
            if not isinstance(cycle_id, str) or stage != self.controller.active_stage:
                _fail(f"shipped trace {trace_id!r} has stale or malformed runtime provenance")
            mapping.setdefault(cycle_id, []).append(trace_id)
        if sum(len(values) for values in mapping.values()) != len(trace_ids):
            _fail("runtime trace provenance lost a shipped trace")
        return mapping

    def _confirmed_batch_cycles_ready(self, confirmed: Mapping[str, Any]) -> bool:
        trace_ids_by_cycle = confirmed.get("trace_ids_by_cycle")
        if not isinstance(trace_ids_by_cycle, Mapping) or not trace_ids_by_cycle:
            _fail("confirmed TrainBatch has malformed cycle provenance")
        completed = set(self.controller.completed_cycle_ids)
        for cycle_id, trace_ids in trace_ids_by_cycle.items():
            if cycle_id not in completed or not isinstance(trace_ids, Sequence):
                return False
            inventory = self.controller.trace_ids_by_cycle.get(cycle_id, [])
            if any(trace_id not in inventory for trace_id in trace_ids):
                return False
        return True

    def _process_batch(self, sink: Any) -> Any:
        if self.pending_batch_receipt is not None or self.confirmed_policy_update is not None:
            _fail("a second Prime TrainBatch was selected before its policy update was confirmed")
        if not self.batch_cut_allowed:
            _fail("Prime attempted to ship a batch before a complete six-group cycle pass")
        if self._current_cycle_has_unobserved_groups():
            _fail("Prime attempted to ship a batch before the active six-group cycle was observed")
        predicted = self._predict_batch_ids(sink)
        result = self._original_process_batch()
        actual = _batch_trace_ids(result)
        if set(actual) != set(predicted):
            _fail("native TrainBatch IDs differ from TrainSink selection; refusing false attribution")
        trace_ids_by_cycle = self._runtime_trace_ids_by_cycle(actual)
        if len(actual) != BATCH_SIZE:
            _fail(f"native optimizer batch has {len(actual)} traces; expected {BATCH_SIZE}")
        receipt = {
            "schema": "staged-family-balanced-runtime-batch-v1",
            "status": "train_batch_observed",
            "stage": self.controller.active_stage,
            "trace_ids_by_cycle": {key: list(value) for key, value in trace_ids_by_cycle.items()},
            "trace_ids": list(actual),
            "trace_count": len(actual),
            "optimizer_update_expected": self.controller.optimizer_updates_observed + 1,
            "expected_policy_version": self._expected_policy_version(),
            "policy_version_at_selection": int(self.orchestrator.policy.version),
            "controller_trace_ids_by_cycle_at_selection": {
                cycle_id: [trace_id for trace_id in ids if trace_id in self.controller.trace_ids_by_cycle.get(cycle_id, [])]
                for cycle_id, ids in trace_ids_by_cycle.items()
            },
        }
        self.receipts.append(receipt)
        self.pending_batch_receipt = receipt
        self.batch_cut_allowed = False
        self.dispatch_paused = True
        self._close_sampling_gate("train_batch_waiting_for_policy_update")
        self._persist("train_batch_selected", receipt, batch_receipts=[receipt], pending_batch=receipt)
        return result

    def _expected_policy_version(self) -> int:
        progress = getattr(self.orchestrator, "progress", None)
        step = getattr(progress, "step", None)
        if type(step) is not int or step < 1:
            _fail("Prime progress.step is unavailable for batch/update binding")
        return step

    def _reconcile_pending_startup(self) -> None:
        pending = self.pending_batch_receipt
        if pending is None:
            return
        expected = pending.get("expected_policy_version")
        progress = getattr(self.orchestrator, "progress", None)
        progress_step = getattr(progress, "step", None)
        current = getattr(getattr(self.orchestrator, "policy", None), "version", None)
        if type(expected) is not int or type(progress_step) is not int or type(current) is not int:
            _fail("pending TrainBatch resume lacks native policy/progress versions")
        if progress_step != expected + 1:
            self._halt_dispatch("pending_batch_resume_progress_mismatch")
            _fail("pending TrainBatch resume has an impossible progress.step")
        if current < expected:
            self._persist("pending_batch_resume_waiting_for_policy", {
                "expected_policy_version": expected,
                "current_policy_version": current,
                "optimizer_update_recorded": False,
            })
            return
        if current > expected:
            self._halt_dispatch("pending_batch_resume_policy_gap")
            _fail(f"pending TrainBatch expected policy v{expected}, startup has v{current}")
        self._confirm_pending_update(expected, startup_reconciled=True)
        self._run_stage_transition_if_ready(getattr(self.orchestrator, "train_sink", None))

    def _confirm_pending_update(self, step: int, *, startup_reconciled: bool = False) -> None:
        pending = self.pending_batch_receipt
        if pending is None:
            self._halt_dispatch("unexpected_policy_update_without_train_batch")
            _fail("Prime policy update has no pending shipped TrainBatch")
        expected_step = pending.get("expected_policy_version")
        if step != expected_step:
            self._halt_dispatch("policy_update_step_mismatch")
            _fail(f"Prime applied policy version {step}, expected {expected_step}")
        current_policy_version = getattr(getattr(self.orchestrator, "policy", None), "version", None)
        if current_policy_version != step:
            self._halt_dispatch("policy_update_not_applied")
            _fail(f"Prime watcher reported v{step} before native policy reached that version")
        expected_update = pending.get("optimizer_update_expected")
        if expected_update != self.controller.optimizer_updates_observed + 1:
            self._halt_dispatch("policy_update_count_mismatch")
            _fail("Prime policy update skipped or duplicates a controller optimizer update")
        self.pending_batch_receipt = None
        self.dispatch_paused = True
        confirmation = {
            "schema": "staged-family-balanced-update-confirmation-v1",
            "policy_version": step,
            "expected_policy_version": expected_step,
            "optimizer_update": expected_update,
            "controller_recorded": False,
            "startup_reconciled": startup_reconciled,
            "trace_ids_by_cycle": pending["trace_ids_by_cycle"],
        }
        self._persist("policy_update_confirmed", {
            "policy_version": step,
            "expected_policy_version": expected_step,
            "optimizer_update": expected_update,
            "controller_recorded": False,
            "startup_reconciled": startup_reconciled,
        }, pending_batch=None, update_confirmations=[confirmation])
        self.confirmed_policy_update = dict(pending)
        self.confirmed_policy_update["policy_version"] = step
        self._persist("policy_update_confirmation_pending_drain", {"policy_version": step}, confirmed_policy_update=self.confirmed_policy_update)

    async def _on_watcher_update(self, step: int) -> None:
        if type(step) is not int or step < 0:
            _fail("Prime watcher supplied a malformed policy version")
        # setup() has already completed by the time the run-owned binding is
        # attached.  The current policy version is therefore the startup sync;
        # a test or alternate setup path may still replay that notification.
        if self.startup_policy_version is not None and step == self.startup_policy_version and not self.update_hook_seen:
            self.update_hook_seen = True
            self._persist("startup_policy_sync_ignored", {"policy_version": step, "optimizer_update_recorded": False})
            return
        self.update_hook_seen = True
        self._confirm_pending_update(step)
        await self._maybe_finish_stage()

    async def _maybe_finish_stage(self) -> None:
        sink = getattr(self.orchestrator, "train_sink", None)
        self._run_stage_transition_if_ready(sink)

    def _run_stage_transition_if_ready(self, sink: Any) -> None:
        if self.active_group_ids:
            return
        if self.pending_tail_purge is not None:
            tail = commit_purged_stage_tail(
                self.controller,
                self.pending_tail_purge,
                reason="resume committed previously verified Prime stage-tail purge",
            )
            self.pending_tail_purge = None
            self._persist(
                "stage_tail_purged_and_committed",
                tail,
                pending_stage_tail_commit=None,
            )
            self.dispatch_paused = bool(self.controller.halted)
            self.dispatch_slots_sent = 0
            if self.controller.halted:
                self._halt_dispatch("curriculum_complete")
            else:
                self._open_sampling_gate("resumed_stage_tail_commit")
            return
        if self.confirmed_policy_update is None:
            return
        confirmed = self.confirmed_policy_update
        if not self._confirmed_batch_cycles_ready(confirmed):
            # A carry-over batch may be selected only after the prior cycle's
            # pass, but keep this guard for resume/native ordering anomalies:
            # let the already-started cycle finish without resetting its cursor.
            self.dispatch_paused = False
            if self.dispatch_slots_sent < len(self._stage_slots()):
                self._open_sampling_gate("confirmed_batch_waiting_for_cycle_observation")
            return
        trace_ids_by_cycle = {key: list(value) for key, value in confirmed["trace_ids_by_cycle"].items()}
        update = self.controller.record_optimizer_update(trace_ids_by_cycle=trace_ids_by_cycle)
        self.confirmed_policy_update = None
        self._persist("optimizer_update_recorded_after_policy_confirmation", update, confirmed_policy_update=None, update_confirmations=[{
            "schema": "staged-family-balanced-update-confirmation-v1",
            "policy_version": confirmed["policy_version"],
            "optimizer_update": update["optimizer_update"],
            "controller_recorded": True,
            "decision": update.get("decision"),
            "trace_ids_by_cycle": trace_ids_by_cycle,
        }])
        if update.get("decision") == "drain_stage_tail":
            if sink is None:
                _fail("stage tail needs the native TrainSink")
            remainder = {key: list(value) for key, value in self.controller.trace_ids_by_cycle.items()}
            purge_history = self.journal.state.get("purge_receipts") or []
            purge = next(
                (
                    value for value in purge_history
                    if isinstance(value, Mapping)
                    and value.get("controller_stage") == self.controller.active_stage
                    and value.get("trace_ids_by_cycle") == remainder
                ),
                None,
            )
            if purge is None:
                purge = purge_pending_stage_tail(
                    sink,
                    self.controller,
                    remainder,
                    prime_revision=PRIME_REVISION,
                    receipt_id=f"{RUN_ID}-s{self.controller.active_stage}-tail-{self.controller.stage_optimizer_updates_observed}",
                )
                self._persist(
                    "stage_tail_physically_purged",
                    {"receipt_id": purge["receipt_id"], "removed_trace_count": purge["removed_trace_count"]},
                    purge_receipts=[purge],
                    pending_stage_tail_commit=purge,
                )
                self.pending_tail_purge = purge
            tail = commit_purged_stage_tail(self.controller, purge, reason="optimizer target reached; purge Prime tail before stage transition")
            self._persist(
                "stage_tail_purged_and_committed",
                tail,
                pending_stage_tail_commit=None,
            )
        else:
            self._persist("optimizer_update_recorded", update)
        self.dispatch_paused = bool(self.controller.halted)
        self.dispatch_slots_sent = 0
        if self.controller.halted:
            self._halt_dispatch("curriculum_complete")
        else:
            self._open_sampling_gate("policy_update_confirmed_and_stage_ready")

    def _halt_dispatch(self, reason: str) -> None:
        self.dispatch_paused = True
        dispatcher = getattr(self.orchestrator, "dispatcher", None)
        if dispatcher is not None and hasattr(dispatcher, "disable_train_scheduling"):
            dispatcher.disable_train_scheduling()
        self._persist("dispatch_halted", {"reason": reason, "stage": self.controller.active_stage})

    def _persist(self, event_type: str, payload: Mapping[str, Any], **updates: Any) -> None:
        updates.setdefault("dispatch_slots_sent", self.dispatch_slots_sent)
        updates.setdefault("dispatch_paused", self.dispatch_paused)
        updates.setdefault("batch_cut_allowed", self.batch_cut_allowed)
        updates.setdefault("batch_release_cycle_id", self.batch_release_cycle_id)
        self.journal.write(
            controller=self._controller_state(),
            event={"type": event_type, **dict(payload)},
            **updates,
        )


def bind_orchestrator(orchestrator: Any, *, root: Path, state_path: Path) -> RuntimeBinding:
    """Bind an already-created pinned Prime orchestrator; used by canaries/tests."""

    existing = getattr(orchestrator, INSTALL_SENTINEL, None)
    if isinstance(existing, RuntimeBinding):
        return existing
    manifest, protocol = _protocol_root(root)
    return RuntimeBinding(orchestrator=orchestrator, root=root, state_path=state_path, manifest=manifest, protocol=protocol)


def install_from_environment() -> dict[str, Any]:
    """Install the run-local setup hook only under exact explicit activation."""

    if os.environ.get(ENABLE_ENV) != "1":
        return {"installed": False, "reason": "explicit enable variable is absent"}
    required = {
        RUN_ENV: RUN_ID,
        PROTOCOL_ENV: EXPECTED_PROTOCOL,
        ADMISSION_ENV: "1",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise RuntimeBindingError(f"{name} must equal the reviewed staged-run value")
    root_value = os.environ.get(ROOT_ENV)
    state_value = os.environ.get(STATE_ENV)
    if not root_value or not state_value:
        raise RuntimeBindingError(f"{ROOT_ENV} and {STATE_ENV} are required for staged activation")
    root = Path(root_value).expanduser().resolve()
    state_path = Path(state_value).expanduser().resolve()
    _protocol_root(root)
    try:
        from painter.consistent_group_admission import install_prime_guard
    except ModuleNotFoundError as exc:
        if exc.name != "painter":
            raise
        from consistent_group_admission import install_prime_guard
    guard = install_prime_guard(group_size=GROUP_SIZE)
    if not guard.get("installed") and not guard.get("already_installed"):
        raise RuntimeBindingError(f"consistent admission guard did not install: {guard}")
    try:
        from prime_rl.orchestrator.orchestrator import Orchestrator
    except ModuleNotFoundError as exc:
        raise RuntimeBindingError(f"pinned Prime orchestrator import unavailable: {exc}") from exc
    if getattr(Orchestrator, INSTALL_SENTINEL, False):
        return {"installed": False, "already_installed": True, "run_id": RUN_ID}
    original_setup = Orchestrator.setup

    async def setup(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original_setup(self, *args, **kwargs)
        bind_orchestrator(self, root=root, state_path=state_path)
        return result

    Orchestrator.setup = setup
    setattr(Orchestrator, INSTALL_SENTINEL, {"original_setup": original_setup, "root": str(root)})
    return {"installed": True, "already_installed": False, "run_id": RUN_ID, "protocol": EXPECTED_PROTOCOL}


__all__ = [
    "ADMISSION_ENV", "ENABLE_ENV", "EXPECTED_PROTOCOL", "ROOT_ENV", "RUN_ENV", "STATE_ENV",
    "RuntimeBinding", "RuntimeBindingError", "bind_orchestrator", "install_from_environment",
]
