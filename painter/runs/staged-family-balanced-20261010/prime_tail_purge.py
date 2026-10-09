"""Fail-closed Prime TrainSink tail purge adapter.

This module intentionally imports no Prime or CUDA package. It validates the
native ``TrainSink`` shape at the boundary, removes only trace IDs explicitly
attributed to the active-stage inventory, and returns a receipt that the
controller requires before stage progression. The caller must persist or
forward that receipt to ``StageController.record_stage_tail_drop``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from staged_curriculum import PRIME_REVISION, CurriculumError, StageController


class PrimeTailPurgeError(CurriculumError):
    """Raised when a native TrainSink tail cannot be proven safe to purge."""


def _fail(message: str) -> None:
    raise PrimeTailPurgeError(message)


def _trace_from_episode(episode: Any, trace_id: str) -> Any:
    traces = getattr(episode, "traces", None)
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        _fail("unsupported Prime episode shape: traces must be a sequence")
    matches = [trace for trace in traces if getattr(trace, "id", None) == trace_id]
    if len(matches) != 1:
        _fail(f"trace {trace_id!r} is missing or ambiguous in episode traces")
    return matches[0]


def _payload_tokens(samples: Any, trace: Any) -> int:
    if not isinstance(samples, list):
        _fail("unsupported Prime pending_batch shape: samples must be a list")
    total = 0
    for sample in samples:
        token_ids = getattr(sample, "token_ids", None)
        if token_ids is None:
            _fail("unsupported Prime TrainingSample shape: token_ids are missing")
        try:
            count = len(token_ids)
        except TypeError as exc:
            _fail(f"unsupported Prime token_ids shape: {exc}")
        if count < 0:
            _fail("Prime token count cannot be negative")
        total += count
    if total:
        return total
    fallback = getattr(trace, "num_total_tokens", None)
    if type(fallback) is not int or fallback < 0:
        _fail("unsupported Prime trace shape: num_total_tokens is missing")
    return fallback


def purge_pending_stage_tail(
    sink: Any,
    controller: StageController,
    trace_ids_by_cycle: Mapping[str, Sequence[str]],
    *,
    prime_revision: str,
    receipt_id: str,
) -> dict[str, Any]:
    """Physically remove an exact active-stage tail from a native TrainSink.

    No controller state is changed. A successful receipt is accepted by
    ``record_stage_tail_drop`` only after all requested IDs are removed and
    the pending trace/token invariants are rechecked.
    """

    if controller.halted:
        _fail("controller is halted")
    if prime_revision != PRIME_REVISION:
        _fail("unsupported Prime revision")
    if not isinstance(receipt_id, str) or not receipt_id.strip():
        _fail("tail purge needs a non-empty receipt_id")
    stage = controller._stage()
    if controller.stage_optimizer_updates_observed != stage["optimizer_updates_target"]:
        _fail("stage tail purge is allowed only after the optimizer target")
    inventory = {cycle_id: count for cycle_id, count in controller.trace_inventory_by_cycle.items() if count}
    if not inventory:
        _fail("stage tail inventory is empty")
    if not isinstance(trace_ids_by_cycle, Mapping) or set(trace_ids_by_cycle) != set(inventory):
        _fail("tail trace cycle keys do not match the active-stage inventory")
    all_trace_ids: list[str] = []
    for cycle_id, expected_count in inventory.items():
        trace_ids = trace_ids_by_cycle[cycle_id]
        if not isinstance(trace_ids, Sequence) or isinstance(trace_ids, (str, bytes)):
            _fail("tail trace IDs must be sequences")
        if len(trace_ids) != expected_count:
            _fail(f"tail trace count mismatch for {cycle_id}")
        if any(not isinstance(trace_id, str) or not trace_id for trace_id in trace_ids):
            _fail("tail trace IDs must be non-empty strings")
        all_trace_ids.extend(trace_ids)
    if len(set(all_trace_ids)) != len(all_trace_ids):
        _fail("tail trace IDs must be unique across cycles")

    if {cycle_id: list(trace_ids_by_cycle[cycle_id]) for cycle_id in inventory} != controller.trace_ids_by_cycle:
        _fail("tail trace IDs do not exactly match controller stage/cycle provenance")

    pending_batch = getattr(sink, "pending_batch", None)
    episode_by_trace = getattr(sink, "episode_by_trace", None)
    if not isinstance(pending_batch, Mapping) or not isinstance(episode_by_trace, Mapping):
        _fail("unsupported Prime TrainSink shape: pending maps are missing")
    if set(pending_batch) != set(episode_by_trace):
        _fail("TrainSink pending_batch and episode_by_trace keys are inconsistent")
    if any(trace_id not in pending_batch for trace_id in all_trace_ids):
        _fail("tail includes an unknown or already-purged pending trace ID")

    batch_size = getattr(sink, "batch_size", None)
    token_batch_size = getattr(sink, "token_batch_size", None)
    batch_mode = type(batch_size) is int and batch_size > 0 and token_batch_size is None
    token_mode = type(token_batch_size) is int and token_batch_size > 0 and batch_size is None
    if not (batch_mode or token_mode):
        _fail("unsupported Prime TrainSink batch/token mode shape")
    pending_tokens = getattr(sink, "pending_tokens", None)
    if type(pending_tokens) is not int or pending_tokens < 0:
        _fail("unsupported Prime pending_tokens accounting")

    token_counts: dict[str, int] = {}
    for trace_id in all_trace_ids:
        trace = _trace_from_episode(episode_by_trace[trace_id], trace_id)
        token_counts[trace_id] = _payload_tokens(pending_batch[trace_id], trace)
    removed_tokens = sum(token_counts.values())
    if token_mode and removed_tokens > pending_tokens:
        _fail("tail purge would make pending token accounting negative")

    before_ids = list(pending_batch)
    before_pending_tokens = pending_tokens
    for trace_id in all_trace_ids:
        del pending_batch[trace_id]
        del episode_by_trace[trace_id]
    if token_mode:
        sink.pending_tokens = pending_tokens - removed_tokens
    if any(trace_id in pending_batch or trace_id in episode_by_trace for trace_id in all_trace_ids):
        _fail("Prime tail purge did not remove every requested trace")
    if set(pending_batch) != set(episode_by_trace):
        _fail("TrainSink pending maps diverged after tail purge")

    return {
        "schema": "staged-family-balanced-prime-tail-purge-v1",
        "status": "purged_pending_controller_commit",
        "receipt_id": receipt_id,
        "prime_revision": prime_revision,
        "controller_stage": controller.active_stage,
        "controller_inventory_before": inventory,
        "controller_trace_ids_before": {cycle_id: list(trace_ids) for cycle_id, trace_ids in controller.trace_ids_by_cycle.items()},
        "trace_ids_by_cycle": {cycle_id: list(trace_ids_by_cycle[cycle_id]) for cycle_id in inventory},
        "removed_trace_ids_by_cycle": {cycle_id: list(trace_ids_by_cycle[cycle_id]) for cycle_id in inventory},
        "removed_trace_count": len(all_trace_ids),
        "removed_token_count": removed_tokens,
        "batch_mode": batch_mode,
        "token_mode": token_mode,
        "pending_trace_ids_before": before_ids,
        "pending_trace_count_after": len(pending_batch),
        "pending_tokens_before": before_pending_tokens,
        "pending_tokens_after": getattr(sink, "pending_tokens"),
        "physical_removal_verified": True,
        "controller_commit_required": True,
        "no_reward_fabrication": True,
    }


def commit_purged_stage_tail(
    controller: StageController,
    purge_receipt: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Commit a previously verified physical purge to the controller."""

    if not isinstance(purge_receipt, Mapping):
        _fail("purge receipt is missing")
    return controller.record_stage_tail_drop(
        trace_ids_by_cycle=purge_receipt.get("trace_ids_by_cycle", {}),
        reason=reason,
        purge_receipt=purge_receipt,
    )
