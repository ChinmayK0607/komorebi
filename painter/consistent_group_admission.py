"""Whole-group admission for the consistent visual-judge RL run.

Prime's pinned ``TrainSink`` calls ``Algorithm.finalize_group`` before
``TrainSource.on_result``.  A failed or uncertain reward would therefore be
omitted from GRPO's reward tensor before the curriculum hook could reject the
group.  This module provides a small, opt-in guard for the run launcher:

* the original ``on_result`` still observes every completed group and updates
  sampler/accounting state;
* invalid groups return ``False`` from admission and skip group scoring; and
* complete groups of the configured size delegate both hooks unchanged.

The Prime classes are imported only by :func:`install_prime_guard`, so this
module remains usable for local preflight tests without Prime installed.
"""

from __future__ import annotations

import os
import logging
import json
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable, Iterable


CONSISTENT_PROTOCOL = "text-paint-sft-pairwise-consistent-v1"
GROUP_SIZE_ENV = "PAINTER_WHOLE_GROUP_SIZE"
ENABLE_ENV = "PAINTER_WHOLE_GROUP_ADMISSION"
INSTALL_SENTINEL = "_painter_consistent_group_guard"

_UNCERTAIN_REWARD_STATUSES = frozenset({
    "judge_uncertain",
    "uncertain",
    "censored",
    "missing",
})
_UNCERTAIN_EVIDENCE_STATUSES = frozenset({
    "uncertain",
    "censored",
    "missing",
})
_LOGGER = logging.getLogger(__name__)


def _emit_group_log(message: str, *args: Any) -> None:
    """Use Prime's configured sink in production, stdlib logging in tests."""
    rendered = message % args if args else message
    try:
        from prime_rl.utils.logger import get_logger
    except ModuleNotFoundError:
        _LOGGER.warning(rendered)
    else:
        get_logger().warning(rendered)


def _write_telemetry(guard: "WholeGroupAdmissionGuard", check: GroupCheck, decision: str) -> None:
    """Optionally append compact group decisions for early stall diagnosis."""
    target = os.environ.get("PAINTER_CONSISTENT_GROUP_TELEMETRY")
    if not target:
        return
    record = {
        "schema": "consistent-group-admission-telemetry-v1",
        "group_size": guard.group_size,
        "observed_size": check.observed_size,
        "decision": decision,
        "reason": check.reason,
        "groups_observed": guard.groups_observed,
        "groups_admitted": guard.groups_admitted,
        "groups_rejected": guard.groups_rejected,
        "delegate_rejections": guard.delegate_rejections,
    }
    path = os.path.abspath(target)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass(frozen=True)
class GroupCheck:
    """Auditable result of the pre-GRPO whole-group check."""

    valid: bool
    reason: str | None = None
    observed_size: int = 0
    invalid_members: tuple[str, ...] = ()


def _member_id(member: Any, index: int) -> str:
    value = getattr(member, "id", None)
    return str(value) if value is not None else f"member-{index}"


def _trace_uncertainty(trace: Any) -> str | None:
    if bool(getattr(trace, "has_error", False)):
        return "trace_error"
    error = getattr(trace, "error", None)
    if error is not None and error is not False and error != "":
        return "trace_error"
    info = getattr(trace, "info", None)
    if not isinstance(info, dict):
        return None
    if info.get("service_failure"):
        return "service_failure"
    if info.get("reward_status") in _UNCERTAIN_REWARD_STATUSES:
        return "uncertain_reward"
    evidence = info.get("score_evidence")
    if isinstance(evidence, dict) and evidence.get("status") in _UNCERTAIN_EVIDENCE_STATUSES:
        return "uncertain_reward"
    # A present null terminal visual score is an unresolved judgment.  The
    # genuine model-invalid path writes 0.0 and therefore remains admissible.
    if "final_canvas_score" in info and info.get("final_canvas_score") is None:
        return "null_reward"
    return None


def check_group(group: Iterable[Any], *, group_size: int = 8) -> GroupCheck:
    """Validate group cardinality and reward/error state before GRPO.

    ``group`` is materialized once because Prime supplies a list but tests and
    future adapters may pass another iterable.  A missing trace is treated as
    an unresolved rollout; an explicit model-invalid canvas is valid when the
    trace itself completed and carries its real zero score.
    """

    if isinstance(group, (str, bytes)):
        members = [group]
    else:
        members = list(group)
    if len(members) != group_size:
        return GroupCheck(False, "wrong_group_size", len(members))
    invalid: list[str] = []
    for index, episode in enumerate(members):
        member = _member_id(episode, index)
        if getattr(episode, "ok", True) is False:
            invalid.append(f"{member}:episode_error")
            continue
        episode_error = getattr(episode, "error", None)
        if episode_error is not None and episode_error is not False and episode_error != "":
            invalid.append(f"{member}:episode_error")
            continue
        traces = getattr(episode, "traces", None)
        if traces is None:
            # Some local doubles represent an episode directly as a trace.
            traces = [episode]
        else:
            traces = list(traces)
        if not traces:
            invalid.append(f"{member}:missing_trace")
            continue
        reasons = [reason for trace in traces if (reason := _trace_uncertainty(trace))]
        if reasons:
            invalid.append(f"{member}:{reasons[0]}")
    if invalid:
        return GroupCheck(False, "uncertain_or_error", len(members), tuple(invalid))
    return GroupCheck(True, observed_size=len(members))


class WholeGroupAdmissionGuard:
    """Call the original Prime admission hook while rejecting unsafe groups."""

    def __init__(self, original: Callable[[list[Any]], bool], *, group_size: int = 8) -> None:
        if not callable(original):
            raise TypeError("original admission hook must be callable")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        self.original = original
        self.group_size = group_size
        self.last_check: GroupCheck | None = None
        self.groups_observed = 0
        self.groups_admitted = 0
        self.groups_rejected = 0
        self.delegate_rejections = 0
        self.rejections_by_reason: Counter[str] = Counter()

    def __call__(self, group: list[Any]) -> bool:
        check = check_group(group, group_size=self.group_size)
        self.last_check = check
        self.groups_observed += 1
        if not check.valid:
            self.groups_rejected += 1
            self.rejections_by_reason[check.reason or "invalid_group"] += 1
            _emit_group_log(
                "consistent whole-group admission rejected group: size=%d reason=%s "
                "observed=%d rejected=%d",
                check.observed_size, check.reason or "invalid_group",
                self.groups_observed, self.groups_rejected,
            )
            _write_telemetry(self, check, "reject_uncertain_or_error")
        # Invoke Prime's hook for every completed group so sampler observation
        # and admission counters remain intact.  The guard overrides only the
        # final decision for incomplete/error/uncertain groups.
        delegated = bool(self.original(group))
        if not check.valid:
            return False
        if not delegated:
            self.delegate_rejections += 1
            _emit_group_log(
                "consistent whole-group admission delegated rejection: size=%d "
                "observed=%d admitted=%d rejected=%d delegate_rejections=%d",
                check.observed_size, self.groups_observed, self.groups_admitted,
                self.groups_rejected, self.delegate_rejections,
            )
            _write_telemetry(self, check, "reject_delegate")
            return False
        self.groups_admitted += 1
        _emit_group_log(
            "consistent whole-group admission accepted: size=%d observed=%d "
            "admitted=%d rejected=%d",
            check.observed_size, self.groups_observed, self.groups_admitted,
            self.groups_rejected,
        )
        _write_telemetry(self, check, "admit")
        return True

    def stats(self) -> dict[str, Any]:
        """Return JSON-friendly cumulative counters for run receipts/logging."""

        return {
            "group_size": self.group_size,
            "groups_observed": self.groups_observed,
            "groups_admitted": self.groups_admitted,
            "groups_rejected": self.groups_rejected,
            "delegate_rejections": self.delegate_rejections,
            "rejections_by_reason": dict(self.rejections_by_reason),
        }


def _group_guarded_finalize(original: Callable[..., Any], *, group_size: int) -> Callable[..., Any]:
    async def guarded(self: Any, group: list[Any]) -> Any:
        if not check_group(group, group_size=group_size).valid:
            return None
        return await original(self, group)

    return guarded


def install_prime_guard(*, group_size: int = 8) -> dict[str, Any]:
    """Install the guard on the pinned Prime classes exactly once.

    This is intentionally explicit rather than imported from normal painter
    modules.  A launcher or ``sitecustomize`` must call it after checking the
    run protocol.  It patches only the current interpreter; no Prime checkout
    is modified.
    """

    from prime_rl.orchestrator.algo.base import Algorithm
    from prime_rl.orchestrator.train_source import TrainSource

    if getattr(TrainSource, INSTALL_SENTINEL, None):
        return {
            "installed": False,
            "already_installed": True,
            "group_size": int(getattr(TrainSource, INSTALL_SENTINEL)["group_size"]),
            "protocol": CONSISTENT_PROTOCOL,
        }
    original_on_result = TrainSource.on_result
    original_finalize_group = Algorithm.finalize_group

    def guarded_on_result(self: Any, group: list[Any]) -> bool:
        guard = getattr(self, "_painter_whole_group_admission_guard", None)
        if guard is None:
            guard = WholeGroupAdmissionGuard(
                lambda rows: original_on_result(self, rows), group_size=group_size
            )
            setattr(self, "_painter_whole_group_admission_guard", guard)
        return guard(group)

    TrainSource.on_result = guarded_on_result
    Algorithm.finalize_group = _group_guarded_finalize(original_finalize_group, group_size=group_size)
    metadata = {
        "group_size": group_size,
        "protocol": CONSISTENT_PROTOCOL,
        "original_on_result": original_on_result,
        "original_finalize_group": original_finalize_group,
    }
    setattr(TrainSource, INSTALL_SENTINEL, metadata)
    _LOGGER.info(
        "consistent whole-group admission installed: protocol=%s group_size=%d",
        CONSISTENT_PROTOCOL, group_size,
    )
    return {"installed": True, "already_installed": False, "group_size": group_size,
            "protocol": CONSISTENT_PROTOCOL}


def install_from_environment() -> dict[str, Any]:
    """Opt-in launcher entry point, gated to the consistent judge protocol."""

    protocol = os.environ.get("PAINTER_JUDGE_PROTOCOL", "")
    enabled = os.environ.get(ENABLE_ENV) == "1"
    if protocol != CONSISTENT_PROTOCOL or not enabled:
        return {
            "installed": False,
            "enabled": False,
            "protocol": protocol or None,
            "required_protocol": CONSISTENT_PROTOCOL,
        }
    raw_size = os.environ.get(GROUP_SIZE_ENV, "8")
    try:
        group_size = int(raw_size)
    except ValueError as exc:
        raise ValueError(f"{GROUP_SIZE_ENV} must be a positive integer") from exc
    return install_prime_guard(group_size=group_size)


__all__ = [
    "CONSISTENT_PROTOCOL",
    "ENABLE_ENV",
    "GROUP_SIZE_ENV",
    "GroupCheck",
    "WholeGroupAdmissionGuard",
    "check_group",
    "install_from_environment",
    "install_prime_guard",
]
