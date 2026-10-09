"""Admission for the consistent visual-judge RL run.

Prime's pinned ``TrainSink`` calls ``Algorithm.finalize_group`` before
``TrainSource.on_result``.  A failed or uncertain reward would therefore be
omitted from GRPO's reward tensor before the curriculum hook could reject the
group.  This module provides a small, opt-in guard for the run launcher:

* uncertain/error traces are marked non-trainable, without changing their
  recorded reward evidence or removing the surrounding episode;
* Prime's group algorithm sees only the scored traces, so its native
  group-centered advantages remain intact; and
* groups with at least three scored traces can train as partial groups, while
  smaller groups still return ``False`` from admission.

The three-trace lower bound is deliberate: it avoids fabricating a reward for
an unresolved judgment while retaining enough within-prompt contrast for a
stable centered group.  A full group is still bounded by the configured Prime
group size.  Telemetry records every excluded trace ID and the resulting
scored count so a run can be audited without guessing a score.

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
MIN_SCORED_GROUP_SIZE = 3

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
        "scored_size": check.scored_size,
        "minimum_scored_size": guard.minimum_scored_size,
        "scored_members": list(check.scored_members),
        "excluded_members": list(check.excluded_members),
        "invalid_members": list(check.invalid_members),
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
    """Auditable result of the pre-GRPO scored-subset check."""

    valid: bool
    reason: str | None = None
    observed_size: int = 0
    invalid_members: tuple[str, ...] = ()
    scored_size: int = 0
    scored_members: tuple[str, ...] = ()
    excluded_members: tuple[str, ...] = ()


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


def _trace_is_trainable(trace: Any) -> bool:
    agent = getattr(trace, "agent", None)
    return getattr(agent, "trainable", True) is not False


def _trace_id(trace: Any, index: int) -> str:
    value = getattr(trace, "id", None)
    return str(value) if value is not None else f"trace-{index}"


def _mark_excluded_trace(trace: Any, reason: str) -> None:
    """Exclude one unsafe trace while preserving its raw evidence.

    Prime's GRPO implementation filters on ``trace.agent.trainable`` before
    constructing the reward tensor.  Changing that flag is the smallest
    run-local hook that preserves the trace and its ``None``/uncertain reward,
    while ensuring it cannot enter scoring, transport, or batching.
    """

    agent = getattr(trace, "agent", None)
    if agent is not None and hasattr(agent, "trainable"):
        agent.trainable = False
    info = getattr(trace, "info", None)
    if isinstance(info, dict):
        info.setdefault(
            "consistent_group_admission",
            {
                "status": (
                    "excluded_error_trace"
                    if reason in {"episode_error", "trace_error"}
                    else "excluded_uncertain_trace"
                ),
                "reason": reason,
            },
        )


def _exclude_group_traces(group: Iterable[Any]) -> None:
    """Mark error/uncertain traces before any Prime group algorithm runs."""

    for episode in group:
        traces = getattr(episode, "traces", None)
        traces = [episode] if traces is None else list(traces)
        episode_error = (
            getattr(episode, "ok", True) is False
            or (
                getattr(episode, "error", None) is not None
                and getattr(episode, "error", None) is not False
                and getattr(episode, "error", None) != ""
            )
        )
        for trace in traces:
            reason = "episode_error" if episode_error else _trace_uncertainty(trace)
            if reason is not None:
                _mark_excluded_trace(trace, reason)


def _collect_group_members(group: Iterable[Any]) -> tuple[list[Any], list[str], list[str], list[str]]:
    """Return materialized episodes plus scored, excluded, and invalid IDs."""

    members = [group] if isinstance(group, (str, bytes)) else list(group)
    scored: list[str] = []
    excluded: list[str] = []
    invalid: list[str] = []
    for episode_index, episode in enumerate(members):
        episode_id = _member_id(episode, episode_index)
        if getattr(episode, "ok", True) is False:
            invalid.append(f"{episode_id}:episode_error")
            continue
        episode_error = getattr(episode, "error", None)
        if episode_error is not None and episode_error is not False and episode_error != "":
            invalid.append(f"{episode_id}:episode_error")
            continue
        traces = getattr(episode, "traces", None)
        if traces is None:
            # Some local doubles represent an episode directly as a trace.
            traces = [episode]
        else:
            traces = list(traces)
        if not traces:
            invalid.append(f"{episode_id}:missing_trace")
            continue
        for trace_index, trace in enumerate(traces):
            trace_id = _trace_id(trace, trace_index)
            reason = _trace_uncertainty(trace)
            if reason is not None:
                excluded.append(f"{trace_id}:{reason}")
                continue
            if _trace_is_trainable(trace):
                scored.append(trace_id)
    return members, scored, excluded, invalid


def check_group(
    group: Iterable[Any], *, group_size: int = 8, minimum_scored: int = MIN_SCORED_GROUP_SIZE
) -> GroupCheck:
    """Validate group bounds and reward/error state before GRPO.

    ``group`` is materialized once because Prime supplies a list but tests and
    future adapters may pass another iterable.  A missing trace is treated as
    an unresolved rollout; an explicit model-invalid canvas is valid when the
    trace itself completed and carries its real zero score.  Prime's native
    GRPO implementation centers over the trainable traces it receives, so a
    group is admissible whenever that scored subset has at least
    ``minimum_scored`` members and the observed episode count does not exceed
    the configured full-group size.
    """

    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if isinstance(minimum_scored, bool) or not isinstance(minimum_scored, int) or minimum_scored < 1:
        raise ValueError("minimum_scored must be a positive integer")
    members, scored, excluded, invalid = _collect_group_members(group)
    observed_size = len(members)
    all_invalid = tuple(invalid) + tuple(excluded)
    if observed_size > group_size:
        return GroupCheck(
            False, "wrong_group_size", observed_size, all_invalid, len(scored),
            tuple(scored), tuple(excluded),
        )
    if len(scored) < minimum_scored:
        reason = "insufficient_scored_members" if (excluded or invalid) else "insufficient_group_size"
        return GroupCheck(
            False, reason, observed_size, all_invalid, len(scored),
            tuple(scored), tuple(excluded),
        )
    return GroupCheck(
        True, observed_size=observed_size, invalid_members=all_invalid,
        scored_size=len(scored), scored_members=tuple(scored), excluded_members=tuple(excluded),
    )


class WholeGroupAdmissionGuard:
    """Call Prime's admission hook while excluding unsafe traces."""

    def __init__(
        self,
        original: Callable[[list[Any]], bool],
        *,
        group_size: int = 8,
        minimum_scored: int = MIN_SCORED_GROUP_SIZE,
    ) -> None:
        if not callable(original):
            raise TypeError("original admission hook must be callable")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        if isinstance(minimum_scored, bool) or not isinstance(minimum_scored, int) or minimum_scored < 1:
            raise ValueError("minimum_scored must be a positive integer")
        self.original = original
        self.group_size = group_size
        self.minimum_scored_size = minimum_scored
        self.last_check: GroupCheck | None = None
        self.groups_observed = 0
        self.groups_admitted = 0
        self.groups_rejected = 0
        self.delegate_rejections = 0
        self.rejections_by_reason: Counter[str] = Counter()

    def __call__(self, group: list[Any]) -> bool:
        check = check_group(
            group, group_size=self.group_size, minimum_scored=self.minimum_scored_size
        )
        self.last_check = check
        self.groups_observed += 1
        # Mark unresolved traces before delegating, including rejected groups.
        # This keeps curriculum observation safe and leaves the full episode
        # objects available for receipt/accounting inspection.
        _exclude_group_traces(group)
        if not check.valid:
            self.groups_rejected += 1
            self.rejections_by_reason[check.reason or "invalid_group"] += 1
            _emit_group_log(
                "consistent group admission rejected group: size=%d scored=%d reason=%s "
                "observed=%d rejected=%d",
                check.observed_size, check.scored_size, check.reason or "invalid_group",
                self.groups_observed, self.groups_rejected,
            )
            _write_telemetry(
                self, check, f"reject_{check.reason or 'invalid_group'}"
            )
        # Invoke Prime's hook for every completed group so sampler observation
        # and admission counters remain intact.  Uncertain traces have already
        # been marked non-trainable, so Prime's own group logic observes only
        # the scored subset without receiving a fabricated reward.
        delegated = bool(self.original(group))
        if not check.valid:
            return False
        if not delegated:
            self.delegate_rejections += 1
            _emit_group_log(
                "consistent group admission delegated rejection: size=%d scored=%d "
                "observed=%d admitted=%d rejected=%d delegate_rejections=%d",
                check.observed_size, check.scored_size, self.groups_observed, self.groups_admitted,
                self.groups_rejected, self.delegate_rejections,
            )
            _write_telemetry(self, check, "reject_delegate")
            return False
        self.groups_admitted += 1
        _emit_group_log(
            "consistent group admission accepted: size=%d scored=%d excluded=%d observed=%d "
            "admitted=%d rejected=%d",
            check.observed_size, check.scored_size, len(check.excluded_members),
            self.groups_observed, self.groups_admitted,
            self.groups_rejected,
        )
        _write_telemetry(self, check, "admit")
        return True

    def stats(self) -> dict[str, Any]:
        """Return JSON-friendly cumulative counters for run receipts/logging."""

        return {
            "group_size": self.group_size,
            "minimum_scored_size": self.minimum_scored_size,
            "groups_observed": self.groups_observed,
            "groups_admitted": self.groups_admitted,
            "groups_rejected": self.groups_rejected,
            "delegate_rejections": self.delegate_rejections,
            "rejections_by_reason": dict(self.rejections_by_reason),
        }


def _group_guarded_finalize(original: Callable[..., Any], *, group_size: int) -> Callable[..., Any]:
    async def guarded(self: Any, group: list[Any]) -> Any:
        check = check_group(group, group_size=group_size)
        # Exclude unresolved traces before Prime's GRPO code builds its reward
        # tensor.  Prime then centers advantages over the remaining traces.
        _exclude_group_traces(group)
        if not check.valid:
            return None
        return await original(self, group)

    return guarded


def _episode_guarded_finalize(original: Callable[..., Any]) -> Callable[..., Any]:
    """Exclude unsafe traces before Prime snapshots group survivors.

    Prime's ``TrainSink.process_group`` computes its local ``survivors`` list
    before invoking ``Algorithm.finalize_group``.  Group-time exclusion alone
    is therefore too late: an info-only uncertain trace could remain in that
    stale list and reach ``trace_to_samples`` without an advantage stream.
    ``process_episode`` runs this hook before the group is assembled, so the
    existing Prime survivor snapshot already omits the trace.
    """

    async def guarded(self: Any, episode: Any) -> Any:
        _exclude_group_traces([episode])
        return await original(self, episode)

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
        metadata = getattr(TrainSource, INSTALL_SENTINEL)
        return {
            "installed": False,
            "already_installed": True,
            "group_size": int(metadata["group_size"]),
            "episode_finalize_guard": bool(metadata.get("episode_finalize_guard")),
            "protocol": CONSISTENT_PROTOCOL,
        }
    original_on_result = TrainSource.on_result
    original_finalize_episode = getattr(Algorithm, "finalize_episode", None)
    original_finalize_group = Algorithm.finalize_group
    if not callable(original_finalize_episode):
        raise RuntimeError("pinned Prime Algorithm.finalize_episode is unavailable")

    def guarded_on_result(self: Any, group: list[Any]) -> bool:
        guard = getattr(self, "_painter_whole_group_admission_guard", None)
        if guard is None:
            guard = WholeGroupAdmissionGuard(
                lambda rows: original_on_result(self, rows), group_size=group_size
            )
            setattr(self, "_painter_whole_group_admission_guard", guard)
        return guard(group)

    TrainSource.on_result = guarded_on_result
    Algorithm.finalize_episode = _episode_guarded_finalize(original_finalize_episode)
    Algorithm.finalize_group = _group_guarded_finalize(original_finalize_group, group_size=group_size)
    metadata = {
        "group_size": group_size,
        "protocol": CONSISTENT_PROTOCOL,
        "episode_finalize_guard": True,
        "original_on_result": original_on_result,
        "original_finalize_episode": original_finalize_episode,
        "original_finalize_group": original_finalize_group,
    }
    setattr(TrainSource, INSTALL_SENTINEL, metadata)
    _LOGGER.info(
        "consistent whole-group admission installed: protocol=%s group_size=%d",
        CONSISTENT_PROTOCOL, group_size,
    )
    return {
        "installed": True,
        "already_installed": False,
        "group_size": group_size,
        "episode_finalize_guard": True,
        "protocol": CONSISTENT_PROTOCOL,
    }


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
    "MIN_SCORED_GROUP_SIZE",
    "GroupCheck",
    "WholeGroupAdmissionGuard",
    "check_group",
    "install_from_environment",
    "install_prime_guard",
]
