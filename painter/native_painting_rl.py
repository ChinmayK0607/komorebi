"""Native painting RL taskset with one terminal aesthetic reward.

The interaction loop belongs to :mod:`painter.native_painting`.  This module
only supplies the larger curriculum's normalized rows and the task reward
which consumes the existing image-hash-bound Luna pairwise queue.  Keeping the
reward on the task means the native interaction trace still owns all sampled
tokens, masks, and log probabilities; the queue owns the external judgment.

Rows are finite and are returned in a family/source-group round-robin order.
Prime's pinned ``StandardSampler`` cycles a finite taskset in source order, so
this ordering controls early exposure and never silently cycles a small prefix
of a larger manifest.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

try:
    from painter.native_painting import (
        DEFAULT_MAX_TURNS,
        NativePaintingData,
        NativePaintingEnv,
        NativePaintingEnvConfig,
        NativePaintingTask,
        TEXT_NEXT_VERSION,
        TEXT_SYSTEM,
        _image,
        _reference_parts,
        _resolve,
        _text,
    )
except ImportError:
    from native_painting import (  # type: ignore
        DEFAULT_MAX_TURNS,
        NativePaintingData,
        NativePaintingEnv,
        NativePaintingEnvConfig,
        NativePaintingTask,
        TEXT_NEXT_VERSION,
        TEXT_SYSTEM,
        _image,
        _reference_parts,
        _resolve,
        _text,
    )

try:
    from painter.contract import NEXT_VERSION, SYSTEM
except ImportError:
    from contract import NEXT_VERSION, SYSTEM  # type: ignore

try:
    from painter.frozen_opponent import (
        FrozenOpponentProvider,
        OpponentResult,
        OpponentSpec,
        deterministic_seed,
        generate_frozen_opponent,
        validate_seed,
    )
except ImportError:
    from frozen_opponent import (  # type: ignore
        FrozenOpponentProvider,
        OpponentResult,
        OpponentSpec,
        deterministic_seed,
        generate_frozen_opponent,
        validate_seed,
    )


RL_PROTOCOL_VERSION = "native-painting-rl-v2"
DEFAULT_JUDGE_TIMEOUT_SECONDS = 1800.0
DEFAULT_MANIFEST = "scaled-training-manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest_rows(raw: Any) -> list[dict[str, Any]]:
    """Extract normalized rows from the supported scaled-manifest wrappers."""

    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = None
        for key in ("training_rows", "rows", "tasks", "training_states"):
            if key in raw:
                rows = raw[key]
                break
        if rows is None and isinstance(raw.get("modes"), dict):
            rows = []
            for mode, value in raw["modes"].items():
                if isinstance(value, dict):
                    value = value.get("training_rows") or value.get("rows") or value.get("tasks")
                if not isinstance(value, list):
                    raise TypeError(f"manifest mode {mode!r} must contain a row list")
                for row in value:
                    if not isinstance(row, dict):
                        raise TypeError("scaled manifest rows must be objects")
                    item = dict(row)
                    item.setdefault("mode", mode)
                    rows.append(item)
        if rows is None:
            raise TypeError(
                "scaled manifest must be a list or contain training_rows/rows/tasks/training_states"
            )
    else:
        raise TypeError("scaled manifest must be a list or JSON object")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("scaled manifest rows must be a non-empty list of objects")
    return [dict(row) for row in rows]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read either the scaled JSON wrapper or its resumable JSONL shards."""

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            rows.append(value)
        return _manifest_rows(rows)
    return _manifest_rows(json.loads(path.read_text()))


def _asset_value(value: Any, *, role: str) -> str | None:
    if isinstance(value, dict):
        for key in ("path", "png", "canvas", role, "reference", "file"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        return None
    return value if isinstance(value, str) and value else None


def _resolve_asset(root: Path, manifest_path: Path, value: Any, *, role: str) -> Path:
    text = _asset_value(value, role=role)
    if not text:
        raise ValueError(f"{role} image path is missing")
    candidate = Path(text)
    if not candidate.is_absolute():
        manifest_candidate = manifest_path.parent / candidate
        candidate = manifest_candidate if manifest_candidate.is_file() else root / candidate
    if candidate.is_symlink():
        raise ValueError(f"{role} image path is a symlink: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{role} image is missing: {candidate}")
    return candidate.absolute()


def _hash_from_row(row: dict[str, Any], role: str, value: Any = None) -> str | None:
    role_keys = {
        "reference": ("reference_sha256", "reference_hash"),
        "opponent": ("opponent_sha256", "opponent_hash", "anchor_sha256", "anchor_hash"),
    }[role]
    for key in role_keys:
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    hashes = row.get("image_sha256")
    if hashes is None:
        hashes = row.get("sha256")
    if isinstance(hashes, dict):
        aliases = {
            "reference": ("reference", "reference_png", "png"),
            "opponent": ("opponent", "anchor", "A", "B", "opponent_png"),
        }[role]
        for key in aliases:
            if isinstance(hashes.get(key), str) and hashes[key]:
                return hashes[key]
    elif role == "reference" and isinstance(hashes, str) and hashes:
        return hashes
    if isinstance(value, dict):
        for key in ("sha256", "hash", "png_sha256", "canvas_sha256"):
            if isinstance(value.get(key), str) and value[key]:
                return value[key]
    return None


def _opponent_value(row: dict[str, Any]) -> tuple[Any, str | None]:
    for key in ("opponent", "opponent_png", "fixed_opponent", "anchor", "anchor_png", "baseline", "baseline_png"):
        if key in row and row[key] is not None:
            value = row[key]
            source_group = None
            if isinstance(value, dict):
                source_group = value.get("source_group") or value.get("group")
            return value, source_group if isinstance(source_group, str) else None
    return None, None


def _lazy_opponent_value(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("lazy_opponent") or row.get("opponent_provider") or row.get("frozen_opponent")
    if value is True:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return None


def _load_generator(path: str):
    """Load a node-side generator as ``module:function`` or ``module.function``."""
    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, separator, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError("opponent_generator must be module:function")
    function = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(function):
        raise ValueError(f"opponent generator is not callable: {path}")
    return function


_FROZEN_PROVIDERS: dict[tuple[str, ...], FrozenOpponentProvider] = {}


def _frozen_provider(data: "NativePaintingRLData") -> FrozenOpponentProvider:
    cache_root = str(data.opponent_cache_root or (Path(data.root) / "frozen-opponents"))
    key = (
        cache_root,
        str(data.opponent_model_alias),
        data.opponent_generator,
        str(data.opponent_checkpoint_sha256),
        str(data.opponent_protocol_version),
        str(data.opponent_render_protocol),
        str(data.opponent_decoding_config),
    )
    provider = _FROZEN_PROVIDERS.get(key)
    if provider is None:
        generator = _load_generator(data.opponent_generator) if data.opponent_generator else generate_frozen_opponent
        provider = FrozenOpponentProvider(
            Path(cache_root),
            model_alias=str(data.opponent_model_alias),
            generator=generator,
            lock_timeout_seconds=float(data.opponent_lock_timeout_seconds),
            checkpoint_sha256=data.opponent_checkpoint_sha256,
            protocol_version=data.opponent_protocol_version,
            render_protocol=data.opponent_render_protocol,
            decoding_config=data.opponent_decoding_config,
        )
        _FROZEN_PROVIDERS[key] = provider
    return provider


def _interleave(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin families/groups, exhausting one row per group per round."""

    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        family = str(row["family"])
        group = str(row["source_group"])
        row = dict(row)
        row["_input_index"] = index
        buckets[family][group].append(row)
    families = sorted(buckets)
    group_names = {
        family: sorted(
            groups,
            key=lambda group: (groups[group][0]["_input_index"], group),
        )
        for family, groups in buckets.items()
    }
    max_rounds = max(
        len(buckets[family][group])
        for family in families
        for group in group_names[family]
    )
    max_groups = max(len(group_names[family]) for family in families)
    output: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        # Rotate group positions across families before taking a second row
        # from any group.  A family with hundreds of groups must not occupy
        # the entire prefix before the other families get their first sample.
        for group_index in range(max_groups):
            for family in families:
                groups = group_names[family]
                if group_index >= len(groups):
                    continue
                group = groups[group_index]
                values = buckets[family][group]
                if round_index < len(values):
                    output.append(values[round_index])
    return output


class NativePaintingRLData(NativePaintingData):
    """One normalized training row and its eager or lazy same-source opponent."""

    opponent: str | None = None
    opponent_sha256: str | None = None
    opponent_source_group: str | None = None
    opponent_cache_key: str | None = None
    opponent_cache_root: str | None = None
    opponent_model_alias: str | None = None
    opponent_checkpoint_sha256: str | None = None
    opponent_protocol_version: str | None = None
    opponent_render_protocol: str | None = None
    opponent_decoding_config: str | None = None
    opponent_seed: int | None = None
    opponent_generator: str | None = None
    opponent_lock_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS
    opponent_max_completion_tokens: int = 8192
    opponent_context_length: int = 32768
    rollout_group: str | None = None
    curriculum_order: int | None = None
    source_row_id: str | None = None
    judge_module_dir: str | None = None
    judge_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS


class NativePaintingRLTask(vf.Task[NativePaintingRLData, vf.State, vf.TaskConfig]):
    """Native task with an explicit RL wire-data type.

    The verifier env server reconstructs dispatched tasks from this generic
    parameter.  Inheriting ``NativePaintingTask`` leaves the base
    ``NativePaintingData`` parameter in place, so RL-only opponent fields are
    silently discarded at the RPC boundary before reward evaluation.
    """

    @classmethod
    def data_type(cls) -> type[NativePaintingRLData]:
        """Keep the RL wire schema explicit across verifier RPC boundaries."""

        return NativePaintingRLData

    @property
    def key(self) -> str:
        """Stable identity for one source/mode/opponent training row."""

        data = self.data
        return ":".join(
            (
                str(data.source_group),
                str(data.mode),
                str(data.opponent_sha256 or data.opponent_cache_key or data.opponent),
                str(data.task_id),
            )
        )

    @vf.reward(weight=1.0)
    async def final_canvas_aesthetic(self, trace: vf.Trace) -> float:
        """Return the two-orientation Luna pairwise fraction for one final canvas."""

        info = trace.info
        if info.get("service_failure"):
            # Renderer/judge outages are operational failures, never visual scores.
            raise RuntimeError(f"service failure is not a visual score: {info['service_failure']}")
        final_canvas_value = info.get("final_canvas")
        if not final_canvas_value or not info.get("final_canvas_valid"):
            # A model can finish after emitting invalid JavaScript or after a
            # failed repair turn.  That is a model outcome and must consume a
            # zero reward so the GRPO group remains trainable.  Renderer and
            # judge outages are marked as service_failure above and still
            # propagate as operational errors rather than becoming scores.
            info["final_canvas_score"] = 0.0
            info["reward_status"] = "model_invalid_canvas"
            return 0.0

        root = Path(self.data.root)
        reference = Path(self.data.reference)
        candidate = Path(str(final_canvas_value))
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            info["service_failure"] = "renderer service failure: advertised final canvas is missing"
            raise RuntimeError(info["service_failure"])
        opponent_value = self.data.opponent
        opponent: Path | None = Path(opponent_value) if opponent_value else None
        if self.data.reference_kind == "text" and opponent is None:
            raise ValueError("text painting requires a pre-rendered fixed SFT opponent")
        opponent_result: OpponentResult | None = None
        if opponent is None:
            if (
                not self.data.opponent_model_alias
                or not self.data.opponent_checkpoint_sha256
                or not self.data.opponent_protocol_version
                or not self.data.opponent_render_protocol
                or self.data.opponent_decoding_config is None
                or self.data.opponent_seed is None
            ):
                raise ValueError("fixed or lazy same-source_group opponent is missing; reward is unavailable")
            provider = _frozen_provider(self.data)
            spec = OpponentSpec(
                source_group=str(self.data.source_group),
                reference=reference,
                reference_sha256=str(self.data.reference_sha256),
                model_alias=str(self.data.opponent_model_alias),
                checkpoint_sha256=str(self.data.opponent_checkpoint_sha256),
                protocol_version=str(self.data.opponent_protocol_version),
                render_protocol=str(self.data.opponent_render_protocol),
                decoding_config=str(self.data.opponent_decoding_config),
                seed=int(self.data.opponent_seed),
                max_turns=int(self.data.max_turns),
                max_completion_tokens=int(self.data.opponent_max_completion_tokens),
                context_length=int(self.data.opponent_context_length),
            )
            opponent_result = await provider.resolve(spec)
            info.update(
                opponent_cache_key=opponent_result.cache_key,
                opponent_status=opponent_result.status,
                opponent_reason=opponent_result.reason,
            )
            if opponent_result.status == "unavailable":
                message = opponent_result.reason or "frozen opponent service unavailable"
                info["service_failure"] = f"frozen opponent service failure: {message}"
                raise RuntimeError(info["service_failure"])
            if opponent_result.status == "invalid_model_output":
                # A valid student canvas beats a frozen opponent that failed
                # to produce a canvas.  No synthetic image or judge score is
                # invented; both-invalid cases returned zero above.
                info["final_canvas_score"] = 1.0
                info["reward_status"] = "anchor_invalid_model_output"
                _atomic(
                    root / str(self.data.output_root) / str(getattr(trace, "id", self.data.task_id)) / "reward.json",
                    {
                        "status": "anchor_invalid_model_output",
                        "reward": 1.0,
                        "opponent_cache_key": opponent_result.cache_key,
                        "opponent_reason": opponent_result.reason,
                    },
                )
                return 1.0
            if not opponent_result.valid:
                raise RuntimeError("frozen opponent provider returned an unusable result")
            opponent = opponent_result.path
        if opponent is None or not all(path.is_file() for path in (reference, opponent, candidate)):
            raise ValueError("reference, opponent, or final canvas image is missing; reward is unavailable")
        if reference == opponent or sha(reference) == sha(opponent):
            raise ValueError("reference cannot be used as the RL opponent")

        if self.data.reference_kind == "text":
            text_protocol = os.environ.get(
                "PAINTER_JUDGE_PROTOCOL", "text-paint-sft-pairwise-v1")
            try:
                if text_protocol == "text-paint-sft-pairwise-margin-v1":
                    from painter.text_pairwise_margin_judge import judge_pair
                elif text_protocol == "text-paint-sft-pairwise-grounded-v1":
                    from painter.text_pairwise_grounded_judge import judge_pair
                elif text_protocol == "text-paint-sft-pairwise-consistent-v1":
                    from painter.text_pairwise_consistent_judge import judge_pair
                elif text_protocol == "text-paint-sft-pairwise-v1":
                    from painter.text_pairwise_judge import judge_pair
                else:
                    raise ValueError(f"unsupported text judge protocol: {text_protocol}")
            except ImportError:
                if text_protocol == "text-paint-sft-pairwise-margin-v1":
                    from text_pairwise_margin_judge import judge_pair
                elif text_protocol == "text-paint-sft-pairwise-grounded-v1":
                    from text_pairwise_grounded_judge import judge_pair
                elif text_protocol == "text-paint-sft-pairwise-consistent-v1":
                    from text_pairwise_consistent_judge import judge_pair
                elif text_protocol == "text-paint-sft-pairwise-v1":
                    from text_pairwise_judge import judge_pair
                else:
                    raise
            try:
                result = await judge_pair(
                    brief=reference, anchor=opponent, candidate=candidate,
                    key_file=Path(os.environ["PAINTER_GATEWAY_ENV_FILE"]),
                )
            except Exception as exc:
                info["service_failure"] = f"text judge service failure: {type(exc).__name__}: {exc}"
                raise
            trace_id = str(getattr(trace, "id", self.data.task_id))
            receipt_path = root / str(self.data.output_root) / trace_id / "text-judge.json"
            _atomic(receipt_path, result)
            comparison_images = {
                "reference": str(reference), "anchor": str(opponent),
                "candidate": str(candidate),
                "image_sha256": {"reference": sha(reference),
                                 "anchor": sha(opponent), "candidate": sha(candidate)},
            }
            decision_keys = (
                "status", "protocol", "winner_id", "preference", "consensus", "acceptance",
                "initial_conflict", "initial_votes", "recheck_votes", "format_version",
            )
            decision = {
                key: result.get(key)
                for key in decision_keys
                if isinstance(result, dict) and key in result
            }
            protocol_matches = isinstance(result, dict) and result.get("protocol") == text_protocol
            score_evidence = {
                "status": "scored" if protocol_matches and result.get("status") in {
                    "scored", "identical_canvases",
                } and result.get("score") is not None else "uncertain",
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha(receipt_path),
                "protocol_version": result.get("protocol") if isinstance(result, dict) else None,
                "decision": decision,
            }
            if not isinstance(result, dict) or score_evidence["status"] != "scored":
                info.update(
                    final_canvas_score=None,
                    reward_status="judge_uncertain",
                    score_evidence=score_evidence,
                    comparison_images=comparison_images,
                )
                raise ValueError("text judge judgment is missing or uncertain; reward is unavailable")
            if isinstance(result.get("score"), bool):
                info.update(
                    final_canvas_score=None,
                    reward_status="judge_uncertain",
                    score_evidence={**score_evidence, "status": "uncertain"},
                    comparison_images=comparison_images,
                )
                raise ValueError("text judge score is malformed; reward is unavailable")
            try:
                score = float(result["score"])
            except (KeyError, TypeError, ValueError) as exc:
                info.update(
                    final_canvas_score=None,
                    reward_status="judge_uncertain",
                    score_evidence={**score_evidence, "status": "uncertain"},
                    comparison_images=comparison_images,
                )
                raise ValueError("text judge score is missing or malformed; reward is unavailable") from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                info.update(
                    final_canvas_score=None,
                    reward_status="judge_uncertain",
                    score_evidence={**score_evidence, "status": "uncertain"},
                    comparison_images=comparison_images,
                )
                raise ValueError("text judge score is outside [0, 1]; reward is unavailable")
            info.update(
                final_canvas_score=score,
                reward_status="scored",
                score_evidence=score_evidence,
                comparison_images=comparison_images,
            )
            return score

        contract = _QueueContract(
            root,
            module_dir=(self.data.judge_module_dir or None),
        )
        trace_id = str(getattr(trace, "id", self.data.task_id))
        request_id = await _contract_call(
            contract.enqueue_comparison,
            root=root,
            trace_id=trace_id,
            reference=reference,
            anchor=opponent,
            candidate=candidate,
        )
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("judge queue returned no request id")
        info.update(
            judge_request_id=request_id,
            comparison_images={
                "reference": str(reference),
                "anchor": str(opponent),
                "candidate": str(candidate),
                "image_sha256": {
                    "reference": sha(reference),
                    "anchor": sha(opponent),
                    "candidate": sha(candidate),
                },
            },
        )
        started = time.monotonic()
        try:
            receipt = await _contract_call(
                contract.await_receipt,
                root=root,
                request_id=request_id,
                timeout=float(self.data.judge_timeout_seconds),
            )
        except Exception as exc:
            info["service_failure"] = f"judge service failure: {type(exc).__name__}: {exc}"
            raise
        info["judge_wait_seconds"] = time.monotonic() - started
        if not isinstance(receipt, dict):
            raise ValueError("judge receipt must be a JSON object")
        score = await _contract_call(
            contract.score_receipt,
            receipt=receipt,
            request_id=request_id,
            reference=reference,
            anchor=opponent,
            candidate=candidate,
        )
        receipt_path = root / "judge-queue" / request_id / "receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError("judge service failure: receipt evidence is missing")
        receipt_sha256 = sha(receipt_path)
        queue_evidence = {
            "status": "scored" if score is not None else "uncertain",
            "request_id": request_id,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
            "protocol_version": getattr(contract.judge, "PROTOCOL_VERSION", None),
        }
        if isinstance(score, bool):
            info.update(
                judge_receipt_path=str(receipt_path),
                judge_receipt=receipt,
                final_canvas_score=None,
                reward_status="judge_uncertain",
                score_evidence={**queue_evidence, "status": "uncertain"},
            )
            raise ValueError("judge score is malformed; reward is unavailable")
        if score is None:
            info.update(
                judge_receipt_path=str(receipt_path),
                judge_receipt=receipt,
                final_canvas_score=None,
                reward_status="judge_uncertain",
                score_evidence=queue_evidence,
            )
            raise ValueError("judge judgment is missing or uncertain; reward is unavailable")
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            info.update(
                judge_receipt_path=str(receipt_path),
                judge_receipt=receipt,
                final_canvas_score=None,
                reward_status="judge_uncertain",
                score_evidence={**queue_evidence, "status": "uncertain"},
            )
            raise ValueError("judge score is malformed; reward is unavailable") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            info.update(
                judge_receipt_path=str(receipt_path),
                judge_receipt=receipt,
                final_canvas_score=None,
                reward_status="judge_uncertain",
                score_evidence={**queue_evidence, "status": "uncertain"},
            )
            raise ValueError(f"aesthetic score outside [0, 1]: {score}")

        info.update(
            judge_request_path=str(root / "judge-queue" / request_id / "request.json"),
            judge_receipt_path=str(receipt_path),
            judge_receipt=receipt,
            final_canvas_score=score,
            score_evidence=queue_evidence,
        )
        _atomic(
            root / str(self.data.output_root) / str(trace_id) / "reward.json",
            {
                "status": "scored",
                "reward": score,
                "score_evidence": info["score_evidence"],
            },
        )
        return score


class NativePaintingRLConfig(vf.TasksetConfig):
    root: str = "."
    manifest: str = DEFAULT_MANIFEST
    expected_count: int | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    policy_label: str = "training"
    output_root: str = "rl-rollouts"
    renderer: str | None = None
    renderer_python: str | None = None
    judge_module_dir: str | None = None
    judge_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS
    preserve_manifest_order: bool = False
    lazy_opponent: bool = False
    opponent_cache_root: str | None = None
    opponent_model_alias: str | None = None
    opponent_checkpoint_sha256: str | None = None
    opponent_protocol_version: str = RL_PROTOCOL_VERSION
    opponent_render_protocol: str = "native-painting-v1"
    opponent_decoding_config: str = "{}"
    opponent_seed: int = 20260922
    opponent_generator: str | None = None
    opponent_lock_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS
    opponent_max_completion_tokens: int = 8192
    opponent_context_length: int = 32768


class NativePaintingRLEnvConfig(NativePaintingEnvConfig):
    pass


class NativePaintingRLEnv(NativePaintingEnv):
    """The current native Env loop, with scoring supplied by the task above."""


class NativePaintingRLTaskset(vf.Taskset[NativePaintingRLTask, NativePaintingRLConfig]):
    """Load every normalized row while preserving broad source exposure."""

    INFINITE = False

    def load(self) -> Iterable[NativePaintingRLTask]:
        root = Path(self.config.root).resolve()
        manifest_path = _resolve(root, self.config.manifest)
        rows = _read_manifest(manifest_path)
        if self.config.expected_count is not None and len(rows) != self.config.expected_count:
            raise ValueError(
                f"expected {self.config.expected_count} scaled training rows, found {len(rows)}"
            )
        if isinstance(self.config.max_turns, bool) or self.config.max_turns < 1:
            raise ValueError("max_turns must be a positive integer")

        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        group_reference: dict[str, str] = {}
        rollout_opponent: dict[tuple[str, str], str] = {}
        for index, raw in enumerate(rows):
            task_id = raw.get("task_id") or raw.get("id") or raw.get("case_id")
            source_group = raw.get("source_group") or raw.get("group")
            family = raw.get("family") or source_group
            reference_value = raw.get("reference_png") or raw.get("reference")
            reference_kind = str(raw.get("reference_kind", "image"))
            if reference_kind not in {"image", "text"}:
                raise ValueError(f"unsupported reference kind for {task_id}: {reference_kind}")
            if not all(isinstance(value, str) and value for value in (task_id, source_group, family)):
                raise ValueError(f"scaled row {index} needs id/task_id, source_group, and family")
            task_id = str(task_id)
            source_group = str(source_group)
            family = str(family)
            if raw.get("split") != "train":
                raise ValueError(
                    f"scaled row {task_id} must declare split=train; development/unsplit rows are not trainable"
                )
            if task_id in seen_ids:
                raise ValueError(f"duplicate scaled task id: {task_id}")
            seen_ids.add(task_id)
            reference = _resolve_asset(root, manifest_path, reference_value, role="reference")
            _reference_parts(reference, reference_kind)
            reference_digest = sha(reference)
            expected_reference = _hash_from_row(raw, "reference", reference_value)
            if expected_reference and expected_reference != reference_digest:
                raise ValueError(f"reference hash mismatch for {task_id}: {reference_digest}")

            opponent_value, embedded_group = _opponent_value(raw)
            opponent_group = (
                raw.get("opponent_source_group")
                or raw.get("anchor_source_group")
                or embedded_group
            )
            lazy_value = _lazy_opponent_value(raw)
            lazy_enabled = opponent_value is None and (lazy_value is not None or self.config.lazy_opponent)
            opponent: Path | None = None
            opponent_digest: str | None = None
            lazy_fields: dict[str, Any] = {}
            if opponent_value is None:
                if reference_kind == "text":
                    raise ValueError(f"text row {task_id} requires a pre-rendered fixed opponent")
                if not lazy_enabled:
                    raise ValueError(
                        f"scaled row {task_id} has no fixed opponent; materialize a distinct same-source_group canvas"
                    )
                if opponent_group is not None and str(opponent_group) != source_group:
                    raise ValueError(
                        f"opponent for {task_id} belongs to {opponent_group!r}, expected source_group {source_group!r}"
                    )
                settings = lazy_value or {}
                alias = str(settings.get("model_alias") or settings.get("alias") or self.config.opponent_model_alias or "")
                if not alias:
                    raise ValueError(f"lazy opponent row {task_id} needs an immutable model_alias")
                checkpoint_sha256 = str(
                    settings.get("checkpoint_sha256")
                    or self.config.opponent_checkpoint_sha256
                    or ""
                )
                if not checkpoint_sha256:
                    raise ValueError(f"lazy opponent row {task_id} needs a pinned checkpoint_sha256")
                protocol_version = str(
                    settings.get("protocol_version", self.config.opponent_protocol_version)
                )
                render_protocol = str(
                    settings.get("render_protocol", self.config.opponent_render_protocol)
                )
                decoding_value = settings.get("decoding_config", self.config.opponent_decoding_config)
                decoding_config = (
                    json.dumps(decoding_value, sort_keys=True, separators=(",", ":"))
                    if isinstance(decoding_value, dict)
                    else str(decoding_value)
                )
                base_seed = int(settings.get("base_seed", self.config.opponent_seed))
                explicit_seed = settings.get("seed")
                source_seed = validate_seed(
                    explicit_seed,
                    name=f"opponent seed for {task_id}",
                ) if explicit_seed is not None else deterministic_seed(
                    base_seed, source_group, reference_digest
                )
                cache_root_value = settings.get("cache_root", self.config.opponent_cache_root)
                cache_root = _resolve(root, str(cache_root_value or "frozen-opponents"))
                generator = settings.get("generator", self.config.opponent_generator)
                max_completion_tokens = int(settings.get("max_completion_tokens", self.config.opponent_max_completion_tokens))
                context_length = int(settings.get("context_length", self.config.opponent_context_length))
                spec = OpponentSpec(
                    source_group=source_group,
                    reference=reference,
                    reference_sha256=reference_digest,
                    model_alias=alias,
                    checkpoint_sha256=checkpoint_sha256,
                    protocol_version=protocol_version,
                    render_protocol=render_protocol,
                    decoding_config=decoding_config,
                    seed=source_seed,
                    max_turns=int(raw.get("opponent_max_turns", self.config.max_turns)),
                    max_completion_tokens=max_completion_tokens,
                    context_length=context_length,
                )
                lazy_fields = {
                    "opponent_cache_key": spec.cache_key,
                    "opponent_cache_root": str(cache_root),
                    "opponent_model_alias": alias,
                    "opponent_checkpoint_sha256": checkpoint_sha256,
                    "opponent_protocol_version": protocol_version,
                    "opponent_render_protocol": render_protocol,
                    "opponent_decoding_config": decoding_config,
                    "opponent_seed": source_seed,
                    "opponent_generator": str(generator) if generator else None,
                    "opponent_lock_timeout_seconds": float(settings.get("lock_timeout_seconds", self.config.opponent_lock_timeout_seconds)),
                    "opponent_max_completion_tokens": max_completion_tokens,
                    "opponent_context_length": context_length,
                }
                opponent_identity = spec.cache_key
            else:
                if opponent_group is not None and str(opponent_group) != source_group:
                    raise ValueError(
                        f"opponent for {task_id} belongs to {opponent_group!r}, expected source_group {source_group!r}"
                    )
                opponent = _resolve_asset(root, manifest_path, opponent_value, role="opponent")
                opponent_digest = sha(opponent)
                expected_opponent = _hash_from_row(raw, "opponent", opponent_value)
                if expected_opponent and expected_opponent != opponent_digest:
                    raise ValueError(f"opponent hash mismatch for {task_id}: {opponent_digest}")
                if reference_digest == opponent_digest:
                    raise ValueError(f"reference is also the opponent for {task_id}")
                opponent_identity = opponent_digest
            if source_group in group_reference and group_reference[source_group] != reference_digest:
                raise ValueError(f"source_group {source_group} has multiple reference images")
            rollout_group = str(
                raw.get("rollout_group")
                or raw.get("comparison_group")
                or raw.get("pair_group")
                or raw.get("opponent_group")
                or task_id
            )
            rollout_key = (source_group, rollout_group)
            if rollout_key in rollout_opponent and rollout_opponent[rollout_key] != opponent_identity:
                raise ValueError(f"rollout_group {rollout_group} has multiple fixed opponents")
            group_reference[source_group] = reference_digest
            rollout_opponent[rollout_key] = opponent_identity

            initial_canvas_value = raw.get("initial_canvas") or raw.get("canvas")
            initial_program_value = raw.get("initial_program") or raw.get("program")
            mode = str(raw.get("mode") or ("correction" if initial_canvas_value or initial_program_value else "fresh"))
            if mode not in {"fresh", "correction"}:
                raise ValueError(f"unsupported mode for {task_id}: {mode}")
            initial_canvas: Path | None = None
            initial_program: str | None = None
            if mode == "correction":
                if initial_canvas_value is None or initial_program_value is None:
                    raise ValueError(f"correction row {task_id} needs initial_canvas and initial_program")
                initial_canvas = _resolve_asset(root, manifest_path, initial_canvas_value, role="initial_canvas")
                program_path = _resolve_asset(root, manifest_path, initial_program_value, role="initial_program")
                initial_program = program_path.read_text()
            elif initial_canvas_value is not None or initial_program_value is not None:
                raise ValueError(f"fresh row {task_id} cannot provide an initial canvas/program")

            max_turns = int(raw.get("max_turns", self.config.max_turns))
            if max_turns < 1:
                raise ValueError(f"max_turns must be positive for {task_id}")
            judge_dir = raw.get("judge_module_dir", self.config.judge_module_dir)
            if judge_dir is not None:
                judge_dir = str(_resolve(root, str(judge_dir)))
            normalized.append({
                "task_id": task_id,
                "source_group": source_group,
                "family": family,
                "reference": str(reference),
                "reference_sha256": reference_digest,
                "reference_kind": reference_kind,
                "opponent": str(opponent) if opponent else None,
                "opponent_sha256": opponent_digest,
                "opponent_source_group": source_group,
                **lazy_fields,
                "rollout_group": rollout_group,
                "mode": mode,
                "initial_canvas": str(initial_canvas) if initial_canvas else None,
                "initial_program": initial_program,
                "max_turns": max_turns,
                "curriculum_order": raw.get("order", index),
                "source_row_id": str(raw.get("id") or task_id),
                "renderer": raw.get("renderer", self.config.renderer),
                "renderer_python": raw.get("renderer_python", self.config.renderer_python),
                "judge_module_dir": judge_dir,
            })

        ordered = normalized if self.config.preserve_manifest_order else _interleave(normalized)
        tasks: list[NativePaintingRLTask] = []
        for index, row in enumerate(ordered):
            prompt_parts: list[Any] = _reference_parts(Path(row["reference"]), row["reference_kind"])
            if row["mode"] == "correction":
                prompt_parts.extend(
                    [
                        _text("CURRENT CANVAS"),
                        _image(Path(row["initial_canvas"])),
                        _text("Current program:\n" + str(row["initial_program"])),
                    ]
                )
            prompt_parts.append(_text(TEXT_NEXT_VERSION if row["reference_kind"] == "text" else NEXT_VERSION))
            data = NativePaintingRLData(
                idx=index,
                name=row["task_id"],
                prompt=[vf.UserMessage(content=prompt_parts)],
                system_prompt=TEXT_SYSTEM if row["reference_kind"] == "text" else SYSTEM,
                root=str(root),
                task_id=row["task_id"],
                source_group=row["source_group"],
                family=row["family"],
                reference=row["reference"],
                reference_sha256=row["reference_sha256"],
                reference_kind=row["reference_kind"],
                policy_label=self.config.policy_label,
                output_root=self.config.output_root,
                max_turns=row["max_turns"],
                mode=row["mode"],
                initial_canvas=row["initial_canvas"],
                initial_program=row["initial_program"],
                renderer=(str(_resolve(root, row["renderer"])) if row["renderer"] else None),
                renderer_python=(str(_resolve(root, row["renderer_python"])) if row["renderer_python"] else None),
                opponent=row["opponent"],
                opponent_sha256=row["opponent_sha256"],
                opponent_source_group=row["opponent_source_group"],
                opponent_cache_key=row.get("opponent_cache_key"),
                opponent_cache_root=row.get("opponent_cache_root"),
                opponent_model_alias=row.get("opponent_model_alias"),
                opponent_checkpoint_sha256=row.get("opponent_checkpoint_sha256"),
                opponent_protocol_version=row.get("opponent_protocol_version"),
                opponent_render_protocol=row.get("opponent_render_protocol"),
                opponent_decoding_config=row.get("opponent_decoding_config"),
                opponent_seed=row.get("opponent_seed"),
                opponent_generator=row.get("opponent_generator"),
                opponent_lock_timeout_seconds=float(
                    row.get("opponent_lock_timeout_seconds", self.config.opponent_lock_timeout_seconds)
                ),
                opponent_max_completion_tokens=int(
                    row.get("opponent_max_completion_tokens", self.config.opponent_max_completion_tokens)
                ),
                opponent_context_length=int(
                    row.get("opponent_context_length", self.config.opponent_context_length)
                ),
                rollout_group=row["rollout_group"],
                curriculum_order=row["curriculum_order"],
                source_row_id=row["source_row_id"],
                judge_module_dir=row["judge_module_dir"],
                judge_timeout_seconds=float(self.config.judge_timeout_seconds),
            )
            tasks.append(NativePaintingRLTask(data))
        if not tasks:
            raise ValueError("scaled manifest contains no usable training rows")
        return tasks


class _QueueContract:
    """Thin adapter for the existing aesthetic_judge/judge_queue API."""

    def __init__(self, root: Path, module_dir: str | None = None) -> None:
        location = Path(module_dir) if module_dir else root
        if not location.is_absolute():
            location = root / location
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
        try:
            self.judge = importlib.import_module("aesthetic_judge")
            self.queue = importlib.import_module("judge_queue")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "aesthetic judge queue is unavailable; provide pinned aesthetic_judge.py and judge_queue.py"
            ) from exc
        required = ("enqueue_comparison", "score_receipt")
        if any(not hasattr(self.queue, name) for name in required):
            raise RuntimeError("aesthetic judge queue lacks the pinned enqueue/score API")

    def enqueue_comparison(self, *, root: Path, trace_id: str, reference: Path, anchor: Path, candidate: Path) -> str:
        return self.queue.enqueue_comparison(root, trace_id, reference, anchor, candidate)

    async def await_receipt(self, *, root: Path, request_id: str, timeout: float) -> dict[str, Any]:
        receipt_path = root / "judge-queue" / request_id / "receipt.json"
        started = time.monotonic()
        while not receipt_path.is_file():
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"aesthetic judge receipt timed out: {request_id}")
            await asyncio.sleep(1.0)
        return json.loads(receipt_path.read_text())

    def score_receipt(self, *, receipt: dict[str, Any], request_id: str, reference: Path, anchor: Path, candidate: Path) -> float | None:
        return self.queue.score_receipt(receipt, request_id, reference, anchor, candidate)


async def _contract_call(function: Any, **kwargs: Any) -> Any:
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        selected = kwargs
    else:
        selected = {name: value for name, value in kwargs.items() if name in parameters}
    result = function(**selected)
    return await result if inspect.isawaitable(result) else result


__all__ = [
    "NativePaintingRLConfig",
    "NativePaintingRLData",
    "NativePaintingRLEnv",
    "NativePaintingRLEnvConfig",
    "NativePaintingRLTask",
    "NativePaintingRLTaskset",
    "RL_PROTOCOL_VERSION",
    "_interleave",
]
