"""Lazy, hash-bound cache for a frozen RL opponent canvas.

The provider is deliberately agnostic to the model server.  A node-side
generator receives :class:`OpponentSpec`, including the immutable LoRA alias,
and returns a canvas path or an explicit model-invalid result.  The provider
deduplicates requests within and across processes, validates the returned
bytes, and caches only successful canvases or model-invalid outcomes.  Server
outages are returned as unavailable and are never converted into a visual
score or a cached failure.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shutil
import sys
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

try:  # The runtime is Linux; keeping import optional makes pure tests portable.
    import fcntl
except ImportError:  # pragma: no cover - only non-POSIX development hosts
    fcntl = None  # type: ignore[assignment]


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PROVIDER_VERSION = "frozen-opponent-v1"
MAX_DIAGNOSTIC_BYTES = 64 * 1024
MAX_SIGNED_SEED = (1 << 63) - 1


class OpponentUnavailable(RuntimeError):
    """The separate frozen-model service could not produce an answer."""


class OpponentInvalid(RuntimeError):
    """The frozen model completed but produced no valid canvas."""


@dataclass(frozen=True)
class OpponentSpec:
    source_group: str
    reference: Path
    reference_sha256: str
    model_alias: str
    checkpoint_sha256: str
    protocol_version: str
    render_protocol: str
    decoding_config: str
    seed: int
    work_root: Path | None = None
    max_turns: int = 6
    max_completion_tokens: int = 8192
    context_length: int = 32768

    def __post_init__(self) -> None:
        validate_seed(self.seed)

    @property
    def cache_key(self) -> str:
        material = "|".join(
            (
                self.source_group,
                self.reference_sha256,
                self.model_alias,
                self.checkpoint_sha256,
                self.protocol_version,
                self.render_protocol,
                self.decoding_config,
                str(self.seed),
                str(self.max_turns),
                str(self.max_completion_tokens),
                str(self.context_length),
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpponentResult:
    status: str
    cache_key: str
    path: Path | None = None
    sha256: str | None = None
    reason: str | None = None
    generated: bool = False

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.path is not None and self.sha256 is not None


def deterministic_seed(base_seed: int, source_group: str, reference_sha256: str) -> int:
    """Derive a stable API-safe per-source seed without process-random hashing.

    vLLM/OpenAI seed values are serialized as signed 64-bit integers.  Masking
    to 63 bits keeps every derived seed in that accepted non-negative range.
    """
    material = f"{base_seed}:{source_group}:{reference_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & MAX_SIGNED_SEED


def validate_seed(seed: int, *, name: str = "seed") -> int:
    """Validate an explicit non-negative signed-63-bit API seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= seed <= MAX_SIGNED_SEED:
        raise ValueError(f"{name} must be in [0, {MAX_SIGNED_SEED}]")
    return seed


def _absolute_preserving_symlink(path: str | Path) -> Path:
    """Make an interpreter path absolute without dereferencing its venv link."""
    return Path(path).absolute()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_private_log(path: Path, value: Any) -> None:
    """Keep bounded evaluator diagnostics beside the isolated attempt."""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value or "")
    if len(text.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
        encoded = text.encode("utf-8")[:MAX_DIAGNOSTIC_BYTES]
        text = encoded.decode("utf-8", errors="ignore") + "\n...[truncated]\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _trace_values(trace: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    context = trace.get("_episode_context") if isinstance(trace.get("_episode_context"), dict) else {}
    return data, info, context


def _native_eval_canvas(
    attempt: Path,
    case_id: str,
    workspace_root: Path,
    *,
    trace_bundle: tuple[list[Mapping[str, Any]], list[Any], list[Path]] | None = None,
) -> Path | Mapping[str, Any]:
    """Read the one-case native trace; never infer validity from a PNG glob."""
    attempt = Path(attempt).resolve()
    workspace_root = Path(workspace_root).resolve()
    try:
        from painter.native_painting_report import read_traces, resolve_existing
    except ImportError:  # installed Prime environments may expose flat modules
        try:
            from native_painting_report import read_traces, resolve_existing  # type: ignore
        except ImportError as exc:
            raise OpponentUnavailable(f"native trace parser unavailable: {exc}") from exc
    traces, parse_errors, trace_files = trace_bundle if trace_bundle is not None else read_traces(attempt)
    matches: list[tuple[Mapping[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for trace in traces:
        data, info, context = _trace_values(trace)
        identifiers = {
            str(value)
            for value in (
                info.get("task_id"), info.get("case_id"), info.get("source_group"),
                data.get("task_id"), data.get("case_id"), data.get("name"),
            )
            if isinstance(value, str) and value
        }
        if case_id in identifiers:
            matches.append((trace, data, info, context))
    if not matches:
        if parse_errors:
            detail = str(parse_errors[0])[-600:]
        elif not trace_files:
            detail = "native eval produced no trace stream"
        else:
            detail = "native eval trace did not contain the requested case"
        raise OpponentUnavailable(detail)

    trace, data, info, context = matches[-1]
    errors: list[str] = []
    for value in (trace.get("errors"), context.get("errors")):
        if isinstance(value, list):
            errors.extend(str(item.get("message") if isinstance(item, dict) else item) for item in value)
    if trace.get("ok") is False or context.get("ok") is False:
        errors.append("native eval episode reported ok=false")
    if info.get("service_failure"):
        errors.append(str(info["service_failure"]))
    raw_turns = info.get("turns")
    if isinstance(raw_turns, list):
        for row in raw_turns:
            if isinstance(row, dict) and row.get("failure_kind") == "service":
                errors.append(str(row.get("error") or "renderer service failure"))
    if str(trace.get("stop_condition") or "") in {"error", "timeout", "cancelled", "failed"}:
        errors.append(f"native eval stop_condition={trace.get('stop_condition')}")
    if errors:
        raise OpponentUnavailable("; ".join(errors)[-1200:])

    if not bool(info.get("final_canvas_valid")):
        return {
            "status": "invalid_model_output",
            "reason": "frozen model produced no valid canvas in canonical native eval",
        }
    final_value = info.get("final_canvas")
    if not isinstance(final_value, str) or not final_value:
        raise OpponentUnavailable("native trace advertised a valid canvas without final_canvas")
    source_path = Path(str(trace.get("_source_path", attempt)))
    final_path = resolve_existing(
        final_value,
        evidence_root=attempt,
        base_dir=source_path.parent,
        node_root=str(workspace_root),
    )
    if final_path is None:
        raise OpponentUnavailable("native trace advertised a valid canvas whose file is missing")
    try:
        final_path.relative_to(attempt)
    except ValueError as exc:
        raise OpponentUnavailable("native trace final canvas escaped isolated eval root") from exc
    if final_path.read_bytes()[: len(PNG_MAGIC)] != PNG_MAGIC:
        raise OpponentUnavailable("native trace advertised a non-PNG final canvas")
    return final_path


def _result_from_value(value: Any) -> tuple[str, Path | None, str | None]:
    if isinstance(value, (str, Path)):
        return "valid", Path(value), None
    if not isinstance(value, Mapping):
        raise OpponentInvalid("generator returned neither a canvas path nor a result object")
    status = str(value.get("status", "valid"))
    if status not in {"valid", "invalid_model_output", "unavailable"}:
        raise OpponentInvalid(f"unsupported generator status: {status}")
    path_value = value.get("path") or value.get("canvas") or value.get("opponent")
    path = Path(path_value) if isinstance(path_value, str) and path_value else None
    reason = str(value.get("reason")) if value.get("reason") is not None else None
    return status, path, reason


class FrozenOpponentProvider:
    """Resolve frozen opponents lazily with process and filesystem locking."""

    def __init__(
        self,
        cache_root: Path,
        *,
        model_alias: str,
        generator: Callable[[OpponentSpec], Any] | None = None,
        lock_timeout_seconds: float = 1800.0,
        checkpoint_sha256: str | None = None,
        protocol_version: str | None = None,
        render_protocol: str | None = None,
        decoding_config: str | None = None,
    ) -> None:
        if not model_alias or any(char.isspace() for char in model_alias):
            raise ValueError("model_alias must be a stable non-empty vLLM alias")
        self.cache_root = Path(cache_root).resolve()
        self.model_alias = model_alias
        self.generator = generator
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.checkpoint_sha256 = checkpoint_sha256
        self.protocol_version = protocol_version
        self.render_protocol = render_protocol
        self.decoding_config = decoding_config
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _local_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(key, asyncio.Lock())

    def _paths(self, spec: OpponentSpec) -> tuple[Path, Path, Path]:
        key = spec.cache_key
        return (
            self.cache_root / "images" / f"{key}.png",
            self.cache_root / "receipts" / f"{key}.json",
            self.cache_root / "locks" / f"{key}.lock",
        )

    def _cached(self, spec: OpponentSpec) -> OpponentResult | None:
        image, receipt, _lock = self._paths(spec)
        if not receipt.is_file():
            return None
        try:
            data = json.loads(receipt.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (
            data.get("provider_version") != PROVIDER_VERSION
            or data.get("model_alias") != self.model_alias
            or data.get("checkpoint_sha256") != spec.checkpoint_sha256
            or data.get("protocol_version") != spec.protocol_version
            or data.get("render_protocol") != spec.render_protocol
            or data.get("decoding_config") != spec.decoding_config
        ):
            return None
        status = data.get("status")
        if status == "invalid_model_output":
            return OpponentResult(status=status, cache_key=spec.cache_key, reason=data.get("reason"))
        if status != "valid" or not image.is_file() or image.is_symlink():
            return None
        actual = sha(image)
        if actual != data.get("sha256") or actual == spec.reference_sha256:
            return None
        return OpponentResult(status="valid", cache_key=spec.cache_key, path=image, sha256=actual)

    async def resolve(self, spec: OpponentSpec) -> OpponentResult:
        """Return a cached/generated result; never score an outage as a canvas."""
        if spec.model_alias != self.model_alias:
            raise ValueError("opponent spec alias does not match the frozen provider alias")
        for name, expected, actual in (
            ("checkpoint_sha256", self.checkpoint_sha256, spec.checkpoint_sha256),
            ("protocol_version", self.protocol_version, spec.protocol_version),
            ("render_protocol", self.render_protocol, spec.render_protocol),
            ("decoding_config", self.decoding_config, spec.decoding_config),
        ):
            if expected is not None and expected != actual:
                raise ValueError(f"opponent spec {name} does not match the frozen provider")
        cached = self._cached(spec)
        if cached is not None:
            return cached
        lock = await self._local_lock(spec.cache_key)
        async with lock:
            cached = self._cached(spec)
            if cached is not None:
                return cached
            return await self._resolve_single_flight(spec)

    async def _resolve_single_flight(self, spec: OpponentSpec) -> OpponentResult:
        image, receipt, lock_path = self._paths(spec)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+")
        try:
            if fcntl is not None:
                deadline = time.monotonic() + self.lock_timeout_seconds
                while True:
                    try:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            return OpponentResult("unavailable", spec.cache_key, reason="opponent lock timeout")
                        await asyncio.sleep(0.25)
            cached = self._cached(spec)
            if cached is not None:
                return cached
            if self.generator is None:
                return OpponentResult(
                    "unavailable",
                    spec.cache_key,
                    reason="frozen opponent generator is not configured",
                )
            try:
                generation_spec = replace(spec, work_root=self.cache_root / "work" / spec.cache_key)
                # The native one-case evaluator is synchronous and can spend
                # several minutes in vLLM/rendering.  Never block the event
                # loop that is serving other RL rollouts; custom async
                # generators remain awaitable directly.
                if inspect.iscoroutinefunction(self.generator):
                    produced = self.generator(generation_spec)
                else:
                    produced = await asyncio.to_thread(self.generator, generation_spec)
                if inspect.isawaitable(produced):
                    produced = await produced
                status, produced_path, reason = _result_from_value(produced)
            except OpponentInvalid as exc:
                status, produced_path, reason = "invalid_model_output", None, str(exc)
            except OpponentUnavailable as exc:
                return OpponentResult("unavailable", spec.cache_key, reason=str(exc))
            except Exception as exc:  # generator/service outage; do not cache it
                return OpponentResult("unavailable", spec.cache_key, reason=f"{type(exc).__name__}: {exc}")
            if status == "unavailable":
                return OpponentResult("unavailable", spec.cache_key, reason=reason)
            if status == "invalid_model_output":
                reason = reason or "frozen model produced no valid canvas"
                _json_write(receipt, {
                    "provider_version": PROVIDER_VERSION,
                    "status": "invalid_model_output",
                    "model_alias": self.model_alias,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                    "protocol_version": spec.protocol_version,
                    "render_protocol": spec.render_protocol,
                    "decoding_config": spec.decoding_config,
                    "source_group": spec.source_group,
                    "reference_sha256": spec.reference_sha256,
                    "seed": spec.seed,
                    "reason": reason,
                })
                return OpponentResult("invalid_model_output", spec.cache_key, reason=reason, generated=True)
            if produced_path is None:
                return OpponentResult(
                    "unavailable", spec.cache_key,
                    reason="generator advertised valid output without a canvas path",
                )
            # A generator path is an operational contract.  Missing,
            # symlinked, or malformed output means the evaluator/renderer
            # failed; cache nothing so a retry can recover it.
            if produced_path.is_symlink() or not produced_path.is_file():
                return OpponentResult(
                    "unavailable", spec.cache_key,
                    reason="frozen model output canvas is missing or a symlink",
                )
            produced_path = produced_path.resolve()
            if produced_path.read_bytes()[: len(PNG_MAGIC)] != PNG_MAGIC:
                return OpponentResult(
                    "unavailable", spec.cache_key,
                    reason="frozen model output is not a PNG canvas",
                )
            produced_sha = sha(produced_path)
            if produced_sha == spec.reference_sha256:
                reason = "frozen model output is byte-identical to the reference"
                _json_write(receipt, {
                    "provider_version": PROVIDER_VERSION,
                    "status": "invalid_model_output",
                    "model_alias": self.model_alias,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                    "protocol_version": spec.protocol_version,
                    "render_protocol": spec.render_protocol,
                    "decoding_config": spec.decoding_config,
                    "source_group": spec.source_group,
                    "reference_sha256": spec.reference_sha256,
                    "seed": spec.seed,
                    "reason": reason,
                })
                return OpponentResult("invalid_model_output", spec.cache_key, reason=reason, generated=True)
            image.parent.mkdir(parents=True, exist_ok=True)
            temporary = image.with_name(image.name + f".{os.getpid()}.tmp")
            shutil.copyfile(produced_path, temporary)
            temporary.replace(image)
            cached_sha = sha(image)
            _json_write(receipt, {
                "provider_version": PROVIDER_VERSION,
                "status": "valid",
                "model_alias": self.model_alias,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "protocol_version": spec.protocol_version,
                "render_protocol": spec.render_protocol,
                "decoding_config": spec.decoding_config,
                "source_group": spec.source_group,
                "reference_sha256": spec.reference_sha256,
                "seed": spec.seed,
                "path": str(image),
                "sha256": cached_sha,
            })
            return OpponentResult("valid", spec.cache_key, path=image, sha256=cached_sha, generated=True)
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            lock_handle.close()


def generate_frozen_opponent(spec: OpponentSpec) -> Path | Mapping[str, Any]:
    """Run one canonical native ``vf eval`` case against the frozen alias.

    The node launcher supplies the separate vLLM endpoint and renderer paths
    through environment variables.  The endpoint must already expose
    ``spec.model_alias`` (for this run, ``sft-step512``); this function never
    reloads or mutates an adapter.  A model/renderer result with no canvas is
    explicit ``invalid_model_output``.  Transport or renderer service errors
    raise :class:`OpponentUnavailable` so reward code cannot score an outage.
    """
    if sys.platform != "linux":
        raise OpponentUnavailable("frozen opponent generation is Linux-node only")
    if spec.work_root is None:
        raise OpponentUnavailable("frozen opponent work_root is not configured")
    # The run's frozen alias is served by the direct vLLM instance.  The local
    # router also exposes the alias, but its pinned request schema rejects the
    # VF default ``reasoning_effort=none`` before it reaches vLLM.  Keep this
    # operator-overridable for topologies with a compatible router.
    endpoint = os.environ.get("PAINTER_FROZEN_OPPONENT_ENDPOINT", "http://127.0.0.1:8100/v1")
    renderer = os.environ.get("PAINTER_FROZEN_OPPONENT_RENDERER")
    renderer_python = os.environ.get("PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON")
    if not renderer or not renderer_python:
        raise OpponentUnavailable("frozen opponent renderer paths are not configured")
    workspace_root = Path(
        os.environ.get("PAINTER_FROZEN_OPPONENT_ROOT") or Path(__file__).resolve().parents[1]
    ).resolve()
    prime_root = Path(os.environ.get("PAINTER_FROZEN_OPPONENT_PRIME_ROOT") or workspace_root / "prime-rl").resolve()
    eval_bin = Path(
        os.environ.get("PAINTER_FROZEN_OPPONENT_EVAL")
        or workspace_root / "prime-rl/.venv/bin/eval"
    ).resolve()
    if not eval_bin.is_file() or not prime_root.is_dir():
        raise OpponentUnavailable("native frozen-opponent eval runtime is not installed")
    config_value = os.environ.get("PAINTER_FROZEN_OPPONENT_CONFIG")
    config: dict[str, Any] = {}
    if config_value:
        try:
            config = json.loads(Path(config_value).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OpponentUnavailable(f"frozen opponent config is unreadable: {exc}") from exc
    limits = dict(config.get("operational_limits") or {})
    episode_timeout = int(limits.get("episode_timeout_seconds", 3600))
    render_timeout = int(limits.get("render_timeout_seconds", 180))
    work = Path(spec.work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # Each retry gets a fresh native-eval root.  A previous timeout or outage
    # must never leave a trace/PNG that a later attempt can mistake for the
    # current frozen response.
    attempt = work / f"attempt-{os.getpid()}-{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=True)
    case_id = f"frozen-{spec.cache_key[:20]}"
    manifest = attempt / "native-eval-manifest.json"
    manifest.write_text(json.dumps({
        "cases": [{
            "id": case_id,
            "source_group": spec.source_group,
            "family": "frozen-opponent",
            "reference": str(spec.reference),
            "reference_sha256": spec.reference_sha256,
            "max_turns": spec.max_turns,
            "renderer": str(Path(renderer).resolve()),
            "renderer_python": str(_absolute_preserving_symlink(renderer_python)),
        }]
    }, indent=2, sort_keys=True) + "\n")
    try:
        from painter.native_painting_eval import config_text
    except ImportError:
        painter_dir = str(Path(__file__).resolve().parent)
        if painter_dir not in sys.path:
            sys.path.insert(0, painter_dir)
        from native_painting_eval import config_text  # type: ignore
    config_path = attempt / "native-eval.toml"
    generated = config_text(
        taskset_id="native_painting",
        root=attempt,
        manifest=manifest,
        policy_label="frozen",
        model=spec.model_alias,
        output_dir=attempt / "eval-output",
        count=1,
        max_tokens=spec.max_completion_tokens,
        max_turns=spec.max_turns,
        rollout_timeout=episode_timeout,
        context_length=spec.context_length,
    )
    generated = generated.replace(
        'base_url = "http://127.0.0.1:8000/v1"',
        f"base_url = {json.dumps(endpoint)}",
    ).replace("seed = 918200", f"seed = {spec.seed}")
    # NativePaintingTaskset takes renderer settings from its taskset config
    # (manifest fields are deliberately metadata-only), so make the node
    # renderer explicit in this isolated one-case config.
    taskset_marker = "[env.taskset]\n"
    renderer_config = (
        taskset_marker
        + f"renderer = {json.dumps(str(Path(renderer).resolve()))}\n"
        + f"renderer_python = {json.dumps(str(_absolute_preserving_symlink(renderer_python)))}\n"
    )
    if taskset_marker not in generated:
        raise OpponentUnavailable("native frozen-opponent config has no env.taskset section")
    generated = generated.replace(taskset_marker, renderer_config, 1)
    config_path.write_text(generated)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(workspace_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("WANDB_MODE", "disabled")
    try:
        completed = subprocess.run(
            [str(eval_bin), "@", str(config_path), "--no-serve", "--no-push"],
            cwd=prime_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=episode_timeout + render_timeout + 300,
        )
        _write_private_log(attempt / "native-eval.stdout.log", completed.stdout)
        _write_private_log(attempt / "native-eval.stderr.log", completed.stderr)
    except subprocess.TimeoutExpired as exc:
        _write_private_log(attempt / "native-eval.stdout.log", exc.stdout)
        _write_private_log(attempt / "native-eval.stderr.log", exc.stderr)
        raise OpponentUnavailable("frozen opponent native eval timed out") from exc
    except Exception as exc:
        _write_private_log(attempt / "native-eval.exception.log", f"{type(exc).__name__}: {exc}")
        raise OpponentUnavailable(f"frozen opponent eval unavailable: {type(exc).__name__}: {exc}") from exc
    diagnostics = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    transport_markers = (
        "connection refused",
        "connecterror",
        "connection error",
        "server disconnected",
        "service unavailable",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "status code 500",
        "status code 502",
        "status code 503",
        "status code 504",
    )
    if any(marker in diagnostics for marker in transport_markers):
        raise OpponentUnavailable("native eval diagnostics indicate HTTP/service unavailability")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native eval failed").strip()[-1200:]
        raise OpponentUnavailable(f"frozen opponent native eval failed: {detail}")
    return _native_eval_canvas(attempt, case_id, workspace_root)


__all__ = [
    "FrozenOpponentProvider",
    "OpponentInvalid",
    "OpponentResult",
    "OpponentSpec",
    "OpponentUnavailable",
    "PROVIDER_VERSION",
    "deterministic_seed",
    "generate_frozen_opponent",
]
