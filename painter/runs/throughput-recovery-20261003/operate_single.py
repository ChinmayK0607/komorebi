#!/usr/bin/env python3
"""Operate one finite recovery run on an already rented Lium GPU pod.

The controller owns setup, transfer, bounded foreground execution, evidence
polling, and exact-pod release.  It never creates a lease, starts a watcher, or
stores provider connection details.  The remote run performs its own reviewed
finite archive/public verification before the controller releases the pod.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence


RUN_NAME = "throughput-recovery-20261003"
EXPERIMENT_NAME = "throughput-recovery-flash-20261003"
REMOTE_ROOT = Path("/root/painter")
REMOTE_REPO = REMOTE_ROOT / "repo"
REMOTE_RUN = REMOTE_REPO / "painter/runs/throughput-recovery-20261003"
REMOTE_TOKEN = REMOTE_ROOT / "private/hf-token"
REMOTE_GATEWAY = REMOTE_ROOT / "private/gateway.env"
REMOTE_BUNDLE = REMOTE_ROOT / "source.bundle"
POD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
MIN_GPU_MEMORY_MIB = 80_000
GPU_COUNT = 2
TRAIN_GPU_COUNT = 1
INFER_GPU_COUNT = 1
LIUM_PYTHON = Path("/Users/chinmay/.local/share/uv/tools/lium-io/bin/python")
POLL_SECONDS = 30.0
SSH_TIMEOUT_SECONDS = 60.0
TRANSFER_TIMEOUT_SECONDS = 600.0
RELEASE_TIMEOUT_SECONDS = 90.0
RELEASE_MARGIN_SECONDS = 180.0
INVENTORY_TIMEOUT_SECONDS = 60.0
DIAGNOSTIC_BYTES = 65_536
DIAGNOSTIC_LINES = 8


class OperationError(RuntimeError):
    """A bounded setup, transport, validation, or release failure."""


@dataclass(frozen=True)
class RentalWindow:
    pod_id: str
    rental_start_epoch: float
    price_usd_per_hour: float
    cap_usd: float
    release_margin_seconds: float = RELEASE_MARGIN_SECONDS

    @property
    def rental_deadline_epoch(self) -> float:
        return self.rental_start_epoch + self.cap_usd / self.price_usd_per_hour * 3600.0

    @property
    def release_deadline_epoch(self) -> float:
        return self.rental_deadline_epoch - self.release_margin_seconds


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def make_window(*, pod_id: str, rental_start_epoch: float, price_usd_per_hour: float, cap_usd: float) -> RentalWindow:
    if not isinstance(pod_id, str) or not POD_ID_RE.fullmatch(pod_id) or pod_id.lower() in {"all", "*"}:
        raise ValueError("pod-id must be one exact provider ID")
    start = _positive_finite("rental-start-epoch", rental_start_epoch)
    price = _positive_finite("price-usd-hour", price_usd_per_hour)
    cap = _positive_finite("cap-usd", cap_usd)
    if cap > 20.0:
        raise ValueError("cap-usd must be at most 20")
    window = RentalWindow(pod_id, start, price, cap)
    if window.release_deadline_epoch <= start:
        raise ValueError("release margin leaves no finite operating window")
    return window


def _regular_file(path: Path, label: str, *, owner_only: bool = False) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if owner_only and path.stat().st_mode & 0o077:
        raise ValueError(f"{label} must be owner-only")
    return path.resolve()


def default_hf_token_file() -> Path:
    return Path.home() / ".cache" / "huggingface" / "token"


def default_gateway_env_file() -> Path:
    return Path(__file__).resolve().parents[3] / "painter/benchmarks/openrouter-teachers-20260922/.env.local"


def _bounded_call(call: Callable[[], Any], timeout: float) -> Any:
    if timeout <= 0:
        raise OperationError("operation deadline has elapsed")
    result: list[Any] = []
    failure: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(call())
        except BaseException as exc:  # surfaced in the caller
            failure.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise OperationError("bounded operation timed out")
    if failure:
        raise failure[0]
    return result[0] if result else None


def _load_lium() -> Any:
    try:
        from lium.sdk import Lium
    except ImportError:
        # The provider SDK is installed in the dedicated uv tool runtime.  A
        # caller may invoke this file with the system interpreter, so add only
        # that known site-packages directory before retrying the import.
        site_packages = next(LIUM_PYTHON.parent.parent.glob("lib/python*/site-packages"), None)
        if site_packages is None:
            raise OperationError("lium SDK is unavailable")
        sys.path.insert(0, str(site_packages))
        try:
            from lium.sdk import Lium
        except ImportError as exc:  # pragma: no cover - parent environment supplies SDK
            raise OperationError("lium SDK is unavailable") from exc
    return Lium()


def _pod_id(pod: Any) -> str:
    value = getattr(pod, "id", None)
    return str(value) if value is not None else ""


def resolve_exact_pod(api: Any, pod_id: str) -> Any:
    rows = api.ps()
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, dict)):
        raise OperationError("provider inventory is malformed")
    matches = [pod for pod in rows if _pod_id(pod) == pod_id]
    if len(matches) != 1:
        raise OperationError("provider inventory did not contain exactly one requested pod")
    return matches[0]


def _remaining(window: RentalWindow, now: Callable[[], float]) -> float:
    remaining = window.release_deadline_epoch - now()
    if remaining <= 0:
        raise OperationError("finite rental release deadline reached")
    return remaining


def _append_tail(buffer: bytearray, data: bytes, limit: int = DIAGNOSTIC_BYTES) -> None:
    buffer.extend(data)
    if len(buffer) > limit:
        del buffer[:-limit]


def _sanitized_diagnostics(
    *values: str,
    forbidden: Iterable[str] = (),
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Keep per-stream failure tails while removing secrets and connection IDs.

    Each input is trimmed independently.  Keeping the stderr tail separate
    from a verbose stdout stream prevents a useful bootstrap error from being
    discarded when the remote command emits many progress lines.
    """
    patterns = (
        (re.compile(r"(?i)(?:bearer\s+|hf_)[A-Za-z0-9._:-]+"), "<redacted-token>"),
        (re.compile(r"\bvck_[A-Za-z0-9._:-]+\b"), "<redacted-token>"),
        (re.compile(r"\bsk-or-v1-[A-Za-z0-9._:-]+\b"), "<redacted-token>"),
        (re.compile(r"(?i)([A-Za-z0-9_-]*(?:key|token|password|secret)[A-Za-z0-9_-]*)\s*[=:]\s*[^\s,;]+"), r"\1=<redacted>"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<redacted-address>"),
    )
    blocked = tuple(item for item in forbidden if item)
    lines: list[str] = []
    if labels is not None and len(labels) != len(values):
        raise ValueError("diagnostic labels must match diagnostic streams")
    for index, raw in enumerate(values):
        stream_lines: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            for item in blocked:
                line = line.replace(item, "<redacted-id>")
            for pattern, replacement in patterns:
                line = pattern.sub(replacement, line)
            line = "".join(character if character.isprintable() else " " for character in line)
            stream_lines.append(line[:320])
        prefix = f"{labels[index]}: " if labels is not None else ""
        lines.extend(prefix + line for line in stream_lines[-DIAGNOSTIC_LINES:])
    return lines


def _remote_exec_capture(ssh: Any, command: str, *, timeout: float, now: Callable[[], float], window: RentalWindow) -> tuple[int, str, str]:
    _remaining(window, now)
    try:
        _stdin, stdout, _stderr = ssh.exec_command(command, timeout=min(timeout, SSH_TIMEOUT_SECONDS))
    except Exception as exc:
        raise OperationError("SSH command could not be started") from exc
    channel = stdout.channel
    deadline = time.monotonic() + timeout
    stdout_tail = bytearray()
    stderr_tail = bytearray()
    while not channel.exit_status_ready():
        if time.monotonic() >= deadline or window.release_deadline_epoch <= now():
            try:
                channel.close()
            finally:
                raise OperationError("SSH command exceeded its finite deadline")
        if channel.recv_ready():
            _append_tail(stdout_tail, channel.recv(65536))
        if channel.recv_stderr_ready():
            _append_tail(stderr_tail, channel.recv_stderr(65536))
        time.sleep(0.1)
    while channel.recv_ready():
        _append_tail(stdout_tail, channel.recv(65536))
    while channel.recv_stderr_ready():
        _append_tail(stderr_tail, channel.recv_stderr(65536))
    return (
        int(channel.recv_exit_status()),
        bytes(stdout_tail).decode("utf-8", "replace"),
        bytes(stderr_tail).decode("utf-8", "replace"),
    )


def _remote_exec(ssh: Any, command: str, *, timeout: float, now: Callable[[], float], window: RentalWindow) -> str:
    code, output, error = _remote_exec_capture(ssh, command, timeout=timeout, now=now, window=window)
    if code != 0:
        diagnostics = _sanitized_diagnostics(error, output, labels=("stderr", "stdout"))
        detail = "; ".join(diagnostics)
        raise OperationError("remote command failed" + (f": {detail}" if detail else ""))
    return output


def _sftp_put(sftp: Any, local: Path, remote: str, *, timeout: float) -> None:
    _bounded_call(lambda: sftp.put(str(local), remote), timeout)
    _bounded_call(lambda: sftp.chmod(remote, 0o600), timeout)


def _sftp_read(sftp: Any, remote: str, *, timeout: float) -> bytes | None:
    def read() -> bytes:
        with sftp.open(remote, "rb") as stream:
            try:
                stream.seek(0, 2)
                stream.seek(max(0, int(stream.tell()) - 1_048_576))
            except (AttributeError, OSError, IOError):
                pass
            return stream.read()

    try:
        return _bounded_call(read, timeout)
    except (OSError, IOError):
        return None


def _source_checkout_command(source_bundle: str, repo: str, source_commit: str) -> str:
    """Replay a bundle whose advertised branch is not the remote HEAD."""
    bundle = shlex.quote(source_bundle)
    checkout = shlex.quote(source_commit)
    target = shlex.quote(repo)
    return (
        "test ! -e {target}; git clone --quiet --no-checkout {bundle} {target}; "
        "git -C {target} checkout --quiet --detach {commit}; "
        "test \"$(git -C {target} rev-parse HEAD)\" = {commit}; "
        "test \"$(git -C {target} rev-parse --is-inside-work-tree)\" = true"
    ).format(target=target, bundle=bundle, commit=checkout)


def _json_bytes(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _gpu_memory(raw: str) -> list[int]:
    values: list[int] = []
    for line in raw.splitlines():
        value = line.strip().split(",", 1)[0]
        if value.isdigit():
            values.append(int(value))
    return values


def _queue_summary(raw: bytes | None) -> dict[str, float | int] | None:
    if raw is None:
        return None
    queue: list[float] = []
    active: list[float] = []
    for line in raw.decode("utf-8", "replace").splitlines()[-200:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for target, key in ((queue, "render_queue_seconds"), (active, "render_active_seconds")):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0:
                target.append(float(value))
    if not queue and not active:
        return None
    values: dict[str, float | int] = {"samples": max(len(queue), len(active))}
    if queue:
        values.update({"queue_mean_seconds": sum(queue) / len(queue), "queue_max_seconds": max(queue)})
    if active:
        values.update({"active_mean_seconds": sum(active) / len(active), "active_max_seconds": max(active)})
    return values


def _safe_event(path: Path, event: str, started: float, **fields: Any) -> None:
    payload = {"schema": "throughput-recovery-single-operation-v1", "event": event, "elapsed_seconds": max(0.0, time.time() - started)}
    payload.update({key: value for key, value in fields.items() if key not in {"pod", "pod_id", "host", "address", "ip", "command"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
    print(json.dumps(payload, sort_keys=True), flush=True)


def _remote_summary(sftp: Any, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout))

    def read(remote: str) -> bytes | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OperationError("status observation budget exhausted")
        return _sftp_read(sftp, remote, timeout=remaining)

    state = _json_bytes(read(str(REMOTE_RUN / "supervisor-state.json"))) or {}
    receipt = _json_bytes(read(str(REMOTE_RUN / "terminal-receipt.json"))) or {}
    preparation_raw = read(str(REMOTE_RUN / "preparation-receipt.json"))
    preparation = _json_bytes(preparation_raw)
    config_raw = read(str(REMOTE_RUN / "rl-config-receipt.json"))
    config = _json_bytes(config_raw)
    if config_raw is None:
        config_summary: dict[str, Any] = {}
    elif config is None:
        config_summary = {"_malformed": True}
    else:
        config_summary = {key: config.get(key) for key in ("profile", "checkpoint_interval", "batch_size", "group_size", "initial_inflight", "max_inflight", "activation_cpu_offload", "deployment", "reward")}
    preparation_summary = {} if preparation is None else {
        key: preparation.get(key)
        for key in ("schema", "profile", "config_sha256", "train_manifest_sha256")
    }
    metrics = read(str(REMOTE_RUN / f"async-output/{RUN_NAME}/monitors/file/metrics.jsonl"))
    return {
        "state": {key: state.get(key) for key in ("status", "reason", "completed_optimizer_steps", "last_optimizer_step", "last_admitted_batch", "max_updates")},
        "terminal": {key: receipt.get(key) for key in ("status", "pipeline_exit_code", "archive_exit_code", "public_archive_hash_verified", "checkpoint_step")},
        "config": config_summary,
        "preparation": preparation_summary,
        "phase_ready": preparation is not None and preparation.get("schema") == "throughput-recovery-single-arm-preparation-v1",
        "render": _queue_summary(metrics),
    }


def _observe_summary(
    ssh: Any,
    sftp: Any | None,
    *,
    timeout: float,
    now: Callable[[], float],
    window: RentalWindow,
    events: Path,
    started: float,
    phase: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Read status without making a transient SFTP failure fatal.

    The SSH command channel remains the source of truth for execution and its
    finite deadline.  Status reads get one bounded retry on a fresh SFTP
    session; a failed observation is recorded and the command loop continues.
    """
    try:
        observation_budget = min(float(timeout), _remaining(window, now))
        observation_deadline = time.monotonic() + observation_budget
    except BaseException as exc:
        _safe_event(
            events,
            "observation_error",
            started,
            phase=phase,
            attempt=0,
            failure_diagnostics=_sanitized_diagnostics(str(exc), labels=("controller",)),
        )
        return sftp, None

    def remaining_budget() -> float:
        hard_remaining = _remaining(window, now)
        remaining = min(observation_deadline - time.monotonic(), hard_remaining)
        if remaining <= 0:
            raise OperationError("status observation budget exhausted")
        return remaining

    def close_quickly(handle: Any) -> None:
        """Close a timed-out SFTP channel without consuming its read budget."""
        try:
            hard_remaining = window.release_deadline_epoch - now()
            if hard_remaining > 0:
                _bounded_call(handle.close, min(5.0, hard_remaining))
        except BaseException:
            pass

    first_error: BaseException | None = None
    if sftp is not None:
        try:
            # Reserve half of this observation window for a fresh SFTP retry
            # if one read blocks.  A single status file must not consume the
            # entire poll deadline and strand the replacement attempt.
            return sftp, _remote_summary(sftp, timeout=min(remaining_budget(), observation_budget / 2))
        except BaseException as exc:
            first_error = exc
            _safe_event(
                events,
                "observation_error",
                started,
                phase=phase,
                attempt=1,
                failure_diagnostics=_sanitized_diagnostics(str(exc), labels=("controller",)),
            )
            close_quickly(sftp)

    try:
        fresh = _bounded_call(
            ssh.open_sftp,
            min(SSH_TIMEOUT_SECONDS, remaining_budget()),
        )
    except BaseException as exc:
        _safe_event(
            events,
            "observation_error",
            started,
            phase=phase,
            attempt=2,
            failure_diagnostics=_sanitized_diagnostics(str(exc), labels=("controller",)),
        )
        return None, None
    try:
        summary = _remote_summary(fresh, timeout=remaining_budget())
    except BaseException as exc:
        _safe_event(
            events,
            "observation_error",
            started,
            phase=phase,
            attempt=2,
            failure_diagnostics=_sanitized_diagnostics(str(exc), labels=("controller",)),
        )
        close_quickly(fresh)
        return None, None
    if first_error is not None:
        _safe_event(events, "observation_recovered", started, phase=phase)
    return fresh, summary


def _validate_config_summary(summary: dict[str, Any]) -> None:
    preparation = summary.get("preparation", {})
    if preparation.get("schema") != "throughput-recovery-single-arm-preparation-v1":
        raise OperationError("remote preparation is not profile-ready")
    if preparation.get("profile") != "rl-throughput-recovery-v1":
        raise OperationError("remote preparation did not use the recovery profile")
    config = summary.get("config", {})
    if any(config.get(key) != expected for key, expected in (("checkpoint_interval", 1), ("batch_size", 16), ("group_size", 4), ("initial_inflight", 32), ("max_inflight", 32))):
        raise OperationError("remote config shape does not match the single-arm profile")
    deployment = config.get("deployment") or {}
    if deployment != {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1}:
        raise OperationError("remote config topology is not 2 GPUs with one trainer and one inference GPU")


def operate(*, window: RentalWindow, source_bundle: Path, source_commit: str, hf_token_file: Path, gateway_env_file: Path, output_root: Path, api: Any | None = None, now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> int:
    source_bundle = _regular_file(source_bundle, "source-bundle")
    hf_token_file = _regular_file(hf_token_file, "HF token file", owner_only=True)
    gateway_env_file = _regular_file(gateway_env_file, "Gateway environment file", owner_only=True)
    if not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source-commit must be a full immutable Git commit")
    output_root = output_root.expanduser()
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("output-root must not be a symlink")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)
    events = output_root / "operations-single.jsonl"
    started = now()
    _safe_event(events, "started", started, run_name=RUN_NAME, experiment_name=EXPERIMENT_NAME, source_commit=source_commit, gpu_count=GPU_COUNT, gpu_memory_min_mib=MIN_GPU_MEMORY_MIB)

    api = api or _load_lium()
    # Keep an exact-id release target even if inventory or SSH setup fails
    # before a provider object can be resolved.  Lium's rm operation only
    # needs the object id, and this prevents an inventory timeout from leaking
    # the already-rented pod.
    release_target: Any = SimpleNamespace(id=window.pod_id)
    pod: Any | None = None
    ssh = None
    sftp = None
    channel_error: BaseException | None = None
    command_code: int | None = None
    remote_diagnostics: list[str] = []
    remote_stdout_tail = bytearray()
    remote_stderr_tail = bytearray()
    try:
        _remaining(window, now)
        pod = _bounded_call(
            lambda: resolve_exact_pod(api, window.pod_id),
            min(INVENTORY_TIMEOUT_SECONDS, _remaining(window, now)),
        )
        release_target = pod
        _safe_event(events, "pod_verified", started, topology={"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1})
        with api.ssh_connection(pod, timeout=min(30.0, _remaining(window, now))) as ssh:
            _remote_exec(ssh, "mkdir -p /root/painter/private /root/painter/logs && chmod 700 /root/painter/private", timeout=SSH_TIMEOUT_SECONDS, now=now, window=window)
            sftp = ssh.open_sftp()
            _sftp_put(sftp, source_bundle, str(REMOTE_BUNDLE), timeout=min(TRANSFER_TIMEOUT_SECONDS, _remaining(window, now)))
            _sftp_put(sftp, hf_token_file, str(REMOTE_TOKEN), timeout=min(TRANSFER_TIMEOUT_SECONDS, _remaining(window, now)))
            _sftp_put(sftp, gateway_env_file, str(REMOTE_GATEWAY), timeout=min(TRANSFER_TIMEOUT_SECONDS, _remaining(window, now)))
            _safe_event(events, "inputs_transferred", started, source_bundle_bytes=source_bundle.stat().st_size)

            verify = "set -Eeuo pipefail; %s; nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits" % _source_checkout_command(str(REMOTE_BUNDLE), str(REMOTE_REPO), source_commit)
            verify_code, gpu_raw, verify_error = _remote_exec_capture(ssh, verify, timeout=min(TRANSFER_TIMEOUT_SECONDS, _remaining(window, now)), now=now, window=window)
            if verify_code != 0:
                remote_diagnostics = _sanitized_diagnostics(
                    verify_error,
                    gpu_raw,
                    forbidden=(window.pod_id,),
                    labels=("stderr", "stdout"),
                )
                detail = "; ".join(remote_diagnostics)
                raise OperationError("source or GPU verification failed" + (f": {detail}" if detail else ""))
            memories = _gpu_memory(gpu_raw)
            if len(memories) != GPU_COUNT or min(memories, default=0) < MIN_GPU_MEMORY_MIB:
                raise OperationError("remote GPU inventory is not exactly two 80GB-class devices")
            _safe_event(events, "source_and_gpu_verified", started, source_commit=source_commit, gpu_count=len(memories), gpu_memory_min_mib=min(memories))

            command = "exec env PAINTER_SOURCE_COMMIT=%s PAINTER_NODE_ROOT=/root/painter HF_TOKEN_PATH=/root/painter/private/hf-token PAINTER_GATEWAY_ENV_FILE=/root/painter/private/gateway.env PAINTER_GPUS_PER_NODE=2 PAINTER_NUM_TRAIN_GPUS=1 PAINTER_NUM_INFER_GPUS=1 bash /root/painter/repo/painter/runs/throughput-recovery-20261003/execute_single.sh %s %s %s" % (
                shlex.quote(source_commit),
                shlex.quote(str(window.rental_start_epoch)),
                shlex.quote(str(window.price_usd_per_hour)),
                shlex.quote(str(window.cap_usd)),
            )
            _stdin, stdout, _stderr = ssh.exec_command(command, timeout=SSH_TIMEOUT_SECONDS)
            channel = stdout.channel
            remote_stdout_tail = bytearray()
            remote_stderr_tail = bytearray()
            _safe_event(events, "remote_execute_started", started)
            checked_config = False
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    _append_tail(remote_stdout_tail, channel.recv(65536))
                if channel.recv_stderr_ready():
                    _append_tail(remote_stderr_tail, channel.recv_stderr(65536))
                current = now()
                if current >= window.release_deadline_epoch:
                    try:
                        channel.close()
                    finally:
                        raise OperationError("remote execute reached the finite release deadline")
                sftp, summary = _observe_summary(
                    ssh,
                    sftp,
                    timeout=min(SSH_TIMEOUT_SECONDS, window.release_deadline_epoch - current),
                    now=now,
                    window=window,
                    events=events,
                    started=started,
                    phase="poll",
                )
                if not checked_config and summary is not None and summary.get("phase_ready"):
                    _validate_config_summary(summary)
                    checked_config = True
                    _safe_event(
                        events,
                        "profile_verified",
                        started,
                        profile=summary.get("preparation", {}).get("profile"),
                        topology=summary.get("config", {}).get("deployment"),
                    )
                state = summary.get("state", {}) if summary is not None else {}
                _safe_event(events, "poll", started, observation_available=summary is not None, state_status=state.get("status"), state_reason=state.get("reason"), completed_optimizer_steps=state.get("completed_optimizer_steps"), last_admitted_batch=state.get("last_admitted_batch"), render=summary.get("render") if summary is not None else None)
                sleep(min(POLL_SECONDS, max(0.0, window.release_deadline_epoch - now())))
            while channel.recv_ready():
                _append_tail(remote_stdout_tail, channel.recv(65536))
            while channel.recv_stderr_ready():
                _append_tail(remote_stderr_tail, channel.recv_stderr(65536))
            command_code = int(channel.recv_exit_status())
            final_remaining = window.release_deadline_epoch - now()
            if final_remaining > 0:
                sftp, summary = _observe_summary(
                    ssh,
                    sftp,
                    timeout=min(SSH_TIMEOUT_SECONDS, final_remaining),
                    now=now,
                    window=window,
                    events=events,
                    started=started,
                    phase="final",
                )
                summary = summary or {}
            else:
                summary = {}
            if command_code != 0:
                remote_diagnostics = _sanitized_diagnostics(
                    bytes(remote_stderr_tail).decode("utf-8", "replace"),
                    bytes(remote_stdout_tail).decode("utf-8", "replace"),
                    forbidden=(window.pod_id,),
                    labels=("stderr", "stdout"),
                )
            finished_fields = {
                "exit_code": command_code,
                "state": summary.get("state"),
                "terminal": summary.get("terminal"),
                "render": summary.get("render"),
            }
            if remote_diagnostics:
                finished_fields["failure_diagnostics"] = remote_diagnostics
            _safe_event(events, "remote_execute_finished", started, **finished_fields)
    except BaseException as exc:
        channel_error = exc
        diagnostics = remote_diagnostics or _sanitized_diagnostics(
            bytes(remote_stderr_tail).decode("utf-8", "replace"),
            bytes(remote_stdout_tail).decode("utf-8", "replace"),
            str(exc),
            forbidden=(window.pod_id,),
            labels=("stderr", "stdout", "controller"),
        )
        _safe_event(events, "operation_error", started, error_type=type(exc).__name__, failure_diagnostics=diagnostics)
    finally:
        if sftp is not None:
            try:
                _bounded_call(sftp.close, min(SSH_TIMEOUT_SECONDS, max(1.0, window.release_deadline_epoch - now())))
            except BaseException:
                pass
        if ssh is not None:
            try:
                _bounded_call(ssh.close, min(SSH_TIMEOUT_SECONDS, max(1.0, window.release_deadline_epoch - now())))
            except BaseException:
                pass
        # Release is deliberately in the outer finally: SSH failures, source
        # mismatches, deadline stops, and archive failures all remove only the
        # exact object selected before setup.  Provider output is discarded.
        release_error: BaseException | None = None
        try:
            _bounded_call(lambda: api.rm(release_target), RELEASE_TIMEOUT_SECONDS)
            _safe_event(events, "pod_release_requested", started)
        except BaseException as exc:
            release_error = exc
            _safe_event(events, "pod_release_error", started, error_type=type(exc).__name__)
        if release_error is not None and channel_error is None:
            channel_error = OperationError("exact pod release failed")

    if channel_error is not None:
        raise channel_error
    return command_code or 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pod-id", required=True)
    value.add_argument("--rental-start-epoch", type=float, required=True)
    value.add_argument("--price-usd-hour", type=float, required=True)
    value.add_argument("--cap-usd", type=float, required=True)
    value.add_argument("--source-bundle", type=Path, required=True)
    value.add_argument("--source-commit", required=True)
    value.add_argument("--hf-token-file", type=Path, default=default_hf_token_file())
    value.add_argument("--gateway-env-file", type=Path, default=default_gateway_env_file())
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--plan-only", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    window = make_window(
        pod_id=args.pod_id,
        rental_start_epoch=args.rental_start_epoch,
        price_usd_per_hour=args.price_usd_hour,
        cap_usd=args.cap_usd,
    )
    _regular_file(args.source_bundle, "source-bundle")
    _regular_file(args.hf_token_file, "HF token file", owner_only=True)
    _regular_file(args.gateway_env_file, "Gateway environment file", owner_only=True)
    if not COMMIT_RE.fullmatch(args.source_commit):
        raise SystemExit("source-commit must be a full immutable Git commit")
    if args.plan_only:
        print(json.dumps({
            "schema": "throughput-recovery-single-operation-plan-v1",
            "run_name": RUN_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "gpu_cap_usd": window.cap_usd,
            "topology": {"gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
            "max_updates": 4,
            "judge": "xiaomi/mimo-v2.6-flash",
        }, sort_keys=True))
        return 0
    return operate(
        window=window,
        source_bundle=args.source_bundle,
        source_commit=args.source_commit,
        hf_token_file=args.hf_token_file,
        gateway_env_file=args.gateway_env_file,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OperationError, OSError, ValueError) as exc:
        print(f"single operation: {exc}", file=sys.stderr)
        raise SystemExit(2)
