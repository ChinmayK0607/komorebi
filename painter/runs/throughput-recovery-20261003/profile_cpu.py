#!/usr/bin/env python3
"""Bounded Linux CPU measurements for the renderer recovery derivative.

The profiler compares the frozen centroid runtime with the opt-in cancellable
renderer on the same saved source, seed, deadline, and explicit Linux ANGLE
SwiftShader backend.  The vendored renderer accepts ``swiftshader`` on Linux;
the profiler refuses to substitute another backend.
It performs two serial repeats, a four-process derivative batch, and one
parent-stop lifecycle case.  Every renderer invocation gets its own output
directory and process group.  This module never generates code, scores a
painting, calls a service, or changes a saved source program.

The expected input bundle contains ``cpu-controls.json`` (``manifest.json``
is accepted for older staging) with schema
``cpu-renderer-controls-v1`` and rows such as::

    {"id": "control-01", "program_path": "control-01.js",
     "program_sha256": "...", "kind": "valid", "saved_valid": true,
     "saved_elapsed_seconds": 6.2, "source_archive_revision": "...",
     "source_archive_sha256": "...", "source_archive_member": "..."}

The runtime roots are supplied separately because the public input bundle can
be copied independently of the verified frozen+centroid and cancellable
runtime artifacts.  A baseline root normally contains ``renderer.py`` and a
derivative overlay contains ``cancellable_watercolour_renderer.py``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "cpu-renderer-profile-v1"
INPUT_SCHEMA = "cpu-renderer-controls-v1"
PROFILE_VERSION = "centroid-vs-cancellable-swiftshader-v1"
BACKEND = "swiftshader"
CANVAS = (600, 600)
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_TOTAL_SECONDS = 900.0
DEFAULT_REPEATS = 2
DEFAULT_POOL_WIDTH = 4
DEFAULT_CANCEL_AFTER_SECONDS = 15.0
DEFAULT_CLEANUP_GRACE_SECONDS = 5.0
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class ProfileError(ValueError):
    """Raised for an unsafe or mismatched benchmark request."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


RUNNER_SHA256 = sha256(Path(__file__))


@dataclass(frozen=True)
class Control:
    id: str
    source: Path
    source_relpath: str
    source_sha256: str
    seed: int
    kind: str
    saved_valid: bool | None
    saved_elapsed_seconds: float | None
    source_archive_revision: str | None
    source_archive_sha256: str | None
    source_archive_member: str | None


@dataclass(frozen=True)
class RuntimeSpec:
    baseline: Path
    derivative: Path
    baseline_root: Path
    derivative_root: Path
    baseline_sha256: str
    derivative_sha256: str
    assets_sha256: Mapping[str, str]
    baseline_patch_manifest_sha256: str | None
    derivative_patch_manifest_sha256: str | None


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be a finite number")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ProfileError(f"{name} must be a finite number")
    if number < 0 or (positive and number <= 0):
        raise ProfileError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return number


def _safe_relative(root: Path, value: str, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ProfileError(f"{label} must be a relative path")
    candidate = (root / value).resolve()
    try:
        relative = candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ProfileError(f"{label} escapes the input bundle") from error
    return candidate, relative.as_posix()


def load_controls(input_root: Path, manifest_name: str = "cpu-controls.json") -> tuple[dict[str, Any], list[Control], Path]:
    """Load and hash-check the public control bundle without changing it."""

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise ProfileError(f"Input root does not exist: {root}")
    manifest_path = root / manifest_name
    if not manifest_path.is_file() and manifest_name == "cpu-controls.json":
        # Older staged bundles used the generic name; accepting it keeps the
        # loader deterministic without searching arbitrary files.
        fallback = root / "manifest.json"
        if fallback.is_file():
            manifest_path = fallback
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"Cannot read input manifest: {error}") from error
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ProfileError(f"Expected {INPUT_SCHEMA} manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ProfileError("Input manifest must contain a non-empty rows list")
    default_seed = manifest.get("seed", 0)
    if isinstance(default_seed, bool) or not isinstance(default_seed, int):
        raise ProfileError("Manifest seed must be an integer")

    controls: list[Control] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProfileError(f"Row {index} is not an object")
        case_id = str(raw.get("id", ""))
        if not SAFE_ID.fullmatch(case_id):
            raise ProfileError(f"Row {index} has an unsafe id")
        if case_id in seen:
            raise ProfileError(f"Duplicate control id: {case_id}")
        seen.add(case_id)
        kind = raw.get("kind")
        if kind not in {"valid", "invalid", "slow_cancel"}:
            raise ProfileError(f"{case_id} has unsupported kind: {kind!r}")
        source, source_relpath = _safe_relative(root, raw.get("program_path", ""), f"{case_id}.program_path")
        expected = raw.get("program_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ProfileError(f"{case_id} is missing a valid program_sha256")
        if not source.is_file() or sha256(source) != expected.lower():
            raise ProfileError(f"Program hash mismatch or missing for {case_id}")
        seed = raw.get("seed", default_seed)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ProfileError(f"{case_id}.seed must be an integer")
        saved_valid = raw.get("saved_valid")
        if saved_valid is not None and not isinstance(saved_valid, bool):
            raise ProfileError(f"{case_id}.saved_valid must be boolean when present")
        saved_elapsed = raw.get("saved_elapsed_seconds")
        if saved_elapsed is not None:
            saved_elapsed = _finite_number(saved_elapsed, f"{case_id}.saved_elapsed_seconds")
        controls.append(Control(
            id=case_id,
            source=source,
            source_relpath=source_relpath,
            source_sha256=expected.lower(),
            seed=seed,
            kind=kind,
            saved_valid=saved_valid,
            saved_elapsed_seconds=saved_elapsed,
            source_archive_revision=_optional_string(raw.get("source_archive_revision")),
            source_archive_sha256=_optional_string(raw.get("source_archive_sha256")),
            source_archive_member=_optional_string(raw.get("source_archive_member")),
        ))
    return manifest, controls, manifest_path


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _renderer_candidate(root: Path, preferred: str | None, names: Sequence[str], label: str) -> Path:
    if preferred:
        candidate = Path(preferred)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
    else:
        candidate = next((root / name for name in names if (root / name).is_file()), root / names[0])
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise ProfileError(f"{label} renderer is missing: {candidate}")
    return candidate


def load_runtime(
    runtime_root: Path,
    derivative_runtime_root: Path | None = None,
    *,
    baseline_renderer: str | None = None,
    derivative_renderer: str | None = None,
) -> RuntimeSpec:
    """Resolve both explicit runtime identities and their shared asset hashes."""

    baseline_root = Path(runtime_root).resolve()
    derivative_root = Path(derivative_runtime_root or runtime_root).resolve()
    baseline = _renderer_candidate(baseline_root, baseline_renderer, ("renderer.py", "frozen_watercolour_renderer.py"), "baseline")
    derivative = _renderer_candidate(derivative_root, derivative_renderer, ("cancellable_watercolour_renderer.py",), "derivative")
    assets: dict[str, str] = {}
    for root in (baseline_root, derivative_root):
        assets_root = root / "assets"
        if not assets_root.is_dir():
            continue
        for path in sorted(assets_root.glob("*.js")):
            digest = sha256(path)
            previous = assets.get(path.name)
            if previous and previous != digest:
                raise ProfileError(f"Runtime asset differs between roots: {path.name}")
            assets[path.name] = digest
    if not assets:
        raise ProfileError("Runtime must expose an assets directory with JavaScript assets")

    def manifest_digest(root: Path) -> str | None:
        path = root / "PATCH-MANIFEST.json"
        return sha256(path) if path.is_file() else None

    return RuntimeSpec(
        baseline=baseline,
        derivative=derivative,
        baseline_root=baseline_root,
        derivative_root=derivative_root,
        baseline_sha256=sha256(baseline),
        derivative_sha256=sha256(derivative),
        assets_sha256=dict(sorted(assets.items())),
        baseline_patch_manifest_sha256=manifest_digest(baseline_root),
        derivative_patch_manifest_sha256=manifest_digest(derivative_root),
    )


def resource_snapshot() -> dict[str, Any]:
    """Capture run-level CPU identity without reading credentials."""

    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            affinity = None
    cgroup: dict[str, str] = {}
    for name in ("cpu.max", "cpu.weight", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        try:
            cgroup[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            cgroup[name] = "unavailable"
    return {
        "platform": sys.platform,
        "platform_release": os.uname().release if hasattr(os, "uname") else "unavailable",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "sched_getaffinity": affinity,
        "cgroup": cgroup,
        "measurement_denominator": "whole profiler process elapsed wall time; CPU values are run metadata",
    }


def minimal_env(scratch: Path, browsers_path: Path | None = None) -> dict[str, str]:
    """Build a child environment with no inherited credentials or profiles."""

    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "LANG": "C.UTF-8",
    }
    chosen = browsers_path
    if chosen is None and os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        chosen = Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"])
    if chosen is not None:
        environment["PLAYWRIGHT_BROWSERS_PATH"] = str(Path(chosen).resolve())
    return environment


def renderer_command(
    renderer_python: Path,
    renderer: Path,
    control: Control,
    output: Path,
    timeout: float,
    *,
    backend: str = BACKEND,
) -> list[str]:
    if backend != BACKEND:
        raise ProfileError(f"The CPU profile requires ANGLE backend {BACKEND!r}")
    return [
        str(renderer_python),
        str(renderer),
        str(control.source),
        "--output", str(output),
        "--seed", str(control.seed),
        "--timeout", str(timeout),
        "--backend", backend,
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read PNG dimensions without trusting a renderer's validity flag."""

    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", header[16:24])


def _proc_rows() -> list[dict[str, Any]]:
    """Return a best-effort process table for ownership evidence on Linux."""

    root = Path("/proc")
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat = (entry / "stat").read_text(encoding="utf-8")
            _, fields = stat.split(") ", 1)
            values = fields.split()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            rows.append({"pid": pid, "ppid": int(values[1]), "state": values[0], "pgid": int(values[2]), "sid": int(values[3]), "command": cmdline[:500]})
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            continue
    return rows


def _descendants(root_pid: int, rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = list(rows if rows is not None else _proc_rows())
    by_parent: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_parent.setdefault(int(row["ppid"]), []).append(dict(row))
    result: list[dict[str, Any]] = []
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        parent = pending.pop()
        for row in by_parent.get(parent, []):
            pid = int(row["pid"])
            if pid in seen:
                continue
            seen.add(pid)
            result.append(row)
            pending.append(pid)
    return result


def _session_has_member(pgid: int, sid: int) -> bool:
    return any(
        int(row.get("pgid", -1)) == pgid
        and int(row.get("sid", -1)) == sid
        and str(row.get("state")) not in {"Z", "X"}
        for row in _proc_rows()
    )


def _kill_group(pgid: int, signum: int) -> bool:
    try:
        os.killpg(pgid, signum)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _owned_sessions(process: subprocess.Popen[Any], observed: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    sessions: set[tuple[int, int]] = set()
    try:
        sessions.add((os.getpgid(process.pid), os.getsid(process.pid)))
    except (ProcessLookupError, PermissionError, OSError):
        pass
    for row in observed:
        try:
            sessions.add((int(row["pgid"]), int(row["sid"])))
        except (KeyError, TypeError, ValueError):
            continue
    return [{"pgid": pgid, "sid": sid} for pgid, sid in sorted(sessions)]


def _terminate_owned_sessions(
    process: subprocess.Popen[Any],
    sessions: Sequence[Mapping[str, Any]],
    *,
    grace: float,
) -> dict[str, Any]:
    """Stop only process groups observed below this profiler-owned Popen."""

    started = time.monotonic()
    try:
        process.send_signal(signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=max(0.05, grace))
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + max(0.05, grace)
    while time.monotonic() < deadline:
        if process.poll() is not None and not any(_session_has_member(int(item["pgid"]), int(item["sid"])) for item in sessions):
            break
        time.sleep(0.05)
    remaining = [item for item in sessions if _session_has_member(int(item["pgid"]), int(item["sid"]))]
    used_sigkill = bool(remaining)
    if remaining:
        for item in remaining:
            _kill_group(int(item["pgid"]), signal.SIGKILL)
        try:
            process.wait(timeout=max(0.05, grace))
        except subprocess.TimeoutExpired:
            pass
    final_deadline = time.monotonic() + max(0.05, grace)
    while time.monotonic() < final_deadline and any(_session_has_member(int(item["pgid"]), int(item["sid"])) for item in sessions):
        time.sleep(0.05)
    try:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=max(0.05, grace))
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    survivors = [item for item in sessions if _session_has_member(int(item["pgid"]), int(item["sid"]))]
    return {
        "signal": "SIGTERM_then_SIGKILL" if remaining else "SIGTERM",
        "used_sigkill": used_sigkill,
        "cleanup_seconds": round(time.monotonic() - started, 6),
        "owned_sessions": [dict(item) for item in sessions],
        "surviving_owned_sessions": survivors,
        "owned_processes_gone": not survivors and process.poll() is not None,
    }


def _backend_check(receipt: Mapping[str, Any], backend: str = BACKEND) -> dict[str, Any]:
    """Validate one completed renderer receipt as the single backend check."""

    requested = receipt.get("requested_angle_backend")
    args = receipt.get("browser_args") or []
    gpu = receipt.get("gpu") or {}
    errors: list[str] = []
    if requested != backend:
        errors.append(f"renderer reported backend {requested!r}, expected {backend!r}")
    if f"--use-angle={backend}" not in args:
        errors.append("renderer receipt did not record the requested ANGLE flag")
    renderer_name = str(gpu.get("unmasked_renderer") or "")
    explicit_errors = receipt.get("errors") or []
    if any("fallback" in str(item).lower() or "backend" in str(item).lower() and "requested" in str(item).lower() for item in explicit_errors):
        errors.append("renderer reported an ANGLE backend fallback")
    return {"checked_once": True, "backend": backend, "available": not errors, "errors": errors, "reported_requested_backend": requested, "gpu_renderer": renderer_name or None}


def _receipt_from_attempt(
    control: Control,
    renderer_kind: str,
    renderer: Path,
    output: Path,
    receipt_path: Path,
    log_path: Path,
    started: float,
    process: subprocess.Popen[Any],
    *,
    timed_out: bool,
    lifecycle: bool = False,
    lifecycle_data: Mapping[str, Any] | None = None,
    requested_timeout: float | None = None,
    backend: str = BACKEND,
) -> dict[str, Any]:
    renderer_receipt = _read_json(output.with_suffix(".json"))
    output_exists = output.is_file()
    output_sha = sha256(output) if output_exists else None
    dimensions = _png_dimensions(output) if output_exists else None
    elapsed = round(time.monotonic() - started, 6)
    reported_valid = renderer_receipt.get("valid") is True
    artifact_hash_matches = not renderer_receipt.get("png_sha256") or renderer_receipt.get("png_sha256") == output_sha
    artifact_contract_ok = output_exists and dimensions == CANVAS and artifact_hash_matches
    result: dict[str, Any] = {
        "id": control.id,
        "kind": control.kind,
        "renderer_kind": renderer_kind,
        "renderer_sha256": sha256(renderer),
        "source_relpath": control.source_relpath,
        "source_sha256": control.source_sha256,
        "seed": control.seed,
        "requested_angle_backend": backend,
        "valid": bool(reported_valid and artifact_contract_ok and not timed_out and not lifecycle),
        "renderer_reported_valid": reported_valid,
        "artifact_present": output_exists,
        "artifact_dimensions": list(dimensions) if dimensions else None,
        "artifact_hash_matches": artifact_hash_matches,
        "artifact_accepted": bool(reported_valid and artifact_contract_ok and not timed_out and not lifecycle),
        "png_sha256": output_sha,
        "elapsed_seconds": elapsed,
        "requested_timeout_seconds": requested_timeout,
        "timed_out": timed_out,
        "returncode": process.returncode,
        "errors": renderer_receipt.get("errors") or [],
        "renderer_receipt": renderer_receipt,
        "log_relpath": log_path.name,
        "censored": False,
    }
    if lifecycle:
        result.update({"valid": False, "artifact_accepted": False, "censored": True, "censor_reason": "parent_cancelled", "lifecycle": True})
    if timed_out:
        result.update({"valid": False, "artifact_accepted": False, "censored": True, "censor_reason": "per_job_timeout"})
    if not renderer_receipt and not timed_out:
        result.update({"censored": True, "censor_reason": "missing_renderer_receipt"})
    if reported_valid and not artifact_contract_ok and not timed_out and not lifecycle:
        result.update({"censored": True, "censor_reason": "artifact_contract_mismatch"})
    if receipt_path:
        _write_json(receipt_path, result)
    return result


def run_renderer(
    renderer_python: Path,
    renderer: Path,
    control: Control,
    attempt_dir: Path,
    timeout: float,
    browsers_path: Path | None,
    *,
    renderer_kind: str,
    backend: str = BACKEND,
    lifecycle_stop_after: float | None = None,
    cleanup_grace: float = DEFAULT_CLEANUP_GRACE_SECONDS,
    requested_timeout: float | None = None,
) -> dict[str, Any]:
    """Run one isolated renderer process with a hard parent-owned deadline."""

    attempt_dir.mkdir(parents=True, exist_ok=False)
    output = attempt_dir / "image.png"
    receipt_path = attempt_dir / "receipt.json"
    log_path = attempt_dir / "renderer.log"
    scratch = attempt_dir / "scratch"
    scratch.mkdir()
    command = renderer_command(renderer_python, renderer, control, output, timeout, backend=backend)
    started = time.monotonic()
    observed: list[dict[str, Any]] = []
    timed_out = False
    process: subprocess.Popen[Any]
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=minimal_env(scratch, browsers_path),
            start_new_session=True,
        )
        if lifecycle_stop_after is None:
            try:
                process.wait(timeout=max(0.001, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
                sessions = _owned_sessions(process, _descendants(process.pid))
                _terminate_owned_sessions(process, sessions, grace=cleanup_grace)
        else:
            stop_deadline = time.monotonic() + lifecycle_stop_after
            while process.poll() is None and time.monotonic() < stop_deadline:
                observed = _descendants(process.pid)
                time.sleep(0.05)
            observed = _descendants(process.pid, _proc_rows())
            sessions = _owned_sessions(process, observed)
            lifecycle_data = _terminate_owned_sessions(process, sessions, grace=cleanup_grace)
            lifecycle_data.update({
                "parent_stop_after_seconds": lifecycle_stop_after,
                "observed_descendants": observed,
                "native_controller_pid": os.getpid(),
                "derivative_pid": process.pid,
                "chain_levels_observed": {
                    "native_popen": True,
                    "derivative": True,
                    "worker": any("--worker" in str(row.get("command", "")) for row in observed),
                    "chrome": any("chrome" in str(row.get("command", "")).lower() or "chromium" in str(row.get("command", "")).lower() for row in observed),
                },
            })
            chain = lifecycle_data["chain_levels_observed"]
            lifecycle_data["chain_observed"] = bool(chain.get("native_popen") and chain.get("derivative") and chain.get("worker") and chain.get("chrome"))
            lifecycle_data["cancel_success"] = bool(lifecycle_data["chain_observed"] and lifecycle_data.get("owned_processes_gone") and not lifecycle_data.get("used_sigkill"))
            result = _receipt_from_attempt(control, renderer_kind, renderer, output, receipt_path, log_path, started, process, timed_out=False, lifecycle=True, lifecycle_data=lifecycle_data, requested_timeout=requested_timeout, backend=backend)
            result["lifecycle"] = lifecycle_data
            if not lifecycle_data["cancel_success"]:
                result["censor_reason"] = "lifecycle_chain_not_observed" if not lifecycle_data["chain_observed"] else "cleanup_escalated"
            _write_json(receipt_path, result)
            return result
    result = _receipt_from_attempt(control, renderer_kind, renderer, output, receipt_path, log_path, started, process, timed_out=timed_out, requested_timeout=requested_timeout, backend=backend)
    return result


def _result_censor_reason(result: Mapping[str, Any], control: Control) -> str | None:
    if result.get("censored"):
        return str(result.get("censor_reason") or "censored")
    if control.kind == "valid" and not result.get("valid"):
        return "valid_reference_failed"
    if control.kind == "invalid" and result.get("valid"):
        return "invalid_reference_reported_valid"
    return None


def _row_metadata(control: Control) -> dict[str, Any]:
    return {
        "id": control.id,
        "kind": control.kind,
        "source_relpath": control.source_relpath,
        "source_sha256": control.source_sha256,
        "seed": control.seed,
        "saved_valid": control.saved_valid,
        "saved_elapsed_seconds": control.saved_elapsed_seconds,
        "source_archive_revision": control.source_archive_revision,
        "source_archive_sha256": control.source_archive_sha256,
        "source_archive_member": control.source_archive_member,
    }


def _run_serial(
    controls: Sequence[Control],
    renderer_kind: str,
    renderer: Path,
    renderer_python: Path,
    parent: Path,
    timeout: float,
    repeats: int,
    browsers_path: Path | None,
    deadline: float,
    backend: str,
    cleanup_grace: float = DEFAULT_CLEANUP_GRACE_SECONDS,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for control in controls:
            if control.kind == "slow_cancel":
                continue
            remaining = deadline - time.monotonic() - cleanup_grace
            if remaining <= 0.05:
                results.append({**_row_metadata(control), "renderer_kind": renderer_kind, "repeat": repeat, "valid": False, "censored": True, "censor_reason": "total_deadline", "elapsed_seconds": 0.0})
                continue
            effective_timeout = min(timeout, remaining)
            folder = parent / f"repeat-{repeat}" / control.id
            result = run_renderer(renderer_python, renderer, control, folder, effective_timeout, browsers_path, renderer_kind=renderer_kind, backend=backend, cleanup_grace=cleanup_grace, requested_timeout=timeout)
            result["repeat"] = repeat
            result["timeout_clamped"] = effective_timeout < timeout
            results.append(result)
    return results


def _run_derivative_pool(
    controls: Sequence[Control],
    renderer: Path,
    renderer_python: Path,
    parent: Path,
    timeout: float,
    pool_width: int,
    browsers_path: Path | None,
    deadline: float,
    backend: str,
    cleanup_grace: float = DEFAULT_CLEANUP_GRACE_SECONDS,
) -> dict[str, Any]:
    eligible = [control for control in controls if control.kind == "valid"]
    if not eligible:
        return {"pool_width": pool_width, "case_ids": [], "results": [], "elapsed_seconds": 0.0, "useful_valid_count": 0, "throughput_valid_per_second": None, "censored": True, "censor_reason": "no_valid_controls"}
    started = time.monotonic()
    parent.mkdir(parents=True, exist_ok=True)

    def one(control: Control) -> dict[str, Any]:
        folder = parent / control.id
        remaining = deadline - time.monotonic() - cleanup_grace
        if remaining <= 0.05:
            return {**_row_metadata(control), "renderer_kind": "derivative", "valid": False, "censored": True, "censor_reason": "total_deadline", "elapsed_seconds": 0.0}
        effective_timeout = min(timeout, remaining)
        result = run_renderer(renderer_python, renderer, control, folder, effective_timeout, browsers_path, renderer_kind="derivative", backend=backend, cleanup_grace=cleanup_grace, requested_timeout=timeout)
        result["timeout_clamped"] = effective_timeout < timeout
        return result

    with ThreadPoolExecutor(max_workers=pool_width) as pool:
        results = list(pool.map(one, eligible))
    elapsed = round(time.monotonic() - started, 6)
    useful = sum(bool(row.get("valid")) for row in results)
    return {
        "pool_width": pool_width,
        "case_ids": [control.id for control in eligible],
        "results": results,
        "elapsed_seconds": elapsed,
        "useful_valid_count": useful,
        "throughput_valid_per_second": useful / elapsed if elapsed > 0 else None,
        "censored": any(bool(row.get("censored")) for row in results),
    }


def compare_parity(baseline: Sequence[Mapping[str, Any]], derivative: Sequence[Mapping[str, Any]], controls: Sequence[Control], repeats: int) -> list[dict[str, Any]]:
    left = {(str(row.get("id")), int(row.get("repeat", 0))): row for row in baseline}
    right = {(str(row.get("id")), int(row.get("repeat", 0))): row for row in derivative}
    output: list[dict[str, Any]] = []
    for control in controls:
        for repeat in range(1, repeats + 1):
            if control.kind != "valid":
                continue
            base, candidate = left.get((control.id, repeat)), right.get((control.id, repeat))
            if not base or not candidate or base.get("censored") or candidate.get("censored"):
                status = "censored"
            elif not base.get("valid") or not candidate.get("valid"):
                status = "invalid"
            elif base.get("png_sha256") == candidate.get("png_sha256"):
                status = "equal"
            else:
                status = "different"
            output.append({
                "id": control.id,
                "repeat": repeat,
                "status": status,
                "baseline_valid": bool(base and base.get("valid")),
                "derivative_valid": bool(candidate and candidate.get("valid")),
                "png_sha_equal": bool(base and candidate and base.get("png_sha256") and base.get("png_sha256") == candidate.get("png_sha256")),
                "baseline_elapsed_seconds": base.get("elapsed_seconds") if base else None,
                "derivative_elapsed_seconds": candidate.get("elapsed_seconds") if candidate else None,
            })
    return output


def _summarize_counts(results: Iterable[Mapping[str, Any]], controls: Sequence[Control]) -> dict[str, Any]:
    rows = list(results)
    by_id = {control.id: control for control in controls}
    counts = {"total": len(rows), "valid": 0, "invalid": 0, "timed_out": 0, "censored": 0, "unexpected_valid": 0, "unexpected_invalid": 0}
    censored: list[dict[str, Any]] = []
    for row in rows:
        if row.get("valid"):
            counts["valid"] += 1
        if row.get("timed_out"):
            counts["timed_out"] += 1
        if row.get("censored"):
            counts["censored"] += 1
            censored.append({"id": row.get("id"), "renderer_kind": row.get("renderer_kind"), "repeat": row.get("repeat"), "reason": row.get("censor_reason")})
        control = by_id.get(str(row.get("id")))
        if control and control.kind == "invalid" and not row.get("valid"):
            counts["invalid"] += 1
        if control and control.kind == "valid" and not row.get("valid") and not row.get("censored"):
            counts["unexpected_invalid"] += 1
        if control and control.kind == "invalid" and row.get("valid"):
            counts["unexpected_valid"] += 1
    return {"counts": counts, "censored_cases": censored}


def build_review_html(report: Mapping[str, Any], output_path: Path) -> None:
    """Write a dependency-free review page with only relative image links."""

    cards: list[str] = []
    for group_name in ("baseline_sequential", "derivative_sequential", "derivative_pool", "lifecycle"):
        group = report.get(group_name)
        if not isinstance(group, Mapping):
            continue
        rows = group.get("results") if group_name != "lifecycle" else [group]
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            image = row.get("image_relpath")
            image_markup = f'<img loading="lazy" src="{html.escape(str(image), quote=True)}" alt="{html.escape(str(row.get("id", "")), quote=True)}" width="240">' if image else "<p>No accepted image</p>"
            cards.append("<article><h3>%s / %s</h3>%s<pre>%s</pre></article>" % (
                html.escape(str(row.get("id", ""))), html.escape(str(row.get("renderer_kind", group_name))), image_markup,
                html.escape(json.dumps({"valid": row.get("valid"), "censored": row.get("censored"), "png_sha256": row.get("png_sha256"), "elapsed_seconds": row.get("elapsed_seconds")}, sort_keys=True)),
            ))
    title = html.escape(str(report.get("schema", SCHEMA)))
    body = "\n".join(cards) or "<p>No render artifacts were accepted.</p>"
    output_path.write_text("<!doctype html><meta charset=\"utf-8\"><title>%s</title><style>body{font:14px sans-serif;background:#f5f1ea;color:#25231f}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem}article{background:white;padding:.7rem;border:1px solid #ddd}img{max-width:100%%;height:auto}pre{white-space:pre-wrap;font-size:11px}</style><h1>%s</h1><main>%s</main>" % (title, title, body), encoding="utf-8")


def _attach_relative_artifact(row: dict[str, Any], attempt_root: Path, report_root: Path) -> None:
    if not row.get("artifact_present"):
        return
    attempt_dir = Path(row.get("_attempt_dir", ""))
    image = attempt_dir / "image.png"
    if image.is_file():
        row["image_relpath"] = image.relative_to(report_root).as_posix()
    row.pop("_attempt_dir", None)


def _flatten_for_report(rows: Iterable[dict[str, Any]], report_root: Path) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        clean = dict(row)
        _attach_relative_artifact(clean, report_root, report_root)
        result.append(clean)
    return result


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "linux":
        raise ProfileError("The CPU renderer profile is Linux-only")
    if args.backend != BACKEND:
        raise ProfileError(f"The CPU renderer profile requires --backend {BACKEND}")
    if args.repeats != DEFAULT_REPEATS:
        raise ProfileError("This matched benchmark requires exactly two sequential repeats")
    timeout = _finite_number(args.timeout, "timeout", positive=True)
    total_limit = _finite_number(args.total_limit, "total_limit", positive=True)
    if total_limit > DEFAULT_TOTAL_SECONDS:
        raise ProfileError(f"total_limit cannot exceed {DEFAULT_TOTAL_SECONDS:g} seconds")
    if args.pool_width != DEFAULT_POOL_WIDTH:
        raise ProfileError("This benchmark requires a four-way derivative pool")

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ProfileError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, controls, manifest_path = load_controls(input_root, args.manifest_name)
    runtime = load_runtime(args.runtime_root, args.derivative_runtime_root, baseline_renderer=args.baseline_renderer, derivative_renderer=args.derivative_renderer)
    attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    attempt_root = output_root / "attempts" / attempt_id
    attempt_root.mkdir(parents=True)
    deadline = time.monotonic() + total_limit
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "profile_version": PROFILE_VERSION,
        "status": "running",
        "runner_sha256": RUNNER_SHA256,
        "source_commit": manifest.get("source_commit"),
        "input_manifest": {"name": manifest_path.name, "sha256": sha256(manifest_path), "schema": manifest.get("schema")},
        "controls": [_row_metadata(control) for control in controls],
        "runtime": {
            "baseline_renderer_sha256": runtime.baseline_sha256,
            "derivative_renderer_sha256": runtime.derivative_sha256,
            "assets_sha256": dict(runtime.assets_sha256),
            "baseline_patch_manifest_sha256": runtime.baseline_patch_manifest_sha256,
            "derivative_patch_manifest_sha256": runtime.derivative_patch_manifest_sha256,
            "baseline_renderer_relpath": runtime.baseline.name,
            "derivative_renderer_relpath": runtime.derivative.name,
        },
        "contract": {
        "backend": args.backend,
            "timeout_seconds": timeout,
            "sequential_repeats": args.repeats,
            "pool_width": args.pool_width,
            "total_measurement_limit_seconds": total_limit,
            "canvas": list(CANVAS),
            "network_enabled": False,
            "programs_unchanged": True,
            "separate_attempt_directories": True,
            "throughput_claim_requires_valid_image_parity": True,
            "cancellation_speedup_claim": None,
        },
        "resource_snapshot": resource_snapshot(),
        "backend_check": None,
        "baseline_sequential": {"results": []},
        "derivative_sequential": {"results": []},
        "derivative_pool": None,
        "lifecycle": None,
    }
    _write_json(output_root / "profile.json", report)

    # The slow control is a lifecycle-only input.  It never consumes a 180s
    # sequential slot and cannot inflate a throughput number.
    measured = [control for control in controls if control.kind != "slow_cancel"]
    valid_controls = [control for control in controls if control.kind == "valid"]
    baseline_rows = _run_serial(measured, "baseline", runtime.baseline, Path(args.renderer_python), attempt_root / "baseline-sequential", timeout, args.repeats, args.browsers_path, deadline, args.backend, args.cleanup_grace_seconds)
    for row in baseline_rows:
        row["_attempt_dir"] = str(attempt_root / "baseline-sequential" / f"repeat-{row.get('repeat', 0)}" / str(row.get("id")))
    report["baseline_sequential"] = {"results": _flatten_for_report(baseline_rows, output_root), **_summarize_counts(baseline_rows, controls)}
    _write_json(output_root / "profile.json", report)

    # Exactly one completed receipt is used to decide whether the requested
    # ANGLE backend is available.  Later receipts retain the contract fields but do not
    # silently change backend or rerun a separate probe.
    first_backend_receipt = next((row.get("renderer_receipt") for row in baseline_rows if row.get("renderer_receipt")), {})
    report["backend_check"] = _backend_check(first_backend_receipt or {}, args.backend)
    if not report["backend_check"]["available"]:
        report["status"] = "backend_unavailable"
        report["censored_cases"] = [{"id": control.id, "reason": "backend_unavailable"} for control in controls]
        report["summary"] = {"backend_available": False, "counts": {"total": 0, "valid": 0, "censored": len(controls)}}
        _write_json(output_root / "profile.json", report)
        build_review_html(report, output_root / "profile.html")
        return report

    derivative_rows = _run_serial(measured, "derivative", runtime.derivative, Path(args.renderer_python), attempt_root / "derivative-sequential", timeout, args.repeats, args.browsers_path, deadline, args.backend, args.cleanup_grace_seconds)
    for row in derivative_rows:
        row["_attempt_dir"] = str(attempt_root / "derivative-sequential" / f"repeat-{row.get('repeat', 0)}" / str(row.get("id")))
    report["derivative_sequential"] = {"results": _flatten_for_report(derivative_rows, output_root), **_summarize_counts(derivative_rows, controls)}
    report["parity"] = compare_parity(baseline_rows, derivative_rows, controls, args.repeats)
    report["parity_summary"] = {
        "eligible_cases": len(valid_controls),
        "repeats": args.repeats,
        "equal_count": sum(row["status"] == "equal" for row in report["parity"]),
        "different_count": sum(row["status"] == "different" for row in report["parity"]),
        "censored_count": sum(row["status"] == "censored" for row in report["parity"]),
    }
    _write_json(output_root / "profile.json", report)

    pool = _run_derivative_pool(valid_controls, runtime.derivative, Path(args.renderer_python), attempt_root / "derivative-pool", timeout, args.pool_width, args.browsers_path, deadline, args.backend, args.cleanup_grace_seconds)
    for row in pool.get("results", []):
        row["_attempt_dir"] = str(attempt_root / "derivative-pool" / str(row.get("id")))
    pool["results"] = _flatten_for_report(pool.get("results", []), output_root)
    sequential_first_pass = [row for row in derivative_rows if row.get("repeat") == 1 and str(row.get("id")) in {control.id for control in valid_controls}]
    sequential_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in sequential_first_pass if not row.get("censored"))
    pool["sequential_one_pass_elapsed_seconds"] = round(sequential_elapsed, 6)
    pool["useful_sequential_valid_count"] = sum(bool(row.get("valid")) for row in sequential_first_pass)
    pool["useful_throughput_comparison"] = {
        "sequential_valid_per_second": (pool["useful_sequential_valid_count"] / sequential_elapsed) if sequential_elapsed > 0 else None,
        "pool_valid_per_second": pool.get("throughput_valid_per_second"),
        "speedup": (pool.get("throughput_valid_per_second") / (pool["useful_sequential_valid_count"] / sequential_elapsed)) if sequential_elapsed > 0 and pool.get("throughput_valid_per_second") is not None and pool["useful_sequential_valid_count"] > 0 else None,
        "basis": "valid outputs completed per elapsed wall second; same derivative, source, seed and backend; cancellation excluded",
    }
    report["derivative_pool"] = pool
    _write_json(output_root / "profile.json", report)

    slow = next((control for control in controls if control.kind == "slow_cancel"), None)
    if slow and time.monotonic() + DEFAULT_CANCEL_AFTER_SECONDS < deadline:
        lifecycle_dir = attempt_root / "lifecycle" / slow.id
        lifecycle = run_renderer(Path(args.renderer_python), runtime.derivative, slow, lifecycle_dir, timeout, args.browsers_path, renderer_kind="derivative", backend=args.backend, lifecycle_stop_after=args.cancel_after_seconds, cleanup_grace=args.cleanup_grace_seconds, requested_timeout=timeout)
        lifecycle["_attempt_dir"] = str(lifecycle_dir)
        report["lifecycle"] = _flatten_for_report([lifecycle], output_root)[0]
    elif slow:
        report["lifecycle"] = {**_row_metadata(slow), "renderer_kind": "derivative", "valid": False, "censored": True, "censor_reason": "total_deadline"}
    report["summary"] = {
        "backend_available": True,
        "baseline": report["baseline_sequential"].get("counts", {}),
        "derivative": report["derivative_sequential"].get("counts", {}),
        "pool_useful_valid_count": report.get("derivative_pool", {}).get("useful_valid_count"),
        "pool_censored": report.get("derivative_pool", {}).get("censored"),
        "parity": report.get("parity_summary"),
        "cancellation": {
            "requested": bool(slow),
            "completed": bool(report.get("lifecycle", {}).get("lifecycle", {}).get("cancel_success")),
            "chain_observed": report.get("lifecycle", {}).get("lifecycle", {}).get("chain_observed") if report.get("lifecycle") else None,
            "owned_processes_gone": report.get("lifecycle", {}).get("lifecycle", {}).get("owned_processes_gone") if report.get("lifecycle") else None,
            "cleanup_seconds": report.get("lifecycle", {}).get("lifecycle", {}).get("cleanup_seconds") if report.get("lifecycle") else None,
            "speedup_claim": None,
        },
        "total_elapsed_seconds": round(total_limit - max(0.0, deadline - time.monotonic()), 6),
    }
    report["status"] = "complete"
    _write_json(output_root / "profile.json", report)
    build_review_html(report, output_root / "profile.html")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--derivative-runtime-root", type=Path)
    parser.add_argument("--baseline-renderer")
    parser.add_argument("--derivative-renderer")
    parser.add_argument("--renderer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--browsers-path", type=Path)
    parser.add_argument("--manifest-name", default="cpu-controls.json")
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--total-limit", type=float, default=DEFAULT_TOTAL_SECONDS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--pool-width", type=int, default=DEFAULT_POOL_WIDTH)
    parser.add_argument("--cancel-after-seconds", type=float, default=DEFAULT_CANCEL_AFTER_SECONDS)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=DEFAULT_CLEANUP_GRACE_SECONDS)
    parser.add_argument("--execute", action="store_true", help="Required opt-in for Linux renderer subprocesses")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("profile_cpu.py requires --execute; no renderer was started")
    try:
        report = run_profile(args)
    except ProfileError as error:
        raise SystemExit(f"profile_cpu: {error}") from error
    print(json.dumps({"schema": report["schema"], "status": report["status"], "output_root": str(Path(args.output_root).resolve()), "backend_check": report.get("backend_check"), "summary": report.get("summary")}, indent=2, sort_keys=True))
    return 0 if report.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
