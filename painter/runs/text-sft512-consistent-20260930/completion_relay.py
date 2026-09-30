#!/usr/bin/env python3
"""Finite local relay for post-training evidence preservation.

The parent rental controller owns exact-pod deletion.  This foreground relay
only waits for a terminal finite supervisor state, asks the node to run the
post-training pipeline, retrieves receipts with scp, and writes a release
marker after public-archive verification or a bounded failure capture. Provider
stdout/stderr is captured in memory and never written to durable records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Sequence

RUN_NAME = "text-sft512-consistent-20260930"
DEFAULT_REMOTE_RUN = f"/root/painter/repo/painter/runs/{RUN_NAME}"
POD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


class RelayError(RuntimeError):
    """The finite relay could not preserve the run safely."""


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RelayError(f"required private input is missing: {path.name}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RelayError(f"required private input is empty: {path.name}")
    return value


def load_private_inputs(private_dir: Path) -> dict[str, Any]:
    """Read pod/start inputs without emitting their contents."""

    private_dir = private_dir.resolve(strict=True)
    start = json.loads(_read_text(private_dir / "start.json"))
    if not isinstance(start, dict):
        raise RelayError("start.json must contain an object")
    pod_path = next((private_dir / name for name in ("pod-id.txt", "pod-id", "pod_id.txt")
                     if (private_dir / name).is_file()), None)
    pod_id = _read_text(pod_path) if pod_path is not None else str(start.get("pod_id") or "")
    if not POD_ID_RE.fullmatch(pod_id) or pod_id.lower() in {"all", "*"}:
        raise RelayError("private pod-id input is invalid")

    def number(*names: str, default: float | None = None) -> float:
        for name in names:
            if start.get(name) is not None:
                value = float(start[name])
                if math.isfinite(value):
                    return value
                break
        if default is not None:
            return default
        raise RelayError(f"start.json is missing {names[0]}")

    start_epoch = number("rental_start_epoch", "start_epoch", "started_epoch")
    price = number("price_usd_per_hour", "price")
    cap = number("max_gpu_usd", default=20.0)
    margin = number("release_margin_seconds", default=180.0)
    if start_epoch <= 0 or price <= 0 or cap <= 0 or cap > 20 or margin < 0:
        raise RelayError("start.json rental budget is outside the finite bounds")
    return {
        "private_dir": private_dir,
        "pod_id": pod_id,
        "start_epoch": start_epoch,
        "price": price,
        "cap": cap,
        "margin": margin,
        "remote_run": str(start.get("remote_run") or DEFAULT_REMOTE_RUN),
    }


def _json_values(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_values(child)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return
        yield from _json_values(decoded)


def parse_state_output(raw: str) -> dict[str, Any] | None:
    """Find the remote supervisor JSON in a provider JSON envelope."""

    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(raw.lstrip())
    except json.JSONDecodeError:
        return None
    def find(candidate: Any, failed: bool = False) -> dict[str, Any] | None:
        if isinstance(candidate, dict):
            # A supervisor receipt is the application payload. Its own
            # exit_code (for example 143 after a finite stop) must not be
            # confused with lium's wrapper exit code.
            if not failed and candidate.get("schema") == "finite-text-paint-supervisor-v1":
                return candidate
            code = candidate.get("exit_code", candidate.get("returncode"))
            try:
                failed = failed or (code is not None and int(code) != 0)
            except (TypeError, ValueError):
                failed = True
            if not failed and candidate.get("status") in {"running", "stopped", "completed", "failed"}:
                return candidate
            for child in candidate.values():
                found = find(child, failed)
                if found is not None:
                    return found
        elif isinstance(candidate, list):
            for child in candidate:
                found = find(child, failed)
                if found is not None:
                    return found
        elif isinstance(candidate, str):
            try:
                return find(json.loads(candidate), failed)
            except (TypeError, json.JSONDecodeError):
                return None
        return None

    return find(value)


def provider_success(raw: str) -> bool:
    """Check the top-level lium JSON result without trusting child output."""

    try:
        value, _ = json.JSONDecoder().raw_decode(raw.lstrip())
    except json.JSONDecodeError:
        return True
    if isinstance(value, dict):
        for key in ("exit_code", "returncode"):
            if key in value:
                try:
                    return int(value[key]) == 0
                except (TypeError, ValueError):
                    return False
    return True


def _run_process(run: Callable[..., Any], argv: Sequence[str], *, timeout: float) -> Any:
    # Provider output is intentionally captured in memory: lium may echo
    # connection details, and those must never enter durable logs.
    return run(list(argv), check=False, capture_output=True, text=True, timeout=timeout)


def _remote_exec(lium_bin: str, pod_id: str, command: str, *, run: Callable[..., Any], timeout: float) -> Any:
    return _run_process(run, (lium_bin, "exec", pod_id, "--json", command), timeout=timeout)


def _remote_state_command(remote_run: str) -> str:
    path = json.dumps(f"{remote_run}/supervisor-state.json")
    code = f"from pathlib import Path; p=Path({path}); print(p.read_text() if p.is_file() else {{}})"
    return f"python3 -c {shlex.quote(code)}"


def _remote_post_command(remote_run: str, *, commit: str, start: float, price: float, cap: float) -> str:
    run = shlex.quote(remote_run)
    commit_arg = shlex.quote(commit)
    return (
        "set -Eeuo pipefail && "
        f"cd {run} && export PAINTER_SOURCE_COMMIT={commit_arg} "
        "PAINTER_PIPELINE_POST_TRAINING_ONLY=1 && "
        f"exec >{shlex.quote(remote_run + '/nodecompletion.log')} 2>&1 && "
        f"bash {shlex.quote(remote_run + '/execute_pipeline.sh')} "
        f"{start:.6f} {price:.6f} {cap:.6f}"
    )


def _scp(
    lium_bin: str,
    pod_id: str,
    remote_path: str,
    destination: Path,
    *,
    run: Callable[..., Any],
    timeout: float,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _run_process(
        run, (lium_bin, "scp", pod_id, remote_path, str(destination), "-d"), timeout=timeout,
    )
    return result.returncode == 0 and destination.is_file() and not destination.is_symlink()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise RelayError(f"refusing symlink output: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verified_upload(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("public_hash_verified") is True and value.get("private") is False


def _valid_terminal_receipt(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("schema") == "text-sft512-consistent-terminal-v1"


def _valid_archive(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
            if "evidence-manifest.json" not in names:
                return False
            # Read every member digest once so a truncated/corrupt transfer
            # cannot authorize pod release.
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return bool(digest.hexdigest())
    except (OSError, tarfile.TarError):
        return False


def relay(
    *,
    private_dir: Path,
    lium_bin: str,
    completion_commit: str,
    poll_seconds: float = 20.0,
    run: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> int:
    if not COMMIT_RE.fullmatch(completion_commit):
        raise RelayError("completion commit must be a reviewed hexadecimal commit")
    if not 0 < poll_seconds <= 30:
        raise RelayError("poll interval must be in (0, 30] seconds")
    inputs = load_private_inputs(private_dir)
    private = inputs["private_dir"]
    deadline = inputs["start_epoch"] + inputs["cap"] / inputs["price"] * 3600.0 - inputs["margin"]
    status = "failed"
    reason = "hard_deadline"
    terminal: dict[str, Any] | None = None
    post_result: Any = None
    while now() < deadline:
        remaining = max(1.0, deadline - now())
        try:
            result = _remote_exec(
                lium_bin, inputs["pod_id"], _remote_state_command(inputs["remote_run"]),
                run=run, timeout=min(30.0, remaining),
            )
            terminal = parse_state_output(result.stdout or "")
            if terminal and terminal.get("status") in {"stopped", "completed", "failed"}:
                reason = "supervisor_terminal"
                break
        except (OSError, subprocess.SubprocessError, RelayError):
            pass
        sleep(min(poll_seconds, max(0.0, deadline - now())))

    if terminal is not None and now() < deadline:
        remaining = max(1.0, deadline - now())
        command = _remote_post_command(
            inputs["remote_run"], commit=completion_commit,
            start=inputs["start_epoch"], price=inputs["price"], cap=inputs["cap"],
        )
        try:
            post_result = _remote_exec(
                lium_bin, inputs["pod_id"], command, run=run, timeout=remaining,
            )
            status = "complete" if post_result.returncode == 0 and provider_success(post_result.stdout or "") else "failed"
            reason = "post_training_complete" if status == "complete" else "post_training_failed"
        except (OSError, subprocess.SubprocessError):
            reason = "post_training_timeout_or_transport_failure"

    receipt_dir = private / "receipts"
    terminal_path = receipt_dir / "terminal-receipt.json"
    upload_path = receipt_dir / "evidence.tar.gz.upload.json"
    remote_run = inputs["remote_run"]
    terminal_copied = _scp(lium_bin, inputs["pod_id"], f"{remote_run}/terminal-receipt.json", terminal_path, run=run, timeout=30.0)
    upload_copied = _scp(lium_bin, inputs["pod_id"], f"{remote_run}/local-evidence/{RUN_NAME}-evidence.tar.gz.upload.json", upload_path, run=run, timeout=30.0)
    verified = upload_copied and _verified_upload(upload_path)
    archive_copied = False
    local_evidence = Path(__file__).resolve().parent / "local-evidence"
    if not verified:
        archive_copied = _scp(
            lium_bin, inputs["pod_id"], f"{remote_run}/local-evidence/{RUN_NAME}-evidence.tar.gz",
            local_evidence / f"{RUN_NAME}-evidence.tar.gz", run=run, timeout=120.0,
        )
        status = "failed"
    archive_valid = archive_copied and _valid_archive(local_evidence / f"{RUN_NAME}-evidence.tar.gz")
    terminal_valid = terminal_copied and _valid_terminal_receipt(terminal_path)
    release_evidence_preserved = verified or archive_valid or terminal_valid

    # Marker is intentionally written last. The parent rental controller treats
    # its presence as permission to remove exactly the rented pod.
    _write_json(private / "relay-receipt.json", {
        "schema": "text-sft512-consistent-relay-v1",
        "status": "complete" if status == "complete" and verified else "failed",
        "reason": reason,
        "supervisor_terminal": terminal is not None,
        "terminal_receipt_copied": terminal_copied,
        "upload_receipt_copied": upload_copied,
        "public_archive_hash_verified": verified,
        "failure_archive_copied": archive_valid,
        "terminal_receipt_valid": terminal_valid,
        "release_marker_written": release_evidence_preserved,
        "created_epoch": now(),
    })
    if release_evidence_preserved:
        marker = private / "release.ready"
        if marker.is_symlink():
            raise RelayError("release marker must not be a symlink")
        marker.write_text("verified-public-archive\n" if verified else "failure-evidence-preserved\n", encoding="utf-8")
    return 0 if status == "complete" and verified else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--lium-bin", default="lium")
    parser.add_argument("--completion-commit", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return relay(
        private_dir=args.private_dir,
        lium_bin=args.lium_bin,
        completion_commit=args.completion_commit,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RelayError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"completion relay: {exc}", file=sys.stderr)
        raise SystemExit(2)
