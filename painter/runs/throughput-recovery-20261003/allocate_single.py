#!/usr/bin/env python3
"""Allocate one verified two-GPU offer and hand it to the finite controller.

All validation and bundle checks happen before ``up``.  Once ``up`` returns,
the exact pod id is retained in memory and released from an outer ``finally``
on every failure path.  Provider connection details and credentials never
enter the private receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence


RUN = Path(__file__).resolve().parent
LIUM_PYTHON = Path("/Users/chinmay/.local/share/uv/tools/lium-io/bin/python")
GPU_COUNT = 2
GPU_TYPE = "H100"
MAX_CAP_USD = 20.0
DEFAULT_PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
DEFAULT_PRIME_BUNDLE_SHA256 = "044c9c7acf599674947c5b5242fa82377ca4b05ec1afad6bc498cf111a9cb79a"
READY_TIMEOUT_SECONDS = 180.0
READY_CALL_MAX_SECONDS = 40.0
SDK_READY_TIMEOUT_SECONDS = 30.0
SDK_READY_POLL_SECONDS = 4.0
READY_POLL_SECONDS = 30
OFFER_TIMEOUT_SECONDS = 40.0
ALLOCATION_TIMEOUT_SECONDS = 60.0
RELEASE_TIMEOUT_SECONDS = 40.0
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _load_operation() -> Any:
    spec = importlib.util.spec_from_file_location("throughput_recovery_operation", RUN / "operate_single.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("single operation controller is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_lium() -> Any:
    try:
        from lium.sdk import Lium
    except ImportError:
        site_packages = next(LIUM_PYTHON.parent.parent.glob("lib/python*/site-packages"), None)
        if site_packages is None:
            raise RuntimeError("lium SDK is unavailable")
        sys.path.insert(0, str(site_packages))
        from lium.sdk import Lium
    return Lium()


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _regular(path: Path, label: str, *, owner_only: bool = False) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if owner_only and path.stat().st_mode & 0o077:
        raise ValueError(f"{label} must be owner-only")
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_bundle(bundle: Path, commit: str, *, run: Callable[..., Any] = subprocess.run) -> str:
    """Verify the bundle, exact commit, and complete object database locally."""
    with tempfile.TemporaryDirectory(prefix="throughput-source-check-") as temp:
        bare = Path(temp) / "repo.git"
        init = run(["git", "init", "--bare", "--quiet", str(bare)], capture_output=True, text=True, check=False, timeout=30)
        if init.returncode != 0:
            raise ValueError("could not initialize source verification repository")
        verified = run(["git", "bundle", "verify", str(bundle)], capture_output=True, text=True, check=False, timeout=30)
        if verified.returncode != 0:
            raise ValueError("source bundle verification failed")
        fetched = run(["git", "--git-dir", str(bare), "fetch", "--quiet", str(bundle), commit], capture_output=True, text=True, check=False, timeout=60)
        if fetched.returncode != 0:
            raise ValueError("source bundle does not contain the requested commit")
        actual = run(["git", "--git-dir", str(bare), "rev-parse", "FETCH_HEAD"], capture_output=True, text=True, check=False, timeout=30)
        if actual.returncode != 0 or actual.stdout.strip().lower() != commit.lower():
            raise ValueError("source bundle commit identity mismatch")
        fsck = run(["git", "--git-dir", str(bare), "fsck", "--full"], capture_output=True, text=True, check=False, timeout=60)
        if fsck.returncode != 0:
            raise ValueError("source bundle object check failed")
    return _sha256(bundle)


def verify_prime_bundle(bundle: Path, commit: str, expected_sha256: str) -> str:
    """Verify the pinned Prime bundle, its complete history, and its bytes."""
    if not SHA256_RE.fullmatch(expected_sha256.strip()):
        raise ValueError("prime bundle SHA-256 must be 64 hexadecimal characters")
    actual = verify_source_bundle(bundle, commit)
    if actual.lower() != expected_sha256.lower():
        raise ValueError("prime bundle SHA-256 does not match the expected immutable bundle")
    return actual


def _pod_id(pod: Any) -> str:
    if isinstance(pod, dict):
        return str(pod.get("id", ""))
    return str(getattr(pod, "id", ""))


def _offer_id(offer: Any) -> str:
    value = getattr(offer, "id", None)
    if value is None and isinstance(offer, dict):
        value = offer.get("id")
    return str(value or "")


def _offer_price(offer: Any) -> float:
    value = getattr(offer, "price_per_hour", None)
    if value is None and isinstance(offer, dict):
        value = offer.get("price_per_hour", offer.get("price"))
    return _positive("offer price", float(value))


def _offer_gpu_type(offer: Any) -> str:
    value = getattr(offer, "gpu_type", None)
    if value is None and isinstance(offer, dict):
        value = offer.get("gpu_type")
    return str(value or "").strip().upper()


def select_offer(offers: Sequence[Any], *, max_price_usd_hour: float) -> Any:
    maximum = _positive("max-price-usd-hour", max_price_usd_hour)
    candidates = []
    for offer in offers:
        count = getattr(offer, "gpu_count", None)
        if count is None and isinstance(offer, dict):
            count = offer.get("gpu_count")
        try:
            price = _offer_price(offer)
        except (TypeError, ValueError):
            continue
        if int(count or 0) == GPU_COUNT and _offer_gpu_type(offer) == GPU_TYPE and price <= maximum:
            candidates.append((price, offer))
    if not candidates:
        raise RuntimeError("no two-GPU H100 offer is within the hourly price limit")
    return min(candidates, key=lambda item: item[0])[1]


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists():
        path.chmod(0o600)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


class _ReleaseTrackingAPI:
    def __init__(self, raw: Any):
        self.raw = raw
        self.released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    def rm(self, pod: Any) -> Any:
        result = self.raw.rm(pod)
        self.released = True
        return result


def _public_progress(event: str, **fields: Any) -> None:
    """Print bounded progress without provider IDs, addresses, or credentials."""
    payload = {"schema": "throughput-recovery-single-allocation-v1", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _ready_pod(pod: Any) -> bool:
    if pod is None or not _pod_id(pod):
        return False
    status = getattr(pod, "status", None)
    if status is None and isinstance(pod, dict):
        status = pod.get("status")
    has_ssh = bool(
        getattr(pod, "ssh_cmd", None)
        or getattr(pod, "ssh_command", None)
        or (isinstance(pod, dict) and (pod.get("ssh_cmd") or pod.get("ssh_command")))
    )
    if status is not None:
        normalized = str(status).strip().lower()
        if normalized in {"provisioning", "pending", "starting", "creating", "queued", "initializing"}:
            return False
        if normalized in {"ready", "running", "active", "started"}:
            return has_ssh
    return has_ssh if status is None else False


def wait_ready_bounded(*, api: Any, operation: Any, allocation: Any) -> Any:
    """Poll the SDK in short bounded calls while keeping a finite 180s total."""
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("allocated pod did not become SSH-ready within 180 seconds")
        timeout = min(READY_CALL_MAX_SECONDS, remaining)
        attempt += 1
        _public_progress("ready_wait_started", attempt=attempt, timeout_seconds=round(timeout, 3))
        try:
            ready = operation._bounded_call(
                lambda: api.wait_ready(
                    allocation,
                    timeout=min(SDK_READY_TIMEOUT_SECONDS, timeout),
                    poll_interval=min(SDK_READY_POLL_SECONDS, timeout),
                ),
                timeout,
            )
        except BaseException as exc:
            if type(exc).__name__ != "OperationError" or "bounded operation timed out" not in str(exc):
                raise
            _public_progress("ready_wait_timeout", attempt=attempt)
            continue
        if _ready_pod(ready):
            _public_progress("ready", attempt=attempt)
            return ready
        _public_progress("ready_pending", attempt=attempt)


def stage_prime_bundle(*, api: Any, operation: Any, window: Any, prime_bundle: Path, prime_commit: str, prime_sha256: str, now: Callable[[], float]) -> None:
    """Transfer and replay the verified Prime bundle before the main handoff."""
    pod = operation._bounded_call(
        lambda: operation.resolve_exact_pod(api, window.pod_id),
        min(float(operation.INVENTORY_TIMEOUT_SECONDS), operation._remaining(window, now)),
    )
    remote_root = Path(operation.REMOTE_ROOT)
    remote_bundle = remote_root / "prime.bundle"
    remote_repo = remote_root / "prime"
    sftp = None
    _public_progress("prime_stage_started")
    try:
        with api.ssh_connection(pod, timeout=min(30.0, operation._remaining(window, now))) as ssh:
            operation._remote_exec(
                ssh,
                f"mkdir -p {shlex.quote(str(remote_root))} && chmod 755 {shlex.quote(str(remote_root))}",
                timeout=min(float(operation.SSH_TIMEOUT_SECONDS), operation._remaining(window, now)),
                now=now,
                window=window,
            )
            sftp = ssh.open_sftp()
            operation._sftp_put(
                sftp,
                prime_bundle,
                str(remote_bundle),
                timeout=min(float(operation.TRANSFER_TIMEOUT_SECONDS), operation._remaining(window, now)),
            )
            checkout = operation._source_checkout_command(str(remote_bundle), str(remote_repo), prime_commit)
            verify = (
                f"set -Eeuo pipefail; "
                f"test \"$(sha256sum {shlex.quote(str(remote_bundle))} | awk '{{print $1}}')\" = {shlex.quote(prime_sha256)}; "
                f"{checkout}; "
                f"git -C {shlex.quote(str(remote_repo))} fsck --full; "
                f"test \"$(git -C {shlex.quote(str(remote_repo))} rev-parse HEAD)\" = {shlex.quote(prime_commit)}"
            )
            code, _stdout, _stderr = operation._remote_exec_capture(
                ssh,
                verify,
                timeout=min(float(operation.TRANSFER_TIMEOUT_SECONDS), operation._remaining(window, now)),
                now=now,
                window=window,
            )
            if code != 0:
                raise RuntimeError("Prime bundle staging checkout or fsck failed")
            _public_progress("prime_stage_verified")
    finally:
        if sftp is not None:
            try:
                operation._bounded_call(sftp.close, min(5.0, operation._remaining(window, now)))
            except BaseException:
                pass


def allocate_and_operate(
    *,
    api: Any,
    operation: Any,
    max_price_usd_hour: float,
    cap_usd: float,
    source_bundle: Path,
    source_commit: str,
    prime_bundle: Path,
    prime_commit: str,
    prime_sha256: str,
    hf_token_file: Path,
    gateway_env_file: Path,
    output_root: Path,
    now: Callable[[], float] = time.time,
) -> int:
    source_bundle = _regular(source_bundle, "source-bundle")
    hf_token_file = _regular(hf_token_file, "HF token file", owner_only=True)
    gateway_env_file = _regular(gateway_env_file, "Gateway environment file", owner_only=True)
    if not operation.COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source-commit must be a full immutable Git commit")
    prime_bundle = _regular(prime_bundle, "prime-bundle")
    if not operation.COMMIT_RE.fullmatch(prime_commit):
        raise ValueError("prime-commit must be a full immutable Git commit")
    maximum = _positive("max-price-usd-hour", max_price_usd_hour)
    cap = _positive("cap-usd", cap_usd)
    if cap > MAX_CAP_USD:
        raise ValueError("cap-usd must be at most 20")
    source_sha = verify_source_bundle(source_bundle, source_commit)
    prime_sha = verify_prime_bundle(prime_bundle, prime_commit, prime_sha256)
    output_root = output_root.expanduser()
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("output-root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)

    offers = operation._bounded_call(
        lambda: api.ls(gpu_type=GPU_TYPE, gpu_count=GPU_COUNT), OFFER_TIMEOUT_SECONDS
    )
    offer = select_offer(offers, max_price_usd_hour=maximum)
    price = _offer_price(offer)
    started = float(now())
    if not math.isfinite(started) or started <= 0:
        raise ValueError("rental start time must be positive and finite")
    name = "throughput-recovery-20261003"
    executor_id = _offer_id(offer)
    if not executor_id:
        raise RuntimeError("selected offer did not contain an exact executor id")
    allocation = operation._bounded_call(
        lambda: api.up(executor_id=executor_id, name=name, enable_volume_encryption=True),
        ALLOCATION_TIMEOUT_SECONDS,
    )
    pod_id = _pod_id(allocation)
    if not pod_id:
        raise RuntimeError("provider allocation did not return an exact pod id")
    release_target = SimpleNamespace(id=pod_id)
    receipt = {
        "schema": "throughput-recovery-single-allocation-v1",
        "pod_id": pod_id,
        "rental_start_epoch": started,
        "price_usd_per_hour": price,
        "cap_usd": cap,
        "source_commit": source_commit,
        "source_bundle_sha256": source_sha,
        "prime_commit": prime_commit,
        "prime_bundle_sha256": prime_sha,
        "status": "allocated",
    }
    receipt_path = output_root / "allocation-receipt.json"
    tracked = _ReleaseTrackingAPI(api)
    try:
        _write_receipt(receipt_path, receipt)
        ready = wait_ready_bounded(api=api, operation=operation, allocation=allocation)
        window = operation.make_window(
            pod_id=pod_id,
            rental_start_epoch=started,
            price_usd_per_hour=price,
            cap_usd=cap,
        )
        receipt["status"] = "ready"
        _write_receipt(receipt_path, receipt)
        stage_prime_bundle(
            api=api,
            operation=operation,
            window=window,
            prime_bundle=prime_bundle,
            prime_commit=prime_commit,
            prime_sha256=prime_sha,
            now=now,
        )
        receipt["status"] = "prime_staged"
        _write_receipt(receipt_path, receipt)
        return operation.operate(
            window=window,
            source_bundle=source_bundle,
            source_commit=source_commit,
            hf_token_file=hf_token_file,
            gateway_env_file=gateway_env_file,
            output_root=output_root,
            api=tracked,
        )
    finally:
        if not tracked.released:
            try:
                operation._bounded_call(lambda: api.rm(release_target), RELEASE_TIMEOUT_SECONDS)
            except BaseException:
                pass
        receipt["status"] = "released" if tracked.released else "release_attempted"
        try:
            _write_receipt(receipt_path, receipt)
        except BaseException:
            # A receipt failure must never prevent the exact pod cleanup or
            # replace the operation's original failure.
            pass


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--max-price-usd-hour", type=float, required=True)
    value.add_argument("--cap-usd", type=float, required=True)
    value.add_argument("--source-bundle", type=Path, required=True)
    value.add_argument("--source-commit", required=True)
    value.add_argument("--prime-bundle", type=Path, required=True)
    value.add_argument("--prime-commit", "--prime-pin", dest="prime_commit", default=DEFAULT_PRIME_COMMIT)
    value.add_argument(
        "--prime-sha256",
        "--prime-bundle-sha256",
        "--expected-prime-sha256",
        "--expected-prime-bundle-sha256",
        dest="prime_sha256",
        required=True,
    )
    value.add_argument("--hf-token-file", type=Path, default=Path.home() / ".cache/huggingface/token")
    value.add_argument("--gateway-env-file", type=Path, default=RUN.parents[2] / "painter/benchmarks/openrouter-teachers-20260922/.env.local")
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--plan-only", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    operation = _load_operation()
    source_bundle = _regular(args.source_bundle, "source-bundle")
    prime_bundle = _regular(args.prime_bundle, "prime-bundle")
    hf_token = _regular(args.hf_token_file, "HF token file", owner_only=True)
    gateway = _regular(args.gateway_env_file, "Gateway environment file", owner_only=True)
    if not operation.COMMIT_RE.fullmatch(args.source_commit):
        raise SystemExit("source-commit must be a full immutable Git commit")
    if not operation.COMMIT_RE.fullmatch(args.prime_commit):
        raise SystemExit("prime-commit must be a full immutable Git commit")
    cap = _positive("cap-usd", args.cap_usd)
    maximum = _positive("max-price-usd-hour", args.max_price_usd_hour)
    source_sha = verify_source_bundle(source_bundle, args.source_commit)
    prime_sha = verify_prime_bundle(prime_bundle, args.prime_commit, args.prime_sha256)
    if args.plan_only:
        print(json.dumps({
            "schema": "throughput-recovery-single-allocation-plan-v1",
            "gpu_count": GPU_COUNT,
            "max_price_usd_hour": maximum,
            "cap_usd": cap,
            "source_commit": args.source_commit,
            "source_bundle_sha256": source_sha,
            "prime_commit": args.prime_commit,
            "prime_bundle_sha256": prime_sha,
        }, sort_keys=True))
        return 0
    api = _load_lium()
    return allocate_and_operate(
        api=api, operation=operation, max_price_usd_hour=maximum, cap_usd=cap,
        source_bundle=source_bundle, source_commit=args.source_commit,
        prime_bundle=prime_bundle, prime_commit=args.prime_commit, prime_sha256=args.prime_sha256,
        hf_token_file=hf_token, gateway_env_file=gateway,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"single allocation: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
