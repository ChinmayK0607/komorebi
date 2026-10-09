#!/usr/bin/env python3
"""Take over one already-rented Diverse-flow node without allocating.

This recovery path is intentionally separate from the allocator.  It verifies
the supplied pod ID, reuses or prepares the pinned Prime checkout on node
scratch, then hands the existing finite operation controller the original
rental window.  Every failure path releases exactly that pod.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shlex
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence
import urllib.parse


RUN = Path(__file__).resolve().parent
PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
PRIME_URL = "https://github.com/PrimeIntellect-ai/prime-rl.git"
DEFAULT_PRIME_SEED_ROOT = Path("/tmp/structured-prime-seed/prime")
DEFAULT_REMOTE_SCRATCH_ROOT = Path("/tmp/Diverse-flow-large")


def _load_operation():
    spec = importlib.util.spec_from_file_location("diverse_flow_recovery_operation", RUN / "operate_single.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Diverse-flow operation controller is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


operation = _load_operation()
base = operation.base


def _absolute_remote_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be absolute and cannot contain '..'")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_public_source(url: str, expected_sha256: str) -> tuple[str, str]:
    """Allow only a credential-free immutable Hugging Face resolve URL."""
    if len(expected_sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("source-sha256 must be lowercase hexadecimal SHA-256")
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("public-source-url must be a credential-free HTTPS Hugging Face URL")
    components = parsed.path.split("/")
    try:
        revision_index = components.index("resolve") + 1
        revision = components[revision_index]
    except (ValueError, IndexError):
        raise ValueError("public-source-url must use an immutable Hugging Face resolve revision") from None
    if not base.COMMIT_RE.fullmatch(revision):
        raise ValueError("public-source-url revision must be an immutable Git-style hash")
    return url, expected_sha256


@contextmanager
def _verified_source_sftp_bypass(*, source_bundle: Path, remote_source: Path, expected_sha256: str):
    """Skip exactly one verified source upload; preserve credential uploads."""
    source_bundle = base._regular_file(source_bundle, "source-bundle")
    if _sha256_file(source_bundle) != expected_sha256:
        raise ValueError("local source bundle hash does not match source-sha256")
    original = base._sftp_put

    def put(sftp: Any, local: Path, remote: str, *, timeout: float) -> None:
        if str(remote) == str(remote_source):
            if Path(local).expanduser().resolve() != source_bundle:
                raise ValueError("source SFTP bypass received an unexpected local file")
            if _sha256_file(source_bundle) != expected_sha256:
                raise ValueError("local source bundle changed before handoff")
            return
        original(sftp, local, remote, timeout=timeout)

    base._sftp_put = put
    try:
        yield
    finally:
        base._sftp_put = original


def prime_seed_command(seed_root: Path, commit: str = PRIME_COMMIT) -> str:
    """Build the bounded remote command that reuses or verifies Prime."""
    if not base.COMMIT_RE.fullmatch(commit) or commit != PRIME_COMMIT:
        raise ValueError("recovery is pinned to the reviewed Prime commit")
    seed_root = _absolute_remote_path(seed_root, "Prime seed root")
    parent = seed_root.parent
    seed = shlex.quote(str(seed_root))
    parent_arg = shlex.quote(str(parent))
    commit_arg = shlex.quote(commit)
    return (
        "set -Eeuo pipefail; "
        f"mkdir -p {parent_arg}; "
        f"test ! -L {seed}; "
        f"if [[ ! -d {seed}/.git ]]; then "
        f"  if [[ -e {seed} ]]; then echo 'Prime seed destination is not a Git checkout' >&2; exit 1; fi; "
        f"  git -c http.version=HTTP/1.1 clone -q {shlex.quote(PRIME_URL)} {seed}; "
        f"fi; "
        f"if ! git -C {seed} cat-file -e {shlex.quote(commit + '^{commit}')} 2>/dev/null; then "
        f"  git -C {seed} -c http.version=HTTP/1.1 fetch -q origin {commit_arg}; "
        f"fi; "
        f"git -C {seed} checkout -q --detach {commit_arg}; "
        f"git -C {seed} fsck --full; "
        f"test \"$(git -C {seed} rev-parse HEAD)\" = {commit_arg}"
    )


def _remote_prepare(
    *,
    api: Any,
    pod: Any,
    window: Any,
    seed_root: Path,
    remote_scratch_root: Path,
    public_source_url: str | None = None,
    public_source_sha256: str | None = None,
    now: Callable[[], float] = time.time,
) -> None:
    """Create scratch and verify the pinned Prime checkout on the node."""
    scratch = _absolute_remote_path(remote_scratch_root, "remote scratch root")
    command = (
        "set -Eeuo pipefail; "
        f"mkdir -p {shlex.quote(str(scratch))}; "
        f"chmod 755 {shlex.quote(str(scratch))}"
    )
    with api.ssh_connection(pod, timeout=min(30.0, base._remaining(window, now))) as ssh:
        base._remote_exec(
            ssh,
            command,
            timeout=min(float(base.SSH_TIMEOUT_SECONDS), base._remaining(window, now)),
            now=now,
            window=window,
        )
        base._remote_exec(
            ssh,
            prime_seed_command(seed_root),
            timeout=min(float(base.TRANSFER_TIMEOUT_SECONDS), base._remaining(window, now)),
            now=now,
            window=window,
        )
        if public_source_url is not None or public_source_sha256 is not None:
            if public_source_url is None or public_source_sha256 is None:
                raise ValueError("public-source-url and source-sha256 must be supplied together")
            _validate_public_source(public_source_url, public_source_sha256)
            remote_source = scratch / "source.bundle"
            partial = scratch / ".source.bundle.download.partial"
            download = (
                "set -Eeuo pipefail; "
                f"rm -f {shlex.quote(str(partial))}; "
                f"curl --fail --silent --show-error --location --max-time 300 "
                f"--proto '=https' --tlsv1.2 -o {shlex.quote(str(partial))} "
                f"{shlex.quote(public_source_url)}; "
                f"test \"$(sha256sum {shlex.quote(str(partial))} | awk '{{print $1}}')\" = {shlex.quote(public_source_sha256)}; "
                f"mv -f {shlex.quote(str(partial))} {shlex.quote(str(remote_source))}; "
                f"chmod 600 {shlex.quote(str(remote_source))}"
            )
            base._remote_exec(
                ssh,
                download,
                timeout=min(float(base.TRANSFER_TIMEOUT_SECONDS), base._remaining(window, now)),
                now=now,
                window=window,
            )


class _ReleaseTrackingAPI:
    """Proxy that records whether the finite operation released the pod."""

    def __init__(self, raw: Any):
        self.raw = raw
        self.released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    def rm(self, pod: Any) -> Any:
        result = self.raw.rm(pod)
        self.released = True
        return result


def _terminate_previous_controller(pid: int | None) -> None:
    if pid is None:
        return
    if pid <= 0 or pid == os.getpid():
        raise ValueError("previous-controller-pid must be a positive PID other than this process")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # A paused controller keeps SIGTERM pending until it is continued.  The
    # replacement owns the release guard before this function is called, so
    # it is safe to resume the known predecessor and let the pending term act.
    try:
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        pass


def recover(
    *,
    api: Any,
    pod_id: str,
    rental_start_epoch: float,
    price_usd_hour: float,
    cap_usd: float,
    source_bundle: Path,
    source_commit: str,
    hf_token_file: Path,
    gateway_env_file: Path,
    output_root: Path,
    prime_seed_root: Path = DEFAULT_PRIME_SEED_ROOT,
    remote_scratch_root: Path = DEFAULT_REMOTE_SCRATCH_ROOT,
    public_source_url: str | None = None,
    source_sha256: str | None = None,
    previous_controller_pid: int | None = None,
    now: Callable[[], float] = time.time,
) -> int:
    if not base.POD_ID_RE.fullmatch(pod_id):
        raise ValueError("pod-id is not a valid exact provider identifier")
    if not base.COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source-commit must be a full immutable Git commit")
    source_bundle = base._regular_file(source_bundle, "source-bundle")
    hf_token_file = base._regular_file(hf_token_file, "HF token file", owner_only=True)
    gateway_env_file = base._regular_file(gateway_env_file, "Gateway environment file", owner_only=True)
    if (public_source_url is None) != (source_sha256 is None):
        raise ValueError("public-source-url and source-sha256 must be supplied together")
    if public_source_url is not None and source_sha256 is not None:
        _validate_public_source(public_source_url, source_sha256)
        if _sha256_file(source_bundle) != source_sha256:
            raise ValueError("local source bundle hash does not match source-sha256")
    prime_seed_root = _absolute_remote_path(prime_seed_root, "Prime seed root")
    remote_scratch_root = _absolute_remote_path(remote_scratch_root, "remote scratch root")
    window = base.make_window(
        pod_id=pod_id,
        rental_start_epoch=rental_start_epoch,
        price_usd_per_hour=price_usd_hour,
        cap_usd=cap_usd,
    )
    tracked = _ReleaseTrackingAPI(api)
    release_target: Any = SimpleNamespace(id=pod_id)
    previous_bundle = base.REMOTE_BUNDLE
    environment_keys = {
        "PAINTER_LOCAL_SCRATCH_ROOT": os.environ.get("PAINTER_LOCAL_SCRATCH_ROOT"),
        "PAINTER_PRIME_SEED_ENABLED": os.environ.get("PAINTER_PRIME_SEED_ENABLED"),
        "PAINTER_PRIME_SEED_ROOT": os.environ.get("PAINTER_PRIME_SEED_ROOT"),
    }
    try:
        pod = base._bounded_call(
            lambda: base.resolve_exact_pod(api, pod_id),
            min(float(base.INVENTORY_TIMEOUT_SECONDS), base._remaining(window, now)),
        )
        release_target = pod
        _remote_prepare(
            api=api,
            pod=pod,
            window=window,
            seed_root=prime_seed_root,
            remote_scratch_root=remote_scratch_root,
            public_source_url=public_source_url,
            public_source_sha256=source_sha256,
            now=now,
        )
        # The wrapper now owns the exact-pod release path.  Only after this
        # point is it safe to stop a paused predecessor.
        _terminate_previous_controller(previous_controller_pid)
        os.environ["PAINTER_LOCAL_SCRATCH_ROOT"] = str(remote_scratch_root)
        os.environ["PAINTER_PRIME_SEED_ENABLED"] = "1"
        os.environ["PAINTER_PRIME_SEED_ROOT"] = str(prime_seed_root)
        base.REMOTE_BUNDLE = remote_scratch_root / "source.bundle"
        bypass = (
            _verified_source_sftp_bypass(
                source_bundle=source_bundle,
                remote_source=base.REMOTE_BUNDLE,
                expected_sha256=source_sha256,
            )
            if public_source_url is not None and source_sha256 is not None
            else nullcontext()
        )
        with bypass:
            return base.operate(
                window=window,
                source_bundle=source_bundle,
                source_commit=source_commit,
                hf_token_file=hf_token_file,
                gateway_env_file=gateway_env_file,
                output_root=output_root,
                api=tracked,
                now=now,
            )
    finally:
        base.REMOTE_BUNDLE = previous_bundle
        for key, value in environment_keys.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        release_error: BaseException | None = None
        if not tracked.released:
            try:
                base._bounded_call(
                    lambda: api.rm(release_target),
                    base.RELEASE_TIMEOUT_SECONDS,
                )
            except BaseException as exc:
                release_error = exc
        if release_error is not None:
            print("Diverse-flow recovery: exact pod release failed", file=sys.stderr)
            if sys.exc_info()[0] is None:
                raise RuntimeError("exact pod release failed") from release_error


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pod-id", required=True)
    value.add_argument("--rental-start-epoch", type=float, required=True)
    value.add_argument("--price-usd-hour", type=float, required=True)
    value.add_argument("--cap-usd", type=float, required=True)
    value.add_argument("--source-bundle", type=Path, required=True)
    value.add_argument("--source-commit", required=True)
    value.add_argument("--hf-token-file", type=Path, required=True)
    value.add_argument("--gateway-env-file", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--prime-seed-root", type=Path, default=DEFAULT_PRIME_SEED_ROOT)
    value.add_argument("--remote-scratch-root", type=Path, default=DEFAULT_REMOTE_SCRATCH_ROOT)
    value.add_argument("--public-source-url")
    value.add_argument("--source-sha256")
    value.add_argument("--previous-controller-pid", type=int)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    api = base._load_lium()
    return recover(
        api=api,
        pod_id=args.pod_id,
        rental_start_epoch=args.rental_start_epoch,
        price_usd_hour=args.price_usd_hour,
        cap_usd=args.cap_usd,
        source_bundle=args.source_bundle,
        source_commit=args.source_commit,
        hf_token_file=args.hf_token_file,
        gateway_env_file=args.gateway_env_file,
        output_root=args.output_root,
        prime_seed_root=args.prime_seed_root,
        remote_scratch_root=args.remote_scratch_root,
        public_source_url=args.public_source_url,
        source_sha256=args.source_sha256,
        previous_controller_pid=args.previous_controller_pid,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Diverse-flow recovery: {exc}", file=sys.stderr)
        raise SystemExit(2)
