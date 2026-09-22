"""Finite Luna-low queue bridge for the photo RL run.

The GPU node owns the aesthetic request queue and the local controller owns
the Codex/Luna credentials.  ``pull`` exports and stages one finite queue
batch over an operator-supplied SSH connection, ``run`` invokes the existing
strict local CLI worker, and ``push`` sends only validated label documents
back so the node-side queue can build and install its two-orientation
receipts.  No daemon, scheduler, endpoint, or credential is created here.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import shutil
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
DEFAULT_PROTOCOL_DIR = REPO_ROOT / "painter" / "runs" / "aesthetic-rl-20260921"
DEFAULT_REMOTE_ROOT = Path("/root/painter/runs/photo-curriculum-rl-20260922")
BATCH_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
HEX64 = frozenset("0123456789abcdef")
TRANSPORT_ATTEMPTS = 5
TRANSPORT_BACKOFF_SECONDS = (2.0, 4.0, 8.0, 16.0)
DISK_PRESSURE_BACKOFF_SECONDS = (5.0, 10.0, 20.0, 30.0)
LOCAL_FREE_SPACE_FLOOR_BYTES = 1024 * 1024 * 1024


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_name(value: str) -> str:
    if not isinstance(value, str) or not BATCH_PATTERN.fullmatch(value):
        raise ValueError("batch must be a simple filename-safe name")
    return value


def _ssh(args: argparse.Namespace) -> list[str]:
    if not args.host:
        raise ValueError("--host is required for node transport")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if args.identity:
        command.extend(["-i", str(args.identity)])
    if args.known_hosts:
        command.extend(["-o", f"UserKnownHostsFile={args.known_hosts}"])
    command.extend(["-p", str(args.port), f"{args.user}@{args.host}"])
    return command


def _remote_root(args: argparse.Namespace) -> Path:
    return Path(args.remote_root)


def _queue_script(args: argparse.Namespace) -> Path:
    return Path(args.queue_script) if args.queue_script else _remote_root(args) / "judge_queue.py"


def _remote_queue(args: argparse.Namespace, action: str, batch: str | None = None, *, limit: int | None = None) -> dict[str, Any]:
    command = _ssh(args) + ["python3", str(_queue_script(args)), action]
    if action != "status":
        if batch is None:
            raise ValueError(f"batch is required for remote queue action {action}")
        command.append(batch)
    command.extend(["--root", str(_remote_root(args))])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    output = completed.stdout.strip().splitlines()
    if not output:
        raise ValueError(f"remote queue {action} returned no JSON")
    try:
        value = json.loads(output[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"remote queue {action} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"remote queue {action} returned a non-object")
    return value


def _remote_batch_exists(args: argparse.Namespace, batch: str) -> bool:
    remote_batch = _remote_root(args) / "reward-batches" / batch
    result = subprocess.run(
        _ssh(args) + ["test", "-d", str(remote_batch)],
        check=False, capture_output=True, timeout=120,
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    return result.returncode == 0


def _safe_members(tar: tarfile.TarFile, batch: str) -> None:
    for member in tar.getmembers():
        path = Path(member.name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] != batch
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError(f"unsafe or unsupported queue archive member: {member.name}")


def _copy_tree_verified(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError(f"conflicting queue directory: {target}")
            target.mkdir(parents=True, exist_ok=True)
        else:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsupported queue member: {path}")
            if target.exists():
                if target.is_symlink() or not target.is_file() or _sha(target) != _sha(path):
                    raise ValueError(f"conflicting queue file: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())


def _rewrite_local_paths(batch_dir: Path, batch: str) -> int:
    """Make staged judge docs portable after the node archive is downloaded."""

    changed = 0
    for path in sorted(batch_dir.glob("judge-*.json")):
        document = json.loads(path.read_text())
        comparisons = document.get("comparisons")
        if not isinstance(comparisons, list) or len(comparisons) != 1:
            raise ValueError(f"invalid staged comparison document: {path.name}")
        pair = comparisons[0]
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or pair_id != path.stem.removeprefix("judge-"):
            raise ValueError(f"staged comparison filename/id mismatch: {path.name}")
        for role, expected_hash in pair.get("image_sha256", {}).items():
            if role not in {"reference", "A", "B"}:
                raise ValueError(f"unexpected staged image role: {role}")
            value = pair.get(role)
            if not isinstance(value, str):
                raise ValueError(f"missing staged image path: {path.name}:{role}")
            relative = Path(pair_id) / Path(value).name
            image = batch_dir / relative
            if not image.is_file() or image.is_symlink() or _sha(image) != expected_hash:
                raise ValueError(f"staged image hash mismatch: {path.name}:{role}")
            if value != relative.as_posix():
                pair[role] = relative.as_posix()
                changed += 1
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return changed


def pull(args: argparse.Namespace) -> dict[str, Any]:
    batch = _batch_name(args.batch)
    local_root = Path(args.local_root).resolve()
    local_batches = local_root / "reward-batches"
    local_batches.mkdir(parents=True, exist_ok=True)
    if not _remote_batch_exists(args, batch):
        exported = _remote_queue(args, "export", batch, limit=args.limit)
        if int(exported.get("requests", 0)) == 0:
            return {"batch": batch, "status": "no_pending_work", "queue": exported}
    _remote_queue(args, "stage", batch)

    archive_path = local_root / f".{batch}.tar.gz.part"
    remote_batches = _remote_root(args) / "reward-batches"
    with archive_path.open("wb") as output:
        subprocess.run(
            _ssh(args) + ["tar", "-czf", "-", "-C", str(remote_batches), batch],
            check=True, stdout=output, stderr=subprocess.PIPE, timeout=300,
        )
    archive_sha = _sha(archive_path)
    destination = local_batches / batch
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_members(archive, batch)
        with tempfile.TemporaryDirectory(prefix=f".{batch}.extract-", dir=local_batches) as temporary:
            temporary_root = Path(temporary)
            try:
                archive.extractall(temporary_root, filter="data")
            except TypeError:  # Python 3.11; members were validated above.
                archive.extractall(temporary_root)
            extracted = temporary_root / batch
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                raise ValueError(f"local batch destination is not a directory: {destination}")
            if destination.exists():
                _copy_tree_verified(extracted, destination)
            else:
                extracted.replace(destination)
    archive_path.unlink()
    rewritten = _rewrite_local_paths(destination, batch)
    receipt = {
        "batch": batch,
        "archive_sha256": archive_sha,
        "local_batch": str(destination),
        "rewritten_absolute_paths": rewritten,
        "queue_protocol": "aesthetic-pairwise-v1",
        "credentials": "local_codex_cli_only",
    }
    (local_root / f"pull-{batch}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def run_local(args: argparse.Namespace) -> dict[str, Any]:
    local_root = Path(args.local_root).resolve()
    usage = shutil.disk_usage(local_root)
    if usage.free < LOCAL_FREE_SPACE_FLOOR_BYTES:
        raise OSError(
            errno.ENOSPC,
            f"local judge root has only {usage.free} free bytes; pausing before Luna calls",
        )
    # Import only on the controller, where the Codex CLI and its credentials
    # exist.  The node never receives this process or its environment.
    sys.path.insert(0, str(REPO_ROOT))
    from painter.luna_cli_judge import run_batch

    batch_dir = local_root / "reward-batches" / _batch_name(args.batch)
    summary = run_batch(
        batch_dir,
        output_dir=batch_dir,
        protocol_dir=Path(args.protocol_dir).resolve(),
        max_concurrency=args.max_concurrency,
        codex_bin=args.codex_bin,
        resume=True,
    )
    return {"batch": args.batch, "local_summary": summary, "receipt_installation": "not performed"}


def _labels_archive(batch_dir: Path) -> bytes:
    labels = sorted(batch_dir.glob("labels-*.json"))
    if not labels:
        raise ValueError(f"no labels-*.json found in {batch_dir}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in labels:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"label is not a regular file: {path}")
            archive.add(path, arcname=path.name, recursive=False)
    return buffer.getvalue()


def push(args: argparse.Namespace) -> dict[str, Any]:
    batch = _batch_name(args.batch)
    local_batch = Path(args.local_root).resolve() / "reward-batches" / batch
    payload = _labels_archive(local_batch)
    remote_batch = _remote_root(args) / "reward-batches" / batch
    subprocess.run(
        _ssh(args) + ["tar", "-xzf", "-", "-C", str(remote_batch)],
        input=payload, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    receipts = _remote_queue(args, "receipts", batch)
    installed = _remote_queue(args, "install", batch)
    return {
        "batch": batch,
        "labels_archive_sha256": hashlib.sha256(payload).hexdigest(),
        "receipts": receipts,
        "installed": installed,
    }


def _is_ssh_transport_failure(error: BaseException) -> bool:
    """Recognize only the SSH failures safe to retry at a batch boundary."""
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    return isinstance(error, subprocess.CalledProcessError) and error.returncode == 255


def _is_disk_pressure_failure(error: BaseException) -> bool:
    """Recognize ENOSPC/EDQUOT through the local judge wrapper's cause chain."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in {errno.ENOSPC, errno.EDQUOT}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _remote_exit_probe(args: argparse.Namespace) -> bool:
    """Check the finite-run marker without leaking permanent SSH failures."""
    result = subprocess.run(
        _ssh(args) + ["test", "-f", str(_remote_root(args) / "rl.exit")],
        check=False, capture_output=True, timeout=120,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    if result.returncode == 255:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    raise RuntimeError(f"judge bridge exit probe failed for batch {args.batch}")


def _run_batch_with_transport_retry(
    args: argparse.Namespace,
    *,
    deadline: float,
) -> dict[str, Any]:
    """Run one batch, retrying SSH transport and bounded local disk pressure."""
    batch = args.batch
    transport_attempt = 0
    disk_pressure_attempt = 0
    while True:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"judge bridge transport deadline exceeded for batch {batch}")
        phase = "pull"
        try:
            pulled = pull(args)
            if pulled.get("status") == "no_pending_work":
                phase = "exit_probe"
                return {
                    "batch": batch,
                    "pull": pulled,
                    "run_exit": _remote_exit_probe(args),
                }
            phase = "local"
            local = run_local(args)
            phase = "push"
            pushed = push(args)
            return {"batch": batch, "pull": pulled, "local": local, "push": pushed}
        except Exception as error:
            if phase in {"pull", "local"} and _is_disk_pressure_failure(error):
                disk_pressure_attempt += 1
                delay = DISK_PRESSURE_BACKOFF_SECONDS[
                    min(disk_pressure_attempt - 1, len(DISK_PRESSURE_BACKOFF_SECONDS) - 1)
                ]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"judge bridge disk-pressure deadline exceeded for batch {batch}"
                    ) from error
                delay = min(delay, remaining)
                print(json.dumps({
                    "status": "disk_pressure_retry",
                    "batch": batch,
                    "attempt": disk_pressure_attempt + 1,
                    "delay_seconds": delay,
                }, sort_keys=True), file=sys.stderr)
                time.sleep(delay)
                continue
            # A local judge/validation error is terminal even if it happens to
            # use subprocess and return 255.  Retry only remote pull/push SSH.
            if phase not in {"pull", "push", "exit_probe"} or not _is_ssh_transport_failure(error):
                raise
            transport_attempt += 1
            if transport_attempt >= TRANSPORT_ATTEMPTS:
                raise RuntimeError(
                    f"judge bridge transport failed for batch {batch} after {transport_attempt} attempts"
                ) from None
            delay = TRANSPORT_BACKOFF_SECONDS[transport_attempt - 1]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"judge bridge transport deadline exceeded for batch {batch}"
                ) from None
            delay = min(delay, remaining)
            print(json.dumps({
                "status": "transport_retry",
                "batch": batch,
                "attempt": transport_attempt + 1,
                "delay_seconds": delay,
            }, sort_keys=True), file=sys.stderr)
            time.sleep(delay)


def worker(args: argparse.Namespace) -> dict[str, Any]:
    """Process finite queue batches until node exit or a hard deadline."""

    if args.deadline_seconds <= 0 or args.poll_seconds <= 0 or args.max_batches < 1:
        raise ValueError("deadline, poll interval, and max batches must be positive")
    started = time.monotonic()
    processed: list[dict[str, Any]] = []
    poll_index = 0
    while len(processed) < args.max_batches:
        if time.monotonic() - started >= args.deadline_seconds:
            return {"status": "deadline", "processed": processed}
        # Empty polls do not consume the finite work-batch budget. A long RL
        # run may have no pending requests while its next rollout group is
        # still rendering; the deadline, rather than max_batches, bounds this
        # wait.
        poll_index += 1
        args.batch = f"{args.batch_prefix}-{poll_index:04d}"
        batch_result = _run_batch_with_transport_retry(
            args, deadline=started + args.deadline_seconds
        )
        pulled = batch_result["pull"]
        if pulled.get("status") == "no_pending_work":
            if batch_result.get("run_exit"):
                return {"status": "run_exit", "processed": processed}
            time.sleep(min(args.poll_seconds, max(0.0, args.deadline_seconds - (time.monotonic() - started))))
            continue
        processed.append(batch_result)
    return {"status": "max_batches", "processed": processed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pull", "run", "push", "status", "worker"))
    parser.add_argument("--batch")
    parser.add_argument("--local-root", type=Path, default=RUN_ROOT / "judge-live")
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--queue-script", type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-prefix", default="luna-auto")
    parser.add_argument("--max-batches", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--deadline-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    if args.action in {"pull", "run", "push"} and args.batch is None:
        parser.error(f"--batch is required for {args.action}")
    if args.action == "run":
        value = run_local(args)
    elif args.action == "pull":
        value = pull(args)
    elif args.action == "push":
        value = push(args)
    elif args.action == "worker":
        value = worker(args)
    else:
        value = _remote_queue(args, "status")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
