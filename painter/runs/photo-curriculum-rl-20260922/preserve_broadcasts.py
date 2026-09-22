"""Finite safety backup for completed adapter broadcasts.

This is a node-local, bounded process.  It only consumes completed broadcast
directories; it does not start or stop training, call a judge, or alter native
checkpoint state.  The existing export and upload helpers remain the source of
truth for model validation and Hugging Face verification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - production runs on Linux
    fcntl = None  # type: ignore[assignment]


STEP_RE = re.compile(r"^step_(\d+)$")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


class PreservationError(RuntimeError):
    """A completed broadcast cannot be copied or verified safely."""


def _load_function(path: Path, function: str, module_name: str) -> Callable[..., Any]:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreservationError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        return getattr(module, function)
    except AttributeError as exc:
        raise PreservationError(f"helper lacks {function}: {path}") from exc


def load_helpers(export_helper: Path, upload_helper: Path) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load the reviewed node helpers without copying them into this run."""

    validate = _load_function(export_helper, "validate_files", "painter_rl_export_adapter")
    upload = _load_function(upload_helper, "upload_checkpoint", "painter_rl_upload_checkpoints")
    return validate, upload


def _step(path: Path) -> int | None:
    match = STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _marker_snapshot(path: Path) -> tuple[int, int]:
    if path.is_symlink() or not path.is_file():
        raise PreservationError(f"completion marker is missing or unsafe: {path}")
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def completed_steps(root: Path, *, first_step: int, max_step: int) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise PreservationError(f"broadcast root is not a regular directory: {root}")
    result: list[Path] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        step = _step(path)
        if step is None or not first_step <= step <= max_step:
            continue
        if path.is_symlink() or not path.is_dir():
            continue
        if (path / ".finished").is_file() and (path / ".receiver_ready").is_file():
            result.append(path)
    return sorted(result, key=lambda item: int(_step(item) or -1))


def pending_backups(root: Path, *, first_step: int, max_step: int) -> list[Path]:
    """Return staged step directories, including ones whose source aged out."""

    if not root.is_dir() or root.is_symlink():
        return []
    return sorted(
        (path for path in root.iterdir()
         if _step(path) is not None and first_step <= int(_step(path) or -1) <= max_step
         and path.is_dir() and not path.is_symlink()),
        key=lambda item: int(_step(item) or -1),
    )


def _copy_stable(source: Path, target: Path) -> None:
    before = (_marker_snapshot(source / ".finished"), _marker_snapshot(source / ".receiver_ready"))
    if not target.is_dir():
        raise PreservationError(f"temporary backup directory is missing: {target}")
    for name in ADAPTER_FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise PreservationError(f"broadcast adapter file is missing or symlinked: {path}")
        shutil.copy2(path, target / name)
    after = (_marker_snapshot(source / ".finished"), _marker_snapshot(source / ".receiver_ready"))
    if before != after:
        raise PreservationError(f"broadcast changed while copied: {source}")


def preserve_one(
    source: Path,
    backup_root: Path,
    *,
    repo: str,
    run: str,
    validate_files: Callable[[Path, int], dict[str, Any]],
    upload_checkpoint: Callable[[Path, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Copy, validate, and immutably upload one completed broadcast."""

    step = _step(source)
    if step is None:
        raise PreservationError(f"expected step_N directory: {source}")
    backup_root.mkdir(parents=True, exist_ok=True)
    final = backup_root / source.name
    if final.exists():
        if final.is_symlink() or not final.is_dir():
            raise PreservationError(f"existing backup is unsafe: {final}")
        # A crash after the rename but before the manifest write leaves a
        # complete local copy. Revalidate and re-upload idempotently rather
        # than refusing recovery forever.
        report = validate_files(final, step)
        receipt = upload_checkpoint(final, repo, run)
        _require_public_verified(receipt, step)
        return {
            "step": step,
            "source": str(source),
            "local_backup": str(final),
            "adapter_sha256": report.get("sha256"),
            "huggingface": receipt,
        }
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{source.name}.", dir=backup_root))
    # upload_checkpoint validates the basename as exactly step_N. Keep the
    # random staging component in the parent, not in the checkpoint name.
    temporary = temporary_parent / source.name
    try:
        temporary.mkdir()
        _copy_stable(source, temporary)
        temporary.replace(final)
        temporary_parent.rmdir()
    except Exception:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    # Keep the atomically staged final directory if validation or the network
    # upload fails. A later poll can retry it even if the source broadcast is
    # rotated out by the trainer.
    report = validate_files(final, step)
    receipt = upload_checkpoint(final, repo, run)
    _require_public_verified(receipt, step)
    return {
        "step": step,
        "source": str(source),
        "local_backup": str(final),
        "adapter_sha256": report.get("sha256"),
        "huggingface": receipt,
    }


def _require_public_verified(receipt: Any, step: int) -> None:
    if not isinstance(receipt, dict) or receipt.get("verified") is not True or receipt.get("private") is not False:
        raise PreservationError(f"checkpoint upload was not verified public for step {step}")


def _read_state(path: Path, *, run: str, repo: str, retention: int) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": "photo-curriculum-rl-preservation-v1", "run": run, "repo": repo,
                "retention": retention, "optimizer_state": "not_preserved_adapter_only_run", "records": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run") != run or value.get("repo") != repo:
        raise PreservationError(f"preservation manifest identity mismatch: {path}")
    value.setdefault("records", [])
    return value


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _apply_retention(state: dict[str, Any], backup_root: Path, retention: int) -> None:
    records = sorted(state["records"], key=lambda item: int(item["step"]))
    keep = {int(item["step"]) for item in records[-retention:]}
    for item in records:
        step = int(item["step"])
        directory = backup_root / f"step_{step}"
        if step not in keep and directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)
            item["local_retained"] = False
        else:
            item["local_retained"] = directory.is_dir()


@contextmanager
def _worker_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise PreservationError(f"preservation worker already running: {path}") from exc
    try:
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def run_worker(
    *,
    broadcast_root: Path,
    backup_root: Path,
    manifest: Path,
    repo: str,
    run: str,
    first_step: int,
    max_step: int,
    retention: int,
    deadline_seconds: float,
    poll_seconds: float,
    validate_files: Callable[[Path, int], dict[str, Any]],
    upload_checkpoint: Callable[[Path, str, str], dict[str, Any]],
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if first_step < 0 or max_step < first_step or retention < 1:
        raise ValueError("invalid step or retention bounds")
    with _worker_lock(manifest.with_name(manifest.name + ".lock")):
        state = _read_state(manifest, run=run, repo=repo, retention=retention)
        completed = {int(item["step"]) for item in state["records"]}
        deadline = time.monotonic() + max(0.0, deadline_seconds)
        expected = set(range(first_step, max_step + 1))
        while True:
            candidates = pending_backups(backup_root, first_step=first_step, max_step=max_step)
            candidates.extend(completed_steps(broadcast_root, first_step=first_step, max_step=max_step))
            seen_candidates: set[int] = set()
            for source in candidates:
                step = int(_step(source) or -1)
                if step in completed or step in seen_candidates:
                    continue
                seen_candidates.add(step)
                try:
                    record = preserve_one(source, backup_root, repo=repo, run=run,
                                          validate_files=validate_files, upload_checkpoint=upload_checkpoint)
                except Exception as exc:
                    # Keep retrying until the explicit finite deadline. Only
                    # the exception type is durable; helper messages can carry
                    # paths or provider details that do not belong in records.
                    state["last_error"] = {"step": step, "type": type(exc).__name__}
                    state["missing_steps"] = sorted(expected - completed)
                    _write_state(manifest, state)
                    continue
                state["records"].append(record)
                state["records"].sort(key=lambda item: int(item["step"]))
                completed.add(step)
                state.pop("last_error", None)
                _apply_retention(state, backup_root, retention)
                _write_state(manifest, state)
            if expected.issubset(completed):
                state["status"] = "complete"
                state.pop("missing_steps", None)
                _write_state(manifest, state)
                return state
            if time.monotonic() >= deadline:
                state["status"] = "deadline"
                state["missing_steps"] = sorted(expected - completed)
                _write_state(manifest, state)
                return state
            sleep(max(0.0, poll_seconds))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broadcast-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", default=os.environ.get("HF_CHECKPOINT_REPO", ""))
    parser.add_argument("--run", default=os.environ.get("HF_CHECKPOINT_RUN", ""))
    parser.add_argument("--export-helper", type=Path, required=True)
    parser.add_argument("--upload-helper", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=32)
    parser.add_argument("--retention", type=int, default=2)
    parser.add_argument("--deadline-seconds", type=float, default=172800.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.repo or not args.run:
        raise SystemExit("--repo/--run or HF_CHECKPOINT_REPO/HF_CHECKPOINT_RUN are required")
    validate, upload = load_helpers(args.export_helper.resolve(strict=True), args.upload_helper.resolve(strict=True))
    state = run_worker(
        broadcast_root=args.broadcast_root.expanduser().resolve(),
        backup_root=args.backup_root.expanduser().resolve(),
        manifest=args.manifest.expanduser().resolve(), repo=args.repo, run=args.run,
        first_step=args.first_step, max_step=args.max_step, retention=args.retention,
        deadline_seconds=args.deadline_seconds, poll_seconds=args.poll_seconds,
        validate_files=validate, upload_checkpoint=upload,
    )
    print(json.dumps({"status": state.get("status"), "records": len(state.get("records", [])),
                      "missing_steps": state.get("missing_steps", [])}, sort_keys=True))
    return 0 if state.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
