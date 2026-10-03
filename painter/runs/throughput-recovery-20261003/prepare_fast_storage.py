#!/usr/bin/env python3
"""Prepare the recovery profile's optional local fast-storage links.

The helper only selects the uv cache before bootstrap and manages the pinned
Prime virtualenv after checkout/submodule verification.  It never moves files
or touches the run's credential paths.  Existing Prime environments are
rejected so a live environment cannot be migrated underneath a running
process; an existing uv cache is preserved and simply left unused by the
explicit fast-cache override.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any


SCHEMA = "throughput-recovery-fast-storage-v1"
PROFILE = "throughput-recovery-20261003"


def _ensure_fast_root(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise RuntimeError(f"fast-storage root must not be a symlink: {path}")
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    # A later renderer phase may deliberately make this root traversable for
    # the unprivileged browser account.  Do not undo that on the idempotent
    # Prime phase; cache children remain private.
    if not existed or not path.stat().st_mode & 0o111:
        path.chmod(0o700)
    return path


def _select_cache(logical: Path, target: Path) -> dict[str, str | bool]:
    logical = logical.expanduser()
    target = target.expanduser()
    if target.is_symlink():
        raise RuntimeError(f"uv cache fast target must not be a symlink: {target}")
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    return {
        "status": "selected_fast_target",
        "logical": str(logical),
        "target": str(target),
        "existing_preserved": logical.exists(),
    }


def _ensure_link(logical: Path, target: Path, label: str) -> dict[str, str]:
    logical = logical.expanduser()
    target = target.expanduser()
    if target.is_symlink():
        raise RuntimeError(f"{label} fast target must not be a symlink: {target}")
    if logical.is_symlink():
        if logical.resolve(strict=False) != target.resolve(strict=False):
            raise RuntimeError(f"{label} link points somewhere else: {logical}")
        return {"status": "reused_symlink", "logical": str(logical), "target": str(target)}
    if logical.exists():
        raise RuntimeError(f"{label} exists; refusing live migration: {logical}")
    logical.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    logical.symlink_to(target, target_is_directory=True)
    return {"status": "created_symlink", "logical": str(logical), "target": str(target)}


def _write_receipt(run_root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    output = run_root / "fast-storage-receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined: dict[str, Any] = {}
    if output.is_file():
        try:
            previous = json.loads(output.read_text())
            if isinstance(previous, dict) and previous.get("schema") == SCHEMA:
                combined.update(previous)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    combined.update(receipt)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return combined


def prepare(
    *,
    phase: str,
    run_root: Path,
    fast_root: Path,
    uv_source: Path,
    prime_root: Path | None = None,
    system: str | None = None,
    nofile_soft: int | None = None,
    nofile_hard: int | None = None,
    nofile_target: int | None = None,
    uv_concurrent_installs: int | None = None,
) -> dict[str, Any]:
    """Create or verify one profile-bound fast-storage placement."""
    if phase not in {"cache", "prime", "render"}:
        raise ValueError("phase must be cache, prime, or render")
    if (system or platform.system()) != "Linux":
        raise RuntimeError("fast-storage preparation requires Linux")
    run_root = run_root.expanduser().resolve()
    fast_root = _ensure_fast_root(fast_root)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "phase": phase,
        "fast_root": str(fast_root),
        "probe": {"status": "not-run", "reason": "placement-only"},
    }
    if nofile_soft is not None or nofile_hard is not None:
        receipt["nofile"] = {"soft": nofile_soft, "hard": nofile_hard, "target": nofile_target}
    if uv_concurrent_installs is not None:
        receipt["uv_concurrent_installs"] = uv_concurrent_installs
    if phase == "cache":
        uv_target = fast_root / "uv-cache"
        receipt["uv_cache"] = _select_cache(uv_source, uv_target)
        receipt["credentials_touched"] = False
    elif phase == "prime":
        if prime_root is None:
            raise ValueError("prime_root is required for the prime phase")
        prime_root = prime_root.expanduser().resolve()
        receipt["prime_venv"] = _ensure_link(prime_root / ".venv", fast_root / "prime-venv", "Prime environment")
        receipt["credentials_touched"] = False
    else:
        browser_root = fast_root / "renderer-browsers"
        if browser_root.is_symlink():
            raise RuntimeError(f"renderer browser target must not be a symlink: {browser_root}")
        browser_root.mkdir(parents=True, exist_ok=True)
        browser_root.chmod(0o755)
        fast_root.chmod(0o711)
        receipt["renderer"] = {
            "browser_root": str(browser_root),
            "uv_cache": str(fast_root / "uv-cache"),
            "status": "public_traversal_root",
        }
        receipt["credentials_touched"] = False
    return _write_receipt(run_root, receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cache", "prime", "render"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fast-root", type=Path, required=True)
    parser.add_argument("--uv-source", type=Path, default=Path.home() / ".cache/uv")
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--nofile-soft", type=int)
    parser.add_argument("--nofile-hard", type=int)
    parser.add_argument("--nofile-target", type=int)
    parser.add_argument("--uv-concurrent-installs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepare(
        phase=args.phase,
        run_root=args.run_root,
        fast_root=args.fast_root,
        uv_source=args.uv_source,
        prime_root=args.prime_root,
        nofile_soft=args.nofile_soft,
        nofile_hard=args.nofile_hard,
        nofile_target=args.nofile_target,
        uv_concurrent_installs=args.uv_concurrent_installs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
