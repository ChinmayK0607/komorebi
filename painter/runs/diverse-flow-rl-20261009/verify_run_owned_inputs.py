#!/usr/bin/env python3
"""Verify run-owned launch files against the checked-out source blob."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import subprocess


SCHEMA = "Diverse-flow-run-owned-inputs-v1"


def _relative(root: str, name: str) -> str:
    base = PurePosixPath(root)
    child = PurePosixPath(name)
    if child.is_absolute() or any(part in {"", ".", ".."} for part in child.parts):
        raise ValueError(f"invalid run-owned file name: {name!r}")
    return str(base / child)


def verify(*, repo: Path, run_root: Path, relative_root: str, files: list[str]) -> dict[str, object]:
    repo = repo.expanduser().resolve(strict=True)
    run_root = run_root.expanduser().resolve(strict=True)
    if not files:
        raise ValueError("at least one run-owned file is required")
    hashes: dict[str, str] = {}
    for name in files:
        local = run_root / name
        if local.is_symlink() or not local.is_file():
            raise ValueError(f"run-owned source is missing or symlinked: {local}")
        relative = _relative(relative_root, name)
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"checked-out run-owned source blob is missing: {relative}")
        expected = bytes(result.stdout)
        actual = local.read_bytes()
        if actual != expected:
            raise ValueError(f"run-owned source differs from checked-out blob: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return {"schema": SCHEMA, "status": "verified", "files": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--relative-root", required=True)
    parser.add_argument("--file", dest="files", action="append", required=True)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(verify(repo=args.repo, run_root=args.run_root,
                           relative_root=args.relative_root, files=args.files), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
