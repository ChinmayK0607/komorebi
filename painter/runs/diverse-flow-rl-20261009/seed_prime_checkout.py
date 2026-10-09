#!/usr/bin/env python3
"""Seed a scratch Prime checkout from a verified local Git checkout.

Only tracked Git source is copied.  The destination virtualenv, submodule
working trees, caches, and other generated state are created by the normal
bootstrap afterward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


EXPECTED_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
UPSTREAM_URL = "https://github.com/PrimeIntellect-ai/prime-rl.git"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_source(source: Path, expected_commit: str) -> str:
    if source.is_symlink() or not source.is_dir() or not (source / ".git").exists():
        raise ValueError(f"Prime seed source is not a regular Git checkout: {source}")
    top = Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    if top != source.resolve():
        raise ValueError(f"Prime seed source resolves outside its declared root: {source}")
    commit = _git(source, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"Prime seed source is not pinned to {expected_commit}: {commit}")
    if _git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Prime seed source has uncommitted or untracked files")
    return commit


def _verify_destination(destination: Path, expected_commit: str) -> str:
    if destination.is_symlink() or not destination.is_dir() or not (destination / ".git").exists():
        raise ValueError(f"existing Prime destination is not a regular Git checkout: {destination}")
    commit = _git(destination, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(f"existing Prime destination is not pinned to {expected_commit}: {commit}")
    if _git(destination, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("existing Prime destination has tracked modifications")
    return commit


def _set_upstream(destination: Path) -> None:
    """Make relative .gitmodules URLs resolve against the public Prime repo."""
    try:
        _git(destination, "remote", "set-url", "origin", UPSTREAM_URL)
    except subprocess.CalledProcessError:
        _git(destination, "remote", "add", "origin", UPSTREAM_URL)


def seed(*, source: Path, destination: Path, expected_commit: str = EXPECTED_COMMIT) -> dict[str, object]:
    source = source.expanduser()
    if source.is_symlink():
        raise ValueError(f"Prime seed source must not be a symlink: {source}")
    source = source.resolve()
    destination = destination.expanduser()
    if not expected_commit or len(expected_commit) != 40 or any(char not in "0123456789abcdef" for char in expected_commit):
        raise ValueError("expected Prime commit must be a lowercase full Git hash")
    source_commit = _verify_source(source, expected_commit)
    if destination.parent.exists() and destination.parent.is_symlink():
        raise ValueError("Prime seed destination parent must not be a symlink")
    destination_parent = destination.parent.resolve()
    destination = destination_parent / destination.name
    if destination == source or source in destination.parents:
        raise ValueError("Prime seed destination must be outside the source checkout")
    if destination.exists() or destination.is_symlink():
        existing_commit = _verify_destination(destination, expected_commit)
        _set_upstream(destination)
        return {
            "schema": "Diverse-flow-prime-seed-v1",
            "status": "existing_verified",
            "source_commit": source_commit,
            "destination": str(destination),
            "destination_commit": existing_commit,
            "generated_environment_copied": False,
        }
    destination_parent.mkdir(parents=True, exist_ok=True)
    try:
        # A local clone copies Git objects and tracked worktree files only;
        # ignored .venv/cache/submodule environments never enter scratch.
        subprocess.run(
            ["git", "clone", "--local", "--no-hardlinks", str(source), str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
        _set_upstream(destination)
        destination_commit = _verify_destination(destination, expected_commit)
        if (destination / ".venv").exists() or (destination / "venv").exists():
            raise ValueError("Prime seed unexpectedly copied a virtualenv")
        return {
            "schema": "Diverse-flow-prime-seed-v1",
            "status": "seeded_verified",
            "source_commit": source_commit,
            "destination": str(destination),
            "destination_commit": destination_commit,
            "generated_environment_copied": False,
        }
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    args = parser.parse_args()
    print(json.dumps(seed(source=args.source, destination=args.destination, expected_commit=args.expected_commit), sort_keys=True))


if __name__ == "__main__":
    main()
