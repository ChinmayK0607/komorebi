#!/usr/bin/env python3
"""Use the reviewed archive screen while keeping this run's path local."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "safe_archive.py"
spec = importlib.util.spec_from_file_location("structured_safe_archive_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"archive helper unavailable: {SOURCE}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
globals().update({name: value for name, value in vars(base).items() if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(base.main())
