#!/usr/bin/env python3
"""Parameterize the reviewed finite supervisor for the four-update pilot."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


BASE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "finite_supervisor.py"
MAX_UPDATES = int(os.environ.get("PAINTER_RL_MAX_STEPS", "4"))
if MAX_UPDATES != 4:
    raise SystemExit("single-arm supervisor is fixed to four updates")

args = list(sys.argv[1:])
for index, value in enumerate(args):
    if value == "--max-updates" and index + 1 < len(args):
        args[index + 1] = str(MAX_UPDATES)
        break
else:
    args.extend(["--max-updates", str(MAX_UPDATES)])
sys.argv[1:] = args
runpy.run_path(str(BASE), run_name="__main__")
