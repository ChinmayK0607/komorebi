#!/usr/bin/env python3
"""Reuse the reviewed foreground completion relay for this run root."""
from pathlib import Path
import runpy

SOURCE = Path(__file__).resolve().parents[1] / "text-sft512-consistent-20260930" / "completion_relay.py"
runpy.run_path(str(SOURCE), run_name="__main__")
