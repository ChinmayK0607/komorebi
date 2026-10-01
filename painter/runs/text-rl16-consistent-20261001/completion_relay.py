#!/usr/bin/env python3
"""Run the shared finite completion relay for the RL16 continuation."""

from __future__ import annotations

import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
BASE = RUN.parent / "text-sft512-consistent-20260930"
os.environ.setdefault("PAINTER_RUN_NAME", "text-rl16-consistent-20261001")
os.environ.setdefault("PAINTER_RELAY_EVIDENCE_DIR", str(RUN / "local-evidence"))
sys.path.insert(0, str(BASE))

from completion_relay import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
