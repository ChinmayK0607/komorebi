#!/usr/bin/env python3
"""Show matched reference, first paint and revision for baseline versus SFT."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "quality-curriculum-20260924/build_astra_firstpaint_gallery.py"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--node-root", type=Path, default=Path("/root/brush-sft-20260927"))
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("matched_gallery", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.NODE_ROOT = a.node_root
    print(json.dumps(module.build(a.evidence, a.output)))
