#!/usr/bin/env python3
"""Summarize and display the three frozen base-control evaluations."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICIES = ("base", "step40", "step160")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(evidence: Path, output: Path) -> dict:
    evidence = evidence.resolve(strict=True)
    if not (evidence / "eval-base-control").is_dir():
        raise FileNotFoundError("base-control evaluations not extracted")
    alias = evidence / "eval"
    if not alias.exists():
        alias.symlink_to("eval-base-control", target_is_directory=True)
    elif alias.resolve() != (evidence / "eval-base-control").resolve():
        raise ValueError("eval alias points to unrelated evidence")
    summary_module = load(HERE / "summarize_brush_sft_eval.py", "base_control_summary")
    summary_module.POLICIES = POLICIES
    summary = summary_module.summarize(evidence)
    summary["schema"] = "painter.base-control-eval-summary.v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    gallery = load(HERE.parent / "quality-curriculum-20260924/build_multiturn_gallery.py", "base_control_gallery")
    gallery.NODE_ROOT = Path("/root/brush-base-control-20260927")
    gallery.POLICIES = POLICIES
    gallery.build(evidence, output / "index.html")
    return {"summary": str(output / "summary.json"), "gallery": str(output / "index.html"),
            "counts": summary["policies"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.evidence, args.output)))
