#!/usr/bin/env python3
"""Render one saved teacher program through the pinned Linux sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run import render_program


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path, required=True)
    parser.add_argument("--browser-path", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--run-as-user", default="painter")
    args = parser.parse_args(argv)
    result = render_program(
        root=Path(__file__).resolve().parent,
        source=args.source,
        output=args.output,
        renderer=args.renderer,
        renderer_python=args.renderer_python,
        timeout=args.timeout,
        browser_path=args.browser_path,
        run_as_user=args.run_as_user,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
