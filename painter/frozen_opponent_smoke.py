#!/usr/bin/env python3
"""Run one frozen-opponent native eval smoke case on the Linux node.

This is a thin CLI around the production generator; it does not implement a
second rollout path or bypass the native trace validity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from painter.frozen_opponent import (
        OpponentSpec,
        OpponentUnavailable,
        generate_frozen_opponent,
        sha,
    )
except ImportError:  # direct ``python painter/frozen_opponent_smoke.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from painter.frozen_opponent import (  # type: ignore[no-redef]
        OpponentSpec,
        OpponentUnavailable,
        generate_frozen_opponent,
        sha,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--model-alias", default="sft-step512")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8100/v1",
        help="direct vLLM endpoint (the pinned local router rejects VF reasoning_effort=none)",
    )
    parser.add_argument("--root", default="/root/painter")
    parser.add_argument("--prime-root", default="/root/painter/prime-rl")
    parser.add_argument("--eval", dest="eval_bin", default="/root/painter/prime-rl/.venv/bin/eval")
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--renderer-python", required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    reference = args.reference.resolve()
    if not reference.is_file():
        parser.error(f"reference does not exist: {reference}")
    os.environ.update({
        "PAINTER_FROZEN_OPPONENT_ENDPOINT": args.endpoint,
        "PAINTER_FROZEN_OPPONENT_ROOT": args.root,
        "PAINTER_FROZEN_OPPONENT_PRIME_ROOT": args.prime_root,
        "PAINTER_FROZEN_OPPONENT_EVAL": args.eval_bin,
        "PAINTER_FROZEN_OPPONENT_RENDERER": args.renderer,
        "PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON": args.renderer_python,
    })
    if args.config is not None:
        os.environ["PAINTER_FROZEN_OPPONENT_CONFIG"] = str(args.config.resolve())
    spec = OpponentSpec(
        source_group=args.source_group,
        reference=reference,
        reference_sha256=args.reference_sha256,
        model_alias=args.model_alias,
        checkpoint_sha256=args.checkpoint_sha256,
        protocol_version="native-painting-rl-v1",
        render_protocol="native-painting-v1",
        decoding_config="{}",
        seed=args.seed,
        work_root=args.work_root.resolve(),
    )
    try:
        result = generate_frozen_opponent(spec)
    except OpponentUnavailable as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}))
        return 2
    if isinstance(result, Path):
        print(json.dumps({"status": "valid", "path": str(result), "sha256": sha(result)}))
    else:
        print(json.dumps(dict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
