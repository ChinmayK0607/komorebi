"""Derive a flat TrainerConfig JSON for the one-update native replay smoke."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="resolved trainer JSON, or an RL JSON containing a trainer object")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=16384)
    return parser


def build_config(baseline: dict[str, Any], *, model_path: str, output_dir: str, seq_len: int) -> dict[str, Any]:
    # Resolved RL configs carry the actual trainer settings under `trainer`,
    # while a resolved TrainerConfig is already flat. Keep the baseline's
    # optimizer, LoRA, VLM, loss, and model implementation fields intact.
    has_nested_trainer = isinstance(baseline.get("trainer"), dict)
    config = copy.deepcopy(baseline.get("trainer", baseline))
    if isinstance(baseline.get("model"), dict):
        if has_nested_trainer and isinstance(config.get("model"), dict):
            merged_model = copy.deepcopy(baseline["model"])
            merged_model.update(config["model"])
            config["model"] = merged_model
        elif "model" not in config:
            config["model"] = copy.deepcopy(baseline["model"])
    model = config.setdefault("model", {})
    model["name"] = model_path
    model["seq_len"] = seq_len
    model["fsdp_cpu_offload"] = False
    model["optim_cpu_offload"] = False
    model["full_offload"] = None
    config["output_dir"] = output_dir
    config["max_steps"] = 1
    config["resume"] = None
    config["ckpt"] = {"interval": 1, "keep_last": 1}
    # Filesystem is the only valid broadcast schema for LoRA. launch_replay.sh
    # supplies a local receiver-ready sentinel so no inference process is
    # contacted during this diagnostic.
    config["weight_broadcast"] = {"type": "filesystem", "timeout": 30}
    config["rollout_transport"] = {"type": "filesystem"}
    config["data"] = config.get("data") or {}
    config["data"].pop("fake", None)
    return config


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with open(args.baseline, encoding="utf-8") as handle:
        baseline = json.load(handle)
    if not isinstance(baseline, dict):
        raise SystemExit("baseline must be a JSON object")
    config = build_config(baseline, model_path=args.model_path, output_dir=args.output_dir, seq_len=args.seq_len)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
