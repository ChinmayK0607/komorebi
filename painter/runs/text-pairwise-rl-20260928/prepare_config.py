"""Write the finite fixed-anchor text RL config from the pinned Prime profile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PHOTO = Path(__file__).resolve().parents[1] / "photo-curriculum-rl-20260922" / "prepare_rl_config.py"
spec = importlib.util.spec_from_file_location("painter_photo_rl_config", PHOTO)
assert spec and spec.loader
photo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = photo
spec.loader.exec_module(photo)

RUN_NAME = "text-pairwise-rl-20260928"


def prepare(*, output: Path, model: Path, run_root: Path, manifest: Path,
            prime_root: Path | None = None, max_steps: int = 20) -> dict:
    rows = json.loads(manifest.read_text())["rows"]
    if len(rows) != 40 or len({row["id"] for row in rows}) != 40:
        raise ValueError("expected 40 unique fixed-anchor training rows")
    if any(row.get("reference_kind") != "text" or not row.get("opponent_sha256") for row in rows):
        raise ValueError("every row must have a text brief and hashed fixed opponent")
    relative = str(manifest.resolve().relative_to(run_root.resolve()))
    config = photo.build_config(
        model=str(model), run_root=run_root, manifest=relative,
        max_steps=max_steps, checkpoint_interval=4, run_name=RUN_NAME,
        expected_count=40, batch_size=8, group_size=4, max_turns=3,
        max_completion_tokens=8192, learning_rate=1e-6,
        initial_inflight=8, max_inflight=8,
        adapter_revision="adf687e5b321514eda7be6eab050434fdc8e44ff",
        adapter_prefix="photo-curriculum-sft-20260922-balanced/step_512",
        adapter_sha256="bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
    )
    source = config["orchestrator"]["train"]["source"][0]
    source["name"] = "text-paint-fixed-sft-pairwise"
    source["env"]["taskset"]["lazy_opponent"] = False
    source["env"]["taskset"]["preserve_manifest_order"] = False
    source["env"]["agent"]["timeout"]["scoring"] = 600
    config["orchestrator"]["max_off_policy_steps"] = 1
    message = photo.write_config(output, config, validate_prime=prime_root is not None,
                                 prime_root=prime_root)
    receipt = {
        "run_name": RUN_NAME, "prime_revision": photo.PRIME_COMMIT,
        "model_revision": photo.BASE_MODEL_REVISION,
        "adapter_revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
        "adapter_prefix": "photo-curriculum-sft-20260922-balanced/step_512",
        "adapter_sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
        "mode": "fresh", "optimizer_resume": False,
        "frozen_anchor": {"alias": "sft-step512", "immutable": True,
                          "checkpoint_sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"},
        "deployment": config["deployment"], "manifest": relative,
        "reward": "text-paint-sft-pairwise-v1", "prime_validation": message,
    }
    output.with_name("rl-config-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
