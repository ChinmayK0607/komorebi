"""Prime SFT from the verified public 27B midpoint adapter on one Linux GPU."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BRUSH_ROOT = Path(os.environ.get("BRUSH_ROOT", ROOT / "source/brush-rl"))
sys.path[:0] = [
    str(BRUSH_ROOT / "integrations/speedpainting/mixed_sft"),
    str(BRUSH_ROOT / "integrations/prime/watercolour_sft"),
    str(BRUSH_ROOT / "scripts"),
]


def main() -> None:
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.utils.config import cli

    source = BRUSH_ROOT / "integrations/speedpainting/mixed_sft/train.py"
    spec = importlib.util.spec_from_file_location("reviewed_multiturn_sft", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing pinned strict SFT source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["reviewed_multiturn_sft"] = module
    spec.loader.exec_module(module)
    module.install()

    config = cli(SFTConfig)
    if config.resume is not None or config.model.lora is None:
        raise ValueError("this run initializes a portable LoRA adapter with a fresh optimizer")
    if config.deployment.num_train_gpus != 1 or config.deployment.gpus_per_node != 1:
        raise ValueError("pinned portable initializer is audited for one GPU")
    if config.data.micro_batch_size != 1 or config.data.shuffle:
        raise ValueError("strict finite training requires one-example microbatches and no shuffle")
    if os.environ.get("CAPACITY_STATE_COMPLETE") != "1":
        raise ValueError("state-complete examples require exact generation-prefix masking")
    adapter = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not adapter:
        raise ValueError("set the verified public step-512 initializer path")
    from prime_rl.trainer.sft.train import train
    from export_adapter import install_export_hook
    from init_adapter import install_init_hook

    install_export_hook(config)
    install_init_hook(config, adapter)
    train(config)


if __name__ == "__main__":
    main()
