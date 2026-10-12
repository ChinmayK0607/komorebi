#!/usr/bin/env python3
"""Prime trainer wrapper with distributed staged-adapter initialization."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


RUN = Path(__file__).resolve().parent


def _trainer_dependency_paths() -> tuple[Path, Path, Path]:
    root = RUN
    brush_root = Path(os.environ.get("BRUSH_ROOT", root.parents[2] / "rl_helpers")).resolve(strict=True)
    helpers = brush_root / "integrations" / "prime" / "watercolour_sft"
    override = root / "prime_runtime" / "integrations" / "prime" / "watercolour_sft"
    if not (override / "init_adapter.py").is_file():
        raise FileNotFoundError(f"staged adapter override is missing under {override}")
    for name in ("init_adapter.py", "export_adapter.py", "model_profiles.py"):
        if not (helpers / name).is_file():
            raise FileNotFoundError(f"reviewed Prime helper is missing under {helpers}")
    return root, brush_root, helpers


def main() -> None:
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    config = cli(TrainerConfig)
    root, brush_root, helpers = _trainer_dependency_paths()
    override = root / "prime_runtime" / "integrations" / "prime" / "watercolour_sft"
    mixed = brush_root / "integrations" / "speedpainting" / "mixed_sft"
    sys.path[:0] = [str(override), str(helpers), str(mixed), str(root)]

    from model_profiles import profile, verify_snapshot

    verify_snapshot(Path(config.model.name))
    if profile()["name"] != "qwen38_27b":
        raise ValueError("Unexpected model profile")
    if config.resume is not None or config.model.lora is None:
        raise ValueError("RL must use a fresh optimizer with the explicit SFT LoRA initializer")
    if config.ckpt is not None and getattr(config.ckpt, "resume_step", None) is not None:
        raise ValueError("RL launch does not resume native optimizer/checkpoint state")
    from init_adapter import validate_trainer_world_size, install_init_hook

    expected_world = int(os.environ.get("PAINTER_NUM_TRAIN_GPUS", "4"))
    validate_trainer_world_size(int(os.environ.get("WORLD_SIZE", "1")), expected_world)
    if expected_world != 4:
        raise ValueError(f"staged-family-balanced RL requires exactly four trainer ranks; got {expected_world}")
    if config.model.optim_cpu_offload or config.model.full_offload:
        raise ValueError("staged 4-rank FSDP keeps optimizer/model offload disabled")

    adapter = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not adapter:
        raise ValueError("WATERCOLOUR_INIT_ADAPTER is required")
    adapter_path = Path(adapter).resolve(strict=True)
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError("WATERCOLOUR_INIT_ADAPTER lacks adapter_model.safetensors")

    proxy = SimpleNamespace(model=config.model, resume=config.resume, run_dir=config.output_dir, max_steps=config.max_steps)
    install_init_hook(proxy, str(adapter_path))
    from export_adapter import install_export_hook

    install_export_hook(proxy)
    import prime_rl.trainer.rl.train as rl_train

    set_proc_title("PainterRLTrainer")
    rl_train.train(config)


if __name__ == "__main__":
    main()
