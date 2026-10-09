"""Prime trainer child wrapper for fresh RL optimizer + SFT LoRA weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    config = cli(TrainerConfig)
    root = Path(__file__).resolve().parent
    brush_root = Path(os.environ.get("BRUSH_ROOT", root / "source" / "brush-rl")).resolve(strict=True)
    helpers = brush_root / "integrations" / "prime" / "watercolour_sft"
    mixed = brush_root / "integrations" / "speedpainting" / "mixed_sft"
    if not (helpers / "init_adapter.py").is_file() or not (helpers / "export_adapter.py").is_file():
        raise FileNotFoundError(f"reviewed Prime adapter helpers are missing under {brush_root}")
    sys.path[:0] = [str(helpers), str(mixed), str(root)]

    from model_profiles import profile, verify_snapshot

    verify_snapshot(Path(config.model.name))
    if profile()["name"] != "qwen38_27b":
        raise ValueError("Unexpected model profile")
    if config.resume is not None or config.model.lora is None:
        raise ValueError("RL must use a fresh optimizer with the explicit SFT LoRA initializer")
    # The pinned Prime 26b trainer keeps resume state in the top-level
    # ``resume`` block; its ``ckpt`` block only controls save cadence and
    # retention.  Use getattr so this wrapper remains explicit about the
    # fresh optimizer without assuming the newer candidate schema.
    if config.ckpt is not None and getattr(config.ckpt, "resume_step", None) is not None:
        raise ValueError("RL launch does not resume native optimizer/checkpoint state")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world = int(os.environ.get("PAINTER_NUM_TRAIN_GPUS", "2"))
    if world_size != expected_world or expected_world != 2:
        raise ValueError(f"Diverse-flow RL requires two Prime trainer ranks; got WORLD_SIZE={world_size}, expected={expected_world}")
    if config.model.optim_cpu_offload or config.model.full_offload:
        raise ValueError("Diverse-flow 2-rank FSDP keeps optimizer/model offload disabled")
    # fsdp_cpu_offload is a valid model sharding option for the two-rank
    # trainer.  Prime owns the FSDP process group; this wrapper only verifies
    # rank count and adapter initialization before delegating to Prime.

    adapter = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not adapter:
        raise ValueError("WATERCOLOUR_INIT_ADAPTER is required")
    adapter_path = Path(adapter).resolve(strict=True)
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError("WATERCOLOUR_INIT_ADAPTER lacks adapter_model.safetensors")

    proxy = SimpleNamespace(
        model=config.model,
        resume=config.resume,
        run_dir=config.output_dir,
        max_steps=config.max_steps,
    )
    from init_adapter import install_init_hook
    from export_adapter import install_export_hook

    # install_init_hook replaces Prime's one native reset with an exact copy
    # from the reviewed SFT adapter and rejects every second reset.  The
    # optimizer remains fresh, while export_adapter publishes verified LoRA
    # adapters at Prime's configured checkpoint interval (8 updates).
    install_init_hook(proxy, str(adapter_path))
    install_export_hook(proxy)

    import prime_rl.trainer.rl.train as rl_train

    # Prime owns data admission, masking, forward/backward, and checkpoint
    # progress.  Keep this wrapper limited to initialization and export hooks;
    # no observer can turn missing optional evidence into a training failure.
    set_proc_title("PainterRLTrainer")
    rl_train.train(config)


if __name__ == "__main__":
    main()
