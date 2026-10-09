"""Prime trainer child wrapper for fresh RL optimizer + SFT LoRA weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _trainer_dependency_paths() -> tuple[Path, Path, Path]:
    """Resolve the exact helper paths used by the Prime trainer child."""

    root = Path(__file__).resolve().parent
    brush_root = Path(os.environ.get("BRUSH_ROOT", root / "source" / "brush-rl")).expanduser()
    if brush_root.is_symlink() or not brush_root.is_dir():
        raise FileNotFoundError(f"reviewed brush helper root is missing or symlinked: {brush_root}")
    brush_root = brush_root.resolve(strict=True)
    helpers = brush_root / "integrations" / "prime" / "watercolour_sft"
    for name in ("init_adapter.py", "export_adapter.py", "model_profiles.py"):
        path = helpers / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"reviewed Prime adapter helper is missing or symlinked: {path}")
    return root, brush_root, helpers


def main() -> None:
    root, brush_root, helpers = _trainer_dependency_paths()
    sys.path[:0] = [str(helpers), str(root)]

    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    config = cli(TrainerConfig)

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
    if config.model.optim_cpu_offload or config.model.fsdp_cpu_offload or config.model.full_offload:
        raise ValueError("This 1-GPU trainer topology keeps model and optimizer on the trainer GPU")

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
