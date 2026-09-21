"""Prime trainer child with the explicit portable SFT adapter initializer.

This uses the same pinned brush-model helper APIs as the previous aesthetic RL
run.  It does not resume an optimizer or data state and does not replace
Prime's native batch, loss, or optimizer implementation.
"""

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
    root = Path(os.environ.get("PAINTER_RUN_ROOT", Path(__file__).resolve().parent)).resolve()
    helpers = root / "source/brush-rl/integrations/prime/watercolour_sft"
    sys.path[:0] = [
        str(helpers),
        str(root / "source/brush-rl/integrations/speedpainting/mixed_sft"),
    ]
    from model_profiles import profile, verify_snapshot

    verify_snapshot(Path(config.model.name))
    if profile()["name"] != "qwen38_27b":
        raise ValueError("unexpected model profile")
    if config.resume is not None or config.model.lora is None:
        raise ValueError("use a fresh RL optimizer and the explicit SFT LoRA initializer")
    if config.model.optim_cpu_offload or config.model.fsdp_cpu_offload or config.model.full_offload:
        raise ValueError("this topology keeps model and optimizer on the trainer GPU")

    adapter = os.environ.get("WATERCOLOUR_INIT_ADAPTER")
    if not adapter:
        raise ValueError("WATERCOLOUR_INIT_ADAPTER is required")
    proxy = SimpleNamespace(
        model=config.model,
        resume=config.resume,
        run_dir=config.output_dir,
        max_steps=config.max_steps,
    )
    from init_adapter import install_init_hook
    from export_adapter import install_export_hook

    install_init_hook(proxy, adapter)
    install_export_hook(proxy)
    import prime_rl.trainer.rl.train as rl_train

    set_proc_title("PainterTrainer")
    rl_train.train(config)


if __name__ == "__main__":
    main()
