"""Native Prime RL training, initialized strictly from the selected SFT adapter."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title
    config = cli(TrainerConfig)
    root = Path(__file__).resolve().parent
    helpers = root / 'source/brush-rl/integrations/prime/watercolour_sft'
    sys.path[:0] = [str(helpers), str(root / 'source/brush-rl/integrations/speedpainting/mixed_sft')]
    from model_profiles import profile, verify_snapshot
    verify_snapshot(Path(config.model.name))
    if profile()['name'] != 'qwen38_27b':
        raise ValueError('Unexpected model profile')
    if config.resume is not None or config.model.lora is None:
        raise ValueError('Use a fresh RL optimizer and the explicit SFT LoRA initializer')
    if config.model.optim_cpu_offload or config.model.fsdp_cpu_offload or config.model.full_offload:
        raise ValueError('This topology keeps model and optimizer on the trainer GPU')
    proxy = SimpleNamespace(model=config.model, resume=config.resume,
                            run_dir=config.output_dir, max_steps=config.max_steps)
    from init_adapter import install_init_hook
    install_init_hook(proxy, os.environ['WATERCOLOUR_INIT_ADAPTER'])
    from export_adapter import install_export_hook
    install_export_hook(proxy)
    from prime_rl.trainer.rl.train import train
    from update_audit import install
    install(proxy, train)
    set_proc_title('PainterTrainer')
    train(config)


if __name__ == '__main__':
    main()
