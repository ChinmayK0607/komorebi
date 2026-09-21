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
    import prime_rl.trainer.rl.train as rl_train
    from training_handoff import record_microbatches

    # Prime's native DataLoader is the first point where a producer payload is
    # decoded as MicroBatch.  Capture and summarize that exact payload before
    # the native trainer enters forward/backward.  This is intentionally an
    # observational adapter: Prime remains responsible for score eligibility,
    # packing, CUDA execution and optimizer updates.
    original_get_batch = rl_train.DataLoader.get_batch

    def guarded_get_batch(loader):
        training_step = loader.receiver.current_step
        path_reader = getattr(loader.receiver, "_get_micro_batch_path", None)
        rank_path = path_reader() if path_reader is not None else None
        micro_batches = original_get_batch(loader)
        try:
            record_microbatches(root, micro_batches, training_step=training_step,
                                max_off_policy_steps=2,
                                native_rank_files=[rank_path] if rank_path is not None else None)
        except Exception as exc:  # the observer must never gate Prime's native trainer
            print(f"[training_handoff] observation unavailable: {type(exc).__name__}: {exc}", flush=True)
        return micro_batches

    if not getattr(rl_train.DataLoader.get_batch, "_painter_handoff_guard", False):
        guarded_get_batch._painter_handoff_guard = True
        rl_train.DataLoader.get_batch = guarded_get_batch
    from update_audit import install
    install(proxy, rl_train.train)
    set_proc_title('PainterTrainer')
    rl_train.train(config)


if __name__ == '__main__':
    main()
