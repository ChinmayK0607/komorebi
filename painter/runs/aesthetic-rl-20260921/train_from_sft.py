"""Native Prime RL trainer wrapper with a non-gating batch observer.

The observer retains native batch evidence only after Prime's own
``DataLoader.get_batch`` has returned.  It cannot turn missing observer
metadata into a trainer failure, and it never catches a native dataloader
exception.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _observer_metadata(loader) -> dict[str, Any]:
    """Best-effort metadata capture for filesystem evidence only."""
    try:
        receiver = loader.receiver
    except Exception as exc:
        print(
            f"[training_handoff] observer receiver unavailable: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {"training_step": None, "rank_path": None}

    step = None
    try:
        for name in ("current_step", "_current_step", "counter"):
            try:
                value = getattr(receiver, name)
            except Exception:
                continue
            if callable(value) or isinstance(value, bool):
                continue
            try:
                step = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            break
    except Exception as exc:
        print(
            f"[training_handoff] observer step metadata unavailable: {type(exc).__name__}: {exc}",
            flush=True,
        )

    rank_path = None
    try:
        path_reader = getattr(receiver, "_get_micro_batch_path", None)
        if callable(path_reader):
            rank_path = path_reader()
    except Exception as exc:
        print(
            f"[training_handoff] observer path metadata unavailable: {type(exc).__name__}: {exc}",
            flush=True,
        )
    return {"training_step": step, "rank_path": rank_path}


def install_batch_observer(rl_train, root: Path, record_microbatches: Callable[..., Any]) -> None:
    """Install an observer while leaving Prime's native get_batch authoritative."""
    original_get_batch = rl_train.DataLoader.get_batch
    if getattr(original_get_batch, "_painter_handoff_guard", False):
        return

    def guarded_get_batch(loader):
        metadata = _observer_metadata(loader)
        # This call is intentionally outside every observer try/except: native
        # data-loader errors must propagate, and a metadata failure must never
        # prevent Prime from receiving its decoded MicroBatch.
        micro_batches = original_get_batch(loader)
        try:
            training_step = metadata.get("training_step")
            if training_step is None:
                print("[training_handoff] batch observed without step metadata", flush=True)
                return micro_batches
            rank_path = metadata.get("rank_path")
            record_microbatches(
                root,
                micro_batches,
                training_step=training_step,
                max_off_policy_steps=1,
                native_rank_files=[rank_path] if rank_path is not None else None,
            )
        except Exception as exc:
            # Evidence retention is advisory; native training already owns the
            # batch and must continue even if the observer cannot serialize it.
            print(
                f"[training_handoff] observation unavailable: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return micro_batches

    guarded_get_batch._painter_handoff_guard = True
    rl_train.DataLoader.get_batch = guarded_get_batch


def main() -> None:
    from prime_rl.configs.trainer import TrainerConfig
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    config = cli(TrainerConfig)
    root = Path(__file__).resolve().parent
    helpers = root / "source/brush-rl/integrations/prime/watercolour_sft"
    sys.path[:0] = [
        str(helpers),
        str(root / "source/brush-rl/integrations/speedpainting/mixed_sft"),
    ]
    from model_profiles import profile, verify_snapshot

    verify_snapshot(Path(config.model.name))
    if profile()["name"] != "qwen38_27b":
        raise ValueError("Unexpected model profile")
    if config.resume is not None or config.model.lora is None:
        raise ValueError("Use a fresh RL optimizer and the explicit SFT LoRA initializer")
    if config.model.optim_cpu_offload or config.model.fsdp_cpu_offload or config.model.full_offload:
        raise ValueError("This topology keeps model and optimizer on the trainer GPU")

    proxy = SimpleNamespace(
        model=config.model,
        resume=config.resume,
        run_dir=config.output_dir,
        max_steps=config.max_steps,
    )
    from init_adapter import install_init_hook
    from export_adapter import install_export_hook

    install_init_hook(proxy, os.environ["WATERCOLOUR_INIT_ADAPTER"])
    install_export_hook(proxy)
    import prime_rl.trainer.rl.train as rl_train
    from training_handoff import record_microbatches

    install_batch_observer(rl_train, root, record_microbatches)
    from update_audit import install

    install(proxy, rl_train.train)
    set_proc_title("PainterTrainer")
    rl_train.train(config)


if __name__ == "__main__":
    main()
