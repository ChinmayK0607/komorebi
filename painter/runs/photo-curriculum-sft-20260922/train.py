"""Run the pinned mixed SFT trainer from a fresh Qwen base snapshot."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BRUSH_ROOT = Path(os.environ.get("BRUSH_ROOT", ROOT / "source/brush-rl"))
sys.path[:0] = [
    str(ROOT), str(BRUSH_ROOT / "integrations/speedpainting/mixed_sft"),
    str(BRUSH_ROOT / "integrations/prime/watercolour_sft"), str(BRUSH_ROOT / "scripts"),
]


def main() -> None:
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.utils.config import cli

    source = BRUSH_ROOT / "integrations/speedpainting/mixed_sft/train.py"
    spec = importlib.util.spec_from_file_location("mixed_sft_train", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing pinned mixed SFT source: {source}")
    mixed_sft_train = importlib.util.module_from_spec(spec)
    sys.modules["mixed_sft_train"] = mixed_sft_train
    spec.loader.exec_module(mixed_sft_train)
    mixed_sft_train.install()

    config = cli(SFTConfig)
    resuming = config.resume is not None
    fresh_base = os.environ.get("FRESH_BASE", "1")
    if resuming and fresh_base != "0":
        raise ValueError("checkpoint resume requires FRESH_BASE=0")
    if not resuming and fresh_base != "1":
        raise ValueError("a new run requires FRESH_BASE=1")
    if config.data.micro_batch_size != 1 or (config.val and config.val.data.micro_batch_size != 1):
        raise ValueError("one-example microbatches are required for exact full-target masking")
    if config.data.shuffle and os.environ.get("ALLOW_DATA_SHUFFLE") != "1":
        raise ValueError("data shuffle is disabled by default for resumable finite admission")
    from prime_rl.trainer.sft.train import train
    from export_adapter import install_export_hook
    import fresh_base_ddp
    import profile_adapter

    export = install_export_hook(config)
    if not resuming:
        fresh_base_ddp.install(config, train, export, _ProfileAdapter(profile_adapter))
    # Prime owns DDP/FSDP setup and all-reduces. The launch sidecar records
    # per-GPU utilization; the historical shared-file update hook is not safe
    # when several ranks append to one file.
    train(config)


class _ProfileAdapter:
    def __init__(self, module):
        self.module = module

    def verify_snapshot(self, path):
        from model_profiles import verify_snapshot
        return verify_snapshot(path)

    def audit_fresh_adapter(self, state):
        return self.module.audit_fresh_adapter(state)


if __name__ == "__main__":
    main()
