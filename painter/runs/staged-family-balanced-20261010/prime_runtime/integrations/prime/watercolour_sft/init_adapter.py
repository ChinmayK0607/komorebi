"""Distributed portable LoRA initialization for the staged 4+4 run.

Prime's archived helper only accepts one trainer rank.  This run-owned hook
keeps the pinned adapter identity and copies each full CPU tensor into Prime's
existing DTensor placement on every rank in deterministic key order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from model_profiles import profile


BASE_REVISION = profile()["revision"]
BASE_MODEL = profile()["model_id"]
TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
}
OVERRIDE_SCHEMA = "staged-family-balanced-multirank-lora-init-v1"


def _world():
    from prime_rl.trainer.world import get_world

    return get_world()


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def validate_trainer_world_size(world_size: int, expected: int = 4) -> dict[str, int | str]:
    """Validate the staged trainer degree without importing CUDA/model code."""

    if type(world_size) is not int or type(expected) is not int or world_size != expected:
        raise ValueError(
            f"staged-family-balanced RL requires {expected} Prime trainer ranks; "
            f"got WORLD_SIZE={world_size}"
        )
    return {"status": "passed", "world_size": world_size, "expected_world_size": expected}


def _distribute_full_tensor(source, destination) -> None:
    import torch

    if hasattr(destination, "to_local") and hasattr(destination, "full_tensor"):
        try:
            from torch.distributed.tensor import distribute_tensor
        except ImportError:  # pragma: no cover - old pinned torch spelling
            from torch.distributed._tensor import distribute_tensor
        if tuple(destination.shape) != tuple(source.shape):
            raise ValueError(f"DTensor global shape mismatch: {tuple(destination.shape)} != {tuple(source.shape)}")
        full = source.detach().to(device=destination.device, dtype=destination.dtype)
        sharded = distribute_tensor(full, device_mesh=destination.device_mesh, placements=destination.placements)
        with torch.no_grad():
            destination.copy_(sharded)
        return
    if tuple(destination.shape) != tuple(source.shape):
        raise ValueError(
            "multi-rank portable initialization requires DTensor destinations "
            f"with global shape {tuple(source.shape)}, got {tuple(destination.shape)}"
        )
    with torch.no_grad():
        destination.copy_(source.to(destination.device, dtype=destination.dtype))


def copy_and_verify(source, destination, reread):
    import torch

    if set(source) != set(destination):
        raise ValueError(
            f"Adapter key mismatch: missing={sorted(set(destination) - set(source))}, "
            f"unexpected={sorted(set(source) - set(destination))}"
        )
    for name in source:
        if source[name].shape != destination[name].shape or source[name].dtype != destination[name].dtype:
            raise ValueError(f"Adapter shape/dtype mismatch: {name}")
        if not torch.isfinite(source[name]).all().item():
            raise ValueError(f"Non-finite source adapter: {name}")
    # Validate all global shapes before any rank enters a collective.
    for name in sorted(source):
        tensor = destination[name]
        if tuple(tensor.shape) != tuple(source[name].shape) or tensor.dtype != source[name].dtype:
            raise ValueError(f"Adapter shape/dtype mismatch: {name}")
    for name in sorted(destination):
        _distribute_full_tensor(source[name], destination[name])
    _barrier()
    verified = reread()
    if set(verified) != set(source):
        raise ValueError("Post-load adapter key set changed")
    for name in sorted(verified):
        tensor = verified[name]
        actual = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
        if not torch.equal(actual.cpu(), source[name]):
            raise ValueError(f"Post-load adapter differs from source: {name}")
    b_names = [name for name in source if ".lora_B." in name]
    nonzero = sum(bool(torch.count_nonzero(source[name]).item()) for name in b_names)
    expected_pairs = profile()["tensors"] // 2
    if len(b_names) != expected_pairs or nonzero != expected_pairs:
        raise ValueError(f"Expected all {expected_pairs} trained B matrices, found {len(b_names)} with {nonzero} nonzero")
    return {
        "tensor_count": len(source),
        "exact_tensor_matches": len(verified),
        "b_tensors": len(b_names),
        "nonzero_b_tensors": nonzero,
        "parameters": sum(value.numel() for value in source.values()),
        "all_finite": True,
        "override_schema": OVERRIDE_SCHEMA,
        "distributed_copy": True,
    }


def install_init_hook(config, adapter):
    from prime_rl.trainer.lora import LoRAState
    from safetensors.torch import load_file

    if config.resume is not None or config.model.lora is None:
        raise ValueError("Portable init requires LoRA and fresh optimizer, not native resume")
    adapter = Path(adapter).resolve(strict=True)
    metadata = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    lora = config.model.lora
    if metadata.get("r") != lora.rank or metadata.get("lora_alpha") != lora.alpha or set(metadata["target_modules"]) != set(lora.target_modules):
        raise ValueError("Source adapter rank/alpha/target modules differ from config")
    weights_path = adapter / "adapter_model.safetensors"
    source = load_file(str(weights_path), device="cpu")
    original_reset = LoRAState.reset_adapter_parameters
    loaded = False

    def reset(self):
        nonlocal loaded
        if loaded:
            raise RuntimeError("Refusing a second reset after portable adapter initialization")
        original_reset(self)
        report = copy_and_verify(source, self.adapter_state_dict(), self.adapter_state_dict)
        world = _world()
        report.update(
            initialization_mode="trained_adapter",
            source=str(adapter),
            source_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            optimizer="fresh",
            native_resume=False,
            before_first_forward=True,
            ddp_rank=getattr(world, "rank", 0),
            world_size=getattr(world, "world_size", 1),
        )
        output = config.run_dir / "artifacts" / f"initial-adapter-audit.rank{getattr(world, 'rank', 0)}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        loaded = True

    LoRAState.reset_adapter_parameters = reset
