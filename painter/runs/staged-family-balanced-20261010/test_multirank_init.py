"""CPU-only checks for the staged 4-rank portable LoRA initializer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


RUN = Path(__file__).resolve().parent


def _load_override():
    import sys

    override = RUN / "prime_runtime/integrations/prime/watercolour_sft"
    sys.path.insert(0, str(override))
    sys.path.insert(1, str(RUN.parents[1] / "rl_helpers/integrations/prime/watercolour_sft"))
    spec = importlib.util.spec_from_file_location("staged_multirank_init", override / "init_adapter.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(override)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(rank: int, world_size: int, init_file: str, rows: int = 4) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        hooks = _load_override()
        hooks.profile = lambda: {"tensors": 2, "parameters": rows * 4}
        mesh = init_device_mesh("cpu", (world_size,))
        source = {
            "layer.lora_A.weight": torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
            "layer.lora_B.weight": torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) + 1,
        }
        destination = {
            name: distribute_tensor(torch.zeros_like(value), mesh, placements=[Shard(0)])
            for name, value in source.items()
        }
        report = hooks.copy_and_verify(source, destination, lambda: destination)
        if not report.get("distributed_copy") or report.get("override_schema") != hooks.OVERRIDE_SCHEMA:
            raise AssertionError(report)
        for name, value in source.items():
            expected = value.chunk(world_size, dim=0)[rank]
            if not torch.equal(destination[name].to_local(), expected):
                raise AssertionError(f"rank {rank} has the wrong shard for {name}")
    finally:
        dist.destroy_process_group()


class StagedMultirankInitTests(unittest.TestCase):
    def test_world_size_contract_is_exactly_four(self):
        hooks = _load_override()
        self.assertEqual(hooks.validate_trainer_world_size(4)["world_size"], 4)
        with self.assertRaises(ValueError):
            hooks.validate_trainer_world_size(2)

    def test_dtensor_copy_preserves_rank_local_shards(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(2, str(Path(directory) / "gloo-init")), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
