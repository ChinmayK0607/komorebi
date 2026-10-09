"""CPU-only executable checks for the run-owned DTensor adapter hooks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


RUN = Path(__file__).resolve().parent


def _load_override():
    path = RUN / "prime_runtime/integrations/prime/watercolour_sft/init_adapter.py"
    import sys
    sys.path.insert(0, str(RUN / "prime_runtime/integrations/prime/watercolour_sft"))
    sys.path.insert(1, str(RUN.parents[1] / "rl_helpers/integrations/prime/watercolour_sft"))
    spec = importlib.util.spec_from_file_location("diverse_flow_multirank_init", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(rank: int, world_size: int, init_file: str, rows: int = 4) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
    )
    try:
        hooks = _load_override()
        hooks.profile = lambda: {"tensors": 2, "parameters": 16}
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
            raise AssertionError(f"missing distributed receipt fields: {report}")
        for name, value in source.items():
            local = destination[name].to_local()
            expected = value.chunk(world_size, dim=0)[rank]
            if not torch.equal(local, expected):
                raise AssertionError(f"rank {rank} does not own its shard for {name}")
    finally:
        dist.destroy_process_group()


class MultiRankHookTests(unittest.TestCase):
    def test_cpu_dtensor_full_source_copy_is_sharded_by_rank(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "gloo-init")
            mp.spawn(_worker, args=(2, init_file), nprocs=2, join=True)

    def test_cpu_dtensor_three_rank_uneven_source_shards(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "gloo-init")
            mp.spawn(_worker, args=(3, init_file, 5), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
