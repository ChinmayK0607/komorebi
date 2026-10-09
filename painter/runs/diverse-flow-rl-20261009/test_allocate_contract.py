"""Offline regression checks for the four-L40S allocation boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


RUN = Path(__file__).resolve().parent


def load_allocator():
    spec = importlib.util.spec_from_file_location("diverse_flow_allocate_contract", RUN / "allocate_single.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("allocator module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FourL40SSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_allocator()

    def test_selects_full_capacity_offer_over_cheaper_46gb_offer(self):
        offers = [
            {"id": "small", "gpu_type": "L40S", "gpu_count": 4,
             "gpu_memory_mib": "46068 MiB", "price_per_hour": 1.10},
            {"id": "full", "gpu_type": "NVIDIA L40S", "gpu_count": 4,
             "gpu_memory_mib": "49140 MiB", "price_per_hour": 1.52},
            {"id": "h100", "gpu_type": "H100", "gpu_count": 4,
             "gpu_memory_mib": "80 GiB", "price_per_hour": 1.20},
        ]
        chosen = self.module._select_l40s_offer(offers, max_price_usd_hour=1.59)
        self.assertEqual(chosen["id"], "full")
        self.assertIs(self.module.select_offer(offers, max_price_usd_hour=1.59), chosen)

    def test_accepts_readable_binary_capacity_from_nested_sdk_shape(self):
        offer = {
            "id": "nested-full", "gpu_count": 4, "price_per_hour": 1.52,
            "specs": {"gpu_type": "L40S", "memory": "48 GiB"},
        }
        self.assertIs(self.module._select_l40s_offer([offer], max_price_usd_hour=1.59), offer)

    def test_accepts_actual_lium_details_list_and_checks_weakest_gpu(self):
        # The live SDK exposes per-device records below specs.gpu.details;
        # UUID/provider fields are intentionally absent from this regression
        # fixture so the selector only depends on safe offer metadata.
        details = [{"name": "L40S", "capacity": "49140 MiB"} for _ in range(4)]
        offer = {
            "id": "sdk-details-full", "price_per_hour": 1.52,
            "specs": {"gpu": {"details": details}},
        }
        self.assertIs(self.module.select_offer([offer], max_price_usd_hour=1.59), offer)
        object_offer = SimpleNamespace(
            id="sdk-object-details-full", price_per_hour=1.52,
            specs={"gpu": {"details": details}},
        )
        self.assertIs(self.module.select_offer([object_offer], max_price_usd_hour=1.59), object_offer)

        undersized = {
            "id": "sdk-details-low", "price_per_hour": 1.52,
            "specs": {"gpu": {"details": [
                {"name": "L40S", "capacity": "49140 MiB"},
                {"name": "L40S", "capacity": "46068 MiB"},
                {"name": "L40S", "capacity": "49140 MiB"},
                {"name": "L40S", "capacity": "49140 MiB"},
            ]}},
        }
        with self.assertRaises(RuntimeError):
            self.module.select_offer([undersized], max_price_usd_hour=1.59)

    def test_rejects_capacity_and_strict_price_boundary(self):
        low_memory = {"id": "low", "gpu_type": "L40S", "gpu_count": 4,
                      "gpu_memory_mib": 46068, "price_per_hour": 1.10}
        at_boundary = {"id": "boundary", "gpu_type": "L40S", "gpu_count": 4,
                       "gpu_memory_mib": 49140, "price_per_hour": 1.60}
        with self.assertRaises(RuntimeError):
            self.module._select_l40s_offer([low_memory], max_price_usd_hour=1.59)
        with self.assertRaises(ValueError):
            self.module._select_l40s_offer([at_boundary], max_price_usd_hour=1.60)

        host_ram_trap = {"id": "host-ram-trap", "gpu_count": 4,
                         "gpu_type": "L40S", "memory": "280 GB",
                         "specs": {"gpu_memory_mib": 46068},
                         "price_per_hour": 1.20}
        with self.assertRaises(RuntimeError):
            self.module._select_l40s_offer([host_ram_trap], max_price_usd_hour=1.59)

    def test_h100_profile_requires_exactly_two_80gb_class_cards(self):
        offers = [
            {"id": "one", "gpu_type": "H100", "gpu_count": 1,
             "gpu_memory_mib": 80000, "price_per_hour": 1.20},
            {"id": "three", "gpu_type": "H100", "gpu_count": 3,
             "gpu_memory_mib": 80000, "price_per_hour": 1.20},
            {"id": "small", "gpu_type": "H100", "gpu_count": 2,
             "gpu_memory_mib": 79000, "price_per_hour": 1.20},
            {"id": "good", "gpu_type": "NVIDIA H100", "gpu_count": 2,
             "gpu_memory_mib": 80000, "price_per_hour": 3.70},
        ]
        chosen = self.module._select_h100_offer(offers, max_price_usd_hour=3.80)
        self.assertEqual(chosen["id"], "good")
        with self.assertRaises(RuntimeError):
            self.module._select_h100_offer(offers[:3], max_price_usd_hour=3.80)
        at_boundary = {"id": "boundary", "gpu_type": "H100", "gpu_count": 2,
                       "gpu_memory_mib": 80000, "price_per_hour": 4.00}
        with self.assertRaises(ValueError):
            self.module._select_h100_offer([at_boundary], max_price_usd_hour=4.00)

    def test_controller_topology_and_remote_memory_guard_are_l40s_compatible(self):
        self.assertEqual(self.module.GPU_COUNT, 4)
        self.assertEqual(self.module.GPU_TYPE, "L40S")
        self.assertEqual(self.module.MIN_GPU_MEMORY_MIB, 49140)
        self.assertEqual(self.module.operation.base.MIN_GPU_MEMORY_MIB, 49140)
        self.assertEqual(self.module.operation.base.GPU_COUNT, 4)


if __name__ == "__main__":
    unittest.main()
