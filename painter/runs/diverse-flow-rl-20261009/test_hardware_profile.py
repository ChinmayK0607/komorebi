"""Execution regressions for the run-local hardware profile boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardwareProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load("diverse_flow_hardware_profile", RUN / "hardware_profile.py")

    def test_profiles_bind_exact_roles_and_tp(self):
        l40s = self.profile.get("l40s_4")
        h100 = self.profile.get("h100_2")
        self.assertEqual((l40s["gpus_per_node"], l40s["num_train_gpus"], l40s["num_infer_gpus"], l40s["inference_tensor_parallel"]), (4, 2, 2, 2))
        self.assertEqual((h100["gpus_per_node"], h100["num_train_gpus"], h100["num_infer_gpus"], h100["inference_tensor_parallel"]), (2, 1, 1, 1))
        self.assertEqual(h100["train_gpu_ids"], ("0",))
        self.assertEqual(h100["infer_gpu_ids"], ("1",))
        self.assertTrue(h100["sft_resume_required"])
        self.assertEqual(h100["warmup_world_size"], 0)

    def test_h100_binding_requires_resume_and_accepts_only_one_plus_one(self):
        values = {
            "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1,
            "train_gpu_ids": "0", "infer_gpu_ids": "1",
            "inference_tensor_parallel": 1,
        }
        with self.assertRaisesRegex(ValueError, "RESUME"):
            self.profile.validate_bindings("h100_2", values, environ={})
        self.profile.validate_bindings(
            "h100_2", values, environ={"PAINTER_RESUME_SAME_RUN_ARCHIVE": "1"}
        )
        # The shell boundary exercises the same values with an explicit
        # resume flag, including the sourced default resolver.
        command = (
            f"PAINTER_HARDWARE_PROFILE=h100_2 PAINTER_RESUME_SAME_RUN_ARCHIVE=1 "
            f"bash -c 'source {RUN / 'runtime_env.sh'}; validate_structured_bindings; "
            "printf \"%s %s %s %s %s\\n\" \"$PAINTER_GPUS_PER_NODE\" "
            "\"$PAINTER_NUM_TRAIN_GPUS\" \"$PAINTER_NUM_INFER_GPUS\" "
            "\"$PAINTER_TRAIN_GPU_IDS\" \"$PAINTER_INFER_GPU_IDS\"'"
        )
        result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "2 1 1 0 1")

    def test_h100_rejects_three_rank_warmup_without_resume(self):
        values = {
            "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1,
            "train_gpu_ids": "0", "infer_gpu_ids": "1",
            "inference_tensor_parallel": 1,
        }
        with self.assertRaisesRegex(ValueError, "RESUME"):
            self.profile.validate_bindings("h100_2", values, environ={})


if __name__ == "__main__":
    unittest.main()
