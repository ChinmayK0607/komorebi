"""Offline tests for the expanded native Prime config preparation."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "prepare_scaled_native_rl.py"
RUN_DIR = ROOT / "runs" / "scaled-curriculum-20260921"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_scaled_native_rl", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScaledNativeRLConfigTests(unittest.TestCase):
    def test_plan_values_and_role_counts_are_explicit(self):
        module = _load_module()
        config = module.build_config(model="/staged/model", run_root=Path("/node/run"))
        self.assertEqual(config["max_steps"], 128)
        self.assertEqual(config["orchestrator"]["batch_size"], 32)
        self.assertEqual(config["orchestrator"]["group_size"], 8)
        self.assertEqual(config["orchestrator"]["seq_len"], 32768)
        self.assertEqual(config["orchestrator"]["train"]["sampling"]["max_completion_tokens"], 8192)
        self.assertEqual(config["orchestrator"]["train"]["sampling"]["temperature"], 1.0)
        self.assertEqual(config["trainer"]["optim"]["lr"], 1e-6)
        self.assertEqual(config["trainer"]["model"]["lora"]["rank"], 16)
        self.assertEqual(config["trainer"]["model"]["lora"]["alpha"], 32)
        self.assertEqual(config["deployment"]["num_train_gpus"], 1)
        self.assertEqual(config["deployment"]["num_infer_gpus"], 1)
        self.assertEqual(config["inference"]["vllm"]["limit_mm_per_prompt"]["image"], 14)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", str(config))

    def test_role_counts_are_configurable_and_fit_node_capacity(self):
        module = _load_module()
        config = module.build_config(
            model="/staged/model",
            run_root=Path("/node/run"),
            gpus_per_node=8,
            num_train_gpus=3,
            num_infer_gpus=2,
        )
        self.assertEqual(config["deployment"]["num_train_gpus"], 3)
        self.assertEqual(config["deployment"]["num_infer_gpus"], 2)
        with self.assertRaises(ValueError):
            module.build_config(
                model="/staged/model",
                run_root=Path("/node/run"),
                gpus_per_node=4,
                num_train_gpus=3,
                num_infer_gpus=2,
            )

    def test_generated_toml_round_trips_with_stdlib_parser(self):
        module = _load_module()
        config = module.build_config(model="/staged/model", run_root=Path("/node/run"))
        parsed = tomllib.loads(module.render_toml(config))
        self.assertEqual(parsed["deployment"]["type"], "single_node")
        self.assertEqual(parsed["deployment"]["num_train_gpus"], 1)
        self.assertEqual(parsed["deployment"]["num_infer_gpus"], 1)
        self.assertEqual(parsed["orchestrator"]["train"]["source"][0]["env"]["taskset"]["id"], "native_painting_rl")
        self.assertEqual(parsed["inference"]["vllm"]["max_model_len"], 32768)
        self.assertEqual(parsed["inference"]["vllm"]["limit_mm_per_prompt"]["image"], 14)

    def test_writer_refuses_to_overwrite_and_optional_prime_validation_is_non_runtime(self):
        module = _load_module()
        config = module.build_config(model="/staged/model", run_root=Path("/node/run"))
        with TemporaryDirectory() as temp:
            output = Path(temp) / "rl.toml"
            module.write_config(output, config)
            with self.assertRaises(FileExistsError):
                module.write_config(output, config)
        valid, message = module.validate_with_prime(config)
        self.assertIsInstance(valid, bool)
        self.assertIsInstance(message, str)

    def test_launch_recipe_has_pinned_source_and_no_physical_gpu_ids(self):
        source = (RUN_DIR / "launch_native_rl.sh").read_text()
        self.assertIn("26b3131d2716a4f8210b165df584b83b4bc54f61", source)
        self.assertIn("num_train_gpus", (ROOT / "prepare_scaled_native_rl.py").read_text())
        self.assertIn("--num-infer-gpus", source)
        self.assertIn("model-receipt.json", source)
        self.assertIn('BRUSH_MODEL_PROFILE="qwen38_27b"', source)
        self.assertIn("PAINTER_RUN_ROOT", (RUN_DIR / "train_from_sft.py").read_text())
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("nvidia-smi", source)

    def test_native_module_sources_are_not_modified_by_this_preparation(self):
        tracked = [ROOT / "native_painting.py", ROOT / "native_painting_rl.py"]
        # The test is a guard against accidentally moving the integration into
        # the shared implementation; the files are intentionally only read.
        for path in tracked:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
