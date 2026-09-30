import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest


SOURCE = Path(__file__).parents[1] / "text-pairwise-rl-20260928" / "prepare_config.py"
SPEC = importlib.util.spec_from_file_location("text_prepare_config", SOURCE)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class PrepareConfigTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        path = root / "train-manifest.json"
        path.write_text(json.dumps({"rows": [
            {"id": f"brief-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
            for index in range(40)
        ]}) + "\n")
        return path

    def test_defaults_keep_original_two_gpu_8_by_4_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            output = root / "native.toml"
            module.prepare(output=output, model=root / "model", run_root=root, manifest=manifest)
            config = tomllib.loads(output.read_text())
            self.assertEqual({key: config["deployment"][key] for key in (
                "gpus_per_node", "num_train_gpus", "num_infer_gpus")}, {
                "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1,
            })
            self.assertEqual(config["orchestrator"]["batch_size"], 8)
            self.assertEqual(config["orchestrator"]["group_size"], 4)
            self.assertEqual(config["orchestrator"]["concurrency"]["initial_inflight"], 8)

    def test_new_run_can_request_batch16_group8_and_role_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            output = root / "native.toml"
            module.prepare(
                output=output, model=root / "model", run_root=root, manifest=manifest,
                run_name="text-sft512-consistent-20260930", max_steps=50,
                checkpoint_interval=5, judge_protocol="text-paint-sft-pairwise-consistent-v1",
                gpus_per_node=2, num_train_gpus=1, num_infer_gpus=1,
                batch_size=16, group_size=8, initial_inflight=16, max_inflight=16,
            )
            config = tomllib.loads(output.read_text())
            self.assertEqual(config["orchestrator"]["batch_size"], 16)
            self.assertEqual(config["orchestrator"]["group_size"], 8)
            self.assertEqual(config["orchestrator"]["max_steps"], 50)
            self.assertEqual(config["ckpt"]["interval"], 5)
            self.assertEqual(config["orchestrator"]["concurrency"]["max_inflight"], 16)
            self.assertEqual(json.loads((root / "rl-config-receipt.json").read_text())["reward"],
                             "text-paint-sft-pairwise-consistent-v1")


if __name__ == "__main__":
    unittest.main()
