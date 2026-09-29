import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


SCRIPT = Path(__file__).with_name("merge_adapters.py")
SPEC = importlib.util.spec_from_file_location("merge_adapters", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeAdaptersTests(unittest.TestCase):
    def test_effective_delta_is_exact_convex_combination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"r": 2, "lora_alpha": 4, "peft_type": "LORA", "bias": "none",
                      "target_modules": ["q_proj"], "base_model_name_or_path": "test/base"}
            tensors = []
            for name, offset in (("sft", 1), ("rl", 3)):
                folder = root / name
                folder.mkdir()
                (folder / "adapter_config.json").write_text(json.dumps(config))
                weights = {
                    "base_model.model.q_proj.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset,
                    "base_model.model.q_proj.lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2) - offset,
                }
                save_file(weights, str(folder / "adapter_model.safetensors"))
                tensors.append(weights)
            output = root / "mix"
            receipt = MODULE.merge(root / "sft", root / "rl", output, 0.25)
            actual = load_file(str(output / "adapter_model.safetensors"))
            a_key, b_key = ("base_model.model.q_proj.lora_A.weight", "base_model.model.q_proj.lora_B.weight")
            effective = (8 / 4) * actual[b_key] @ actual[a_key]
            expected = sum(fraction * (4 / 2) * w[b_key] @ w[a_key]
                           for fraction, w in ((0.75, tensors[0]), (0.25, tensors[1])))
            self.assertTrue(torch.allclose(effective, expected, atol=1e-5))
            self.assertEqual(receipt["rank"], 4)
            self.assertEqual(json.loads((output / "adapter_config.json").read_text())["lora_alpha"], 8)

    def test_endpoints_are_existing_adapters(self):
        with self.assertRaisesRegex(ValueError, "strictly between"):
            MODULE.merge(Path("sft"), Path("rl"), Path("mix"), 0)


if __name__ == "__main__":
    unittest.main()
