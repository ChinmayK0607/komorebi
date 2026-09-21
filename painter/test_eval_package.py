"""Pure packaging checks for the controlled evaluation input bundle."""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import eval_package


class EvalPackageTests(unittest.TestCase):
    def test_manifest_is_exactly_twelve_suffix_007_cases(self):
        raw, cases = eval_package._load_manifest(eval_package.ROOT)
        self.assertEqual(raw["schema_version"], "controlled-comparison-eval-v1")
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["family"] for case in cases}), 12)
        self.assertTrue(all(case["id"].endswith("-007") for case in cases))

    def test_build_has_only_explicit_assets_and_no_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "eval.tar.gz"
            receipt = eval_package.build(output=archive)
            self.assertFalse(receipt["includes_weights"])
            self.assertFalse(receipt["includes_credentials"])
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
                assets = sorted(name for name in names if name.startswith("eval-assets/"))
                self.assertEqual(len(assets), 12)
                self.assertIn("native-eval-tasks.json", names)
                self.assertIn("eval-provenance.json", names)
                self.assertIn("bundle-file-manifest.json", names)
                self.assertIn("hardware.py", names)
                self.assertIn("test_contract.py", names)
                self.assertIn("native_painting_eval.py", names)
                self.assertFalse(any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in names))
                self.assertFalse(any(".." in Path(name).parts or Path(name).is_absolute() for name in names))
                derived = json.load(bundle.extractfile("native-eval-tasks.json"))
                self.assertEqual(len(derived["cases"]), 12)
                self.assertEqual({case["max_turns"] for case in derived["cases"]}, {6})
                self.assertEqual(
                    derived["protocol"],
                    {
                        "context_length": 32768,
                        "max_images_per_prompt": 14,
                        "max_tokens_per_turn": 8192,
                        "max_turns": 6,
                        "temperature": 1.0,
                    },
                )

    def test_safety_filter_rejects_credentials_and_cache_state(self):
        self.assertFalse(eval_package._safe(Path("private/hf-token")))
        self.assertFalse(eval_package._safe(Path("cache/huggingface/model.safetensors")))
        self.assertFalse(eval_package._safe(Path("logs/server.log")))
        self.assertTrue(eval_package._safe(eval_package.ROOT / "native_painting.py"))


if __name__ == "__main__":
    unittest.main()
