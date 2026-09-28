import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


def _module():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("calibrate_human_categories", root / "calibrate_human_categories.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HumanCategoryCalibrationTests(unittest.TestCase):
    def test_reverse_orientation_maps_to_original_candidate(self):
        module = _module()
        with TemporaryDirectory() as temp:
            receipts = Path(temp)
            for prompt, winners in {"default": ("A", "B"),
                                    "subject_first": ("B", "A")}.items():
                for orientation, winner in zip(("ab", "ba"), winners):
                    (receipts / f"sample-{prompt}-{orientation}.json").write_text(
                        json.dumps({"label": {"winner": winner}}))
            result = module.summarize([{"id": "sample", "category": "layout", "human": "A"}], receipts)
            self.assertEqual(result["by_prompt"]["default"]["exact_agreement"], 1)
            self.assertEqual(result["by_prompt"]["subject_first"]["exact_agreement"], 0)


if __name__ == "__main__":
    unittest.main()
