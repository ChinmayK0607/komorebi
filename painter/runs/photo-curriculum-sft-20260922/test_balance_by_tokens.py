import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("balance_by_tokens.py")
SPEC = importlib.util.spec_from_file_location("photo_curriculum_balance", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BalanceByTokensTests(unittest.TestCase):
    def test_coverage_and_windows_are_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, examples = [], []
            for bucket, kind in (("real", "real_photo_teacher"), ("foundation", "diverse_foundation"), ("retention", "retention")):
                for index in range(4):
                    ident = f"{bucket}-{index}"
                    rows.append({"id": ident, "source_kind": kind, "source_group": bucket, "messages": []})
                    examples.append({"id": ident, "split": "train", "assistant_tokens": 10 + index})
            candidates = root / "candidates-train.jsonl"
            candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "candidates-validation.jsonl").write_text(json.dumps({"id": "dev"}) + "\n")
            audit = root / "audit.json"
            audit.write_text(json.dumps({"examples": examples}))
            output = root / "train.jsonl"
            receipt = MODULE.build(candidates, audit, output, steps=6, batch_size=2,
                                   targets={"real": 0.4, "foundation": 0.5, "retention": 0.1})
            admitted = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertTrue(receipt["coverage_satisfied"])
        self.assertEqual(receipt["uncovered_source_examples"], [])
        self.assertEqual([row["admission_index"] for row in admitted], list(range(len(admitted))))
        self.assertTrue(all("teacher_metadata" not in row and "contract" not in row for row in admitted))
        for window in (admitted[:4], admitted[len(admitted) // 2 - 2:len(admitted) // 2 + 2], admitted[-4:]):
            self.assertGreaterEqual(len({row["admission_bucket"] for row in window}), 2)


if __name__ == "__main__":
    unittest.main()
