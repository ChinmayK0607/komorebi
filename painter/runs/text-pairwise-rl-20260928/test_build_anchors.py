from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


SCRIPT = Path(__file__).with_name("build_anchors.py")
SPEC = importlib.util.spec_from_file_location("text_anchor_builder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnchorBuilderTest(unittest.TestCase):
    def test_scaled_selection_caps_each_family_before_training(self) -> None:
        rows = ([{"id": f"s{i}", "split": "train", "family": "simple"}
                 for i in range(24)] +
                [{"id": f"i{i}", "split": "train", "family": "intermediate"}
                 for i in range(16)] +
                [{"id": f"c{i}", "split": "train", "family": "complex"}
                 for i in range(80)])
        chosen = MODULE.choose_training_rows(rows, (12, 8, 48))
        self.assertEqual(len(chosen), 68)
        self.assertEqual([sum(row["family"] == family for row in chosen)
                          for family in ("simple", "intermediate", "complex")],
                         [12, 8, 48])
        reversed_choice = MODULE.choose_training_rows(list(reversed(rows)), (12, 8, 48))
        self.assertEqual([row["id"] for row in chosen if row["family"] == "complex"],
                         [row["id"] for row in reversed_choice if row["family"] == "complex"])

    def test_freezes_same_prompt_sft_and_replaces_failed_train_from_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = Path(__file__).parent / "prepared/pool.json"
            pool = json.loads(source.read_text())
            for row in pool["cases"]:
                destination = root / row["reference"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((source.parent / row["reference"]).read_bytes())
            (root / "pool.json").write_text(json.dumps(pool))
            eval_root = root / "eval"
            eval_root.mkdir()
            traces = []
            rejected = next(row for row in pool["cases"] if row["split"] == "train")
            for index, row in enumerate(pool["cases"]):
                canvas = eval_root / (row["id"] + ".png")
                image = Image.new("RGB", (256, 256), "white")
                ImageDraw.Draw(image).ellipse((20 + index % 10, 20, 200, 220), fill="red")
                image.save(canvas)
                valid = row["id"] != rejected["id"]
                traces.append({"task": {"data": {"task_id": row["id"]}},
                               "info": {"final_canvas_valid": valid,
                                        "final_canvas": str(canvas) if valid else None}})
            (eval_root / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
            audit = MODULE.build(pool=root / "pool.json", eval_root=eval_root,
                                 output=root / "frozen")
            train = json.loads((root / "frozen/train-manifest.json").read_text())["rows"]
            val = json.loads((root / "frozen/validation-manifest.json").read_text())["cases"]
            self.assertEqual((len(train), len(val)), (40, 16))
            self.assertNotIn(rejected["id"], {row["id"] for row in train})
            self.assertEqual(sum(bool(row.get("reserve_replacement")) for row in train), 1)
            self.assertEqual(audit["validation_usable"], 16)
            self.assertTrue(all(Path(row["opponent"]).is_file() for row in train))
            retry = root / "retry"
            retry.mkdir()
            retry_canvas = retry / "repaint.png"
            image = Image.new("RGB", (256, 256), "white")
            ImageDraw.Draw(image).rectangle((30, 30, 210, 210), fill="blue")
            image.save(retry_canvas)
            (retry / "traces.jsonl").write_text(json.dumps({
                "task": {"data": {"task_id": rejected["id"]}},
                "info": {"final_canvas_valid": True, "final_canvas": str(retry_canvas)},
            }) + "\n")
            retry_audit = MODULE.build(pool=root / "pool.json", eval_root=eval_root,
                                       retry_roots=(retry,), output=root / "frozen-retry")
            retry_train = json.loads((root / "frozen-retry/train-manifest.json").read_text())["rows"]
            self.assertIn(rejected["id"], {row["id"] for row in retry_train})
            self.assertEqual(sum(bool(row.get("reserve_replacement")) for row in retry_train), 0)
            record = next(row for row in retry_audit["decisions"] if row["id"] == rejected["id"])
            self.assertEqual(record["chosen_attempt"], 1)


if __name__ == "__main__":
    unittest.main()
