import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from painter.contract import NEXT_VERSION, SYSTEM, paint_target


MODULE_PATH = Path(__file__).with_name("prepare_photo_curriculum.py")
SPEC = importlib.util.spec_from_file_location("photo_curriculum_prepare", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PROGRAM = """function setup() { createCanvas(600, 600, WEBGL); }
function draw() { translate(-300, -300); noLoop(); }
"""


class PhotoCurriculumPreparationTests(unittest.TestCase):
    def test_full_target_uses_canonical_contract_and_prompt_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference.png"
            reference.write_bytes(b"png")
            row = MODULE.full_row(
                ident="example__full", source_group="example", split="train", family="box",
                reference=reference, target=PROGRAM, category="real_photo_teacher",
                metadata={"source": "test"}, instruction=NEXT_VERSION, plan="Match the reference.",
            )
        self.assertEqual(row["messages"][0]["content"], [{"type": "text", "text": SYSTEM}])
        self.assertEqual(row["messages"][1]["content"][0], {"type": "text", "text": "REFERENCE"})
        self.assertEqual(row["messages"][1]["content"][2], {"type": "text", "text": NEXT_VERSION})
        self.assertEqual(row["messages"][2]["content"][0]["text"], paint_target("Match the reference.", PROGRAM))

    def test_procedural_dev_is_validation_and_train_is_train(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root / "shards" / "shard-000"
            rows = []
            for ident, split in (("sc-train", "train"), ("sc-dev", "dev")):
                folder = shard / ident
                folder.mkdir(parents=True)
                (folder / "turn-3.js").write_text(PROGRAM)
                (folder / "turn-3.png").write_bytes(b"png")
                rows.append({
                    "id": ident, "source_group": ident, "split": split, "family": "box",
                    "programs": [f"{ident}/turn-3.js"], "reference": f"{ident}/turn-3.png",
                    "plans": ["Match the reference."],
                })
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            prepared, rejected = MODULE.load_procedural(root, manifest, include_corrections=False)
        self.assertEqual(rejected, [])
        self.assertEqual({row["split"] for row in prepared}, {"train", "validation"})
        self.assertEqual(sum(row["split"] == "train" for row in prepared), 1)
        self.assertEqual(sum(row["split"] == "validation" for row in prepared), 1)

    def test_loader_skips_failed_wrapper_receipt_for_valid_js_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "teachers"
            folder = root / "example"
            folder.mkdir(parents=True)
            reference = folder / "reference.png"
            reference.write_bytes(b"reference")
            program = folder / "program.js"
            program.write_text(PROGRAM)
            valid_canvas = folder / "program.js.png"
            valid_canvas.write_bytes(b"valid canvas")
            (folder / "program.png").write_bytes(b"failed canvas")
            (folder / "program.json").write_text(json.dumps({"valid": False, "error": "old wrapper"}))
            (folder / "program.js.json").write_text(json.dumps({
                "valid": True,
                "source_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
                "png_sha256": hashlib.sha256(valid_canvas.read_bytes()).hexdigest(),
            }))
            (folder / "metadata.json").write_text(json.dumps({
                "assignment_id": "example", "reference": str(reference),
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
                "split": "train", "family": "box",
            }))
            rows, rejected = MODULE.load_teacher_rows(root, include_corrections=False, reference_roots=[root])
        self.assertEqual(rejected, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_kind"], "real_photo_teacher")


if __name__ == "__main__":
    unittest.main()
