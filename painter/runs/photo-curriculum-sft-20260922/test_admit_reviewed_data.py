import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("admit_reviewed_data.py")
SPEC = importlib.util.spec_from_file_location("photo_curriculum_admit", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class AdmitReviewedDataTests(unittest.TestCase):
    def test_keeps_only_hash_bound_review_and_preserves_other_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            program, reference, render = ("a" * 64, "b" * 64, "c" * 64)
            # The real rows carry the provenance hashes emitted by preparation.
            real = {"id": "real-astra-coco128-000000000001", "source_kind": "real_photo_teacher",
                    "source_group": "coco128:train:000000000001", "reference_sha256": reference,
                    "teacher_metadata": {"id": "coco128-000000000001", "program_sha256": program,
                                         }, "messages": []}
            weak = dict(real, id="real-astra-coco128-000000000002",
                        source_group="coco128:train:000000000002",
                        reference_sha256="x" * 64,
                        teacher_metadata={"id": "coco128-000000000002", "program_sha256": "y" * 64,
                                          "render_sha256": "z" * 64})
            foundation = {"id": "foundation-1", "source_kind": "procedural_foundation", "messages": []}
            dev = {"id": "dev-1", "source_kind": "procedural_foundation", "messages": []}
            train = root / "candidates-train.jsonl"
            validation = root / "candidates-validation.jsonl"
            train.write_bytes(b"".join((json.dumps(row, separators=(",", ":")).encode() + b"\n") for row in (real, weak, foundation)))
            validation.write_bytes(json.dumps(dev, separators=(",", ":")).encode() + b"\n")
            audit = root / "audit.json"
            audit.write_text(json.dumps({
                "dataset_sha256": {"train": digest(train.read_bytes()), "validation": digest(validation.read_bytes())},
                "examples": [{"id": row["id"], "split": split} for row, split in ((real, "train"), (weak, "train"), (foundation, "train"), (dev, "validation"))],
            }))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [{
                "id": "coco128-000000000001", "source_image_id": "000000000001", "pair_status": "complete",
                "reference": {"status": "valid", "expected_sha256": reference, "actual_sha256": reference},
                "teacher_source": {"sha256": program}, "source_sha256": program,
                "render": {"status": "valid", "sha256": render, "receipt": {"source_sha256": program, "png_sha256": render}},
            }]}))
            labels = root / "labels.json"
            labels.write_text(json.dumps({"reviews": {
                "000000000001": {"decision": "keep", "program_sha256": program,
                                  "reference_sha256": reference, "render_sha256": render},
                # A stale keep decision must not admit the second row.
                "000000000002": {"decision": "keep", "program_sha256": "0" * 64,
                                  "reference_sha256": "1" * 64, "render_sha256": "2" * 64},
            }}))
            output = root / "final"
            result = MODULE.admit(train=train, validation=validation, audit=audit,
                                  review_manifest=manifest, labels=[labels], output=output)
            kept = [json.loads(line) for line in (output / "candidates-train.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in kept], [real["id"], foundation["id"]])
            self.assertEqual(json.loads((output / "candidates-validation.jsonl").read_text())["id"], "dev-1")
            self.assertEqual(result["kept_real"], 1)
            self.assertEqual(result["rejected_real"], 1)
            token_audit = json.loads((output / "token-audit.json").read_text())
            self.assertEqual([row["id"] for row in token_audit["examples"]], [real["id"], foundation["id"], dev["id"]])


if __name__ == "__main__":
    unittest.main()
