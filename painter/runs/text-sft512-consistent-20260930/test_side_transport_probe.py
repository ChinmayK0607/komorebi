import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


MODULE_PATH = Path(__file__).with_name("run_side_transport_probe.py")
SPEC = importlib.util.spec_from_file_location("side_transport_probe", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class SideTransportProbeTests(unittest.TestCase):
    def test_dry_run_selects_four_failed_pairs_and_writes_no_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            cases = []
            receipts = root / "receipts"
            receipts.mkdir()
            for index in range(4):
                case_id = f"case-{index}"
                brief = f"Paint a readable subject {index}."
                image_rows = {}
                for role, color in (("frozen SFT", "red"), ("RL-step-16", "blue")):
                    path = images / f"{case_id}-{role}.png"
                    Image.new("RGB", (8, 8), color).save(path)
                    image_rows[role] = {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                cases.append({"id": case_id, "brief": brief, "brief_sha256": hashlib.sha256(brief.encode()).hexdigest(), "images": image_rows})
                pair_id = f"{case_id}__frozen-sft__rl-step-16"
                (receipts / f"{index}.json").write_text(json.dumps({
                    "identity": {"pair_id": pair_id, "image_sha256": {role: image_rows[role]["sha256"] for role in image_rows}},
                    "judge": {"consensus": "failed_initial_call"},
                }))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": cases}))
            output = root / "out"
            pairs = probe._prepare_pairs(manifest_path=manifest, receipts=receipts, evidence_root=root)
            self.assertEqual(len(pairs), 4)
            summary = asyncio.run(probe.run_probe(pairs=pairs, output=output, key_file=None, dry_run=True))
            self.assertEqual(summary["calls_planned"], 8)
            self.assertEqual(summary["calls_completed"], 0)
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual(plan["format_version"], probe.judge.FORMAT_VERSION)
            self.assertFalse((output / "summary.json").exists())

    def test_call_record_preserves_transport_fields_and_selects_cost_once(self):
        pair = {"pair_id": "case-0__frozen-sft__rl-step-16", "case_id": "case-0", "brief_sha256": "b" * 64, "left_role": "frozen SFT", "right_role": "RL-step-16"}
        call = {"image_sha256": {"A": "a" * 64, "B": "b" * 64}, "image_ids": {"A": "img_a", "B": "img_b"}, "sheet_sha256": "s" * 64, "prompt_sha256": "p" * 64, "format_version": "fmt", "source_format_sha256": "src" * 16, "status": "parsed", "usage": {"cost": 0.2, "gateway_cost": 0.3}, "raw_response": "{}", "assessment": {"winner_id": "img_a"}}
        record = probe._call_record(pair, side="left-to-right", call=call)
        self.assertEqual(record["cost"], 0.2)
        self.assertEqual(record["request_order"]["A"], "frozen SFT")
        self.assertEqual(record["raw_response"], "{}")


if __name__ == "__main__":
    unittest.main()
