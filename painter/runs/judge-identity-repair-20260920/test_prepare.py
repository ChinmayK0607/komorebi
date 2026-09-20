import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from prepare import import_manifest, prepare, resolve_manifest  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfixture-"


class IdentityRepairPreparationTest(unittest.TestCase):
    def make_selection(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        source = root / "source"
        source.mkdir()
        counter = 0

        def image(label):
            nonlocal counter
            counter += 1
            path = source / f"{counter:03d}-{label}.png"
            path.write_bytes(PNG + label.encode())
            return str(path)

        real = []
        for number in range(6):
            real.append({
                "id": f"real_{number}",
                "reference": image(f"real-{number}-reference"),
                "A": image(f"real-{number}-A"),
                "B": image(f"real-{number}-B"),
            })
        controls = [
            {"id": "identity", "reference": image("identity-ref"),
             "A": image("identity-candidate"), "B": image("identity-candidate"),
             "expected_winner": "tie"},
            {"id": "exact-both", "reference": image("exact-both"),
             "A": str(source / "000-nope.png"), "B": str(source / "000-nope.png"),
             "expected_winner": "tie"},
        ]
        # Reuse the exact reference path for the all-identical control.  This
        # avoids fabricating image bytes while keeping its expected answer.
        controls[1]["A"] = controls[1]["reference"]
        controls[1]["B"] = controls[1]["reference"]
        for expected in ("A", "B", "A", "B"):
            reference = image(f"match-{expected}-reference")
            other = image(f"match-{expected}-other")
            controls.append({"id": f"match-{len(controls)}", "reference": reference,
                             "A": reference if expected == "A" else other,
                             "B": other if expected == "A" else reference,
                             "expected_winner": expected})
        selection = root / "selection.json"
        selection.write_text(json.dumps({"controls": controls, "real_pairs": real}))
        return tmp, root, selection

    def test_role_bijection_and_reversal_keep_hash_bindings(self):
        tmp, root, selection = self.make_selection()
        try:
            output = root / "prepared"
            prepare(selection, output, seed=7, require_real_count=6)
            first = json.loads((output / "judge-01.json").read_text())
            second = json.loads((output / "judge-02.json").read_text())
            bindings = json.loads((output / "bindings.json").read_text())
            self.assertEqual(len(first), 6)
            self.assertEqual(len(second), 6)
            self.assertEqual({x["ordinal"] for x in first}, {x["ordinal"] for x in second})
            for source_id, pair in bindings["real_pairs"].items():
                view_ids = set(pair["views"])
                shard_ids = {x["id"] for x in first + second if x["id"] in view_ids}
                self.assertEqual(shard_ids, view_ids)
                views = list(pair["views"].values())
                self.assertEqual({v["orientation"] for v in views}, {"ab", "ba"})
                by_orientation = {v["orientation"]: v for v in views}
                canonical = pair["canonical_rolemap"]
                self.assertEqual(by_orientation["ab"]["rolemap"], canonical)
                self.assertEqual(by_orientation["ba"]["rolemap"]["reference"], canonical["reference"])
                self.assertEqual(by_orientation["ba"]["rolemap"]["A"], canonical["B"])
                self.assertEqual(by_orientation["ba"]["rolemap"]["B"], canonical["A"])
                self.assertEqual(set(canonical), {"reference", "A", "B"})
                self.assertEqual(set(pair["source_hashes"]), {"reference", "A", "B"})
        finally:
            tmp.cleanup()

    def test_unique_assets_hash_check_and_no_path_claims(self):
        tmp, root, selection = self.make_selection()
        try:
            output = root / "prepared"
            prepare(selection, output, seed=8, require_real_count=6)
            assets = json.loads((output / "assets.json").read_text())["assets"]
            unique_hashes = {asset["sha256"] for asset in assets}
            self.assertEqual(len(assets), len(unique_hashes))
            for filename in ("judge-01.json", "judge-02.json"):
                items = json.loads((output / filename).read_text())
                for item in items:
                    self.assertEqual(set(item), {"id", "ordinal", "rolemap", "asset_ids", "display_order"})
                    self.assertEqual(item["display_order"], ["reference", "A", "B"])
                    self.assertEqual(item["asset_ids"], list(dict.fromkeys(
                        item["rolemap"][role] for role in ("reference", "A", "B"))))
                    self.assertNotIn("reviewed_images", item)
                    self.assertNotIn("path", item)
                self.assertEqual(import_manifest(output / "assets.json", output / filename)["views"], 6)
            path = Path(assets[0]["path"])
            original = path.read_bytes()
            path.write_bytes(PNG + b"changed")
            with self.assertRaises(ValueError):
                import_manifest(output / "assets.json", output / "judge-01.json")
            path.write_bytes(original)
        finally:
            tmp.cleanup()

    def test_deterministic_controls_are_separate_and_not_visual_successes(self):
        tmp, root, selection = self.make_selection()
        try:
            output = root / "prepared"
            prepare(selection, output, seed=9, require_real_count=6)
            controls = json.loads((output / "deterministic-controls.json").read_text())
            self.assertEqual(len(controls["controls"]), 6)
            self.assertEqual(controls["visual_judging"], "excluded")
            classifications = {row["deterministic"]["classification"] for row in controls["controls"]}
            self.assertIn("identical_candidates_tie", classifications)
            self.assertIn("both_candidates_exact_reference", classifications)
            self.assertIn("exact_reference_match_A", classifications)
            self.assertIn("exact_reference_match_B", classifications)
            shard_ids = {row["id"] for name in ("judge-01.json", "judge-02.json")
                         for row in json.loads((output / name).read_text())}
            self.assertTrue(shard_ids)
            self.assertTrue(shard_ids.isdisjoint({row["id"] for row in controls["controls"]}))
            resolved = resolve_manifest(output / "assets.json", output / "judge-01.json")
            self.assertEqual(len(resolved), 6)
            self.assertEqual(set(resolved[0]), {"id", "ordinal", "rolemap", "display"})
            self.assertTrue(all(set(item["rolemap"]) == {"reference", "A", "B"} for item in resolved))
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
