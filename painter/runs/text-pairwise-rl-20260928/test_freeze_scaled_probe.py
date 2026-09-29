"""The fixed training probe must sample prompts the short curriculum visits."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("freeze_scaled_probe", HERE / "freeze_scaled_probe.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FreezeProbeTest(unittest.TestCase):
    def test_picks_early_training_rows_from_both_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = ([{"id": f"s{i}", "family": "simple"} for i in range(6)] +
                    [{"id": f"i{i}", "family": "intermediate"} for i in range(5)] +
                    [{"id": f"c{i}", "family": "complex"} for i in range(20)])
            train = root / "train.json"
            output = root / "probe.json"
            train.write_text(json.dumps({"rows": rows}))
            MODULE.freeze(train, output)
            selected = [row["id"] for row in json.loads(output.read_text())["cases"]]
            self.assertEqual(selected, ["s0", "s1", "s2", "s3", "c0", "c1", "c2", "c3"])


if __name__ == "__main__":
    unittest.main()
