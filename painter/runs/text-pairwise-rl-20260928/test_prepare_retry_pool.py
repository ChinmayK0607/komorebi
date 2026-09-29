"""Retry manifests must keep prompt files valid after moving directories."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare_retry_pool", HERE / "prepare_retry_pool.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RetryPoolTest(unittest.TestCase):
    def test_resolves_prompts_from_source_pool_before_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            rows = []
            for ident in ("one", "two"):
                prompt = source / f"{ident}.txt"
                prompt.write_text(f"Paint {ident}.")
                rows.append({"id": ident, "split": "train", "reference": prompt.name,
                             "reference_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()})
            pool = source / "pool.json"
            pool.write_text(json.dumps({"cases": rows}))
            output = root / "relocated" / "retry.json"
            with patch.object(MODULE, "read_traces", return_value=([], [], [Path("trace.jsonl")])):
                MODULE.prepare(pool, [root / "prior"], output)
            retried = json.loads(output.read_text())["cases"]
            self.assertEqual([Path(row["reference"]) for row in retried],
                             [(source / "one.txt").resolve(),
                              (source / "two.txt").resolve()])


if __name__ == "__main__":
    unittest.main()
