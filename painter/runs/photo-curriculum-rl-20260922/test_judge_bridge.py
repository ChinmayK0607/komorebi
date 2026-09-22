"""Offline tests for finite queue transport invariants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("photo_rl_judge_bridge", ROOT / "judge_bridge.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JudgeBridgeTests(unittest.TestCase):
    def test_status_does_not_supply_a_batch_argument(self):
        args = type("Args", (), {
            "host": "node.example", "port": 22, "user": "root",
            "identity": None, "known_hosts": None, "remote_root": Path("/root/painter"),
            "queue_script": Path("/root/painter/judge_queue.py"),
        })()
        completed = type("Completed", (), {"stdout": '{"pending": 0}\n'})()
        with patch.object(MODULE.subprocess, "run", return_value=completed) as run:
            result = MODULE._remote_queue(args, "status")
        command = run.call_args.args[0]
        self.assertEqual(result, {"pending": 0})
        self.assertEqual(command[command.index("status") + 1], "--root")

    def test_pull_rewrite_binds_relative_paths_to_hashes(self):
        with TemporaryDirectory() as temp:
            batch = Path(temp) / "batch"
            pair_id = "a" * 64
            pair_dir = batch / pair_id
            pair_dir.mkdir(parents=True)
            hashes = {}
            for role in ("reference", "A", "B"):
                data = b"\x89PNG\r\n\x1a\n" + role.encode()
                path = pair_dir / f"{role}.png"
                path.write_bytes(data)
                hashes[role] = hashlib.sha256(data).hexdigest()
            document = {
                "protocol_version": "aesthetic-pairwise-v1",
                "judge_model": "gpt-5.6-luna",
                "reasoning_effort": "low",
                "comparisons": [{
                    "id": pair_id,
                    "reference": f"/node/reward-batches/b/{pair_id}/reference.png",
                    "A": f"/node/reward-batches/b/{pair_id}/A.png",
                    "B": f"/node/reward-batches/b/{pair_id}/B.png",
                    "image_sha256": hashes,
                    "protocol_version": "aesthetic-pairwise-v1",
                }],
            }
            (batch / f"judge-{pair_id}.json").write_text(json.dumps(document))
            self.assertEqual(MODULE._rewrite_local_paths(batch, "b"), 3)
            updated = json.loads((batch / f"judge-{pair_id}.json").read_text())["comparisons"][0]
            self.assertEqual(updated["A"], f"{pair_id}/A.png")
            self.assertEqual(updated["image_sha256"], hashes)


if __name__ == "__main__":
    unittest.main()
