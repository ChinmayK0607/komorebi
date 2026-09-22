"""Offline tests for finite queue transport invariants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("photo_rl_judge_bridge", ROOT / "judge_bridge.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JudgeBridgeTests(unittest.TestCase):
    def _worker_args(self, temp: str) -> SimpleNamespace:
        return SimpleNamespace(
            batch=None, batch_prefix="test-batch", max_batches=1,
            deadline_seconds=60.0, poll_seconds=20.0,
            local_root=Path(temp),
            host="node.example", port=22, user="root",
            identity=None, known_hosts=None, remote_root=Path("/root/painter"),
        )

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

    def test_worker_retries_ssh_transport_at_same_batch_boundary(self):
        with TemporaryDirectory() as temp:
            for transient in (
                subprocess.CalledProcessError(255, ["ssh", "private-host"]),
                subprocess.TimeoutExpired(["ssh", "private-host"], 120),
            ):
                with self.subTest(transient=type(transient).__name__):
                    args = self._worker_args(temp)
                    with patch.object(MODULE, "pull", side_effect=[
                        transient, {"batch": "test-batch-0001", "status": "staged"}
                    ]) as pull, patch.object(
                        MODULE, "run_local", return_value={"status": "labels"}
                    ), patch.object(
                        MODULE, "push", return_value={"status": "installed"}
                    ), patch.object(MODULE.time, "sleep") as sleep:
                        result = MODULE.worker(args)
                    self.assertEqual(result["status"], "max_batches")
                    self.assertEqual(len(result["processed"]), 1)
                    self.assertEqual(pull.call_count, 2)
                    sleep.assert_called_once_with(2.0)

    def test_worker_does_not_retry_local_judge_failure(self):
        with TemporaryDirectory() as temp:
            args = self._worker_args(temp)
            local_failure = subprocess.CalledProcessError(255, ["local-judge"])
            with patch.object(MODULE, "pull", return_value={"batch": "test-batch-0001"}) as pull, patch.object(
                MODULE, "run_local", side_effect=local_failure
            ) as run_local, patch.object(MODULE.time, "sleep") as sleep:
                with self.assertRaises(subprocess.CalledProcessError):
                    MODULE.worker(args)
            self.assertEqual(pull.call_count, 1)
            self.assertEqual(run_local.call_count, 1)
            sleep.assert_not_called()

    def test_worker_retries_empty_queue_exit_probe(self):
        with TemporaryDirectory() as temp:
            args = self._worker_args(temp)
            timeout = subprocess.TimeoutExpired(["ssh", "node.example"], 120)
            probe_clear = SimpleNamespace(returncode=0, args=["ssh"], stdout=b"", stderr=b"")
            with patch.object(
                MODULE, "pull", return_value={"batch": "test-batch-0001", "status": "no_pending_work"}
            ), patch.object(MODULE.subprocess, "run", side_effect=[timeout, probe_clear]) as remote_run, patch.object(
                MODULE.time, "sleep"
            ) as sleep:
                result = MODULE.worker(args)
            self.assertEqual(result["status"], "run_exit")
            self.assertEqual(remote_run.call_count, 2)
            self.assertEqual(sleep.call_args_list[0].args, (2.0,))


if __name__ == "__main__":
    unittest.main()
