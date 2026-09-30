import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

MODULE_PATH = Path(__file__).with_name("completion_relay.py")
SPEC = importlib.util.spec_from_file_location("completion_relay_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relay)


class CompletionRelayTests(unittest.TestCase):
    def test_parse_state_from_lium_json_envelope(self):
        raw = json.dumps({"result": {"stdout": json.dumps({
            "schema": "finite-text-paint-supervisor-v1", "status": "stopped"
        })}})
        state = relay.parse_state_output(raw)
        self.assertEqual(state["status"], "stopped")
        failed = json.dumps({"exit_code": 1, "stdout": json.dumps({"status": "stopped"})})
        self.assertIsNone(relay.parse_state_output(failed))
        self.assertIsNone(relay.parse_state_output("provider warning only"))

    def test_remote_post_redirects_node_output_and_keeps_commit_explicit(self):
        command = relay._remote_post_command(
            "/root/painter/repo/painter/runs/text-sft512-consistent-20260930",
            commit="a" * 40, start=1000, price=2.38, cap=19.5,
        )
        self.assertIn("PAINTER_PIPELINE_POST_TRAINING_ONLY=1", command)
        self.assertIn("nodecompletion.log", command)
        self.assertIn("a" * 40, command)

    def test_relay_retrieves_verified_receipts_before_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp)
            (private / "pod-id.txt").write_text("pod-test\n")
            (private / "start.json").write_text(json.dumps({
                "rental_start_epoch": 1000, "price_usd_per_hour": 360,
                "max_gpu_usd": 20, "release_margin_seconds": 60,
            }))
            calls = []

            def fake_run(argv, **kwargs):
                calls.append(argv)
                if argv[:2] == ["lium", "exec"]:
                    return types.SimpleNamespace(returncode=0, stdout=json.dumps({
                        "stdout": json.dumps({"schema": "finite-text-paint-supervisor-v1", "status": "stopped"})
                    }), stderr="")
                destination = Path(argv[-2])
                destination.parent.mkdir(parents=True, exist_ok=True)
                remote = argv[3]
                if remote.endswith("terminal-receipt.json"):
                    destination.write_text('{"status":"complete"}\n')
                elif remote.endswith("upload.json"):
                    destination.write_text(json.dumps({"private": False, "public_hash_verified": True}))
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            result = relay.relay(
                private_dir=private, lium_bin="lium", completion_commit="b" * 40,
                run=fake_run, sleep=lambda _seconds: None, now=lambda: 1000.0,
            )
            self.assertEqual(result, 0)
            self.assertTrue((private / "release.ready").is_file())
            receipt = json.loads((private / "relay-receipt.json").read_text())
            self.assertTrue(receipt["public_archive_hash_verified"])
            self.assertNotIn("pod-test", (private / "relay-receipt.json").read_text())
            self.assertTrue(any(call[:2] == ["lium", "scp"] for call in calls))

    def test_relay_does_not_mark_release_without_preserved_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp)
            (private / "pod-id.txt").write_text("pod-test\n")
            (private / "start.json").write_text(json.dumps({
                "rental_start_epoch": 1000, "price_usd_per_hour": 360,
                "max_gpu_usd": 20, "release_margin_seconds": 180,
            }))
            clock = [1000.0]

            def fake_run(argv, **_kwargs):
                if argv[:2] == ["lium", "exec"]:
                    return types.SimpleNamespace(
                        returncode=1,
                        stdout=json.dumps({"exit_code": 1, "stdout": ""}), stderr="",
                    )
                return types.SimpleNamespace(returncode=1, stdout="", stderr="")

            result = relay.relay(
                private_dir=private, lium_bin="lium", completion_commit="c" * 40,
                run=fake_run, sleep=lambda _seconds: None,
                now=lambda: (clock.__setitem__(0, clock[0] + 100) or clock[0] - 100),
            )
            self.assertEqual(result, 1)
            self.assertFalse((private / "release.ready").exists())


if __name__ == "__main__":
    unittest.main()
