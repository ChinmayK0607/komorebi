"""Offline tests for the finite parent-side rental controller."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("rental_controller.py")
SPEC = importlib.util.spec_from_file_location("rental_controller_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)


class FakeResult:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "provider-secret"):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeProcess:
    pid = 424242

    def __init__(self, code: int = 0):
        self.code = code
        self.commands: list[tuple[str, ...]] = []

    def poll(self):
        return self.code

    def wait(self, **_kwargs):
        return self.code


class FakeOwner:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid
        self.pgid = process.pid

    def stop(self, **_kwargs):
        self.process.code = 0


class RentalControllerTests(unittest.TestCase):
    def _budget(self, **overrides):
        values = {
            "pod_id": "pod-123",
            "price_usd_per_hour": 3600.0,
            "rental_start_epoch": 1000.0,
            "max_gpu_usd": 20.0,
        }
        values.update(overrides)
        return controller.make_budget(**values)

    def test_budget_uses_sixty_second_release_margin(self):
        budget = self._budget(max_gpu_usd=0.01)
        self.assertAlmostEqual(budget.rental_deadline_epoch, 1000.01)
        self.assertAlmostEqual(budget.release_deadline_epoch, budget.rental_deadline_epoch - 60.0)

    def test_exact_id_matching_and_json_prefix_are_strict(self):
        payload = '[{"id":"pod-123"}]\nprovider warning\n'
        parsed = controller.parse_ps_prefix(payload)
        self.assertTrue(controller.exact_pod_present(parsed, "pod-123"))
        self.assertFalse(controller.exact_pod_present(parsed, "pod-12"))
        with self.assertRaises(controller.RentalControllerError):
            controller.validate_pod_id("all")
        with self.assertRaises(controller.RentalControllerError):
            controller.exact_pod_present([{"id": "pod-123"}, {"id": "pod-123"}], "pod-123")

    def test_pod_present_accepts_valid_exact_inventory_with_warning_exit(self):
        def fake_run(_argv, **_kwargs):
            return FakeResult('[{"id":"pod-123"}]\n', returncode=1, stderr="warning: stale cache")

        self.assertTrue(controller.pod_present(pod_id="pod-123", lium_bin="lium", run=fake_run))

    def test_pod_present_keeps_nonzero_absent_inventory_as_failure(self):
        def fake_run(_argv, **_kwargs):
            return FakeResult('[{"id":"other-pod"}]\n', returncode=1, stderr="error: unavailable")

        with self.assertRaises(controller.RentalControllerError):
            controller.pod_present(pod_id="pod-123", lium_bin="lium", run=fake_run)

    def test_release_calls_exact_rm_and_verifies_absence_without_logging_output(self):
        calls: list[list[str]] = []
        inventory = [[{"id": "pod-123"}], []]

        def fake_run(argv, **_kwargs):
            calls.append(argv)
            if argv[1] == "ps":
                return FakeResult(json.dumps(inventory.pop(0)), stderr="secret-host")
            self.assertEqual(argv, ["lium", "rm", "pod-123", "-y"])
            return FakeResult("secret-remove-output", stderr="secret-token")

        status = controller.release_exact_pod(
            pod_id="pod-123",
            run=fake_run,
            now=lambda: 1000.0,
            sleep=lambda _seconds: None,
            poll_seconds=0.01,
        )
        self.assertEqual(status, "released")
        self.assertEqual(calls, [
            ["lium", "ps", "--format", "json"],
            ["lium", "rm", "pod-123", "-y"],
            ["lium", "ps", "--format", "json"],
        ])

    def test_marker_releases_without_starting_second_process_and_receipt_is_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "release.ready"
            marker.write_text("hash-verified-preservation")
            receipt = root / "rental-receipt.jsonl"
            calls: list[list[str]] = []
            inventory = [[{"id": "pod-123"}], []]
            started = False

            def fake_run(argv, **_kwargs):
                calls.append(argv)
                if argv[1] == "ps":
                    return FakeResult(json.dumps(inventory.pop(0)), stderr="secret-host")
                return FakeResult("secret-provider-output", stderr="secret-token")

            def fake_popen(*_args, **_kwargs):
                nonlocal started
                started = True
                raise AssertionError("marker should release before starting a local process")

            result = controller.supervise(
                budget=self._budget(),
                command=["local", "run"],
                release_marker=marker,
                receipt_path=receipt,
                run=fake_run,
                popen=fake_popen,
                now=lambda: 1000.0,
                sleep=lambda _seconds: None,
                poll_seconds=20.0,
            )
            self.assertEqual(result, 0)
            self.assertFalse(started)
            events = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["started", "released"])
            self.assertEqual(events[-1]["release_status"], "released")
            self.assertNotIn("secret", receipt.read_text())

    def test_deadline_includes_setup_and_does_not_start_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "rental-receipt.jsonl"
            calls: list[list[str]] = []
            inventory = [[{"id": "pod-123"}], []]

            def fake_run(argv, **_kwargs):
                calls.append(argv)
                if argv[1] == "ps":
                    return FakeResult(json.dumps(inventory.pop(0)))
                return FakeResult("")

            process_started = False

            def fake_popen(*_args, **_kwargs):
                nonlocal process_started
                process_started = True
                raise AssertionError("expired budget must not start setup")

            # $20 is intentionally tiny here, so the 60-second release margin
            # makes the release deadline earlier than this invocation's clock.
            budget = self._budget(max_gpu_usd=0.005)
            result = controller.supervise(
                budget=budget,
                command=["local", "run"],
                release_marker=root / "missing",
                receipt_path=receipt,
                run=fake_run,
                popen=fake_popen,
                now=lambda: 1000.0,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(result, 0)
            self.assertFalse(process_started)
            events = [json.loads(line) for line in receipt.read_text().splitlines()]
            self.assertEqual(events[-1]["reason"], "rental_deadline")

    def test_running_process_is_foreground_and_receipt_has_no_connection_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "rental-receipt.jsonl"
            calls: list[list[str]] = []
            inventory = [[{"id": "pod-123"}], []]
            fake_process = FakeProcess(code=0)
            original_owner = controller.OwnedChild

            def fake_run(argv, **_kwargs):
                calls.append(argv)
                if argv[1] == "ps":
                    return FakeResult(json.dumps(inventory.pop(0)))
                return FakeResult("")

            def fake_popen(command, **kwargs):
                self.assertEqual(command, ["local", "run"])
                self.assertTrue(kwargs["start_new_session"])
                return fake_process

            try:
                controller.OwnedChild = FakeOwner
                result = controller.supervise(
                    budget=self._budget(),
                    command=["local", "run"],
                    release_marker=root / "missing",
                    receipt_path=receipt,
                    run=fake_run,
                    popen=fake_popen,
                    now=lambda: 1000.0,
                    sleep=lambda _seconds: None,
                )
            finally:
                controller.OwnedChild = original_owner
            self.assertEqual(result, 0)
            self.assertIn(["lium", "rm", "pod-123", "-y"], calls)
            text = receipt.read_text()
            self.assertNotIn("ssh", text.lower())
            self.assertNotIn("host", text.lower())
            self.assertNotIn("token", text.lower())

    def test_poll_interval_cannot_exceed_thirty_seconds(self):
        with self.assertRaises(ValueError):
            controller.release_exact_pod(
                pod_id="pod-123", run=lambda *_args, **_kwargs: None, poll_seconds=30.1
            )


if __name__ == "__main__":
    unittest.main()
