"""CPU-only allocation wrapper contracts with a fake provider SDK."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest


RUN = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


operation = load("allocation_operation", RUN / "operate_single.py")
allocate = load("single_allocator", RUN / "allocate_single.py")


class FakeChannel:
    def __init__(self, *, exit_code=0):
        self.exit_code = exit_code

    def exit_status_ready(self):
        return True

    def recv_exit_status(self):
        return self.exit_code

    def recv_ready(self):
        return False

    def recv_stderr_ready(self):
        return False

    def close(self):
        return None


class FakeStream:
    def __init__(self, *, exit_code=0):
        self.channel = FakeChannel(exit_code=exit_code)


class FakeSFTP:
    def __init__(self, provider):
        self.provider = provider

    def put(self, local, remote):
        self.provider.sftp_calls.append(("put", local, remote))

    def chmod(self, remote, mode):
        self.provider.sftp_calls.append(("chmod", remote, mode))

    def close(self):
        self.provider.sftp_calls.append(("close",))


class FakeSSH:
    def __init__(self, provider):
        self.provider = provider

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.provider.ssh_closed += 1
        return False

    def exec_command(self, command, timeout=None):
        self.provider.ssh_commands.append(command)
        failed = self.provider.stage_error and "git -C" in command
        return None, FakeStream(exit_code=1 if failed else 0), None

    def open_sftp(self):
        return FakeSFTP(self.provider)


class FakeProvider:
    def __init__(self, *, wait_error: BaseException | None = None, stage_error: bool = False):
        self.offers = [SimpleNamespace(id="executor-2", gpu_type="H100", gpu_count=2, price_per_hour=3.8)]
        self.wait_error = wait_error
        self.stage_error = stage_error
        self.calls = []
        self.removed = []
        self.wait_calls = []
        self.sftp_calls = []
        self.ssh_commands = []
        self.ssh_closed = 0

    def ls(self, *, gpu_type, gpu_count):
        self.calls.append(("ls", gpu_type, gpu_count))
        return self.offers

    def up(self, **kwargs):
        self.calls.append(("up", kwargs))
        return {"id": "pod-allocated", "status": "PROVISIONING"}

    def wait_ready(self, pod, *, timeout, poll_interval):
        self.wait_calls.append((pod, timeout, poll_interval))
        if self.wait_error is not None:
            raise self.wait_error
        if len(self.wait_calls) == 1:
            return SimpleNamespace(id="pod-allocated", status="PROVISIONING")
        return SimpleNamespace(id="pod-allocated", status="RUNNING", ssh_cmd="ssh root@unpublished")

    def ps(self):
        return [SimpleNamespace(id="pod-allocated")]

    def ssh_connection(self, pod, timeout=None):
        return FakeSSH(self)

    def rm(self, pod):
        self.removed.append(str(pod.id))
        return {"status": "removed"}


class FakeOperation:
    COMMIT_RE = operation.COMMIT_RE
    _bounded_call = staticmethod(operation._bounded_call)
    make_window = staticmethod(operation.make_window)
    resolve_exact_pod = staticmethod(operation.resolve_exact_pod)
    _remaining = staticmethod(operation._remaining)
    _remote_exec = staticmethod(operation._remote_exec)
    _remote_exec_capture = staticmethod(operation._remote_exec_capture)
    _sftp_put = staticmethod(operation._sftp_put)
    _source_checkout_command = staticmethod(operation._source_checkout_command)
    REMOTE_ROOT = operation.REMOTE_ROOT
    INVENTORY_TIMEOUT_SECONDS = operation.INVENTORY_TIMEOUT_SECONDS
    SSH_TIMEOUT_SECONDS = operation.SSH_TIMEOUT_SECONDS
    TRANSFER_TIMEOUT_SECONDS = operation.TRANSFER_TIMEOUT_SECONDS

    def __init__(self, *, error: BaseException | None = None, window_error: BaseException | None = None):
        self.error = error
        self.window_error = window_error
        self.window = None

    def operate(self, **kwargs):
        self.window = kwargs["window"]
        if self.error is not None:
            raise self.error
        return 0

    def make_window(self, **kwargs):
        if self.window_error is not None:
            raise self.window_error
        return operation.make_window(**kwargs)


class AllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source_repo = cls.root / "source-repo"
        cls.source_repo.mkdir()
        subprocess.run(["git", "init", "--quiet", str(cls.source_repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(cls.source_repo), "config", "user.email", "fixture@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(cls.source_repo), "config", "user.name", "fixture"], check=True)
        (cls.source_repo / "tracked.txt").write_text("allocator fixture\n")
        subprocess.run(["git", "-C", str(cls.source_repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(cls.source_repo), "commit", "--quiet", "-m", "fixture"], check=True)
        cls.bundle = cls.root / "source.bundle"
        cls.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cls.source_repo, text=True).strip()
        subprocess.run(["git", "bundle", "create", str(cls.bundle), "--all"], cwd=cls.source_repo, check=True, capture_output=True)
        cls.token = cls.root / "hf-token"
        cls.gateway = cls.root / "gateway.env"
        cls.token.write_text("token-placeholder\n")
        cls.gateway.write_text("gateway-placeholder\n")
        cls.token.chmod(0o600)
        cls.gateway.chmod(0o600)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_allocation(self, provider, fake_operation):
        return allocate.allocate_and_operate(
            api=provider,
            operation=fake_operation,
            max_price_usd_hour=4.0,
            cap_usd=20.0,
            source_bundle=self.bundle,
            source_commit=self.commit,
            prime_bundle=self.bundle,
            prime_commit=self.commit,
            prime_sha256=allocate._sha256(self.bundle),
            hf_token_file=self.token,
            gateway_env_file=self.gateway,
            output_root=self.root / f"receipt-{len(provider.calls)}",
            now=lambda: 1_800_000_000.0,
        )

    def test_pending_ssh_then_ready_uses_bounded_sdk_wait_and_releases_exact_pod(self):
        provider = FakeProvider()
        fake_operation = FakeOperation()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(self.run_allocation(provider, fake_operation), 0)
        self.assertEqual(len(provider.wait_calls), 2)
        self.assertTrue(all(call[1] <= 40.0 for call in provider.wait_calls))
        self.assertTrue(all(call[1] <= 40.0 and call[2] <= 4.0 for call in provider.wait_calls))
        self.assertEqual(provider.removed, ["pod-allocated"])
        self.assertEqual(provider.calls[1][1]["executor_id"], "executor-2")
        self.assertEqual(fake_operation.window.price_usd_per_hour, 3.8)
        self.assertIn("ready_pending", output.getvalue())
        self.assertIn("prime_stage_verified", output.getvalue())
        self.assertNotIn("pod-allocated", output.getvalue())
        self.assertTrue(any(call[0] == "put" and call[2].endswith("/prime.bundle") for call in provider.sftp_calls))
        stage_commands = " ".join(provider.ssh_commands)
        self.assertIn("sha256sum", stage_commands)
        self.assertIn("git clone --quiet --no-checkout", stage_commands)
        self.assertIn("git -C /root/painter/prime fsck --full", stage_commands)
        receipt_path = self.root / "receipt-0" / "allocation-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["pod_id"], "pod-allocated")
        self.assertEqual(receipt["source_commit"], self.commit)
        self.assertNotIn("token-placeholder", receipt_path.read_text())
        self.assertNotIn("gateway-placeholder", receipt_path.read_text())

    def test_handoff_exception_still_releases_exact_pod(self):
        provider = FakeProvider()
        fake_operation = FakeOperation(error=RuntimeError("handoff failed"))
        with self.assertRaisesRegex(RuntimeError, "handoff failed"):
            self.run_allocation(provider, fake_operation)
        self.assertEqual(provider.removed, ["pod-allocated"])

    def test_window_construction_exception_still_releases_exact_pod(self):
        provider = FakeProvider()
        fake_operation = FakeOperation(window_error=ValueError("window rejected"))
        with self.assertRaisesRegex(ValueError, "window rejected"):
            self.run_allocation(provider, fake_operation)
        self.assertEqual(provider.removed, ["pod-allocated"])

    def test_wait_ready_failure_still_releases_after_allocation(self):
        provider = FakeProvider(wait_error=RuntimeError("ssh handoff failed"))
        with self.assertRaisesRegex(RuntimeError, "ssh handoff failed"):
            self.run_allocation(provider, FakeOperation())
        self.assertEqual(provider.removed, ["pod-allocated"])

    def test_prime_staging_failure_still_releases_exact_pod(self):
        provider = FakeProvider(stage_error=True)
        with self.assertRaisesRegex(RuntimeError, "staging"):
            self.run_allocation(provider, FakeOperation())
        self.assertEqual(provider.removed, ["pod-allocated"])

    def test_invalid_cap_is_rejected_before_provider_calls(self):
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "cap-usd"):
            allocate.allocate_and_operate(
                api=provider,
                operation=FakeOperation(),
                max_price_usd_hour=4.0,
                cap_usd=20.01,
                source_bundle=self.bundle,
                source_commit=self.commit,
                prime_bundle=self.bundle,
                prime_commit=self.commit,
                prime_sha256=allocate._sha256(self.bundle),
                hf_token_file=self.token,
                gateway_env_file=self.gateway,
                output_root=self.root / "invalid",
            )
        self.assertEqual(provider.calls, [])

    def test_invalid_hourly_limit_is_rejected_before_provider_calls(self):
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "max-price-usd-hour"):
            allocate.allocate_and_operate(
                api=provider,
                operation=FakeOperation(),
                max_price_usd_hour=0.0,
                cap_usd=20.0,
                source_bundle=self.bundle,
                source_commit=self.commit,
                prime_bundle=self.bundle,
                prime_commit=self.commit,
                prime_sha256=allocate._sha256(self.bundle),
                hf_token_file=self.token,
                gateway_env_file=self.gateway,
                output_root=self.root / "invalid-hourly-limit",
            )
        self.assertEqual(provider.calls, [])

    def test_prime_bundle_hash_is_rejected_before_provider_calls(self):
        provider = FakeProvider()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            allocate.allocate_and_operate(
                api=provider,
                operation=FakeOperation(),
                max_price_usd_hour=4.0,
                cap_usd=20.0,
                source_bundle=self.bundle,
                source_commit=self.commit,
                prime_bundle=self.bundle,
                prime_commit=self.commit,
                prime_sha256="0" * 64,
                hf_token_file=self.token,
                gateway_env_file=self.gateway,
                output_root=self.root / "invalid-prime-hash",
            )
        self.assertEqual(provider.calls, [])

    def test_offer_selection_rejects_wrong_gpu_count_and_price(self):
        with self.assertRaisesRegex(RuntimeError, "two-GPU"):
            allocate.select_offer(
                [SimpleNamespace(id="one", gpu_type="H100", gpu_count=1, price_per_hour=1.0),
                 SimpleNamespace(id="two-expensive", gpu_type="H100", gpu_count=2, price_per_hour=4.0)],
                max_price_usd_hour=3.8,
            )

    def test_offer_selection_rejects_cheaper_non_h100_offer(self):
        selected = allocate.select_offer(
            [SimpleNamespace(id="cheap-4090", gpu_type="RTX 4090", gpu_count=2, price_per_hour=1.0),
             SimpleNamespace(id="pilot-h100", gpu_type="H100", gpu_count=2, price_per_hour=3.8)],
            max_price_usd_hour=4.0,
        )
        self.assertEqual(selected.id, "pilot-h100")


if __name__ == "__main__":
    unittest.main()
