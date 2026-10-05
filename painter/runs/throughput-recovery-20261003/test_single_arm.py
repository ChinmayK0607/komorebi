"""CPU-only contracts for the single-arm recovery launcher and controller."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_operator():
    spec = importlib.util.spec_from_file_location("single_operator", RUN / "operate_single.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


operator = load_operator()


def load_profile_preparer():
    spec = importlib.util.spec_from_file_location(
        "single_profile_preparer", RUN / "prepare_profile_single.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_preparer = load_profile_preparer()


def load_fast_storage():
    spec = importlib.util.spec_from_file_location(
        "single_fast_storage", RUN / "prepare_fast_storage.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fast_storage = load_fast_storage()


class FakeSSHFailure:
    def __enter__(self):
        raise RuntimeError("transport unavailable")

    def __exit__(self, *_args):
        return False


class FakePod:
    id = "pod-test"


class FakeAPI:
    def __init__(self):
        self.removed = []

    def ps(self):
        return [FakePod()]

    def ssh_connection(self, _pod, timeout):
        self.timeout = timeout
        return FakeSSHFailure()

    def rm(self, pod):
        self.removed.append(pod)


class InventoryFailureAPI(FakeAPI):
    def ps(self):
        raise RuntimeError("inventory unavailable")


class SummaryChannel:
    def __init__(self, *, output: bytes = b"", running: bool = False):
        self.output = output
        self.running = running
        self.polls = 0
        self.closed = False

    def exit_status_ready(self):
        if self.running:
            self.polls += 1
            return self.polls >= 2
        return True

    def recv_ready(self):
        return bool(self.output)

    def recv_stderr_ready(self):
        return False

    def recv(self, _size):
        value, self.output = self.output, b""
        return value

    def recv_stderr(self, _size):
        return b""

    def recv_exit_status(self):
        return 0

    def close(self):
        self.closed = True


class SummaryStream:
    def __init__(self, payload: bytes):
        self.payload = payload

    def seek(self, offset, whence=0):
        if whence == 2:
            self.position = len(self.payload) + offset
        else:
            self.position = offset

    def tell(self):
        return getattr(self, "position", 0)

    def read(self, size=-1):
        position = getattr(self, "position", 0)
        if size is None or size < 0:
            size = len(self.payload) - position
        value = self.payload[position:position + size]
        self.position = position + len(value)
        return value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TailStream:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.position = 0
        self.read_sizes = []

    def seek(self, offset, whence=0):
        if whence == 2:
            self.position = len(self.payload) + offset
        elif whence == 1:
            self.position += offset
        else:
            self.position = offset

    def tell(self):
        return self.position

    def read(self, size=-1):
        self.read_sizes.append(size)
        if size is None or size < 0:
            size = len(self.payload) - self.position
        value = self.payload[self.position:self.position + size]
        self.position += len(value)
        return value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TailSFTP:
    def __init__(self, payload: bytes):
        self.stream = TailStream(payload)

    def open(self, _remote, _mode):
        return self.stream


class BlockingTailStream(TailStream):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.started = threading.Event()

    def read(self, size=-1):
        self.started.set()
        time.sleep(1.0)
        return super().read(size)


class BlockingTailSFTP(TailSFTP):
    def __init__(self, payload: bytes):
        self.stream = BlockingTailStream(payload)


class SummarySFTP:
    def __init__(self, contents, *, fail_reads=False, block_reads=False):
        self.contents = contents
        self.fail_reads = fail_reads
        self.block_reads = block_reads
        self.closed_event = threading.Event()
        self.closed = False

    def put(self, _local, _remote):
        return None

    def chmod(self, _remote, _mode):
        return None

    def open(self, remote, _mode):
        if self.block_reads:
            self.closed_event.wait(5.0)
            raise operator.OperationError("blocked status read cancelled")
        if self.fail_reads:
            raise operator.OperationError("bounded status read timed out")
        for suffix, payload in self.contents.items():
            if remote.endswith(suffix):
                return SummaryStream(payload)
        return SummaryStream(b"{}")

    def close(self):
        self.closed = True
        self.closed_event.set()


class SummarySSH:
    def __init__(self, contents, *, block_first_read=False):
        self.contents = contents
        self.sftp_calls = 0
        self.stale = SummarySFTP(contents, fail_reads=not block_first_read, block_reads=block_first_read)
        self.healthy = SummarySFTP(contents)
        self.execute_channel = SummaryChannel(running=True)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False

    def exec_command(self, command, timeout=None):
        if "nvidia-smi" in command:
            channel = SummaryChannel(output=b"81920\n81920\n")
        elif "execute_single.sh" in command:
            channel = self.execute_channel
        else:
            channel = SummaryChannel()
        return None, type("Stdout", (), {"channel": channel})(), None

    def open_sftp(self):
        self.sftp_calls += 1
        return self.stale if self.sftp_calls == 1 else self.healthy

    def close(self):
        self.closed = True


class DropoutChannel(SummaryChannel):
    def exit_status_ready(self):
        return True

    def recv_exit_status(self):
        return -1


class DropoutSFTP(SummarySFTP):
    def open(self, _remote, _mode):
        raise operator.OperationError("SSH inactive")


class DropoutSSH(SummarySSH):
    def __init__(self, contents, *, inactive):
        super().__init__(contents)
        self.inactive = inactive

    def exec_command(self, command, timeout=None):
        if "execute_single.sh" in command:
            return None, type("Stdout", (), {"channel": DropoutChannel()})(), None
        return super().exec_command(command, timeout=timeout)

    def open_sftp(self):
        if self.inactive:
            self.sftp_calls += 1
            return DropoutSFTP(self.contents)
        return super().open_sftp()


class AmbiguousLaunchSSH(SummarySSH):
    def __init__(self, contents):
        super().__init__(contents)
        self.execute_attempts = 0

    def exec_command(self, command, timeout=None):
        if "execute_single.sh" in command:
            self.execute_attempts += 1
            raise RuntimeError("SSH disconnected after remote fork")
        return super().exec_command(command, timeout=timeout)

    def open_sftp(self):
        self.sftp_calls += 1
        return DropoutSFTP(self.contents)


class SummaryAPI(FakeAPI):
    def __init__(self, contents, *, block_first_read=False):
        super().__init__()
        self.ssh = SummarySSH(contents, block_first_read=block_first_read)

    def ssh_connection(self, _pod, timeout):
        self.timeout = timeout
        return self.ssh


class DropoutAPI(FakeAPI):
    def __init__(self, contents):
        super().__init__()
        self.connections = [DropoutSSH(contents, inactive=True), DropoutSSH(contents, inactive=False)]
        self.connection_calls = 0

    def ssh_connection(self, _pod, timeout):
        self.timeout = timeout
        connection = self.connections[min(self.connection_calls, len(self.connections) - 1)]
        self.connection_calls += 1
        return connection


class DeadlineClock:
    def __init__(self, deadline):
        self.deadline = deadline
        self.launched = False
        self.stop_reads = 0

    def __call__(self):
        if not self.launched:
            return 1000.0
        self.stop_reads += 1
        if self.stop_reads <= 2:
            return self.deadline + 1.0 if self.stop_reads == 1 else self.deadline - 1.0
        return self.deadline + 1.0


class DeadlineLaunchChannel(SummaryChannel):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def exit_status_ready(self):
        self.clock.launched = True
        return True


class DeadlineSSH(SummarySSH):
    def __init__(self, contents, clock):
        super().__init__(contents)
        self.clock = clock
        self.commands = []

    def exec_command(self, command, timeout=None):
        self.commands.append(command)
        if "execute_single.sh" in command:
            return None, type("Stdout", (), {"channel": DeadlineLaunchChannel(self.clock)})(), None
        return super().exec_command(command, timeout=timeout)


class DeadlineAPI(FakeAPI):
    def __init__(self, contents, clock):
        super().__init__()
        self.ssh = DeadlineSSH(contents, clock)

    def ssh_connection(self, _pod, timeout):
        self.timeout = timeout
        return self.ssh


class AmbiguousLaunchAPI(FakeAPI):
    def __init__(self, contents):
        super().__init__()
        self.connections = [AmbiguousLaunchSSH(contents), SummarySSH(contents)]
        self.connection_calls = 0

    def ssh_connection(self, _pod, timeout):
        self.timeout = timeout
        connection = self.connections[min(self.connection_calls, len(self.connections) - 1)]
        self.connection_calls += 1
        return connection


class SingleArmTests(unittest.TestCase):
    def test_detached_job_real_subprocess_writes_valid_zero_and_nonzero_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shim_dir = root / "shim-bin"
            shim_dir.mkdir()
            setsid = shim_dir / "setsid"
            setsid.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "os.setsid()\n"
                "os.execvp(sys.argv[1], sys.argv[1:])\n"
            )
            setsid.chmod(0o755)
            for exit_code in (0, 17):
                case = root / f"case-{exit_code}"
                execute = case / "repo/painter/runs/throughput-recovery-20261003/execute_single.sh"
                execute.parent.mkdir(parents=True)
                execute.write_text(f"#!/usr/bin/env bash\nsleep 0.05\nexit {exit_code}\n")
                execute.chmod(0o755)
                (case / "private").mkdir(parents=True)
                command = operator._detached_job_command(
                    source_commit="a" * 40,
                    rental_start_epoch=1000.0,
                    price_usd_per_hour=3.8,
                    cap_usd=14.0,
                ).replace("/root/painter", str(case))
                environment = {**os.environ, "PATH": f"{shim_dir}:{os.environ.get('PATH', '')}"}
                result = subprocess.run(["bash", "-Eeuo", "pipefail", "-c", command], env=environment, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                receipt = case / "private/throughput-recovery-remote-job-exit.json"
                deadline = time.monotonic() + 5.0
                while not receipt.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(receipt.is_file())
                self.assertEqual(json.loads(receipt.read_text())["exit_code"], exit_code)
                self.assertEqual(json.loads((case / "private/throughput-recovery-remote-job-start.json").read_text())["status"], "started")

    def test_detached_job_command_is_shell_valid_and_durable(self):
        command = operator._detached_job_command(
            source_commit="a" * 40,
            rental_start_epoch=1000.0,
            price_usd_per_hour=3.8,
            cap_usd=14.0,
        )
        result = subprocess.run(["bash", "-n", "-c", command], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nohup setsid bash -c", command)
        self.assertIn("throughput-recovery-remote-job-exit.json", command)
        self.assertIn("throughput-recovery-remote-job.stderr.log", command)

    def test_fast_storage_links_are_profile_bound_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            home = root / "home"
            fast = root / "fast"
            uv_source = home / ".cache/uv"
            cache_receipt = fast_storage.prepare(
                phase="cache",
                run_root=run,
                fast_root=fast,
                uv_source=uv_source,
                system="Linux",
                nofile_soft=131072,
                nofile_hard=524288,
            )
            self.assertFalse(uv_source.is_symlink())
            self.assertFalse(cache_receipt["credentials_touched"])
            self.assertEqual(cache_receipt["nofile"]["soft"], 131072)
            reused = fast_storage.prepare(
                phase="cache", run_root=run, fast_root=fast, uv_source=uv_source, system="Linux"
            )
            self.assertEqual(reused["uv_cache"]["status"], "selected_fast_target")
            prime = root / "prime"
            prime_receipt = fast_storage.prepare(
                phase="prime", run_root=run, fast_root=fast, uv_source=uv_source,
                prime_root=prime, system="Linux",
            )
            self.assertTrue((prime / ".venv").is_symlink())
            self.assertEqual(prime_receipt["uv_cache"]["status"], "selected_fast_target")
            self.assertEqual(prime_receipt["prime_venv"]["status"], "created_symlink")
            render_receipt = fast_storage.prepare(
                phase="render", run_root=run, fast_root=fast, uv_source=uv_source, system="Linux"
            )
            self.assertEqual(render_receipt["renderer"]["status"], "public_traversal_root")
            self.assertTrue((fast / "renderer-browsers").is_dir())
            self.assertTrue(fast.stat().st_mode & 0o111)

    def test_fast_storage_refuses_nonempty_live_paths_and_nonlinux(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uv_source = root / "home/.cache/uv"
            uv_source.mkdir(parents=True)
            (uv_source / "active-cache-entry").write_text("preserve\n")
            cache_receipt = fast_storage.prepare(
                phase="cache", run_root=root / "run", fast_root=root / "fast",
                uv_source=uv_source, system="Linux", nofile_soft=65536,
                nofile_hard=65536, nofile_target=65536, uv_concurrent_installs=16,
            )
            self.assertEqual(cache_receipt["uv_cache"]["status"], "selected_fast_target")
            self.assertTrue(cache_receipt["uv_cache"]["existing_preserved"])
            self.assertEqual((uv_source / "active-cache-entry").read_text(), "preserve\n")
            self.assertEqual(cache_receipt["nofile"]["target"], 65536)
            self.assertEqual(cache_receipt["uv_concurrent_installs"], 16)
            with self.assertRaisesRegex(RuntimeError, "Linux"):
                fast_storage.prepare(
                    phase="cache", run_root=root / "run", fast_root=root / "fast2",
                    uv_source=root / "other-cache", system="Darwin",
                )
            prime = root / "prime"
            (prime / ".venv").mkdir(parents=True)
            (prime / ".venv/active").write_text("preserve\n")
            with self.assertRaisesRegex(RuntimeError, "live migration"):
                fast_storage.prepare(
                    phase="prime", run_root=root / "run", fast_root=root / "fast3",
                    uv_source=root / "other-cache", prime_root=prime, system="Linux",
                )

    def test_fast_storage_binding_is_limited_to_recovery_bootstrap(self):
        prepare = (RUN / "prepare_single.sh").read_text()
        runtime = (REPO / "painter/runs/photo-curriculum-rl-20260922/setup_rl_runtime.sh").read_text()
        self.assertIn('FAST_STORAGE_MODE="${PAINTER_FAST_STORAGE_MODE:-profile_default}"', prepare)
        self.assertIn("ulimit -S -n", runtime_env_text := (RUN / "runtime_env.sh").read_text())
        self.assertIn("apply_recovery_resource_limits", prepare)
        self.assertIn("soft / 64", runtime_env_text)
        self.assertIn("PAINTER_RENDERER_RUNTIME_HELPER", runtime)
        self.assertIn("PAINTER_UV_CONCURRENCY_LIMIT", runtime_env_text)
        self.assertIn('PAINTER_FAST_STORAGE_HELPER', prepare)
        self.assertIn('PAINTER_FAST_STORAGE_HELPER', runtime)
        self.assertIn('apply_recovery_resource_limits', (RUN / "execute_single.sh").read_text())
        self.assertIn('--sandbox-only', runtime)
        self.assertIn('retained base preparation may skip', prepare)
        self.assertIn('prepare_renderer_runtime.py', prepare)

    def test_resource_limits_adapt_to_lower_hard_descriptor_limit(self):
        runtime = RUN / "runtime_env.sh"
        result = subprocess.run(
            ["bash", "-c", f"ulimit -S -n 1024; ulimit -H -n 65536; source {runtime!s}; unset UV_CONCURRENT_INSTALLS; apply_recovery_resource_limits; printf '%s %s %s %s' \"$PAINTER_NOFILE_HARD\" \"$PAINTER_NOFILE_TARGET\" \"$PAINTER_NOFILE_SOFT\" \"$UV_CONCURRENT_INSTALLS\""],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        hard, target, soft, concurrency = map(int, result.stdout.split())
        self.assertEqual((hard, target, soft), (65536, 65536, 65536))
        self.assertGreaterEqual(concurrency, 1)
        self.assertLessEqual(concurrency, min(32, soft // 64))

    def test_renderer_runtime_binds_bundled_node_and_records_metadata(self):
        renderer_runtime_spec = importlib.util.spec_from_file_location(
            "single_renderer_runtime", RUN / "prepare_renderer_runtime.py"
        )
        assert renderer_runtime_spec and renderer_runtime_spec.loader
        renderer_runtime = importlib.util.module_from_spec(renderer_runtime_spec)
        renderer_runtime_spec.loader.exec_module(renderer_runtime)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer"
            node = renderer / "renderer-env/lib/python3.12/site-packages/playwright/driver/node"
            node.parent.mkdir(parents=True)
            node.write_text("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo v-test; else exit 0; fi\n")
            node.chmod(0o755)
            asset = renderer / "vendor/integrations/watercolour/assets/p5.brush.js"
            asset.parent.mkdir(parents=True)
            asset.write_text("const brush = 1;\n")
            prime = root / "prime/.venv"
            browsers = root / "browsers"
            browsers.mkdir(parents=True)
            helper = browsers / "chrome-sandbox"
            helper.write_text("sandbox\n")
            helper.chmod(0o4755)
            with patch.object(renderer_runtime, "platform") as platform_mock:
                platform_mock.system.return_value = "Linux"
                with patch.object(renderer_runtime, "_sandbox_helpers", return_value=[helper]):
                    receipt = renderer_runtime.prepare(
                        run_root=root / "run", renderer_root=renderer,
                        renderer_python=Path(sys.executable), browsers=browsers,
                        prime_venv=prime, smoke=False,
                    )
            self.assertTrue((prime / "bin/node").is_symlink())
            self.assertEqual(receipt["node_check"], "passed")
            self.assertEqual(receipt["sandbox_smoke"], "not-run")

    def test_renderer_sandbox_matches_full_helper_to_headless_fixture(self):
        renderer_runtime_spec = importlib.util.spec_from_file_location(
            "single_renderer_runtime_headless", RUN / "prepare_renderer_runtime.py"
        )
        assert renderer_runtime_spec and renderer_runtime_spec.loader
        renderer_runtime = importlib.util.module_from_spec(renderer_runtime_spec)
        renderer_runtime_spec.loader.exec_module(renderer_runtime)
        with tempfile.TemporaryDirectory() as temp:
            browsers = Path(temp) / "browsers"
            full = browsers / "chromium-1208/chrome-linux64"
            headless = browsers / "chromium_headless_shell-1208/chrome-headless-shell-linux64"
            full.mkdir(parents=True)
            headless.mkdir(parents=True)
            source = full / "chrome_sandbox"
            source.write_text("trusted helper\n")
            executable = headless / "chrome-headless-shell"
            executable.write_text("headless\n")
            targets = renderer_runtime._headless_targets(browsers, [source])
            self.assertEqual(targets, [(headless / "chrome-sandbox", source)])

    def test_renderer_sandbox_rejects_symlink_and_untrusted_destination_before_copy(self):
        renderer_runtime_spec = importlib.util.spec_from_file_location(
            "single_renderer_runtime_guards", RUN / "prepare_renderer_runtime.py"
        )
        assert renderer_runtime_spec and renderer_runtime_spec.loader
        renderer_runtime = importlib.util.module_from_spec(renderer_runtime_spec)
        renderer_runtime_spec.loader.exec_module(renderer_runtime)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "trusted/chrome_sandbox"
            source.parent.mkdir(parents=True)
            source.write_text("trusted helper\n")
            symlink_target = root / "trusted/chrome-sandbox"
            symlink_target.symlink_to(root / "missing")
            with self.assertRaisesRegex(RuntimeError, "symlink target"):
                renderer_runtime._install_helper(source, symlink_target, root / "trusted")
            untrusted = root / "untrusted"
            untrusted.mkdir()
            destination = untrusted / "chrome-sandbox"
            with patch.object(renderer_runtime, "_trusted", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "destination parent"):
                    renderer_runtime._install_helper(source, destination, root)
    def test_plan_only_uses_fake_interpreter_without_pydantic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "fake-python"
            log = root / "interpreter.log"
            blocked = root / "pydantic.py"
            blocked.write_text("raise RuntimeError('pydantic imported before bootstrap')\n")
            fake.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >> {str(log)!r}\n"
                f"exec {str(Path(sys.executable))!r} \"$@\"\n"
            )
            fake.chmod(0o755)
            env = {**os.environ, "PAINTER_PREPARE_PYTHON": str(fake), "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}"}
            result = subprocess.run(
                [str(RUN / "prepare_single.sh"), "--plan-only"],
                cwd=RUN,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("THROUGHPUT_RECOVERY_SINGLE_ARM_PLAN_VALIDATED", result.stdout)
            self.assertIn("-", log.read_text())

    def test_plan_binds_four_gpu_topology_as_one_trainer_plus_three_inference(self):
        env = {**os.environ, "PAINTER_GPUS_PER_NODE": "4", "PAINTER_NUM_TRAIN_GPUS": "1", "PAINTER_NUM_INFER_GPUS": "3"}
        result = subprocess.run(
            [str(RUN / "prepare_single.sh"), "--plan-only"],
            cwd=RUN,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"PAINTER_NUM_INFER_GPUS": 3', result.stdout)

    def test_plan_rejects_two_trainer_topology(self):
        env = {**os.environ, "PAINTER_GPUS_PER_NODE": "4", "PAINTER_NUM_TRAIN_GPUS": "2", "PAINTER_NUM_INFER_GPUS": "2"}
        result = subprocess.run(
            [str(RUN / "prepare_single.sh"), "--plan-only"],
            cwd=RUN,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one trainer", result.stderr)

    def test_controller_releases_exact_pod_after_transport_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("AI_GATEWAY_API_KEY=opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = FakeAPI()
            window = operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.0,
            )
            with self.assertRaises(RuntimeError):
                operator.operate(
                    window=window,
                    source_bundle=bundle,
                    source_commit="a" * 40,
                    hf_token_file=token,
                    gateway_env_file=gateway,
                    output_root=root / "ops",
                    api=api,
                    now=lambda: 1000.0,
                )
            self.assertEqual(len(api.removed), 1)
            receipt = (root / "ops/operations-single.jsonl").read_text()
            self.assertNotIn("pod-test", receipt)
            self.assertNotIn("AI_GATEWAY_API_KEY", receipt)

    def test_controller_reconnects_after_detached_launch_channel_minus_one(self):
        contents = {
            "throughput-recovery-remote-job-start.json": b'{"status":"started"}',
            "throughput-recovery-remote-job-exit.json": b'{"exit_code":0}',
            "supervisor-state.json": b'{"status":"completed","completed_optimizer_steps":1}',
            "terminal-receipt.json": b'{"status":"complete","pipeline_exit_code":0,"archive_exit_code":0}',
            "preparation-receipt.json": b'{"schema":"throughput-recovery-single-arm-preparation-v1","profile":"rl-throughput-recovery-v1"}',
            "rl-config-receipt.json": json.dumps({
                "checkpoint_interval": 1,
                "batch_size": 16,
                "group_size": 4,
                "initial_inflight": 32,
                "max_inflight": 32,
                "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
            }).encode(),
            "metrics.jsonl": b'{"render_queue_seconds":0.1,"render_active_seconds":0.2}\n',
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = DropoutAPI(contents)
            window = operator.make_window(pod_id="pod-test", rental_start_epoch=1000.0, price_usd_per_hour=3.8, cap_usd=20.0)
            result = operator.operate(
                window=window,
                source_bundle=bundle,
                source_commit="a" * 40,
                hf_token_file=token,
                gateway_env_file=gateway,
                output_root=root / "ops",
                api=api,
                now=lambda: 1000.0,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(result, 0)
            events = [json.loads(line) for line in (root / "ops/operations-single.jsonl").read_text().splitlines()]
            self.assertTrue(any(event["event"] == "remote_transport_dropped" for event in events))
            self.assertTrue(any(event["event"] == "remote_transport_reconnected" for event in events))
            finished = [event for event in events if event["event"] == "remote_execute_finished"]
            self.assertEqual(finished[-1]["exit_code"], 0)
            self.assertFalse(any(event.get("exit_code") == -1 for event in events))
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])

    def test_controller_returns_durable_nonzero_job_exit(self):
        contents = {
            "throughput-recovery-remote-job-start.json": b'{"status":"started"}',
            "throughput-recovery-remote-job-exit.json": b'{"exit_code":17}',
            "supervisor-state.json": b'{"status":"failed"}',
            "terminal-receipt.json": b'{"status":"failed","pipeline_exit_code":17,"archive_exit_code":0}',
            "preparation-receipt.json": b'{}',
            "rl-config-receipt.json": b'{}',
            "metrics.jsonl": b'{}\n',
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = SummaryAPI(contents)
            window = operator.make_window(pod_id="pod-test", rental_start_epoch=1000.0, price_usd_per_hour=3.8, cap_usd=20.0)
            result = operator.operate(
                window=window,
                source_bundle=bundle,
                source_commit="a" * 40,
                hf_token_file=token,
                gateway_env_file=gateway,
                output_root=root / "ops",
                api=api,
                now=lambda: 1000.0,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(result, 17)
            events = [json.loads(line) for line in (root / "ops/operations-single.jsonl").read_text().splitlines()]
            finished = [event for event in events if event["event"] == "remote_execute_finished"]
            self.assertEqual(finished[-1]["exit_code"], 17)
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])

    def test_controller_reconnects_after_ambiguous_launch_without_relaunching(self):
        contents = {
            "throughput-recovery-remote-job-start.json": b'{"status":"started"}',
            "throughput-recovery-remote-job-exit.json": b'{"exit_code":0}',
            "supervisor-state.json": b'{"status":"completed"}',
            "terminal-receipt.json": b'{"status":"complete","pipeline_exit_code":0,"archive_exit_code":0}',
            "preparation-receipt.json": b'{}',
            "rl-config-receipt.json": b'{}',
            "metrics.jsonl": b'{}\n',
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = AmbiguousLaunchAPI(contents)
            window = operator.make_window(pod_id="pod-test", rental_start_epoch=1000.0, price_usd_per_hour=3.8, cap_usd=20.0)
            result = operator.operate(
                window=window,
                source_bundle=bundle,
                source_commit="a" * 40,
                hf_token_file=token,
                gateway_env_file=gateway,
                output_root=root / "ops",
                api=api,
                now=lambda: 1000.0,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(result, 0)
            self.assertEqual(api.connections[0].execute_attempts, 1)
            self.assertGreaterEqual(api.connection_calls, 2)
            events = [json.loads(line) for line in (root / "ops/operations-single.jsonl").read_text().splitlines()]
            self.assertTrue(any(event["event"] == "remote_launch_ambiguous" for event in events))
            self.assertTrue(any(event["event"] == "remote_transport_reconnected" for event in events))
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])

    def test_controller_requests_remote_stop_at_hard_deadline_before_release(self):
        contents = {
            "throughput-recovery-remote-job-start.json": b'{"status":"started"}',
            "supervisor-state.json": b'{"status":"running"}',
            "terminal-receipt.json": b'{}',
            "preparation-receipt.json": b'{}',
            "rl-config-receipt.json": b'{}',
            "metrics.jsonl": b'{}\n',
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            window = operator.make_window(pod_id="pod-test", rental_start_epoch=1000.0, price_usd_per_hour=3.8, cap_usd=20.0)
            clock = DeadlineClock(window.release_deadline_epoch)
            api = DeadlineAPI(contents, clock)
            with self.assertRaises(operator.OperationError):
                operator.operate(
                    window=window,
                    source_bundle=bundle,
                    source_commit="a" * 40,
                    hf_token_file=token,
                    gateway_env_file=gateway,
                    output_root=root / "ops",
                    api=api,
                    now=clock,
                    sleep=lambda _seconds: None,
                )
            events = [json.loads(line) for line in (root / "ops/operations-single.jsonl").read_text().splitlines()]
            deadline_events = [event for event in events if event["event"] == "remote_hard_deadline_action"]
            self.assertEqual(len(deadline_events), 1)
            self.assertTrue(deadline_events[0]["stop_sent"])
            self.assertTrue(any("kill -TERM" in command for command in api.ssh.commands))
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])

    def test_controller_budget_rejects_cap_above_twenty(self):
        with self.assertRaises(ValueError):
            operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.01,
            )

    def test_controller_replays_bundle_without_remote_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Single Arm Test"], check=True)
            (source / "marker.txt").write_text("bundle replay\n")
            subprocess.run(["git", "-C", str(source), "add", "marker.txt"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "bundle"], check=True)
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(source), "branch", "-M", "codex/single-arm-bundle"], check=True)
            bundle = root / "source.bundle"
            subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "refs/heads/codex/single-arm-bundle"], check=True, capture_output=True, text=True)
            destination = root / "repo"
            command = operator._source_checkout_command(str(bundle), str(destination), commit)
            subprocess.run(["bash", "-Eeuo", "pipefail", "-c", command], check=True)
            self.assertEqual((destination / "marker.txt").read_text(), "bundle replay\n")
            self.assertEqual(subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip(), commit)

    def test_controller_releases_requested_id_when_inventory_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = InventoryFailureAPI()
            window = operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.0,
            )
            with self.assertRaises(RuntimeError):
                operator.operate(
                    window=window,
                    source_bundle=bundle,
                    source_commit="a" * 40,
                    hf_token_file=token,
                    gateway_env_file=gateway,
                    output_root=root / "ops",
                    api=api,
                    now=lambda: 1000.0,
                )
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])
            self.assertNotIn("pod-test", (root / "ops/operations-single.jsonl").read_text())

    def test_controller_waits_for_absent_config_and_rejects_malformed_config(self):
        valid = {
            "checkpoint_interval": 1,
            "batch_size": 16,
            "group_size": 4,
            "initial_inflight": 32,
            "max_inflight": 32,
            "activation_cpu_offload": True,
            "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
        }

        def summary_for(config_raw, *, ready=False):
            preparation = (
                b'{"schema":"throughput-recovery-single-arm-preparation-v1","profile":"rl-throughput-recovery-v1"}'
                if ready else b"{}"
            )
            values = [b"{}", b"{}", None, None, preparation, config_raw, None]
            with patch.object(operator, "_sftp_read", side_effect=values), \
                 patch.object(operator, "_sftp_read_metrics_tail", return_value=None):
                return operator._remote_summary(object(), timeout=1)

        self.assertEqual(summary_for(None)["config"], {})
        valid_summary = summary_for(json.dumps(valid).encode(), ready=True)
        self.assertIsNone(valid_summary["config"]["profile"])
        operator._validate_config_summary(valid_summary)
        wrong_profile = dict(valid_summary, preparation={
            "schema": "throughput-recovery-single-arm-preparation-v1",
            "profile": "wrong-profile",
        })
        with self.assertRaises(operator.OperationError):
            operator._validate_config_summary(wrong_profile)
        malformed_ready = summary_for(b"{not-json", ready=True)
        with self.assertRaises(operator.OperationError):
            operator._validate_config_summary(malformed_ready)
        malformed = summary_for(b"{not-json")
        self.assertEqual(malformed["config"], {"_malformed": True})
        self.assertFalse(malformed["phase_ready"])
        with self.assertRaises(operator.OperationError):
            operator._validate_config_summary(malformed)

    def test_metrics_observation_reads_bounded_tail_and_drops_partial_lines(self):
        records = b"".join(
            json.dumps({
                "index": index,
                "render_queue_seconds": float(index),
                "render_active_seconds": 0.25,
            }, separators=(",", ":")).encode() + b"\n"
            for index in range(12000)
        )
        payload = b"PARTIAL-OLD-RECORD" + (b"x" * 100) + b"\n" + records + b'{"index":12000'
        sftp = TailSFTP(payload)
        data = operator._sftp_read_metrics_tail(sftp, "metrics.jsonl", timeout=1)
        self.assertIsNotNone(data)
        self.assertLessEqual(len(data), operator.METRICS_TAIL_BYTES)
        self.assertEqual(sftp.stream.read_sizes, [1, operator.METRICS_TAIL_BYTES])
        rows = [json.loads(line) for line in data.splitlines()]
        self.assertTrue(rows)
        self.assertEqual(rows[-1]["index"], 11999)
        summary = operator._queue_summary(data)
        self.assertEqual(summary["queue_max_seconds"], 11999.0)
        self.assertEqual(summary["scope"], "recent_bounded_metrics_tail")
        self.assertEqual(summary["tail_bytes_observed"], len(data))
        self.assertEqual(summary["tail_bytes_limit"], operator.METRICS_TAIL_BYTES)

    def test_metrics_observation_timeout_is_bounded(self):
        sftp = BlockingTailSFTP(b'{"render_queue_seconds":1}\n')
        started = time.monotonic()
        with self.assertRaises(operator.OperationError):
            operator._sftp_read_metrics_tail(sftp, "metrics.jsonl", timeout=0.03)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(sftp.stream.started.wait(0.2))

    def test_controller_survives_transient_sftp_observation_timeout(self):
        config = {
            "checkpoint_interval": 1,
            "batch_size": 16,
            "group_size": 4,
            "initial_inflight": 32,
            "max_inflight": 32,
            "activation_cpu_offload": True,
            "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
        }
        contents = {
            "throughput-recovery-remote-job-start.json": b'{"status":"started"}',
            "throughput-recovery-remote-job-exit.json": b'{"exit_code":0}',
            "supervisor-state.json": b'{"status":"running","completed_optimizer_steps":0}',
            "terminal-receipt.json": b"{}",
            "preparation-receipt.json": b'{"schema":"throughput-recovery-single-arm-preparation-v1","profile":"rl-throughput-recovery-v1"}',
            "rl-config-receipt.json": json.dumps(config).encode(),
            "metrics.jsonl": b'{"render_queue_seconds":0.1,"render_active_seconds":0.2}\n',
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"git bundle fixture")
            token = root / "token"
            gateway = root / "gateway.env"
            token.write_text("opaque-token\n")
            gateway.write_text("opaque\n")
            token.chmod(0o600)
            gateway.chmod(0o600)
            api = SummaryAPI(contents)
            window = operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.0,
            )
            result = operator.operate(
                window=window,
                source_bundle=bundle,
                source_commit="a" * 40,
                hf_token_file=token,
                gateway_env_file=gateway,
                output_root=root / "ops",
                api=api,
                now=lambda: 1000.0,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(result, 0)
            self.assertEqual([pod.id for pod in api.removed], ["pod-test"])
            events = [json.loads(line) for line in (root / "ops/operations-single.jsonl").read_text().splitlines()]
            self.assertTrue(any(event["event"] == "observation_error" for event in events))
            self.assertTrue(any(event["event"] == "observation_recovered" for event in events))
            self.assertTrue(any(event["event"] == "remote_execute_finished" and event["exit_code"] == 0 for event in events))
            self.assertFalse(any(event["event"] == "operation_error" for event in events))

    def test_observation_timeout_closes_blocked_sftp_before_retry(self):
        config = {
            "checkpoint_interval": 1,
            "batch_size": 16,
            "group_size": 4,
            "initial_inflight": 32,
            "max_inflight": 32,
            "activation_cpu_offload": True,
            "deployment": {"type": "single_node", "gpus_per_node": 2, "num_train_gpus": 1, "num_infer_gpus": 1},
        }
        contents = {
            "throughput-recovery-remote-job-start.json": b'{"status":"started"}',
            "throughput-recovery-remote-job-exit.json": b'{"exit_code":0}',
            "supervisor-state.json": b'{"status":"running"}',
            "terminal-receipt.json": b"{}",
            "preparation-receipt.json": b'{"schema":"throughput-recovery-single-arm-preparation-v1","profile":"rl-throughput-recovery-v1"}',
            "rl-config-receipt.json": json.dumps(config).encode(),
            "metrics.jsonl": b"{}\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ssh = SummarySSH(contents, block_first_read=True)
            stale = ssh.open_sftp()
            window = operator.make_window(
                pod_id="pod-test",
                rental_start_epoch=1000.0,
                price_usd_per_hour=3.8,
                cap_usd=20.0,
            )
            fresh, summary = operator._observe_summary(
                ssh,
                stale,
                timeout=0.3,
                now=lambda: 1000.0,
                window=window,
                events=root / "events.jsonl",
                started=1000.0,
                phase="poll",
            )
            self.assertIs(fresh, ssh.healthy)
            self.assertEqual(summary["preparation"]["profile"], "rl-throughput-recovery-v1")
            self.assertTrue(ssh.stale.closed)

    def test_wrappers_keep_matched_panel_renderer_and_four_update_bindings(self):
        execute = (RUN / "execute_single.sh").read_text()
        evaluate = (RUN / "evaluate.sh").read_text()
        telemetry = (REPO / "painter/policy_runner.sh").read_text()
        self.assertIn("PAINTER_RL_MAX_STEPS=4", execute)
        self.assertIn("PAINTER_EVAL_PREFERRED_STEP=4", execute)
        self.assertIn("balanced16", evaluate)
        self.assertIn("PAINTER_EVAL_EXPECTED_COUNT=47", evaluate)
        self.assertIn("cancellable_watercolour_renderer.py", evaluate)
        self.assertIn("ADAPTER_ONLY_CHECKPOINTS=1", execute)
        self.assertIn('"average_utilization_percent"', telemetry)
        self.assertIn('"sampled_busy_fraction"', telemetry)
        self.assertIn('"role": "inference" if index < inference_gpu_count else "trainer"', telemetry)

    def test_pilot_and_matched_wrappers_delegate_canonical_policy_runner(self):
        pilot = (RUN / "run_single.sh").read_text()
        matched = (REPO / "painter/runs/rl-judge-matched-20261004/run_arm.sh").read_text()
        canonical = (REPO / "painter/policy_runner.sh").read_text()
        for wrapper in (pilot, matched):
            self.assertIn('exec bash "$REPO/painter/policy_runner.sh" "$@"', wrapper)
            self.assertLess(len(wrapper.splitlines()), 20)
        self.assertIn("PAINTER_RL_PREFLIGHT_ONLY", canonical)
        self.assertIn("nvidia-smi --query-gpu", canonical)
        self.assertIn('bash "$RUN/launch_rl.sh" "$RUN"', canonical)
        self.assertIn("PAINTER_POLICY_MODE", canonical)
        self.assertIn("recovered_policy", canonical)

    def test_remote_failure_diagnostics_are_scrubbed(self):
        diagnostics = operator._sanitized_diagnostics(
            "pod-test bootstrap failed API_GATEWAY_KEY=secret vck_live sk-or-v1-secret 10.1.2.3",
            forbidden=("pod-test",),
        )
        joined = " ".join(diagnostics)
        self.assertNotIn("pod-test", joined)
        self.assertNotIn("secret", joined)
        self.assertNotIn("vck_live", joined)
        self.assertNotIn("sk-or-v1-secret", joined)
        self.assertNotIn("10.1.2.3", joined)
        self.assertIn("bootstrap failed", joined)

    def test_remote_diagnostics_keep_stderr_tail_when_stdout_is_verbose(self):
        stdout = "\n".join(f"progress-{index}" for index in range(32))
        diagnostics = operator._sanitized_diagnostics(
            "fatal profile bootstrap error",
            stdout,
            labels=("stderr", "stdout"),
        )
        self.assertIn("stderr: fatal profile bootstrap error", diagnostics)
        self.assertTrue(any(line.startswith("stdout: progress-") for line in diagnostics))

    def test_profile_stage_replaces_existing_bootstrap_config_only_after_validation(self):
        from painter import throughput_recovery_profile as profile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]}) + "\n")
            source = profile_preparer.load_config_tool(REPO)
            native = root / "native.toml"
            receipt = root / "rl-config-receipt.json"
            native.write_text("bootstrap config\n")
            receipt.write_text(json.dumps({"stage": "base"}) + "\n")
            (root / "frozen-opponent-config.json").write_text(json.dumps({
                "schema": "photo-curriculum-rl-frozen-limits-v1",
                "operational_limits": {
                    "episode_timeout_seconds": 3600,
                    "render_timeout_seconds": 180,
                },
            }, indent=2) + "\n")
            frozen_before = (root / "frozen-opponent-config.json").read_bytes()
            kwargs = {
                "model": root / "model",
                "manifest": manifest,
                "prime_root": None,
                "run_name": "throughput-recovery-20261003",
                "output_root": "rl-rollouts",
                "expected_count": 40,
                "max_steps": 4,
                "learning_rate": 1e-6,
                "renderer": str(root / "renderer.py"),
                "renderer_python": str(root / "renderer.py"),
                "gpus_per_node": 2,
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
            }
            with self.assertRaises(FileExistsError):
                profile.prepare_profile(
                    source.prepare,
                    output=native,
                    run_root=root,
                    **kwargs,
                )
            self.assertEqual(native.read_text(), "bootstrap config\n")
            profile_preparer.prepare_staged_profile(profile, source, run_root=root, **kwargs)
            self.assertIn("batch_size = 16", native.read_text())
            self.assertEqual(json.loads(receipt.read_text())["batch_size"], 16)
            self.assertEqual((root / "frozen-opponent-config.json").read_bytes(), frozen_before)
            guarded = root / "guarded"
            guarded.mkdir()
            guarded_native = guarded / "native.toml"
            guarded_receipt = guarded / "rl-config-receipt.json"
            guarded_native.write_text("bootstrap config\n")
            guarded_receipt.write_text(json.dumps({"stage": "base"}) + "\n")
            (guarded / "async-output").mkdir()
            with self.assertRaisesRegex(RuntimeError, "pretraining-only"):
                profile_preparer.prepare_staged_profile(profile, source, run_root=guarded, **kwargs)
            self.assertEqual(guarded_native.read_text(), "bootstrap config\n")
            self.assertEqual(json.loads(guarded_receipt.read_text())["stage"], "base")

    def test_profile_main_replays_existing_bootstrap_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prime = root / "prime"
            (prime / "src/prime_rl/configs").mkdir(parents=True)
            (prime / "src/prime_rl/__init__.py").write_text("")
            (prime / "src/prime_rl/configs/__init__.py").write_text("")
            (prime / "src/prime_rl/configs/rl.py").write_text(
                "class RLConfig:\n"
                "    @classmethod\n"
                "    def model_validate(cls, config):\n"
                "        return config\n"
            )
            (root / "runtime/watercolour-cancellable-centroid-v1").mkdir(parents=True)
            (root / "native.toml").write_text("bootstrap config\n")
            (root / "rl-config-receipt.json").write_text(json.dumps({"stage": "base"}) + "\n")
            (root / "frozen-opponent-config.json").write_text(json.dumps({
                "schema": "photo-curriculum-rl-frozen-limits-v1",
                "operational_limits": {
                    "episode_timeout_seconds": 3600,
                    "render_timeout_seconds": 180,
                },
            }, indent=2) + "\n")
            (root / "preparation-receipt.json").write_text(
                json.dumps({"schema": "text-paint-consistent-preparation-v2"}) + "\n"
            )
            manifest = root / "train-manifest.json"
            manifest.write_text(json.dumps({"rows": [
                {"id": f"brief-{index}", "reference_kind": "text", "opponent_sha256": "a" * 64}
                for index in range(40)
            ]}) + "\n")
            (root / "runtime/watercolour-cancellable-centroid-v1/PATCH-MANIFEST.json").write_text(
                json.dumps({
                    "runtime_id": "test-runtime",
                    "selected_renderer": {"sha256": "b" * 64},
                    "frozen_baseline": {"sha256": "c" * 64},
                    "centroid_patch": {"after_sha256": "d" * 64},
                }) + "\n"
            )
            env = {
                **os.environ,
                "PAINTER_RUN_NAME": "throughput-recovery-20261003",
                "PAINTER_EXPERIMENT_NAME": "throughput-recovery-flash-20261003",
                "PAINTER_JUDGE_MODEL": "xiaomi/mimo-v2.6-flash",
                "PAINTER_JUDGE_EFFORT": "medium",
                "PAINTER_JUDGE_FORMAT": "identity-joint-side-schema-v2-generic-v1",
                "PAINTER_JUDGE_PROTOCOL": "text-paint-sft-pairwise-consistent-v1",
                "PAINTER_BASE_MODEL_REVISION": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "PAINTER_POLICY_REVISION": "ac4d3eda8e5925e2d23992cfdc1a63620ebde573",
                "PAINTER_POLICY_PREFIX": "text-pairwise-scaled-20260929/step_16",
                "PAINTER_POLICY_SHA256": "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c",
                "PAINTER_FROZEN_SFT_MODEL_REVISION": "adf687e5b321514eda7be6eab050434fdc8e44ff",
                "PAINTER_FROZEN_SFT_ADAPTER_SHA256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
                "PAINTER_RL_CHECKPOINT_INTERVAL": "1",
                "ADAPTER_ONLY_CHECKPOINTS": "1",
                "PAINTER_ACTIVATION_CPU_OFFLOAD": "1",
                "PAINTER_GPUS_PER_NODE": "2",
                "PAINTER_NUM_TRAIN_GPUS": "1",
                "PAINTER_NUM_INFER_GPUS": "1",
                "PYTHONPATH": f"{REPO}:{REPO / 'painter'}",
            }
            command = [
                sys.executable,
                str(RUN / "prepare_profile_single.py"),
                "--repo", str(REPO),
                "--run-root", str(root),
                "--model", str(root / "model"),
                "--prime-root", str(prime),
                "--manifest", str(manifest),
                "--renderer", str(root / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"),
                "--renderer-python", str(root / "runtime/watercolour-cancellable-centroid-v1/cancellable_watercolour_renderer.py"),
            ]
            result = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / "rl-config-receipt.json").read_text())["batch_size"], 16)
            self.assertEqual(
                json.loads((root / "preparation-receipt.json").read_text())["schema"],
                "throughput-recovery-single-arm-preparation-v1",
            )
            self.assertTrue((root / "source-identity.json").is_file())
            self.assertTrue((root / "renderer-receipt.json").is_file())
            config_receipt = (root / "rl-config-receipt.json").read_bytes()
            self.assertNotIn("profile", json.loads(config_receipt))
            with patch.object(operator, "_sftp_read", side_effect=[
                b"{}", b"{}", None, None, (root / "preparation-receipt.json").read_bytes(), config_receipt
            ]), patch.object(operator, "_sftp_read_metrics_tail", return_value=None):
                summary = operator._remote_summary(object(), timeout=1)
            self.assertTrue(summary["phase_ready"])
            operator._validate_config_summary(summary)


if __name__ == "__main__":
    unittest.main()
