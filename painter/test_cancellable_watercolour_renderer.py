from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from painter import cancellable_watercolour_renderer as cancellable
from painter import native_painting_renderer as native_renderer


@unittest.skipUnless(os.name == "posix", "process-group lifecycle tests require POSIX")
class CancellableRendererProcessTests(unittest.TestCase):
    def _tree_code(self) -> str:
        return (
            "import os, pathlib, subprocess, sys, time\n"
            "marker=pathlib.Path(sys.argv[1])\n"
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "marker.write_text(str(child.pid))\n"
            "if len(sys.argv) > 2 and sys.argv[2] == 'exit': os._exit(0)\n"
            "time.sleep(30)\n"
        )

    def _wait_for_pid(self, marker: Path) -> int:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if marker.is_file():
                return int(marker.read_text())
            time.sleep(0.01)
        self.fail("stub process did not publish child pid")

    def _wait_for_file(self, marker: Path) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if marker.is_file():
                return
            time.sleep(0.01)
        self.fail(f"stub process did not publish marker: {marker}")

    def _pid_live(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            try:
                state = proc_stat.read_text().split(") ", 1)[1].split()[0]
                return state != "Z"
            except (FileNotFoundError, IndexError):
                return False
        return True

    def _wait_dead(self, pid: int) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not self._pid_live(pid):
                return
            time.sleep(0.02)
        self.fail(f"owned child {pid} survived cleanup")

    def test_leader_exit_with_live_child_is_cleaned_by_recorded_session(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "child.pid"
            leader = subprocess.Popen(
                [sys.executable, "-c", self._tree_code(), str(marker), "exit"],
                start_new_session=True,
            )
            identity = cancellable._record_owned_process_group(leader)
            self.assertIsNotNone(identity)
            child_pid = self._wait_for_pid(marker)
            self.assertEqual(leader.wait(timeout=3), 0)
            self.assertTrue(self._pid_live(child_pid))
            cancellable._terminate_owned_process(leader, identity, grace=0.4)
            self._wait_dead(child_pid)

    def test_repeated_term_cleanup_is_idempotent_for_owned_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "child.pid"
            leader = subprocess.Popen(
                [sys.executable, "-c", self._tree_code(), str(marker)],
                start_new_session=True,
            )
            identity = cancellable._record_owned_process_group(leader)
            child_pid = self._wait_for_pid(marker)
            cancellable._terminate_owned_process(leader, identity, grace=0.4)
            cancellable._terminate_owned_process(leader, identity, grace=0.4)
            self._wait_dead(child_pid)

    def test_nested_fresh_sessions_forward_repeated_term_to_owned_worker_tree(self):
        """Exercise the relay used when the worker starts its own session.

        The outer process models the renderer controller.  Its worker starts a
        separate session and then starts a grandchild in that worker group;
        the controller's guarded SIGTERM handler forwards cleanup by the
        recorded worker group.  An unrelated fresh-session process proves the
        forwarding does not broaden to neighboring process groups.
        """

        worker_code = (
            "import pathlib, subprocess, sys, time\n"
            "marker=pathlib.Path(sys.argv[1])\n"
            "grandchild=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "marker.write_text(str(grandchild.pid))\n"
            "time.sleep(30)\n"
        )
        outer_code = (
            "import os, pathlib, signal, subprocess, sys, time\n"
            f"worker_code={worker_code!r}\n"
            "worker_marker=pathlib.Path(sys.argv[2])\n"
            "worker_pid_marker=pathlib.Path(sys.argv[3])\n"
            "cleanup_marker=pathlib.Path(sys.argv[4])\n"
            "worker=subprocess.Popen([sys.executable, '-c', worker_code, str(worker_marker)], start_new_session=True)\n"
            "worker_pid_marker.write_text(str(worker.pid))\n"
            "cleanup_started=False\n"
            "def on_term(signum, frame):\n"
            "    global cleanup_started\n"
            "    if cleanup_started:\n"
            "        return\n"
            "    cleanup_started=True\n"
            "    cleanup_marker.write_text('started')\n"
            "    try:\n"
            "        os.killpg(worker.pid, signal.SIGTERM)\n"
            "    except ProcessLookupError:\n"
            "        pass\n"
            "    time.sleep(0.4)\n"
            "    raise SystemExit(128 + signum)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            worker_marker = Path(temp) / "grandchild.pid"
            worker_pid_marker = Path(temp) / "worker.pid"
            cleanup_marker = Path(temp) / "cleanup.started"
            unrelated = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                start_new_session=True,
            )
            outer = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    outer_code,
                    str(Path(temp) / "unused"),
                    str(worker_marker),
                    str(worker_pid_marker),
                    str(cleanup_marker),
                ],
                start_new_session=True,
            )
            try:
                grandchild_pid = self._wait_for_pid(worker_marker)
                worker_pid = self._wait_for_pid(worker_pid_marker)
                outer.send_signal(signal.SIGTERM)
                self._wait_for_file(cleanup_marker)
                try:
                    outer.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.assertIsNotNone(outer.wait(timeout=3))
                self._wait_dead(worker_pid)
                self._wait_dead(grandchild_pid)
                self.assertTrue(self._pid_live(unrelated.pid))
            finally:
                if outer.poll() is None:
                    outer.kill()
                outer.wait()
                if unrelated.poll() is None:
                    unrelated.terminate()
                unrelated.wait()

    def test_cancellation_event_terminates_real_nested_stub_process(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "child.pid"
            cancel_event = threading.Event()
            outcome: list[BaseException] = []

            def run_stub():
                try:
                    native_renderer._run_renderer_process(
                        [sys.executable, "-c", self._tree_code(), str(marker)],
                        env={"PATH": os.environ.get("PATH", "")},
                        timeout=10,
                        cancel_event=cancel_event,
                    )
                except BaseException as exc:
                    outcome.append(exc)

            thread = threading.Thread(target=run_stub)
            thread.start()
            child_pid = self._wait_for_pid(marker)
            cancel_event.set()
            thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertTrue(any(isinstance(exc, native_renderer.RenderCancelled) for exc in outcome))
            self._wait_dead(child_pid)

    def test_timeout_terminates_real_nested_stub_process(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "child.pid"
            with self.assertRaises(subprocess.TimeoutExpired):
                native_renderer._run_renderer_process(
                    [sys.executable, "-c", self._tree_code(), str(marker)],
                    env={"PATH": os.environ.get("PATH", "")},
                    timeout=0.2,
                    cancel_event=threading.Event(),
                )
            child_pid = self._wait_for_pid(marker)
            self._wait_dead(child_pid)


if __name__ == "__main__":
    unittest.main()
