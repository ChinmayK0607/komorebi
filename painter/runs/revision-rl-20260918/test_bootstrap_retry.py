"""Offline coverage for the bounded frozen dependency-sync retry loop."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("setup_async_node.sh")


class BootstrapRetryTest(unittest.TestCase):
    def run_bootstrap(self, behavior):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        try:
            (root / "prime-rl").mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            (fake_bin / "uv").write_text(
                """#!/bin/sh
count_file="$UV_TEST_ROOT/uv-count"
count=$(cat "$count_file" 2>/dev/null || printf 0)
count=$((count + 1))
printf '%s\n' "$count" > "$count_file"
printf '%s\n' "$*" >> "$UV_TEST_ROOT/uv-args"
if [ "$UV_TEST_BEHAVIOR" = transient ] && [ "$count" -lt 2 ]; then
  printf 'transient wheel failure %s\n' "$count" >&2
  exit 75
fi
if [ "$UV_TEST_BEHAVIOR" = exhausted ]; then
  printf 'permanent wheel failure %s\n' "$count" >&2
  exit 42
fi
printf 'sync completed on attempt %s\n' "$count"
"""
            )
            (fake_bin / "sleep").write_text(
                """#!/bin/sh
printf '%s\n' "$1" >> "$UV_TEST_ROOT/sleeps"
"""
            )
            (fake_bin / "flock").write_text("#!/bin/sh\nexit 0\n")
            for command in (fake_bin / "uv", fake_bin / "sleep", fake_bin / "flock"):
                command.chmod(0o755)
            env = os.environ.copy()
            env.update(
                BOOTSTRAP_RETRY_TEST_ONLY="1",
                UV_TEST_BEHAVIOR=behavior,
                UV_TEST_ROOT=str(root),
                PATH=f"{fake_bin}:{env['PATH']}",
            )
            result = subprocess.run(
                ["bash", str(SCRIPT), str(root)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            return temp, root, result
        except Exception:
            temp.cleanup()
            raise

    def test_transient_failure_retries_with_backoff_and_succeeds(self):
        temp, root, result = self.run_bootstrap("transient")
        self.addCleanup(temp.cleanup)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((root / "uv-count").read_text().strip(), "2")
        self.assertEqual((root / "sleeps").read_text().splitlines(), ["10"])
        self.assertEqual(
            (root / "uv-args").read_text().splitlines(),
            ["sync --frozen --extra offline-test", "sync --frozen --extra offline-test"],
        )
        log_dir = next((root / "logs").glob("prime-install-*-*"))
        self.assertIn(
            "transient wheel failure 1",
            (log_dir / "attempt-1.log").read_text(),
        )
        self.assertIn(
            "sync completed on attempt 2",
            (log_dir / "attempt-2.log").read_text(),
        )
        self.assertEqual(
            (root / "logs/prime-install.log").read_text(),
            (log_dir / "attempt-2.log").read_text(),
        )

    def test_exhaustion_is_bounded_and_returns_final_error(self):
        temp, root, result = self.run_bootstrap("exhausted")
        self.addCleanup(temp.cleanup)

        self.assertEqual(result.returncode, 42)
        self.assertEqual((root / "uv-count").read_text().strip(), "3")
        self.assertEqual((root / "sleeps").read_text().splitlines(), ["10", "20"])
        log_dir = next((root / "logs").glob("prime-install-*-*"))
        for attempt in range(1, 4):
            log = log_dir / f"attempt-{attempt}.log"
            self.assertTrue(log.exists())
            self.assertIn(f"permanent wheel failure {attempt}", log.read_text())
        final_log = log_dir / "attempt-3.log"
        self.assertEqual((root / "logs/prime-install.log").read_text(), final_log.read_text())
        self.assertIn("failed after 3 attempts", result.stderr)


if __name__ == "__main__":
    unittest.main()
