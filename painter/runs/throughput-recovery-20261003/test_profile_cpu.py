"""Offline contracts for the bounded CPU renderer profile.

These tests exercise metadata, command construction, output accounting, and
the owned POSIX cleanup path.  They never launch Playwright, Chrome, a model,
or a network/API call.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("profile_cpu", ROOT / "profile_cpu.py")
assert SPEC and SPEC.loader
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class ProfileCpuTests(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        source = root / "control-01.js"
        source.write_text("function setup() { createCanvas(600, 600); }\n", encoding="utf-8")
        manifest = {
            "schema": "cpu-renderer-controls-v1",
            "seed": 0,
            "rows": [{
                "id": "control-01",
                "program_path": source.name,
                "program_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "kind": "valid",
                "saved_elapsed_seconds": 6.2,
                "saved_valid": True,
                "source_archive_revision": "public-revision",
                "source_archive_sha256": "a" * 64,
                "source_archive_member": "programs/control-01.js",
            }],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, source

    def _runtime(self, root: Path) -> tuple[Path, Path]:
        baseline = root / "baseline"
        derivative = root / "derivative"
        for runtime in (baseline, derivative):
            (runtime / "assets").mkdir(parents=True)
            (runtime / "assets" / "p5.min.js").write_text("p5", encoding="utf-8")
            (runtime / "assets" / "p5.brush.js").write_text("brush", encoding="utf-8")
        (baseline / "renderer.py").write_text("baseline", encoding="utf-8")
        (derivative / "cancellable_watercolour_renderer.py").write_text("derivative", encoding="utf-8")
        return baseline, derivative

    def test_load_controls_hashes_sources_and_preserves_archive_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, source = self._bundle(root)
            manifest, controls, loaded = profile.load_controls(root)
            self.assertEqual(manifest["schema"], profile.INPUT_SCHEMA)
            self.assertEqual(loaded, manifest_path.resolve())
            self.assertEqual(controls[0].source, source.resolve())
            self.assertEqual(controls[0].source_archive_member, "programs/control-01.js")
            source.write_text("tampered", encoding="utf-8")
            with self.assertRaises(profile.ProfileError):
                profile.load_controls(root)

    def test_loader_rejects_escape_and_unknown_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._bundle(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["rows"][0]["program_path"] = "../outside.js"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(profile.ProfileError, "escapes"):
                profile.load_controls(root)

    def test_runtime_identity_and_backend_command_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, derivative = self._runtime(root)
            runtime = profile.load_runtime(baseline, derivative)
            _, source = self._bundle(root / "inputs")
            control = profile.Control("control-01", source.resolve(), "control-01.js", profile.sha256(source), 0, "valid", True, 1.0, None, None, None)
            command = profile.renderer_command(Path("/venv/python"), runtime.derivative, control, root / "out.png", 180)
            self.assertEqual(command[command.index("--backend") + 1], "swiftshader")
            self.assertNotEqual(runtime.baseline_sha256, runtime.derivative_sha256)

    def test_backend_check_rejects_silent_fallback(self):
        receipt = {
            "requested_angle_backend": "gl",
            "browser_args": ["--use-gl=angle", "--use-angle=gl"],
            "gpu": {"unmasked_renderer": "SwiftShader Device"},
        }
        checked = profile._backend_check(receipt, "swiftshader")
        self.assertTrue(checked["checked_once"])
        self.assertFalse(checked["available"])
        self.assertIn("expected 'swiftshader'", " ".join(checked["errors"]))

    def test_resource_snapshot_has_affinity_and_cgroup_window(self):
        snapshot = profile.resource_snapshot()
        self.assertIn("cpu_count", snapshot)
        self.assertIn("sched_getaffinity", snapshot)
        self.assertIn("cgroup", snapshot)
        self.assertIn("measurement_denominator", snapshot)

    def test_run_renderer_mock_does_not_claim_valid_without_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, source = self._bundle(root / "inputs")
            renderer = root / "renderer.py"
            renderer.write_text("renderer", encoding="utf-8")
            control = profile.Control("control-01", source.resolve(), "control-01.js", profile.sha256(source), 0, "valid", True, 1.0, None, None, None)

            class FakeProcess:
                pid = 999999
                returncode = 1

                def wait(self, timeout=None):
                    return self.returncode

                def poll(self):
                    return self.returncode

                def send_signal(self, _signal):
                    return None

            def fake_popen(command, **kwargs):
                self.assertIn("--backend", command)
                self.assertEqual(command[command.index("--backend") + 1], "swiftshader")
                return FakeProcess()

            with patch.object(profile.subprocess, "Popen", fake_popen):
                result = profile.run_renderer(Path(sys.executable), renderer, control, root / "attempt", 1, None, renderer_kind="baseline")
            self.assertFalse(result["valid"])
            self.assertTrue(result["censored"])
            self.assertEqual(result["censor_reason"], "missing_renderer_receipt")

    @unittest.skipUnless(sys.platform == "linux" and Path("/proc").is_dir(), "Linux /proc lifecycle contract")
    def test_owned_posix_group_cleanup_reaps_dummy_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "child.pid"
            code = "import pathlib, subprocess, sys, time; child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
            leader = subprocess.Popen([sys.executable, "-c", code, str(marker)], start_new_session=True)
            try:
                deadline = time.monotonic() + 3
                while not marker.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.is_file())
                observed = profile._descendants(leader.pid)
                sessions = profile._owned_sessions(leader, observed)
                cleaned = profile._terminate_owned_sessions(leader, sessions, grace=0.3)
                self.assertTrue(cleaned["owned_processes_gone"])
                self.assertEqual(cleaned["surviving_owned_sessions"], [])
            finally:
                if leader.poll() is None:
                    leader.kill()
                leader.wait()

    def test_html_uses_relative_image_links_and_escapes_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "attempts" / "a" / "image.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            report = {"schema": "profile<test>", "baseline_sequential": {"results": [{"id": "x<1>", "renderer_kind": "baseline", "valid": True, "artifact_present": True, "image_relpath": "attempts/a/image.png", "png_sha256": "abc"}]}}
            output = root / "profile.html"
            profile.build_review_html(report, output)
            text = output.read_text()
            self.assertIn("attempts/a/image.png", text)
            self.assertIn("&lt;test&gt;", text)
            self.assertIn("&lt;1&gt;", text)
            self.assertNotIn(str(root), text)


if __name__ == "__main__":
    unittest.main()
