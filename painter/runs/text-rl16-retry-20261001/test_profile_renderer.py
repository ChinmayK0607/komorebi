"""Pure/watchdog tests; never launch Playwright or render locally."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parent
profile = load_module("profile_renderer", ROOT / "profile_renderer.py")
worker = load_module("profile_worker", ROOT / "profile_worker.py")


class ProfileRendererTests(unittest.TestCase):
    def test_worker_preserves_automatic_html_order_and_has_no_manual_p5_start(self):
        html = worker.build_html("P5_ASSET", "BRUSH_ASSET", "function setup(){}", 0)
        self.assertLess(html.index("P5_ASSET"), html.index("BRUSH_ASSET"))
        self.assertLess(html.index(worker.MARKER), html.index("function setup(){}"))
        self.assertLess(html.index("function setup(){}"), html.index(worker.BOOTSTRAP))
        self.assertNotIn("new p5()", html)

    def test_static_metrics_report_literal_area_and_unresolved_helper_geometry(self):
        source = """
        function draw() {
          brush.fill('#abc', 100);
          brush.polygon([[0,0],[10,0],[10,20],[0,20]]);
          function poly(points) { brush.polygon(points); }
          poly([[0,0],[1,0],[1,1]]);
          brush.line(0, 0, 10, 10);
        }
        """
        result = profile.static_brush_metrics(source)
        self.assertEqual(result["fill_calls"], 1)
        self.assertEqual(result["stroke_calls"], 1)
        self.assertEqual(result["polygon_calls"], 2)
        self.assertEqual(result["literal_polygon_count"], 1)
        self.assertEqual(result["literal_point_count"], 4)
        self.assertEqual(result["literal_polygon_area_px2"], 200.0)
        self.assertEqual(result["unresolved_polygon_count"], 1)

    def test_manifest_runtime_and_program_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_root, runtime = root / "inputs", root / "runtime"
            (manifest_root / "programs").mkdir(parents=True)
            (runtime / "assets").mkdir(parents=True)
            (runtime / "renderer.py").write_text("renderer fixture\n", encoding="utf-8")
            (runtime / "assets" / "p5.min.js").write_text("p5 fixture\n", encoding="utf-8")
            (runtime / "assets" / "p5.brush.js").write_text("brush fixture\n", encoding="utf-8")
            source = manifest_root / "programs" / "case-a.js"
            source.write_text("function setup(){}\n", encoding="utf-8")
            manifest = {
                "schema": "text-paint-first-turn-recovery-inputs-v1",
                "seed": 0,
                "renderer": {
                    "renderer_sha256": profile.sha256(runtime / "renderer.py"),
                    "assets_sha256": {name: profile.sha256(runtime / "assets" / name) for name in ("p5.min.js", "p5.brush.js")},
                },
                "cases": [{"id": "case-a", "seed": 17, "source_sha256": profile.sha256(source)}],
            }
            manifest_path = manifest_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            records = profile.load_profile_inputs(manifest_path, runtime, ["case-a"])
            self.assertEqual(records[0]["seed"], 17)
            source.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(profile.ProfileError):
                profile.load_profile_inputs(manifest_path, runtime, ["case-a"])

    def test_external_watchdog_kills_child_and_keeps_partial_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, receipt, log = root / "case.journal", root / "case.json", root / "case.log"
            code = "import json,sys,time; p=sys.argv[1]; f=open(p,'w'); f.write(json.dumps({'event':'brush_start','id':7,'name':'polygon','start_ms':12.5})+'\\n'); f.flush(); f.close(); time.sleep(2)"
            result = profile.run_watchdog([sys.executable, "-c", code, str(journal)], receipt, journal, log, 0.2, {"PATH": "/usr/bin:/bin"})
            self.assertTrue(result["watchdog_timed_out"])
            self.assertEqual(result["error_code"], "profile_timeout")
            self.assertEqual(result["journal"]["active_brush_call_at_last_event"]["id"], 7)
            self.assertTrue(receipt.is_file())

    def test_worker_exit_without_receipt_gets_error_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, receipt, log = root / "case.journal", root / "case.json", root / "case.log"
            result = profile.run_watchdog([sys.executable, "-c", "raise SystemExit(3)"], receipt, journal, log, 2, {"PATH": "/usr/bin:/bin"})
            self.assertFalse(result["watchdog_timed_out"])
            self.assertEqual(result["error_code"], "profile_worker_failure")
            self.assertEqual(result["returncode"], 3)

    def test_control_command_invokes_frozen_worker_directly(self):
        record = {"source": Path("/tmp/case.js"), "seed": 0}
        command = profile.control_command(record, Path("/tmp/runtime"), Path("/tmp/python"), Path("/tmp/out.png"), 180)
        self.assertIn("--worker", command)
        self.assertIn("/tmp/runtime/renderer.py", command)
        self.assertNotIn("--output-dir", command)

    def test_resource_snapshot_records_cpu_affinity_cgroup_and_python(self):
        snapshot = profile.resource_snapshot()
        self.assertIn("cpu_count", snapshot)
        self.assertIn("sched_getaffinity", snapshot)
        self.assertIn("cgroup", snapshot)
        self.assertIn("python_version", snapshot)
        self.assertIn("platform", snapshot)

    def test_run_controls_returns_report_level_renderer_and_asset_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            (runtime / "assets").mkdir(parents=True)
            (runtime / "renderer.py").write_text("renderer fixture\n", encoding="utf-8")
            (runtime / "assets" / "p5.min.js").write_text("p5 fixture\n", encoding="utf-8")
            (runtime / "assets" / "p5.brush.js").write_text("brush fixture\n", encoding="utf-8")
            record = {"id": "case-a", "source": root / "case-a.js", "source_sha256": "source", "seed": 17}
            record["source"].write_text("function setup(){}\n", encoding="utf-8")
            with patch.object(profile, "run_watchdog", return_value={"valid": True}):
                report = profile.run_controls([record], runtime, Path(sys.executable), root / "control", 1, None)
            self.assertEqual(report["renderer_sha256"], profile.sha256(runtime / "renderer.py"))
            self.assertEqual(report["assets_sha256"], {name: profile.sha256(runtime / "assets" / name) for name in ("p5.min.js", "p5.brush.js")})
            self.assertEqual(report["results"][0]["renderer_sha256"], report["renderer_sha256"])
            self.assertEqual(report["results"][0]["assets_sha256"], report["assets_sha256"])

    def test_control_comparison_requires_matching_png_hashes(self):
        identity = {"source_sha256": "source", "seed": 17, "renderer_sha256": "r", "assets_sha256": {"p5": "p5"}}
        profile_report = {**identity, "case_ids": ["a"], "results": [{"id": "a", **identity, "valid": True, "png_sha256": "same", "watchdog_elapsed_seconds": 3}]}
        control_report = {**identity, "case_ids": ["a"], "results": [{"id": "a", **identity, "valid": True, "png_sha256": "same", "watchdog_elapsed_seconds": 2}]}
        result = profile.compare_control(profile_report, control_report)
        self.assertTrue(result["same_renderer_identity"])
        self.assertTrue(result["same_assets_identity"])
        self.assertTrue(result["cases"][0]["identity_equal"])
        self.assertTrue(result["cases"][0]["png_sha_equal"])

    def test_control_comparison_reports_unequal_and_missing_identities_strictly(self):
        profile_report = {
            "renderer_sha256": "profile-renderer",
            "assets_sha256": {"p5": "profile-p5"},
            "case_ids": ["registered", "baseline-only"],
            "results": [{
                "id": "registered",
                "source_sha256": "source-a",
                "seed": 17,
                "renderer_sha256": "profile-row-renderer",
                "assets_sha256": {"p5": "profile-row-p5"},
                "valid": True,
                "png_sha256": "profile-png",
            }],
        }
        control_report = {
            "renderer_sha256": "control-renderer",
            "assets_sha256": {"p5": "control-p5"},
            "case_ids": ["registered", "control-only"],
            "results": [{
                "id": "registered",
                "source_sha256": "source-b",
                "seed": 18,
                "renderer_sha256": "control-row-renderer",
                "assets_sha256": {"p5": "control-row-p5"},
                "valid": True,
                "png_sha256": "control-png",
            }],
        }
        result = profile.compare_control(profile_report, control_report)
        self.assertFalse(result["same_renderer_identity"])
        self.assertFalse(result["same_assets_identity"])
        self.assertEqual([row["id"] for row in result["cases"]], ["registered"])
        self.assertEqual(result["missing_profile_cases"], ["control-only"])
        self.assertEqual(result["profile_cases_without_registered_controls"], ["baseline-only"])
        self.assertEqual(result["missing_control_results"], ["control-only"])
        row = result["cases"][0]
        self.assertFalse(row["source_sha_equal"])
        self.assertFalse(row["seed_equal"])
        self.assertFalse(row["renderer_sha_equal"])
        self.assertFalse(row["assets_sha_equal"])
        self.assertFalse(row["identity_equal"])

    def test_control_comparison_does_not_treat_missing_identities_as_equal(self):
        profile_report = {"case_ids": ["a"], "results": [{"id": "a", "valid": True}]}
        control_report = {"case_ids": ["a"], "results": [{"id": "a", "valid": True}]}
        result = profile.compare_control(profile_report, control_report)
        self.assertFalse(result["same_renderer_identity"])
        self.assertFalse(result["same_assets_identity"])
        self.assertFalse(result["cases"][0]["identity_equal"])
        self.assertFalse(result["cases"][0]["png_sha_equal"])


if __name__ == "__main__":
    unittest.main()
