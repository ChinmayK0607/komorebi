from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_cpu_replay import (
    RECOVERY_INPUT_SCHEMA,
    ReplayError,
    build_command,
    load_replay_inputs,
    run_replay,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class CpuReplayRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.renderer = self.root / "renderer.py"
        self.renderer.write_text("# pinned renderer fixture\n")
        assets = self.renderer.parent / "assets"
        assets.mkdir()
        (assets / "p5.min.js").write_text("p5 fixture\n")
        (assets / "p5.brush.js").write_text("brush fixture\n")
        self.renderer_python = self.root / "renderer-python"
        self.renderer_python.write_text("fixture executable")
        self.renderer_python.chmod(0o755)

        self.sources: dict[str, tuple[Path, bytes]] = {}
        for case_id, source in (("case-a", b"function setup(){}\n"), ("case-b", b"function setup(){background(1);}\n"), ("case-c", b"function setup(){background(2);}\n")):
            path = self.inputs / f"{case_id}.js"
            path.write_bytes(source)
            self.sources[case_id] = (path, source)
        renderer_assets = {name: digest((self.renderer.parent / "assets" / name).read_bytes()) for name in ("p5.brush.js", "p5.min.js")}
        self.manifest_path = self.inputs / "manifest.json"
        cases = []
        for case_id, (source_path, source) in self.sources.items():
            target = {"case-a": b"replay-a", "case-b": b"recorded-old", "case-c": b"never-rendered"}[case_id]
            cases.append({
                "id": case_id,
                "source_file": source_path.name,
                "source_sha256": digest(source),
                "target_png_sha256": digest(target),
                "seed": 0,
            })
        self.manifest_path.write_text(json.dumps({
            "schema": "text-sft512-cpu-replay-inputs-v1",
            "source_trace_sha256": "0" * 64,
            "protocol_version": "native-painting-eval-v3",
            "seed": 0,
            "renderer": {"renderer_sha256": digest(self.renderer.read_bytes()), "assets_sha256": renderer_assets},
            "cases": cases,
        }))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_linux_runner_validates_hashes_and_labels_replay_results(self) -> None:
        output = self.root / "results"

        def fake_invoke(command: list[str], _browser: Path | None):
            source = Path(command[2] if command[0] == str(self.renderer_python) else command[5])
            case_id = source.parent.name
            target = output / "cases" / case_id / "painting.png"
            receipt = target.with_suffix(".json")
            if case_id == "case-c":
                receipt.write_text(json.dumps({"valid": False, "source": "/private/node/source.js", "error_code": "render_timeout"}))
                return 1, None, 0.5
            pixels = b"replay-a" if case_id == "case-a" else b"new-b"
            target.write_bytes(pixels)
            receipt.write_text(json.dumps({
                "valid": True,
                "source": "/private/node/source.js",
                "source_sha256": digest(self.sources[case_id][1]),
                "png_sha256": digest(pixels),
                "seed": 0,
            }))
            return 0, None, 0.5

        report = run_replay(
            manifest_path=self.manifest_path,
            renderer=self.renderer,
            renderer_python=self.renderer_python,
            output_dir=output,
            workers=2,
            run_as_user=None,
            platform_name="linux",
            invoke=fake_invoke,
        )
        self.assertEqual(report["summary"]["cases"], 3)
        self.assertEqual(report["summary"]["verified"], 2)
        self.assertEqual(report["summary"]["recovered_identical"], 1)
        self.assertEqual(report["summary"]["new_replay"], 1)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertIn("render_timeout", (output / "cases/case-c/painting.json").read_text())
        self.assertNotIn("/private/node", (output / "cases/case-c/painting.json").read_text())
        self.assertNotIn(str(self.root), (output / "manifest.json").read_text())
        self.assertEqual(
            report["cases"][0]["program_file"],
            "cases/case-a/program.js",
        )

    def test_first_turn_recovery_replay_does_not_invent_target_comparisons(self) -> None:
        raw = json.loads(self.manifest_path.read_text())
        raw["schema"] = RECOVERY_INPUT_SCHEMA
        raw["replay_mode"] = "first_turn_recovery"
        for case in raw["cases"]:
            case.pop("target_png_sha256")
            case["original_receipt_sha256"] = "f" * 64
        self.manifest_path.write_text(json.dumps(raw))
        output = self.root / "recovery-results"

        def fake_invoke(command: list[str], _browser: Path | None):
            source = Path(command[2])
            case_dir = source.parent
            pixels = b"first-turn-recovery"
            (case_dir / "painting.png").write_bytes(pixels)
            (case_dir / "painting.json").write_text(json.dumps({
                "valid": True,
                "source_sha256": digest(source.read_bytes()),
                "png_sha256": digest(pixels),
                "seed": 0,
            }))
            return 0, None, 0.1

        report = run_replay(
            manifest_path=self.manifest_path,
            renderer=self.renderer,
            renderer_python=self.renderer_python,
            output_dir=output,
            workers=2,
            run_as_user=None,
            platform_name="linux",
            invoke=fake_invoke,
        )
        self.assertEqual(report["replay_mode"], "first_turn_recovery")
        self.assertEqual(report["summary"]["replayed"], 3)
        self.assertEqual(report["summary"]["not_applicable_comparison"], 3)
        self.assertEqual(report["summary"]["verified"], 0)
        self.assertTrue(all(row["comparison"] == "not-applicable" for row in report["cases"]))
        self.assertTrue(all(row["target_png_sha256"] is None for row in report["cases"]))
        self.assertIn("not a three-turn evaluation", (output / "report.md").read_text())

    def test_runner_refuses_non_linux_hosts_before_rendering(self) -> None:
        with self.assertRaisesRegex(ReplayError, "only on Linux"):
            run_replay(
                manifest_path=self.manifest_path,
                renderer=self.renderer,
                renderer_python=self.renderer_python,
                output_dir=self.root / "results",
                platform_name="darwin",
            )

    def test_runner_bounds_concurrency(self) -> None:
        with self.assertRaisesRegex(ReplayError, "between 2 and 4"):
            run_replay(
                manifest_path=self.manifest_path,
                renderer=self.renderer,
                renderer_python=self.renderer_python,
                output_dir=self.root / "results",
                workers=1,
                platform_name="linux",
            )

    def test_input_hash_mismatch_is_rejected_before_render(self) -> None:
        raw = json.loads(self.manifest_path.read_text())
        raw["cases"][0]["source_sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ReplayError, "source hash mismatch"):
            load_replay_inputs(self.manifest_path, self.renderer)

    def test_venv_python_symlink_is_preserved_and_seed_deadline_are_pinned(self) -> None:
        venv_python = self.root / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(self.renderer_python)
        command = build_command(
            source=self.root / "program.js",
            output=self.root / "painting.png",
            renderer=self.renderer,
            renderer_python=venv_python,
            run_as_user=None,
        )
        self.assertEqual(command[0], str(venv_python))
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertEqual(command[command.index("--timeout") + 1], "180")

        seen: list[list[str]] = []

        def fake_invoke(command: list[str], _browser: Path | None):
            seen.append(command)
            source = Path(command[2])
            png = source.parent / "painting.png"
            pixels = b"venv-replay"
            png.write_bytes(pixels)
            (source.parent / "painting.json").write_text(json.dumps({
                "valid": True,
                "png_sha256": digest(pixels),
                "source_sha256": digest(source.read_bytes()),
                "seed": 0,
                "chromium_version": "fixture",
            }))
            return 0, None, 0.1

        output = self.root / "venv-results"
        run_replay(
            manifest_path=self.manifest_path,
            renderer=self.renderer,
            renderer_python=venv_python,
            output_dir=output,
            workers=2,
            run_as_user=None,
            platform_name="linux",
            invoke=fake_invoke,
        )
        self.assertTrue(seen)
        self.assertTrue(all(command[0] == str(venv_python) for command in seen))
        self.assertIn("chromium_version", (output / "cases/case-a/painting.json").read_text())


if __name__ == "__main__":
    unittest.main()
