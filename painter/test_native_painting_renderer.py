import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


from painter import native_painting_renderer as renderer_module


class NativePaintingRendererTests(unittest.TestCase):
    def test_configured_deadline_is_forwarded_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episode"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                source = Path(command[6])
                output = Path(command[8])
                image = Image.new("RGB", (600, 600), "white")
                image.paste("#315b70", (100, 100, 500, 500))
                image.save(output)
                receipt = {
                    "valid": True,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
                output.with_suffix(".json").write_text(json.dumps(receipt))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.dict(os.environ, {"PAINTER_RENDER_TIMEOUT_SECONDS": "177"}, clear=False):
                with patch.object(renderer_module.subprocess, "run", side_effect=fake_run) as run:
                    result = renderer_module.render_action(
                        root=root, folder=folder,
                        reply="```javascript\nfunction setup() {}\n```",
                        initial_canvas=None, initial_program=None,
                        renderer=renderer, renderer_python=renderer_python,
                    )
            self.assertTrue(result["valid"])
            self.assertEqual(result["render_timeout_seconds"], 177)
            self.assertEqual(result["render_process_timeout_seconds"], 197)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--timeout") + 1], "177")
            self.assertEqual(run.call_args.kwargs["timeout"], 197)

    def test_deadline_rejects_out_of_range_values(self):
        for value in ("0", "181", "not-an-int"):
            with self.subTest(value=value), patch.dict(os.environ, {"PAINTER_RENDER_TIMEOUT_SECONDS": value}, clear=False):
                with self.assertRaises(ValueError):
                    renderer_module.render_timeout_seconds()

    def test_blank_canvas_is_model_failure_even_when_renderer_reports_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                source = Path(command[6])
                output = Path(command[8])
                Image.new("RGB", (600, 600), "black").save(output)
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": True,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(renderer_module.subprocess, "run", side_effect=fake_run):
                result = renderer_module.render_action(
                    root=root, folder=root / "episode",
                    reply="```javascript\nfunction setup() {}\n```",
                    initial_canvas=None, initial_program=None,
                    renderer=renderer, renderer_python=renderer_python,
                )
            self.assertFalse(result["valid"])
            self.assertEqual(result["failure_kind"], "blank_canvas")
            self.assertIsNone(result["canvas"])

    def test_runtime_receipt_failure_is_service_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "episode"
            renderer = root / "renderer.py"
            renderer_python = root / "python"
            renderer.write_text("# test renderer\n")
            renderer_python.write_text("# test python\n")

            def fake_run(command, **kwargs):
                output = Path(command[8])
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": False,
                    "error_code": "renderer_error",
                    "error": "ModuleNotFoundError: No module named 'playwright'",
                }))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(renderer_module.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "renderer service failure.*playwright"):
                    renderer_module.render_action(
                        root=root, folder=folder,
                        reply="```javascript\nfunction setup() {}\n```",
                        initial_canvas=None, initial_program=None,
                        renderer=renderer, renderer_python=renderer_python,
                    )


if __name__ == "__main__":
    unittest.main()
