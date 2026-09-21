import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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
                output.write_bytes(b"png")
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


if __name__ == "__main__":
    unittest.main()
