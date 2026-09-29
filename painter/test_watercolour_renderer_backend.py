import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = (Path(__file__).parent / "runs/text-pairwise-rl-20260928"
          / "renderer_hardware_probe.py")
SPEC = importlib.util.spec_from_file_location("painter_watercolour_renderer", SOURCE)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


class RendererBackendTests(unittest.TestCase):
    def test_hardware_backends_are_opt_in_and_linux_only(self):
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(RENDERER.resolve_backend("gl"), "gl")
            self.assertEqual(RENDERER.resolve_backend("vulkan"), "vulkan")
            self.assertIn("--use-angle=gl", RENDERER.browser_args("gl"))
            self.assertIn("--use-angle=vulkan", RENDERER.browser_args("vulkan"))
        with patch.object(sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "requires Linux"):
                RENDERER.resolve_backend("gl")
        self.assertEqual(RENDERER.resolve_backend("swiftshader"), "swiftshader")


if __name__ == "__main__":
    unittest.main()
