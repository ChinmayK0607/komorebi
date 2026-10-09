from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


RUN = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("diverse_renderer_runtime_wrapper_test", RUN / "prepare_renderer_runtime.py")
assert SPEC is not None and SPEC.loader is not None
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class RendererRuntimeWrapperTests(unittest.TestCase):
    def test_main_forwards_repeatable_old_renderer_roots(self) -> None:
        root = Path("/node")
        old_one = root / "renderer-old"
        old_two = root / "renderer-older"
        with patch.object(WRAPPER.shared, "prepare", return_value={}) as prepare:
            self.assertEqual(
                WRAPPER.main([
                    "--run-root", str(root / "run"),
                    "--renderer-root", str(root / "renderer"),
                    "--renderer-python", str(root / "renderer-env/bin/python"),
                    "--browsers", str(root / "browsers"),
                    "--prime-venv", str(root / "prime/.venv"),
                    "--allowed-old-renderer-root", str(old_one),
                    "--allowed-old-renderer-root", str(old_two),
                    "--no-smoke",
                ]),
                0,
            )
        prepare.assert_called_once()
        self.assertEqual(
            prepare.call_args.kwargs["allowed_old_renderer_roots"],
            [old_one, old_two],
        )


if __name__ == "__main__":
    unittest.main()
