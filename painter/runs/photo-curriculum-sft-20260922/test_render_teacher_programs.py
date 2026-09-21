import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("render_teacher_programs.py")
SPEC = importlib.util.spec_from_file_location("photo_curriculum_render_teachers", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RenderTeacherProgramsTests(unittest.TestCase):
    def test_preserves_venv_python_symlink_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "teacher"
            folder.mkdir()
            (folder / "program.js").write_text("function setup() {}\n")
            renderer = root / "renderer.py"
            renderer.write_text("# renderer\n")
            real_python = root / "venv-python-real"
            real_python.write_text("# python\n")
            symlink_python = root / "venv" / "bin" / "python"
            symlink_python.parent.mkdir(parents=True)
            symlink_python.symlink_to(real_python)

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                source = Path(command[6])
                output.write_bytes(b"png")
                output.with_suffix(".json").write_text(json.dumps({
                    "valid": True,
                    "source_sha256": MODULE.sha(source),
                    "png_sha256": MODULE.sha(output),
                }))
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(MODULE.subprocess, "run", side_effect=fake_run) as run:
                result = MODULE.render(folder, renderer, symlink_python)
            self.assertTrue(result["valid"])
            self.assertEqual(run.call_args.args[0][4], str(symlink_python.absolute()))


if __name__ == "__main__":
    unittest.main()
