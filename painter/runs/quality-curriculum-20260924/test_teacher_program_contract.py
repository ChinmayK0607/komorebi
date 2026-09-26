"""Regression check for a real Astra teacher render failure."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contract import validate_teacher_program


class TeacherProgramContractTest(unittest.TestCase):
    def test_p5_global_helper_collision_is_rejected(self):
        source = "function smooth(p) { return p; }\nfunction draw() { translate(-300,-300); noLoop(); }"
        with self.assertRaisesRegex(ValueError, "redefines p5 global: smooth"):
            validate_teacher_program(source)

    def test_renamed_helper_is_valid(self):
        source = "function smoothPoly(p) { return p; }\nfunction draw() { translate(-300,-300); noLoop(); }"
        validate_teacher_program(source)


if __name__ == "__main__":
    unittest.main()
