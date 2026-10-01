from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).with_name("finalize.sh")


class FinalizeScriptTests(unittest.TestCase):
    def test_checkpoint_receipt_python_receives_run_for_path_binding(self):
        text = SCRIPT.read_text()
        self.assertIn('"$PY" - "$RUN" "$ADAPTER" "$STEP" <<\'PY\'', text)
        self.assertIn("run, adapter, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])", text)
        self.assertIn("f'{run.name}/step_{step}'", text)

    def test_scoring_receipt_keeps_selected_model_and_format(self):
        text = SCRIPT.read_text()
        self.assertIn('JUDGE_FORMAT="${PAINTER_JUDGE_FORMAT:-identity-joint-side-schema-v1}"', text)
        self.assertIn('JUDGE_MODEL="${PAINTER_JUDGE_MODEL:-xiaomi/mimo-v2.6-flash}"', text)
        self.assertIn('PAINTER_JUDGE_FORMAT="$JUDGE_FORMAT"', text)
        self.assertIn('PAINTER_JUDGE_MODEL="$JUDGE_MODEL"', text)
        self.assertIn("'judge_format': judge_format", text)
        self.assertIn("'judge_model': judge_model", text)

    def test_embedded_python_blocks_compile(self):
        blocks = []
        active = None
        for line_number, line in enumerate(SCRIPT.read_text().splitlines(), 1):
            if active is None:
                if re.search(r"<<['\"]PY['\"]\s*$", line):
                    active = (line_number, [])
            elif line == "PY":
                blocks.append(active)
                active = None
            else:
                active[1].append(line)
        self.assertIsNone(active)
        for line_number, body in blocks:
            compile("\n".join(body) + "\n", f"finalize.sh:heredoc:{line_number}", "exec")


if __name__ == "__main__":
    unittest.main()
