from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).with_name("evaluate.sh")


class EvaluateScriptTests(unittest.TestCase):
    def test_default_concurrency_is_explicit_and_environment_is_validated(self):
        text = SCRIPT.read_text()
        self.assertIn('MAX_CONCURRENT="${PAINTER_EVAL_MAX_CONCURRENT:-16}"', text)
        self.assertIn('[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]', text)
        self.assertIn('--max-concurrent "$MAX_CONCURRENT"', text)
        self.assertIn("max_concurrent = {max_concurrent}", Path(__file__).parents[2].joinpath("native_painting_eval.py").read_text())

    def test_seed_and_concurrency_are_checked_before_endpoint_rewrite(self):
        text = SCRIPT.read_text()
        self.assertLess(text.index("unexpected evaluation seed"), text.index("path.write_text"))
        self.assertLess(text.index("unexpected evaluation concurrency"), text.index("path.write_text"))
        self.assertIn("seed = 918200", text)
        self.assertIn("'max_concurrent': max_concurrent", text)

    def test_generation_receipt_requires_run_local_png_backed_artifacts(self):
        text = SCRIPT.read_text()
        self.assertIn('"$OUT/eval/evaluation-rollouts"', text)
        self.assertIn("from native_painting_eval import audit_evaluation_artifacts", text)
        self.assertIn("evaluation artifacts escaped the archived run root", text)
        self.assertIn("valid_receipts_with_png", text)
        self.assertLess(text.index("audit_evaluation_artifacts"), text.index("generation-receipt.json"))
        self.assertIn("evaluation-artifact-receipt.json", text)

    def test_renderer_python_import_is_checked_before_gpu_server_start(self):
        text = SCRIPT.read_text()
        self.assertIn('"$RENDERER_PYTHON" - <<\'PY\'', text)
        self.assertIn("import playwright", text)
        self.assertLess(text.index("import playwright"), text.index('"$VLLM" serve'))

    def test_embedded_python_blocks_compile(self):
        blocks = []
        active = None
        for line_number, line in enumerate(SCRIPT.read_text().splitlines(), 1):
            if active is None:
                if re.search(r"<<['\"]PY['\"]", line):
                    active = (line_number, [])
            elif line == "PY":
                blocks.append(active)
                active = None
            else:
                active[1].append(line)
        self.assertIsNone(active, "unterminated Python heredoc")
        self.assertGreaterEqual(len(blocks), 4)
        for line_number, body in blocks:
            compile("\n".join(body) + "\n", f"evaluate.sh:heredoc:{line_number}", "exec")


if __name__ == "__main__":
    unittest.main()
