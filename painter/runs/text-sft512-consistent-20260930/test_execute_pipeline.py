from pathlib import Path
import re
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("execute_pipeline.sh")


class ExecutePipelineTests(unittest.TestCase):
    def test_refresh_uses_new_prime_executable_in_shell(self):
        text = SCRIPT.read_text()
        start = text.index("refresh_prime_python() {")
        end = text.index("\n}\n", start) + 3
        function = text[start:end]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = f"""#!/usr/bin/env bash
set -Eeuo pipefail
{function}
PRIME_ROOT={root / 'missing'}
PY=python3
if refresh_prime_python; then exit 11; fi
PRIME_ROOT={root / 'prime'}
mkdir -p "$PRIME_ROOT/.venv/bin"
printf '#!/bin/sh\\nexit 0\\n' > "$PRIME_ROOT/.venv/bin/python"
chmod 755 "$PRIME_ROOT/.venv/bin/python"
refresh_prime_python
[[ "$PY" == "$PRIME_ROOT/.venv/bin/python" ]]
"""
            completed = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_every_embedded_python_heredoc_compiles(self):
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
        self.assertIsNone(active, "unterminated Python heredoc")
        self.assertGreater(len(blocks), 5)
        for line_number, body in blocks:
            compile("\n".join(body) + "\n", f"execute_pipeline.sh:heredoc:{line_number}", "exec")

    def test_post_training_mode_keeps_prepare_in_normal_branch(self):
        text = SCRIPT.read_text()
        post = text.index('if [[ "$post_training_only" == 1 ]]')
        prepare = text.index('bash "$RUN/prepare_node.sh"', post)
        self.assertGreater(prepare, text.index("else", post))


if __name__ == "__main__":
    unittest.main()
