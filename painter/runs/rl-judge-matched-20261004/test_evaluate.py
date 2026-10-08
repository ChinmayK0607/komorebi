"""Bounded shell contracts for the matched evaluation wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


RUN = Path(__file__).resolve().parent


def _materialized_wrapper(root: Path) -> tuple[Path, Path]:
    """Create the flat source/arm layout used by the campaign materializer."""
    source = root / "repo/painter/runs/rl-judge-matched-20261004"
    arm = root / "repo/painter/runs/rl-judge-matched-20261004-flash"
    base = root / "repo/painter/runs/text-sft512-consistent-20260930"
    source.mkdir(parents=True)
    arm.mkdir(parents=True)
    base.mkdir(parents=True)

    for name in ("arm_env.sh", "runtime_env.sh", "evaluate_arm.sh"):
        path = source / name
        shutil.copy2(RUN / name, path)
        path.chmod(0o755)
    wrapper = arm / "evaluate.sh"
    shutil.copy2(RUN / "evaluate.sh", wrapper)
    wrapper.chmod(0o755)

    evaluator = base / "evaluate.sh"
    evaluator.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "printf '%s\\n' \"$*\" \"${PAINTER_EVAL_SCOPE:-}\" "
        "\"${PAINTER_EVAL_EXPECTED_COUNT:-}\" \"${PAINTER_EVAL_SEED:-}\" "
        "\"${PAINTER_EVAL_MANIFEST:-}\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    evaluator.chmod(0o755)
    return wrapper, arm


class MatchedEvaluateTests(unittest.TestCase):
    def test_legacy_all_reaches_fake_base_as_frozen_balanced16_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wrapper, arm = _materialized_wrapper(root)
            capture = root / "capture.txt"
            environment = {
                **os.environ,
                "BASH_ENV": "/dev/null",
                "CAPTURE": str(capture),
                "PAINTER_BASE_RUN": str(root / "repo/painter/runs/text-sft512-consistent-20260930"),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(root / "repo/painter/runs"),
                "PAINTER_REPO_ROOT": str(root / "repo"),
            }
            result = subprocess.run(
                [str(wrapper), "12", "all", "--preflight"],
                cwd=arm,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(
                capture.read_text(encoding="utf-8").splitlines(),
                [
                    "12 balanced16 --preflight",
                    "balanced16",
                    "16",
                    "918210",
                    str(arm / "anchors-frozen/validation-matched16.json"),
                ],
            )

            for arguments in (("12", "all"), ("12", "balanced16"), ("12",)):
                result = subprocess.run(
                    [str(wrapper), *arguments],
                    cwd=arm,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                rows = capture.read_text(encoding="utf-8").splitlines()
                self.assertEqual(rows[0].split(), ["12", "balanced16"])
                self.assertEqual(rows[1:], [
                    "balanced16",
                    "16",
                    "918210",
                    str(arm / "anchors-frozen/validation-matched16.json"),
                ])

    def test_wrapper_rejects_non_frozen_scope_and_non_primary_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wrapper, arm = _materialized_wrapper(root)
            capture = root / "capture.txt"
            environment = {
                **os.environ,
                "BASH_ENV": "/dev/null",
                "CAPTURE": str(capture),
                "PAINTER_BASE_RUN": str(root / "repo/painter/runs/text-sft512-consistent-20260930"),
                "PAINTER_MATCHED_ARM_ROOT_BASE": str(root / "repo/painter/runs"),
                "PAINTER_REPO_ROOT": str(root / "repo"),
            }

            for arguments in (("12", "expanded"), ("11", "all"), ("12", "balanced16", "--score")):
                result = subprocess.run(
                    [str(wrapper), *arguments],
                    cwd=arm,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
                self.assertFalse(capture.exists(), arguments)


if __name__ == "__main__":
    unittest.main()
