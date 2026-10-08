"""Offline tests for scratch Prime checkout seeding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


RUN = Path(__file__).resolve().parent
SCRIPT = RUN / "seed_prime_checkout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("structured_flow_prime_seed", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Prime seed helper is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


class PrimeSeedTests(unittest.TestCase):
    def _source(self, root: Path) -> tuple[Path, str]:
        source = root / "staged-prime"
        source.mkdir()
        git(source, "init", "-q")
        git(source, "config", "user.email", "seed-test@example.invalid")
        git(source, "config", "user.name", "seed-test")
        (source / ".gitignore").write_text(".venv/\n", encoding="utf-8")
        (source / "pyproject.toml").write_text("[project]\nname='prime-fixture'\n", encoding="utf-8")
        (source / ".venv").mkdir()
        (source / ".venv/bin").mkdir()
        (source / ".venv/bin/python").write_text("stale env\n", encoding="utf-8")
        git(source, "add", ".gitignore", "pyproject.toml")
        git(source, "commit", "-q", "-m", "fixture")
        return source, git(source, "rev-parse", "HEAD")

    def test_local_seed_copies_tracked_source_without_virtualenv(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            source, commit = self._source(Path(directory))
            result = module.seed(source=source, destination=Path(directory) / "scratch/prime", expected_commit=commit)
            destination = Path(directory) / "scratch/prime"
            self.assertEqual(result["status"], "seeded_verified")
            self.assertEqual(git(destination, "rev-parse", "HEAD"), commit)
            self.assertEqual(git(destination, "remote", "get-url", "origin"), module.UPSTREAM_URL)
            self.assertTrue((destination / "pyproject.toml").is_file())
            self.assertFalse((destination / ".venv").exists())
            # Reuse permits bootstrap-generated ignored state, but never a
            # tracked source edit.
            (destination / ".venv/cache").mkdir(parents=True)
            (destination / ".venv/cache/marker").write_text("generated\n", encoding="utf-8")
            replay = module.seed(source=source, destination=destination, expected_commit=commit)
            self.assertEqual(replay["status"], "existing_verified")
            self.assertEqual(git(destination, "remote", "get-url", "origin"), module.UPSTREAM_URL)
            (destination / "pyproject.toml").write_text("[project]\nname='edited'\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tracked modifications"):
                module.seed(source=source, destination=destination, expected_commit=commit)

    def test_seed_rejects_dirty_source_and_wrong_existing_destination(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, commit = self._source(root)
            (source / "untracked.py").write_text("must reject\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "uncommitted or untracked"):
                module.seed(source=source, destination=root / "scratch/prime", expected_commit=commit)
            (source / "untracked.py").unlink()
            clean = root / "clean"
            clean.mkdir()
            git(clean, "init", "-q")
            git(clean, "config", "user.email", "seed-test@example.invalid")
            git(clean, "config", "user.name", "seed-test")
            (clean / "other").write_text("other\n", encoding="utf-8")
            git(clean, "add", "other")
            git(clean, "commit", "-q", "-m", "other")
            with self.assertRaisesRegex(ValueError, "not pinned"):
                module.seed(source=source, destination=clean, expected_commit=commit)

    def test_shell_bootstrap_has_opt_in_seed_and_fetch_only_when_commit_missing(self):
        setup = (RUN.parent / "text-pairwise-rl-20260928/setup_node.sh").read_text()
        helper = SCRIPT.read_text()
        self.assertIn("PAINTER_PRIME_SEED_ENABLED", setup)
        self.assertIn("--no-hardlinks", helper)
        self.assertIn('git -C "$PRIME_ROOT" cat-file -e "$PRIME_COMMIT^{commit}"', setup)
        self.assertIn('git -C "$PRIME_ROOT" -c http.version=HTTP/1.1 fetch -q origin "$PRIME_COMMIT"', setup)


if __name__ == "__main__":
    unittest.main()
