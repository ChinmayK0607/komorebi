"""Pure packaging allowlist checks; never renders or contacts a node."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


def _module():
    spec = importlib.util.spec_from_file_location("aesthetic_package_run", ROOT / "package_run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackageRunTest(unittest.TestCase):
    def test_current_overlay_excludes_live_generated_directories_and_keeps_anchors(self):
        module = _module()
        files = {}
        module._add_current_overlay(files)
        names = set(files)

        self.assertFalse(
            any(
                name.startswith(prefix)
                for prefix in ("reward-batches/", "collected-initial/", "attempts/", "logs/")
                for name in names
            )
        )
        self.assertTrue(
            all(
                name in module.CURRENT_OVERLAY_FILES or name.startswith(module.ANCHOR_POOL + "/")
                for name in names
            )
        )
        for name in (
            "taskset_impl.py",
            "train_from_sft.py",
            "activate_async.sh",
            "launch_async.sh",
            "prepare_async.py",
        ):
            self.assertIn(name, names)
            self.assertEqual(files[name], ROOT / name)

        rows = json.loads((ROOT / "aesthetic-states.json").read_text())["training_states"]
        expected_anchors = sum(len(row["anchors"]) for row in rows)
        anchor_names = [name for name in names if name.startswith(module.ANCHOR_POOL + "/")]
        self.assertEqual(len(anchor_names), expected_anchors * 2)
        self.assertTrue(all(Path(name).suffix in module.ANCHOR_SUFFIXES for name in anchor_names))

    def test_full_bundle_current_overrides_replace_historical_wrappers(self):
        module = _module()
        files = {}
        module._add_old_wrappers(files)
        old = {name: files.get(name) for name in ("activate_async.sh", "launch_async.sh", "prepare_async.py", "train_from_sft.py")}
        module._add_current_overlay(files)

        for name in ("activate_async.sh", "launch_async.sh", "prepare_async.py", "train_from_sft.py"):
            self.assertIsNotNone(old[name])
            self.assertEqual(files[name], ROOT / name)
        self.assertEqual(files["taskset_impl.py"], ROOT / "taskset_impl.py")


if __name__ == "__main__":
    unittest.main()
