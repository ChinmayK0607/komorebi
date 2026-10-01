"""Guards for the isolated p5.brush centroid derivative.

These tests inspect only the vendored source and a pure JavaScript function.
They do not invoke a renderer, model, network API, or painting workload.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:  # Works both from the repository root and from ``painter/``.
    from painter.patch_brush_centroid import (
        COPIED_FILES,
        EXPECTED_SHA256,
        NEW_FALLBACK,
        OLD_FALLBACK,
        PatchError,
        SOURCE_ROOT,
        _patch_brush,
        build_runtime,
        extract_centroid_function,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by local test layout
    from patch_brush_centroid import (  # type: ignore[no-redef]
        COPIED_FILES,
        EXPECTED_SHA256,
        NEW_FALLBACK,
        OLD_FALLBACK,
        PatchError,
        SOURCE_ROOT,
        _patch_brush,
        build_runtime,
        extract_centroid_function,
    )


class BrushCentroidPatchTests(unittest.TestCase):
    def test_frozen_source_hash_and_exact_guarded_edit(self):
        source = (SOURCE_ROOT / "assets/p5.brush.js").read_bytes()
        self.assertEqual(
            __import__("hashlib").sha256(source).hexdigest(),
            EXPECTED_SHA256["assets/p5.brush.js"],
        )
        function = extract_centroid_function(source.decode("utf-8"))
        self.assertEqual(function.count(OLD_FALLBACK), 1)
        self.assertEqual(function.count(NEW_FALLBACK), 0)

        patched = _patch_brush(source).decode("utf-8")
        patched_function = extract_centroid_function(patched)
        self.assertEqual(patched_function.count(OLD_FALLBACK), 0)
        self.assertEqual(patched_function.count(NEW_FALLBACK), 1)
        self.assertEqual(patched.count(OLD_FALLBACK), 0)
        self.assertEqual(patched.count(NEW_FALLBACK), 1)

    def test_build_copies_renderer_assets_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "watercolour-p5-brush-2.2.1-centroid-fix-v1"
            metadata = build_runtime(SOURCE_ROOT, output)
            self.assertEqual(metadata["runtime_id"], output.name)
            self.assertTrue((output / "renderer.py").is_file())
            self.assertTrue((output / "PATCH-MANIFEST.json").is_file())
            for relative in COPIED_FILES:
                self.assertTrue((output / relative).is_file(), relative)
            manifest = json.loads((output / "PATCH-MANIFEST.json").read_text())
            self.assertEqual(manifest["provenance"]["upstream"]["commit"], "35f7ccc096829ba2d1d7ac8d9feeed7fd882fa82")
            self.assertEqual(manifest["licenses"]["p5.brush"]["spdx"], "MIT")
            self.assertEqual(manifest["patch"]["replacement_count"], 1)
            self.assertEqual(
                manifest["patch"]["before_sha256"],
                EXPECTED_SHA256["assets/p5.brush.js"],
            )
            self.assertNotEqual(
                manifest["patch"]["before_sha256"],
                manifest["patch"]["after_sha256"],
            )
            self.assertNotIn(
                OLD_FALLBACK,
                (output / "assets/p5.brush.js").read_text(),
            )

    def test_changed_or_already_patched_source_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            shutil.copytree(SOURCE_ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
            (root / "renderer.py").write_text(
                (root / "renderer.py").read_text() + "\n# unexpected source change\n"
            )
            with self.assertRaisesRegex(PatchError, "Unexpected pinned source"):
                build_runtime(root, Path(folder) / "output")

            shutil.copytree(SOURCE_ROOT, root / "already", ignore=shutil.ignore_patterns("__pycache__"))
            already = root / "already" / "assets/p5.brush.js"
            already.write_text(already.read_text().replace(OLD_FALLBACK, NEW_FALLBACK))
            with self.assertRaisesRegex(PatchError, "already-patched"):
                build_runtime(root / "already", Path(folder) / "already-output")

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            output.mkdir()
            with self.assertRaisesRegex(PatchError, "overwrite"):
                build_runtime(SOURCE_ROOT, output)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the pure-function regression")
    def test_node_regression_covers_empty_short_zero_area_and_polygon(self):
        fixture = Path(__file__).with_name("brush_centroid_regression.cjs")
        result = subprocess.run(
            [shutil.which("node") or "node", str(fixture)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("p5.brush centroid regression passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
