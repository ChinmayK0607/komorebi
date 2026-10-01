#!/usr/bin/env python3
"""Build the versioned p5.brush centroid-fix runtime.

The checked-in renderer under ``vendor/integrations/watercolour`` is a frozen
baseline.  This helper creates a separate runtime directory containing the
same renderer and assets, with one explicitly guarded upstream-source fix:
the zero-area fallback in ``function xn(t)`` uses ``t[0]`` instead of the
undefined ``v[0]``.  It refuses to patch a changed or already-patched source.

The generated runtime is intentionally versioned and ignored by the repository
(``painter/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1`` by default).
Its ``PATCH-MANIFEST.json`` records the before/after hashes, upstream
provenance, and license files needed to reproduce or audit the derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "painter/vendor/integrations/watercolour"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "painter/runtime/watercolour-p5-brush-2.2.1-centroid-fix-v1"
)

# These are the complete frozen inputs copied by this helper.  Keeping the
# list explicit makes an unexpected upstream file or silently added asset a
# hard failure instead of an unreviewed part of a future runtime.
COPIED_FILES = (
    Path("renderer.py"),
    Path("assets/ATTRIBUTION.md"),
    Path("assets/UPSTREAM-LICENSE"),
    Path("assets/VERSIONS"),
    Path("assets/p5-LICENSE"),
    Path("assets/p5.brush-LICENSE"),
    Path("assets/p5.brush.js"),
    Path("assets/p5.min.js"),
)
# Human-readable instructions live beside the pinned runtime but are not part
# of the renderer payload or its copied hash manifest.
ALLOWED_SOURCE_DOCUMENTS = (Path("PATCHING.md"),)

EXPECTED_SHA256 = {
    "renderer.py": "6181fb904a0c5dcd580f5c3cb01eaf341a14b7a15479e811d99ebb10b70bc841",
    "assets/ATTRIBUTION.md": "5394e87d53545a331765cb9d159e43b59e587b4d0559c89dd5a747f4c2a9af80",
    "assets/UPSTREAM-LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "assets/VERSIONS": "8493f4be8b2714625da05fa870a53b5af2c1963958d2a9f4070be7a8cbd14643",
    "assets/p5-LICENSE": "eb2f25f8f165d5f20682c1ce7e99ce69144ecfd80fbed8c17dfd327ecd8b6cb6",
    "assets/p5.brush-LICENSE": "cabbece1a276329fa28ab40f1c99af3af2b7852a329551a9a485b2199616583a",
    "assets/p5.brush.js": "9c6e271a78e5dcb23c9cfa9d085775317fe5fb2188f976465c83c1f7965d1ec8",
    "assets/p5.min.js": "87adc350e8ec0e9bced22d4f03c181bdd208dc997d3956ab3ec2e90537643c9a",
}

CENTROID_FUNCTION_PREFIX = "function xn(t){"
OLD_FALLBACK = "{x:v[0].x,y:v[0].y}"
NEW_FALLBACK = "{x:t[0].x,y:t[0].y}"
UPSTREAM = {
    "repository": "huggingface/huggingenvs",
    "commit": "35f7ccc096829ba2d1d7ac8d9feeed7fd882fa82",
    "path": "02-watercolour/envs/watercolour/core/vendor",
}
LICENSES = {
    "p5.js": {"version": "2.3.2", "spdx": "LGPL-2.1", "file": "assets/p5-LICENSE"},
    "p5.brush": {"version": "2.2.1", "spdx": "MIT", "file": "assets/p5.brush-LICENSE"},
    "renderer-adaptation": {
        "spdx": "BSD-3-Clause",
        "file": "assets/UPSTREAM-LICENSE",
    },
}


class PatchError(ValueError):
    """Raised when the pinned derivative preconditions do not hold."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_files(root: Path) -> tuple[Path, ...]:
    files = tuple(
        sorted(
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    )
    return files


def extract_centroid_function(source: str) -> str:
    """Return ``function xn(t)`` using a small balanced-brace scanner.

    The function is minified, so line-oriented extraction is intentionally not
    used.  The scanner handles quoted strings and comments to remain a useful
    static test if the vendored formatting changes while the pinned guard is
    still deliberately exact.
    """

    start = source.find(CENTROID_FUNCTION_PREFIX)
    if start < 0:
        raise PatchError("Expected function xn(t) was not found")
    opening = source.find("{", start)
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = opening
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == "/" and next_char == "/":
            line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            block_comment = True
            index += 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1
    raise PatchError("function xn(t) has unbalanced braces")


def _validate_source(source_root: Path) -> dict[str, bytes]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise PatchError(f"Source runtime directory does not exist: {source_root}")
    actual = _relative_files(source_root)
    expected = tuple(sorted((*COPIED_FILES, *ALLOWED_SOURCE_DOCUMENTS)))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(str(path) for path in missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(str(path) for path in unexpected))
        raise PatchError("Unexpected source file set: " + "; ".join(detail))

    values = {str(path): (source_root / path).read_bytes() for path in COPIED_FILES}
    brush = values["assets/p5.brush.js"].decode("utf-8")
    function = extract_centroid_function(brush)
    if NEW_FALLBACK in function:
        raise PatchError(
            "Source already contains the centroid replacement; refusing an already-patched source"
        )
    for relative, value in values.items():
        actual_sha = sha256_bytes(value)
        expected_sha = EXPECTED_SHA256[relative]
        if actual_sha != expected_sha:
            raise PatchError(
                f"Unexpected pinned source for {relative}: expected {expected_sha}, got {actual_sha}"
            )
    if function.count(OLD_FALLBACK) != 1:
        raise PatchError(
            "Expected exactly one v[0] centroid fallback in function xn(t); "
            f"found {function.count(OLD_FALLBACK)}"
        )
    if brush.count(OLD_FALLBACK) != 1:
        raise PatchError(
            "Expected exactly one v[0] centroid fallback in p5.brush.js; "
            f"found {brush.count(OLD_FALLBACK)}"
        )
    return values


def _patch_brush(value: bytes) -> bytes:
    source = value.decode("utf-8")
    function = extract_centroid_function(source)
    if function.count(OLD_FALLBACK) != 1 or function.count(NEW_FALLBACK):
        raise PatchError("Centroid replacement guard failed inside function xn(t)")
    updated_function = function.replace(OLD_FALLBACK, NEW_FALLBACK)
    if updated_function.count(OLD_FALLBACK) or updated_function.count(NEW_FALLBACK) != 1:
        raise PatchError("Centroid replacement did not produce exactly one t[0] fallback")
    start = source.index(function)
    updated = source[:start] + updated_function + source[start + len(function) :]
    if updated.count(OLD_FALLBACK) or updated.count(NEW_FALLBACK) != 1:
        raise PatchError(
            "Centroid replacement changed an unexpected number of source occurrences"
        )
    return updated.encode("utf-8")


def _runtime_metadata(
    source_values: dict[str, bytes], patched_values: dict[str, bytes], output_root: Path
) -> dict[str, Any]:
    files = {
        relative: {
            "before_sha256": sha256_bytes(source_values[relative]),
            "after_sha256": sha256_bytes(patched_values[relative]),
        }
        for relative in sorted(source_values)
    }
    return {
        "schema": "watercolour-centroid-fix-v1",
        "runtime_id": output_root.name,
        "source_root": "painter/vendor/integrations/watercolour",
        "source_files": files,
        "patch": {
            "file": "assets/p5.brush.js",
            "function": "xn",
            "guarded_replacement": "v[0] -> t[0]",
            "replacement_count": 1,
            "before_sha256": files["assets/p5.brush.js"]["before_sha256"],
            "after_sha256": files["assets/p5.brush.js"]["after_sha256"],
        },
        "provenance": {
            "upstream": UPSTREAM,
            "local_change": "one guarded fallback identifier replacement",
        },
        "licenses": LICENSES,
        "renderer_contract": {"canvas": [600, 600], "network": "disabled by renderer"},
    }


def build_runtime(
    source_root: Path = SOURCE_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    """Copy the pinned runtime and return its manifest after an atomic build."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source_root:
        raise PatchError("Output runtime must be separate from the frozen source runtime")
    if output_root.exists():
        raise PatchError(f"Refusing to overwrite existing runtime: {output_root}")
    source_values = _validate_source(source_root)
    patched_values = dict(source_values)
    patched_values["assets/p5.brush.js"] = _patch_brush(source_values["assets/p5.brush.js"])
    metadata = _runtime_metadata(source_values, patched_values, output_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for relative in COPIED_FILES:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(patched_values[str(relative)])
            shutil.copystat(source_root / relative, destination)
        (temporary / "PATCH-MANIFEST.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        metadata = build_runtime(args.source_root, args.output_root)
    except PatchError as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
