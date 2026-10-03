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
CANCELLABLE_RENDERER_SOURCE = REPOSITORY_ROOT / "painter/cancellable_watercolour_renderer.py"
CANCELLABLE_RENDERER_RUNTIME_ID = "watercolour-cancellable-v1"
CANCELLABLE_RENDERER_SHA256 = "374296922790f02e3ba42ee16930488dbf61a686a3cee18c1e74f55d8826d3a7"

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
        # The runtime is consumed by the unprivileged ``painter`` account on
        # Linux.  tempfile.mkdtemp creates a mode-700 root; preserve private
        # build isolation during assembly, then publish a read/traverse-only
        # runtime so that account can load the renderer and its assets.
        for directory in (temporary, *[path for path in temporary.rglob("*") if path.is_dir()]):
            directory.chmod(0o755)
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def build_renderer_overlay(
    source_root: Path = SOURCE_ROOT,
    output_root: Path | None = None,
    renderer_source: Path = CANCELLABLE_RENDERER_SOURCE,
) -> dict[str, Any]:
    """Build the explicit cancellable renderer overlay runtime.

    This copies the frozen renderer under an adjacent stable name so the
    derivative remains runnable when the runtime directory is materialized on
    Linux.  The original centroid-fix builder and its default output are
    untouched; callers must choose a distinct output root for this overlay.
    """

    source_root = Path(source_root).resolve()
    renderer_source = Path(renderer_source).resolve()
    if output_root is None:
        raise PatchError("Cancellable renderer overlay requires an explicit output root")
    output_root = Path(output_root).resolve()
    if output_root == source_root or output_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise PatchError("Cancellable renderer overlay requires a distinct output root")
    source_values = _validate_source(source_root)
    if not renderer_source.is_file():
        raise PatchError(f"Cancellable renderer source is missing: {renderer_source}")
    renderer_bytes = renderer_source.read_bytes()
    renderer_sha = sha256_bytes(renderer_bytes)
    if renderer_sha != CANCELLABLE_RENDERER_SHA256:
        raise PatchError(
            "Unexpected cancellable renderer source: expected "
            f"{CANCELLABLE_RENDERER_SHA256}, got {renderer_sha}"
        )
    if output_root.exists():
        raise PatchError(f"Refusing to overwrite existing runtime: {output_root}")

    patched_values = dict(source_values)
    patched_values["assets/p5.brush.js"] = _patch_brush(source_values["assets/p5.brush.js"])
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for relative in COPIED_FILES:
            destination_relative = (
                Path("frozen_watercolour_renderer.py")
                if relative == Path("renderer.py")
                else relative
            )
            destination = temporary / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(patched_values[str(relative)])
            shutil.copystat(source_root / relative, destination)
        derivative_destination = temporary / "cancellable_watercolour_renderer.py"
        derivative_destination.write_bytes(renderer_bytes)
        shutil.copystat(renderer_source, derivative_destination)
        metadata = {
            "schema": "watercolour-cancellable-renderer-v1",
            "runtime_id": output_root.name,
            "selected_renderer": {
                "runtime_id": CANCELLABLE_RENDERER_RUNTIME_ID,
                "path": "cancellable_watercolour_renderer.py",
                "sha256": renderer_sha,
            },
            "frozen_baseline": {
                "path": "frozen_watercolour_renderer.py",
                "sha256": sha256_bytes(source_values["renderer.py"]),
            },
            "assets": {
                str(relative): {
                    "before_sha256": sha256_bytes(source_values[str(relative)]),
                    "after_sha256": sha256_bytes(patched_values[str(relative)]),
                }
                for relative in COPIED_FILES
                if str(relative).startswith("assets/")
            },
            "source_root": "painter/vendor/integrations/watercolour",
            "centroid_baseline_manifest": {
                "renderer_sha256": EXPECTED_SHA256["renderer.py"],
                "assets_sha256": {
                    str(relative): EXPECTED_SHA256[str(relative)]
                    for relative in COPIED_FILES
                    if str(relative).startswith("assets/")
                },
            },
            "renderer_contract": {"canvas": [600, 600], "network": "disabled by renderer"},
            "centroid_patch": {
                "file": "assets/p5.brush.js",
                "function": "xn",
                "guarded_replacement": "v[0] -> t[0]",
                "replacement_count": 1,
                "before_sha256": sha256_bytes(source_values["assets/p5.brush.js"]),
                "after_sha256": sha256_bytes(patched_values["assets/p5.brush.js"]),
            },
        }
        (temporary / "PATCH-MANIFEST.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for directory in (temporary, *[path for path in temporary.rglob("*") if path.is_dir()]):
            directory.chmod(0o755)
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def validate_renderer_overlay(
    output_root: Path,
    source_root: Path = SOURCE_ROOT,
    renderer_source: Path = CANCELLABLE_RENDERER_SOURCE,
) -> dict[str, Any]:
    """Validate and return an already-materialized cancellable overlay."""

    output_root = Path(output_root).resolve()
    source_root = Path(source_root).resolve()
    renderer_source = Path(renderer_source).resolve()
    metadata_path = output_root / "PATCH-MANIFEST.json"
    if not metadata_path.is_file():
        raise PatchError(f"Cancellable renderer overlay manifest is missing: {output_root}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PatchError(f"Invalid cancellable renderer overlay manifest: {output_root}") from error
    if metadata.get("schema") != "watercolour-cancellable-renderer-v1":
        raise PatchError("Unexpected cancellable renderer overlay schema")
    expected_derivative = sha256_bytes(renderer_source.read_bytes())
    if expected_derivative != CANCELLABLE_RENDERER_SHA256:
        raise PatchError("Unexpected cancellable renderer source")
    expected_baseline = _validate_source(source_root)
    patched_assets = dict(expected_baseline)
    patched_assets["assets/p5.brush.js"] = _patch_brush(expected_baseline["assets/p5.brush.js"])
    checks = {
        "cancellable_watercolour_renderer.py": expected_derivative,
        "frozen_watercolour_renderer.py": EXPECTED_SHA256["renderer.py"],
        **{
            str(relative): sha256_bytes(patched_assets[str(relative)])
            for relative in COPIED_FILES
            if str(relative).startswith("assets/")
        },
    }
    for relative, expected in checks.items():
        path = output_root / relative
        if not path.is_file():
            raise PatchError(f"Cancellable renderer overlay file is missing: {relative}")
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise PatchError(f"Cancellable renderer overlay hash mismatch for {relative}")
    selected = metadata.get("selected_renderer") or {}
    baseline = metadata.get("frozen_baseline") or {}
    if (
        selected.get("path") != "cancellable_watercolour_renderer.py"
        or selected.get("sha256") != expected_derivative
        or selected.get("runtime_id") != CANCELLABLE_RENDERER_RUNTIME_ID
    ):
        raise PatchError("Cancellable renderer overlay manifest does not match its files")
    if (
        baseline.get("path") != "frozen_watercolour_renderer.py"
        or baseline.get("sha256") != EXPECTED_SHA256["renderer.py"]
    ):
        raise PatchError("Cancellable renderer overlay baseline does not match its files")
    for relative, expected in checks.items():
        if relative.startswith("assets/"):
            entry = (metadata.get("assets") or {}).get(relative) or {}
            if entry.get("after_sha256") != expected:
                raise PatchError(f"Cancellable renderer overlay asset manifest mismatch for {relative}")
    centroid_patch = metadata.get("centroid_patch") or {}
    if centroid_patch.get("after_sha256") != checks["assets/p5.brush.js"]:
        raise PatchError("Cancellable renderer overlay centroid manifest mismatch")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--renderer-overlay",
        action="store_true",
        help="Build the opt-in cancellable renderer overlay; requires --output-root",
    )
    args = parser.parse_args(argv)
    try:
        if args.renderer_overlay:
            metadata = build_renderer_overlay(args.source_root, args.output_root)
        else:
            metadata = build_runtime(
                args.source_root,
                args.output_root or DEFAULT_OUTPUT_ROOT,
            )
    except PatchError as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
