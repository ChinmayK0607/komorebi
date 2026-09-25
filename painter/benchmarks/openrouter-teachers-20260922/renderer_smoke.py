#!/usr/bin/env python3
"""Run one explicit Linux-only smoke render through the benchmark renderer.

The smoke sketch is fixed and contains one p5.brush polygon and line.  No
AI Gateway/API/model code is imported or called.  The output directory is
isolated from benchmark episodes and training artifacts; use ``--output-dir``
to point it at a disposable directory on the Linux node.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "renderer-smoke"
PROGRAM = """function setup() {
  createCanvas(600, 600, WEBGL);
  pixelDensity(1);
  randomSeed(17);
  noiseSeed(17);
  brush.seed(17);
  background(248, 245, 237);
}

function draw() {
  translate(-300, -300);
  brush.noStroke();
  brush.noWash();
  brush.fill("#c8704a", 220);
  brush.fillBleed(0.02, "out");
  brush.fillTexture(0.10, 0.05);
  brush.polygon([[100, 220], [300, 100], [500, 220], [380, 430], [180, 430]]);

  brush.noFill();
  brush.noWash();
  brush.set("cpencil", "#403832", 0.8);
  brush.line(100, 220, 500, 220);
  noLoop();
}
"""


def _load_renderer_module() -> Any:
    """Import the existing benchmark runner without making it a package."""

    path = ROOT / "run.py"
    spec = importlib.util.spec_from_file_location("openrouter_teacher_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    """Make a usable absolute path without following venv/runtime symlinks."""

    return path.expanduser().absolute()


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(
    *,
    renderer: Path,
    renderer_python: Path,
    browser_path: Path | None,
    run_as_user: str,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    """Render the fixed sketch once and independently verify its receipt."""

    if sys.platform != "linux":
        raise RuntimeError("renderer smoke tests are Linux-only; no renderer was invoked")
    if not run_as_user or any(char.isspace() for char in run_as_user):
        raise ValueError("--run-as-user must be one non-whitespace username")
    if not 1 <= timeout <= 900:
        raise ValueError("--timeout must be between 1 and 900 seconds")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "smoke-program.js"
    output = output_dir / "smoke.png"
    smoke_receipt = output_dir / "smoke-receipt.json"
    source.write_text(PROGRAM, encoding="utf-8")

    runner = _load_renderer_module()
    renderer_path = _absolute(renderer)
    renderer_python_path = _absolute(renderer_python)
    browser_path_value = _absolute(browser_path) if browser_path else None
    try:
        renderer_result = runner.render_program(
            root=ROOT,
            source=source,
            output=output,
            renderer=renderer_path,
            renderer_python=renderer_python_path,
            browser_path=browser_path_value,
            run_as_user=run_as_user,
            timeout=timeout,
        )
    except Exception as exc:  # Keep a durable, bounded diagnostic for node triage.
        renderer_result = {
            "valid": False,
            "service_failure": True,
            "error_code": "smoke_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }

    source_sha256 = _sha256(source)
    output_exists = output.is_file()
    png_sha256 = _sha256(output) if output_exists else None
    returned_receipt = renderer_result.get("receipt") if isinstance(renderer_result, dict) else None
    receipt_source_sha256 = returned_receipt.get("source_sha256") if isinstance(returned_receipt, dict) else None
    receipt_png_sha256 = returned_receipt.get("png_sha256") if isinstance(returned_receipt, dict) else None
    receipt_hashes_match = (
        isinstance(returned_receipt, dict)
        and receipt_source_sha256 == source_sha256
        and output_exists
        and receipt_png_sha256 == png_sha256
    )
    valid = bool(renderer_result.get("valid")) and output_exists and receipt_hashes_match
    smoke = {
        "schema": "painter.ai-gateway-teacher-renderer-smoke.v1",
        "valid": valid,
        "service_failure": bool(renderer_result.get("service_failure")),
        "error_code": renderer_result.get("error_code"),
        "renderer": str(renderer_path),
        "renderer_python": str(renderer_python_path),
        "browser_path": str(browser_path_value) if browser_path_value else None,
        "run_as_user": run_as_user,
        "timeout_seconds": timeout,
        "source": source.name,
        "source_sha256": source_sha256,
        "output": output.name,
        "output_exists": output_exists,
        "png_sha256": png_sha256,
        "receipt_hashes_match": receipt_hashes_match,
        "renderer_result": renderer_result,
    }
    _write_receipt(smoke_receipt, smoke)
    return {**smoke, "receipt_path": str(smoke_receipt)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True, help="pinned Linux renderer script")
    parser.add_argument("--renderer-python", type=Path, required=True, help="renderer virtualenv Python")
    parser.add_argument("--browser-path", type=Path, help="Playwright browser bundle, if required")
    parser.add_argument("--run-as-user", default="painter", help="unprivileged renderer user (default: painter)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="isolated smoke output directory")
    parser.add_argument("--timeout", type=int, default=180, help="renderer timeout in seconds, 1-900")
    args = parser.parse_args(argv)
    try:
        result = run_smoke(
            renderer=args.renderer,
            renderer_python=args.renderer_python,
            browser_path=args.browser_path,
            run_as_user=args.run_as_user,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"renderer smoke error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
