#!/usr/bin/env python3
"""Bind the reviewed renderer runtime and fail fast on browser startup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import hashlib


SCHEMA = "throughput-recovery-renderer-runtime-v1"


def _node(renderer_root: Path) -> Path:
    candidates = sorted(renderer_root.glob("renderer-env/lib/python*/site-packages/playwright/driver/node"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError("bundled Playwright Node executable is missing")


def _bind_node(prime_venv: Path, source: Path) -> Path:
    link = prime_venv / "bin/node"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve(strict=False) != source.resolve(strict=False):
            raise RuntimeError(f"Prime Node link points somewhere else: {link}")
    elif link.exists():
        if not os.access(link, os.X_OK):
            raise RuntimeError(f"Prime Node path is not executable: {link}")
    else:
        link.symlink_to(source)
    return link


def _trusted(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    current = path
    while True:
        if current.stat().st_uid != 0 or current.stat().st_mode & 0o022:
            return False
        if current == root:
            return True
        if current.parent == current:
            return False
        current = current.parent


def _version_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    for part in relative.parts:
        match = re.search(r"(?:^|[-_])(\d[\w.-]*)$", part)
        if match:
            return match.group(1)
    return ""


def _headless_targets(browsers: Path, sources: list[Path]) -> list[tuple[Path, Path]]:
    targets = []
    for executable in sorted(path for path in browsers.rglob("chrome-headless-shell")
                             if path.is_file() and not path.is_symlink()):
        version = _version_key(executable, browsers)
        matching = [source for source in sources if _version_key(source, browsers) == version]
        if len(matching) != 1:
            raise RuntimeError("could not match headless Chromium to one trusted full-browser sandbox helper")
        targets.append((executable.with_name("chrome-sandbox"), matching[0]))
    return targets


def _install_helper(source: Path, target: Path, browsers: Path) -> None:
    # Check the destination before any privileged mutation.  ``exists()`` is
    # false for a broken symlink, so it must never be the first guard.
    if target.is_symlink():
        raise RuntimeError(f"refusing Chromium sandbox symlink target: {target}")
    if target.exists():
        return
    if not _trusted(target.parent, browsers):
        raise RuntimeError("Chromium sandbox destination parent is not trusted")
    if os.geteuid() != 0 or not _trusted(source, browsers) or source.stat().st_uid != 0:
        raise RuntimeError("trusted Chromium sandbox helper is not root-owned")
    shutil.copy2(source, target)
    os.chown(target, 0, 0)
    target.chmod(0o4755)
    if target.read_bytes() != source.read_bytes():
        raise RuntimeError("Chromium sandbox helper copy hash mismatch")


def _sandbox_helpers(browsers: Path) -> list[Path]:
    browsers = browsers.expanduser().resolve()
    sources = sorted(path for path in browsers.rglob("chrome_sandbox") if path.is_file() and not path.is_symlink())
    expected = sorted(path for path in browsers.rglob("chrome-sandbox") if path.is_file() and not path.is_symlink())
    if os.geteuid() != 0 and sources:
        raise RuntimeError("trusted Chromium sandbox helper requires root setup")
    for candidate in sources:
        if not _trusted(candidate, browsers) or candidate.stat().st_uid != 0:
            raise RuntimeError("trusted Chromium sandbox helper is not root-owned")
        target = candidate.with_name("chrome-sandbox")
        _install_helper(candidate, target, browsers)
        expected.append(target)
    for target, source in _headless_targets(browsers, sources):
        _install_helper(source, target, browsers)
        expected.append(target)
    helpers = sorted({path.resolve() for path in expected if path.is_file()})
    if not helpers:
        raise RuntimeError("Chromium chrome-sandbox helper is missing")
    for helper in helpers:
        mode = helper.stat().st_mode
        if not _trusted(helper, browsers) or helper.stat().st_uid != 0:
            raise RuntimeError("Chromium chrome-sandbox must be root-owned mode 4755")
        matching = [source for source in sources if _version_key(source, browsers) == _version_key(helper, browsers)]
        if len(matching) == 1 and _sha256(helper) != _sha256(matching[0]):
            raise RuntimeError("Chromium chrome-sandbox hash does not match its trusted bundled helper")
        if mode & 0o4755 != 0o4755:
            if os.geteuid() != 0:
                raise RuntimeError("Chromium chrome-sandbox must be mode 4755")
            helper.chmod(0o4755)
            if helper.stat().st_mode & 0o4755 != 0o4755:
                raise RuntimeError("Chromium chrome-sandbox could not be made mode 4755")
    return helpers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _smoke(renderer_python: Path, browsers: Path, prime_venv: Path | None) -> None:
    code = "from playwright.sync_api import sync_playwright\nwith sync_playwright() as p:\n    b = p.chromium.launch(headless=True, chromium_sandbox=True)\n    b.close()\n"
    env = {
        "PATH": f"{prime_venv / 'bin' if prime_venv else '/usr/local/bin'}:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "PLAYWRIGHT_BROWSERS_PATH": str(browsers),
    }
    command = [str(renderer_python), "-c", code]
    if os.geteuid() == 0 and shutil.which("runuser"):
        if subprocess.run(["id", "-u", "painter"], capture_output=True, check=False).returncode != 0:
            raise RuntimeError("painter browser account is missing")
        command = ["runuser", "-u", "painter", "--", "env", "-i", *(
            f"{key}={value}" for key, value in env.items()), *command]
    else:
        command = ["env", "-i", *(f"{key}={value}" for key, value in env.items()), *command]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Chromium sandbox startup smoke timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError("Chromium sandbox startup smoke failed: " + " | ".join(detail))


def prepare(*, run_root: Path, renderer_root: Path, renderer_python: Path,
            browsers: Path, prime_venv: Path, smoke: bool = True) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("renderer runtime preparation requires Linux")
    source = _node(renderer_root)
    link = _bind_node(prime_venv, source)
    node_check = renderer_root / "vendor/integrations/watercolour/assets/p5.brush.js"
    if not node_check.is_file():
        raise RuntimeError(f"reviewed brush asset is missing: {node_check}")
    subprocess.run([str(link), "--check", str(node_check)], check=True, timeout=30,
                   capture_output=True, text=True)
    helpers = _sandbox_helpers(browsers)
    if smoke:
        _smoke(renderer_python, browsers, prime_venv)
    receipt = {
        "schema": SCHEMA,
        "node_source": str(source),
        "prime_node": str(link),
        "node_version": subprocess.check_output([str(link), "--version"], text=True).strip(),
        "node_check": "passed",
        "browser_root": str(browsers),
        "sandbox_helper": str(helpers[0]),
        "sandbox_helpers": [str(helper) for helper in helpers],
        "sandbox_helper_mode": stat.S_IMODE(helpers[0].stat().st_mode),
        "sandbox_helper_sha256": _sha256(helpers[0]),
        "sandbox_helper_sha256s": {str(helper): _sha256(helper) for helper in helpers},
        "sandbox_smoke": "passed" if smoke else "not-run",
        "credentials_touched": False,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = run_root / ".renderer-runtime-receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, run_root / "renderer-runtime-receipt.json")
    return receipt


def sandbox_preflight(*, run_root: Path, renderer_python: Path, browsers: Path) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("renderer sandbox preflight requires Linux")
    helpers = _sandbox_helpers(browsers)
    _smoke(renderer_python, browsers, None)
    receipt = {
        "schema": SCHEMA,
        "browser_root": str(browsers),
        "sandbox_helper": str(helpers[0]),
        "sandbox_helpers": [str(helper) for helper in helpers],
        "sandbox_helper_mode": stat.S_IMODE(helpers[0].stat().st_mode),
        "sandbox_helper_sha256": _sha256(helpers[0]),
        "sandbox_helper_sha256s": {str(helper): _sha256(helper) for helper in helpers},
        "sandbox_smoke": "passed",
        "credentials_touched": False,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = run_root / ".renderer-runtime-receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, run_root / "renderer-runtime-receipt.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path, required=True)
    parser.add_argument("--browsers", type=Path, required=True)
    parser.add_argument("--prime-venv", type=Path, required=True)
    parser.add_argument("--sandbox-only", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()
    if args.sandbox_only:
        sandbox_preflight(run_root=args.run_root, renderer_python=args.renderer_python,
                          browsers=args.browsers)
        return 0
    prepare(run_root=args.run_root, renderer_root=args.renderer_root,
            renderer_python=args.renderer_python, browsers=args.browsers,
            prime_venv=args.prime_venv, smoke=not args.no_smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
