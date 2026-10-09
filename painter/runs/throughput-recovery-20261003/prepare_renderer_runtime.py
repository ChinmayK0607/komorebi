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
import tempfile
from collections.abc import Iterable


SCHEMA = "throughput-recovery-renderer-runtime-v1"
_PYTHON_RUNTIME = re.compile(r"python\d+(?:\.\d+)*$")


def _renderer_root_for_node(path: Path) -> Path | None:
    """Return the lexical renderer root for an exact Playwright driver/node."""

    path = path.absolute()
    if len(path.parents) < 7:
        return None
    if path.name != "node" or path.parent.name != "driver":
        return None
    if path.parents[1].name != "playwright" or path.parents[2].name != "site-packages":
        return None
    if not _PYTHON_RUNTIME.fullmatch(path.parents[3].name):
        return None
    if path.parents[4].name != "lib" or path.parents[5].name != "renderer-env":
        return None
    return path.parents[6]


def _regular_node(path: Path, description: str) -> tuple[Path, Path]:
    """Validate a non-symlink executable and return it with its runtime root."""

    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} is not a regular file: {path}")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"{description} is not executable: {path}")
    resolved = path.resolve(strict=True)
    root = _renderer_root_for_node(resolved)
    if root is None:
        raise RuntimeError(f"{description} is not a Playwright driver/node: {path}")
    return resolved, root


def _safe_renderer_root(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{description} is not a non-symlink directory: {path}")
    return path.resolve(strict=True)


def _safe_target_beneath(target: Path, root: Path) -> None:
    """Reject mutable path components before trusting an old renderer target."""

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"renderer Node target escapes its allowed runtime: {target}") from exc
    current = target
    while True:
        if current.is_symlink() or current.stat().st_mode & 0o022:
            raise RuntimeError(f"renderer Node target path is not trusted: {current}")
        if current == root:
            return
        if current.parent == current:
            raise RuntimeError(f"renderer Node target has no trusted runtime root: {target}")
        current = current.parent


def _node(renderer_root: Path) -> Path:
    renderer_root = _safe_renderer_root(renderer_root, "renderer root")
    candidates = sorted(renderer_root.glob("renderer-env/lib/python*/site-packages/playwright/driver/node"))
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        source, source_root = _regular_node(candidate, "bundled Playwright Node")
        if source_root == renderer_root:
            return source
    raise RuntimeError("bundled Playwright Node executable is missing")


def _bind_node(
    prime_venv: Path,
    source: Path,
    *,
    current_renderer_root: Path | None = None,
    allowed_old_renderer_roots: Iterable[Path] = (),
) -> Path:
    """Bind Prime's Node link, allowing only an explicitly trusted migration.

    A stale generated link can be moved when it resolves to a regular
    Playwright ``driver/node`` beneath an allowlisted old renderer runtime.
    Every other pre-existing destination, including a regular executable or a
    broken/untrusted symlink, is rejected before any mutation.
    """

    source, source_root = _regular_node(source, "current Playwright Node")
    if current_renderer_root is not None and source_root != _safe_renderer_root(current_renderer_root, "current renderer root"):
        raise RuntimeError(f"current Playwright Node is outside the current renderer root: {source}")

    prime_venv = prime_venv.expanduser().absolute()
    if prime_venv.is_symlink():
        raise RuntimeError(f"Prime virtualenv cannot be a symlink: {prime_venv}")
    prime_venv.mkdir(parents=True, exist_ok=True)
    bin_dir = prime_venv / "bin"
    if bin_dir.is_symlink() or (bin_dir.exists() and not bin_dir.is_dir()):
        raise RuntimeError(f"Prime virtualenv bin directory is not a directory: {bin_dir}")
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "node"

    if not link.exists() and not link.is_symlink():
        link.symlink_to(source)
        return link
    if not link.is_symlink():
        raise RuntimeError(f"Prime Node path is not a symlink: {link}")

    try:
        existing_target = link.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Prime Node link is broken or unreadable: {link}") from exc
    if existing_target == source:
        return link

    old_target, old_root = _regular_node(existing_target, "existing Prime Node target")
    allowed_roots = tuple(_safe_renderer_root(root, "allowed old renderer root") for root in allowed_old_renderer_roots)
    if old_root not in allowed_roots:
        raise RuntimeError(f"Prime Node link points outside an allowed old renderer runtime: {link}")
    _safe_target_beneath(old_target, old_root)

    # Recheck the link immediately before replacing it.  ``os.replace`` is
    # atomic, so readers see either the old or the new complete symlink.
    if not link.is_symlink() or link.resolve(strict=True) != old_target:
        raise RuntimeError(f"Prime Node link changed while validating it: {link}")
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{link.name}.", suffix=".tmp", dir=link.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        temporary.unlink()
        temporary.symlink_to(source)
        os.replace(temporary, link)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
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
    runuser = shutil.which("runuser")
    if os.geteuid() == 0 and runuser:
        if subprocess.run(["id", "-u", "painter"], capture_output=True, check=False).returncode != 0:
            raise RuntimeError("painter browser account is missing")
        command = [str(runuser), "-u", "painter", "--", "env", "-i", *(
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
            browsers: Path, prime_venv: Path, smoke: bool = True,
            allowed_old_renderer_roots: Iterable[Path] | None = None) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("renderer runtime preparation requires Linux")
    renderer_root = _safe_renderer_root(renderer_root, "renderer root")
    source = _node(renderer_root)
    if allowed_old_renderer_roots is None:
        # The checked-out painter tree is the only legacy runtime location
        # accepted implicitly.  Callers may provide a narrower explicit list
        # for a relocated deployment; no arbitrary parent directory is ever
        # treated as an old renderer root.
        legacy_root = Path(__file__).resolve().parents[2]
        allowed_old_renderer_roots = (renderer_root, legacy_root)
    link = _bind_node(
        prime_venv,
        source,
        current_renderer_root=renderer_root,
        allowed_old_renderer_roots=allowed_old_renderer_roots,
    )
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
    parser.add_argument(
        "--allowed-old-renderer-root", type=Path, action="append", default=None,
        help="explicit legacy renderer root whose Playwright Node link may be rebound",
    )
    parser.add_argument("--sandbox-only", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()
    if args.sandbox_only:
        sandbox_preflight(run_root=args.run_root, renderer_python=args.renderer_python,
                          browsers=args.browsers)
        return 0
    prepare(run_root=args.run_root, renderer_root=args.renderer_root,
            renderer_python=args.renderer_python, browsers=args.browsers,
            prime_venv=args.prime_venv, smoke=not args.no_smoke,
            allowed_old_renderer_roots=args.allowed_old_renderer_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
