#!/usr/bin/env python3
"""Run-owned renderer runtime wrapper with sandbox diagnostics.

The renderer payload remains the reviewed shared helper and its exact hashes
are unchanged.  This wrapper adds an early Linux capability probe and keeps a
diagnostic record containing both the first and last browser stderr lines;
Chromium startup failures otherwise often reduce to an unhelpful SIGTRAP tail.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time


RUN = Path(__file__).resolve().parent
SHARED = RUN.parent / "throughput-recovery-20261003" / "prepare_renderer_runtime.py"
spec = importlib.util.spec_from_file_location("structured_shared_renderer_runtime", SHARED)
if spec is None or spec.loader is None:
    raise RuntimeError(f"shared renderer runtime helper is missing: {SHARED}")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)

CAPABILITY_SCHEMA = "structured-flow-browser-capability-v1"


def _lines(value: str | bytes | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return int(value) if value else None
    except (OSError, ValueError):
        return None


def _sandbox_candidates(browsers: Path) -> list[Path]:
    if not browsers.exists():
        return []
    return sorted(
        path for path in browsers.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name in {"chrome_sandbox", "chrome-sandbox"}
    )


def capability_preflight(*, run_root: Path, browsers: Path, allow_deferred_browser_setup: bool = False) -> dict[str, object]:
    """Check OS prerequisites before model/Python setup.

    The early node probe may run before the renderer setup creates the
    unprivileged ``painter`` account or downloads Chromium.  In that mode it
    records the kernel evidence and defers only those two checks; the real
    sandbox helper and browser smoke remain fail-closed after setup.
    """

    if sys.platform != "linux":
        raise RuntimeError("structured-flow browser capability probe requires Linux")
    run_root = run_root.expanduser().resolve()
    browsers = browsers.expanduser().resolve()
    runuser = shutil.which("runuser")
    painter_uid: int | None = None
    if runuser:
        completed = subprocess.run([runuser, "-u", "painter", "--", "id", "-u"], capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            try:
                painter_uid = int(completed.stdout.strip())
            except ValueError:
                painter_uid = None
    elif os.geteuid() != 0:
        painter_uid = os.geteuid()

    max_user_namespaces = _read_int(Path("/proc/sys/user/max_user_namespaces"))
    unprivileged_userns_clone = _read_int(Path("/proc/sys/kernel/unprivileged_userns_clone"))
    no_new_privs = None
    seccomp = None
    cap_eff = None
    cap_bnd = None
    for line in Path("/proc/self/status").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("NoNewPrivs:"):
            try:
                no_new_privs = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Seccomp:"):
            try:
                seccomp = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("CapEff:"):
            cap_eff = line.split(":", 1)[1].strip() or None
        elif line.startswith("CapBnd:"):
            cap_bnd = line.split(":", 1)[1].strip() or None

    userns_probe: dict[str, object] = {"attempted": False, "returncode": None, "stderr": []}
    if runuser and painter_uid is not None and shutil.which("unshare"):
        command = [runuser, "-u", "painter", "--", "unshare", "-Ur", "true"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
            userns_probe = {
                "attempted": True,
                "returncode": completed.returncode,
                "stderr": _lines(completed.stderr)[:10],
            }
        except subprocess.TimeoutExpired as exc:
            userns_probe = {"attempted": True, "returncode": "timeout", "stderr": _lines(exc.stderr)[:10]}

    helpers = _sandbox_candidates(browsers)
    helper_details = []
    for helper in helpers:
        mode = stat.S_IMODE(helper.stat().st_mode)
        helper_details.append({
            "path": str(helper),
            "uid": helper.stat().st_uid,
            "mode": oct(mode),
            "setuid_root": helper.stat().st_uid == 0 and bool(mode & stat.S_ISUID),
        })
    setuid_helper = any(item["setuid_root"] for item in helper_details)
    namespace_available = (
        (max_user_namespaces is not None and max_user_namespaces > 0)
        and (unprivileged_userns_clone is None or unprivileged_userns_clone > 0)
    ) or userns_probe.get("returncode") == 0
    failures: list[str] = []
    if not namespace_available and not setuid_helper and not allow_deferred_browser_setup:
        failures.append("browser sandbox has no usable user namespace or setuid helper")
    if no_new_privs == 1 and not namespace_available and not setuid_helper and not allow_deferred_browser_setup:
        failures.append("NoNewPrivs=1 blocks both sandbox paths")
    if os.geteuid() == 0 and not runuser:
        failures.append("root browser setup requires runuser to drop to painter")
    if painter_uid is None and not allow_deferred_browser_setup:
        failures.append("painter browser account is unavailable")

    warnings: list[str] = []
    if allow_deferred_browser_setup and painter_uid is None:
        warnings.append("painter browser account will be created by render setup")
    if allow_deferred_browser_setup and not namespace_available and not setuid_helper:
        warnings.append("sandbox helper will be checked after the pinned browser is installed")

    value: dict[str, object] = {
        "schema": CAPABILITY_SCHEMA,
        "status": "capability_probe_passed" if not failures else "capability_probe_failed",
        "euid": os.geteuid(),
        "painter_uid": painter_uid,
        "runuser": runuser,
        "max_user_namespaces": max_user_namespaces,
        "unprivileged_userns_clone": unprivileged_userns_clone,
        "no_new_privs": no_new_privs,
        "seccomp": seccomp,
        "cap_eff": cap_eff,
        "cap_bnd": cap_bnd,
        "userns_probe": userns_probe,
        "sandbox_helpers": helper_details,
        "namespace_available": namespace_available,
        "setuid_helper_available": setuid_helper,
        "browser_root": str(browsers),
        "failures": failures,
        "warnings": warnings,
        "credentials_touched": False,
    }
    _write_json(run_root / "browser-capability-preflight.json", value)
    if failures:
        raise RuntimeError("; ".join(failures))
    return value


def _browser_tmpdir() -> Path:
    root = Path(os.environ.get("PAINTER_BROWSER_TMPDIR", "/tmp/structured-flow-browser"))
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    if os.geteuid() == 0 and shutil.which("runuser"):
        uid = subprocess.check_output(["runuser", "-u", "painter", "--", "id", "-u"], text=True).strip()
        gid = subprocess.check_output(["runuser", "-u", "painter", "--", "id", "-g"], text=True).strip()
        os.chown(root, int(uid), int(gid))
    return root


def _smoke(renderer_python: Path, browsers: Path, prime_venv: Path | None) -> None:
    """Start Chromium with sandboxing intact and preserve useful diagnostics."""

    scratch = _browser_tmpdir()
    # Match the pinned renderer's production launch flags.  A bare headless
    # launch can initialize the container GPU path differently and produce a
    # misleading SIGTRAP even when the renderer's SwiftShader path is usable.
    code = (
        "from playwright.sync_api import sync_playwright\n"
        "with sync_playwright() as p:\n"
        "    b = p.chromium.launch(headless=True, chromium_sandbox=True, "
        "args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])\n"
        "    b.close()\n"
    )
    env = {
        "PATH": f"{prime_venv / 'bin' if prime_venv else '/usr/local/bin'}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "XDG_RUNTIME_DIR": str(scratch),
        "LANG": "C",
        "LC_ALL": "C",
        "PLAYWRIGHT_BROWSERS_PATH": str(browsers),
    }
    command = [str(renderer_python), "-c", code]
    if os.geteuid() == 0 and shutil.which("runuser"):
        command = ["runuser", "-u", "painter", "--", "env", "-i", *(f"{key}={value}" for key, value in env.items()), *command]
    else:
        command = ["env", "-i", *(f"{key}={value}" for key, value in env.items()), *command]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        timeout = False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(command, returncode=None, stdout=exc.stdout or "", stderr=exc.stderr or "")
        timeout = True
    stdout = _lines(completed.stdout)
    stderr = _lines(completed.stderr)
    diagnostic = {
        "schema": "structured-flow-browser-smoke-v2",
        "status": "passed" if completed.returncode == 0 and not timeout else "failed",
        "returncode": completed.returncode,
        "signal": -completed.returncode if isinstance(completed.returncode, int) and completed.returncode < 0 else None,
        "timeout": timeout,
        "elapsed_seconds": time.monotonic() - started,
        "command": command[:6] + ["..."] if len(command) > 6 else command,
        # Keep the complete startup streams.  The first/last projections make
        # controller summaries useful, while the full fields retain a SIGTRAP
        # diagnostic that happened before Chromium's usual tail.
        "stderr": stderr,
        "stdout": stdout,
        "stderr_first": stderr[:20],
        "stderr_last": stderr[-20:] if stderr else [],
        "stdout_first": stdout[:20],
        "stdout_last": stdout[-20:] if stdout else [],
        "browser_root": str(browsers),
        "scratch": str(scratch),
        "browser_args": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    }
    run_root = Path(os.environ.get("PAINTER_RUN_ROOT", ".")).expanduser().resolve()
    _write_json(run_root / "browser-smoke-diagnostic.json", diagnostic)
    if diagnostic["status"] != "passed":
        first = " | ".join(stderr[:5] or stdout[:5] or ["no browser diagnostic output"])
        last = " | ".join(stderr[-5:] or stdout[-5:] or ["no browser diagnostic output"])
        raise RuntimeError(f"Chromium sandbox startup smoke failed: returncode={completed.returncode} first={first} last={last}; diagnostic={run_root / 'browser-smoke-diagnostic.json'}")


shared._smoke = _smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path, required=True)
    parser.add_argument("--browsers", type=Path, required=True)
    parser.add_argument("--prime-venv", type=Path, required=True)
    parser.add_argument("--sandbox-only", action="store_true")
    parser.add_argument("--capability-only", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("PAINTER_RUN_ROOT", str(args.run_root.resolve()))
    if args.capability_only:
        capability_preflight(run_root=args.run_root, browsers=args.browsers, allow_deferred_browser_setup=True)
        return 0
    if args.sandbox_only:
        shared.sandbox_preflight(run_root=args.run_root, renderer_python=args.renderer_python, browsers=args.browsers)
        return 0
    shared.prepare(
        run_root=args.run_root,
        renderer_root=args.renderer_root,
        renderer_python=args.renderer_python,
        browsers=args.browsers,
        prime_venv=args.prime_venv,
        smoke=not args.no_smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
