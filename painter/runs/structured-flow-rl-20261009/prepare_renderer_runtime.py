#!/usr/bin/env python3
"""Run-owned renderer runtime wrapper with sandbox diagnostics.

The renderer payload remains the reviewed shared helper and its exact hashes
are unchanged.  This wrapper adds an early Linux capability probe and keeps a
diagnostic record containing both the first and last browser stderr lines;
Chromium startup failures otherwise often reduce to an unhelpful SIGTRAP tail.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _resolve_node_executable(prime_venv: Path | None = None) -> Path:
    """Resolve verified Prime Node before falling back to the controller PATH."""
    if prime_venv is not None:
        prime_node = prime_venv.expanduser().resolve() / "bin" / "node"
        if prime_node.is_file() and os.access(prime_node, os.X_OK):
            return prime_node
    selected = shutil.which("node")
    if not selected:
        raise RuntimeError("renderer probe requires Node.js for the syntax preflight")
    node = Path(selected).expanduser().resolve()
    if not node.is_file() or not os.access(node, os.X_OK):
        raise RuntimeError(f"resolved Node.js executable is not runnable: {node}")
    return node


def render_probe(
    *,
    run_root: Path,
    renderer: Path,
    renderer_python: Path,
    browsers: Path,
    expected_renderer_sha256: str,
    prime_venv: Path | None = None,
    expected_assets_sha256: str | None = None,
) -> dict[str, object]:
    """Render one tiny p5.brush program through the native production boundary.

    This deliberately calls ``native_painting_renderer.render_action`` from
    the Prime interpreter rather than invoking the cancellable renderer
    directly.  Native evaluation derives paths from its nested attempt root;
    the explicit browser variables below are therefore part of the probe's
    contract and catch the failure mode before any paid model work starts.
    """
    if sys.platform != "linux":
        raise RuntimeError("structured-flow renderer probe requires Linux")
    run_root = run_root.expanduser().resolve()
    renderer = renderer.expanduser().resolve()
    renderer_python = renderer_python.expanduser().absolute()
    browsers = browsers.expanduser().resolve()
    if not renderer.is_file() or not renderer_python.is_file():
        raise FileNotFoundError("pinned renderer or renderer Python is missing")
    if not browsers.is_dir():
        raise FileNotFoundError(f"pinned browser root is missing: {browsers}")
    actual_renderer_sha256 = _sha256(renderer)
    if actual_renderer_sha256 != expected_renderer_sha256:
        raise RuntimeError("renderer probe source hash does not match the pinned renderer")
    assets = {
        path.relative_to(renderer.parent / "assets").as_posix(): _sha256(path)
        for path in sorted((renderer.parent / "assets").rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    assets_sha256 = hashlib.sha256(
        (json.dumps(assets, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    if expected_assets_sha256 is not None and assets_sha256 != expected_assets_sha256:
        raise RuntimeError("renderer probe asset map hash does not match the pinned renderer assets")
    node = _resolve_node_executable(prime_venv)
    probe_root = run_root / "runtime" / "native-render-probe"
    source = probe_root / "probe.js"
    nested_root = probe_root / "nested-attempt"
    native_folder = nested_root / "native-render"
    native_source = native_folder / "program.js"
    output = native_folder / "painting.png"
    renderer_receipt = output.with_suffix(".json")
    native_row_path = probe_root / "native-row.json"
    receipt_path = probe_root / "renderer-probe-receipt.json"
    stdout_path = probe_root / "renderer.stdout.log"
    stderr_path = probe_root / "renderer.stderr.log"
    if any(path.exists() for path in (source, output, renderer_receipt, receipt_path, native_row_path)):
        raise RuntimeError(f"existing renderer probe evidence must be preserved: {probe_root}")
    probe_root.mkdir(parents=True, exist_ok=False)
    source.write_text(
        "function setup(){createCanvas(600,600,WEBGL);}\n"
        "function draw(){translate(-width/2,-height/2);background('#f5efe4');"
        "brush.set('cpencil','#554d44',0.8);brush.stroke('#554d44');"
        "brush.line(80,80,520,520);noLoop();}\n",
        encoding="utf-8",
    )
    source.chmod(0o644)
    nested_root.mkdir()
    native_folder.mkdir()
    home = probe_root / "home"
    tmp = probe_root / "tmp"
    home.mkdir()
    tmp.mkdir()
    runuser = shutil.which("runuser")
    if os.geteuid() == 0 and runuser:
        uid = int(subprocess.check_output([runuser, "-u", "painter", "--", "id", "-u"], text=True).strip())
        gid = int(subprocess.check_output([runuser, "-u", "painter", "--", "id", "-g"], text=True).strip())
        for path in (probe_root, home, tmp):
            os.chown(path, uid, gid)
    # Renderer children receive only the non-secret runtime needed for the
    # sandbox.  Keep PYTHONPATH explicit so this subprocess exercises the
    # checked-in native renderer with the actual Prime interpreter.  Do not
    # use os.environ.copy(): inherited model/API tokens must not cross this
    # boundary.
    repo_root = Path(__file__).resolve().parents[3]
    # Keep the controller-selected Node binary available to the native
    # syntax preflight even when it lives in the image SDK or Prime toolchain.
    # Only this verified parent directory is admitted; no ambient PATH is
    # copied into the child environment.
    safe_path = os.pathsep.join(dict.fromkeys((str(node.parent), "/usr/local/bin", "/usr/bin", "/bin")))
    environment = {
        "PATH": safe_path,
        "PYTHONPATH": str(repo_root),
        "PAINTER_BROWSER_ROOT": str(browsers),
        "PLAYWRIGHT_BROWSERS_PATH": str(browsers),
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "XDG_RUNTIME_DIR": str(tmp),
        "LANG": "C",
        "LC_ALL": "C",
        "PAINTER_JS_SYNTAX_PREFLIGHT": "1",
        "PAINTER_RENDER_TIMEOUT_SECONDS": "180",
    }
    # Keep this harness small and pass all paths as argv.  render_action then
    # performs the exact production runuser/Popen renderer launch itself,
    # including receipt validation and sandbox handling.
    harness = """
import json
import sys
from pathlib import Path
from painter.native_painting_renderer import render_action

root, folder, source, renderer, renderer_python, row_path = map(Path, sys.argv[1:])
program = source.read_text(encoding="utf-8")
row = render_action(
    root=root,
    folder=folder,
    reply="```javascript\\n" + program + "\\n```",
    initial_canvas=None,
    initial_program=None,
    renderer=renderer,
    renderer_python=renderer_python,
)
row_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
if not row.get("valid"):
    raise SystemExit(2)
"""
    command = [
        str(sys.executable), "-c", harness,
        str(nested_root), str(native_folder), str(source),
        str(renderer), str(renderer_python), str(native_row_path),
    ]
    started = time.monotonic()
    try:
        # Explicit env= is intentional.  A runuser ``env`` argv prefix alone
        # does not stop this controller process from inheriting credentials.
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        stdout = _text(completed.stdout)
        stderr = _text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        completed = subprocess.CompletedProcess(command, returncode=None, stdout=stdout, stderr=stderr)
    except Exception as exc:
        completed = subprocess.CompletedProcess(command, returncode=None, stdout="", stderr=f"{type(exc).__name__}: {exc}")
        stdout = ""
        stderr = completed.stderr
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    stdout_path.chmod(0o600)
    stderr_path.chmod(0o600)
    value: dict[str, object] = {
        "schema": "structured-flow-native-render-probe-v1",
        "status": "failed",
        "renderer": str(renderer),
        "renderer_sha256": actual_renderer_sha256,
        "assets_sha256": assets_sha256,
        "renderer_python": str(renderer_python),
        "node": str(node),
        "node_sha256": _sha256(node),
        "browser_root": str(browsers),
        "source": str(native_source),
        "source_sha256": _sha256(native_source) if native_source.is_file() else None,
        "probe_source": str(source),
        "probe_source_sha256": _sha256(source),
        "output": str(output),
        "nested_root": str(nested_root),
        "native_row": str(native_row_path),
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_first": _lines(stdout)[:10],
        "stdout_last": _lines(stdout)[-10:],
        "stderr_first": _lines(stderr)[:10],
        "stderr_last": _lines(stderr)[-10:],
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    try:
        native_receipt = json.loads(renderer_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        native_receipt = None
    try:
        native_result = json.loads(native_row_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        native_result = None
    value["renderer_receipt"] = native_receipt
    value["native_result"] = native_result
    valid = bool(
        completed.returncode == 0
        and isinstance(native_result, dict)
        and native_result.get("valid") is True
        and isinstance(native_receipt, dict)
        and native_receipt.get("valid") is True
        and output.is_file()
        and output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        and value["source_sha256"] is not None
        and native_receipt.get("source_sha256") == value["source_sha256"]
        and native_receipt.get("png_sha256") == _sha256(output)
        and native_receipt.get("renderer_sha256") == actual_renderer_sha256
    )
    value["png_sha256"] = _sha256(output) if output.is_file() else None
    value["renderer_receipt_sha256"] = _sha256(renderer_receipt) if renderer_receipt.is_file() else None
    value["valid"] = valid
    value["status"] = "verified" if valid else "censored_invalid"
    _probe_write_json(receipt_path, value)
    if not valid:
        raise RuntimeError(f"native renderer probe failed; evidence preserved at {receipt_path}")
    return value


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
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--renderer-sha256")
    parser.add_argument("--assets-sha256")
    parser.add_argument("--render-probe", action="store_true")
    parser.add_argument("--sandbox-only", action="store_true")
    parser.add_argument("--capability-only", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("PAINTER_RUN_ROOT", str(args.run_root.resolve()))
    if args.render_probe:
        if args.renderer is None or not args.renderer_sha256:
            raise SystemExit("--render-probe requires --renderer and --renderer-sha256")
        render_probe(
            run_root=args.run_root,
            renderer=args.renderer,
            renderer_python=args.renderer_python,
            browsers=args.browsers,
            prime_venv=args.prime_venv,
            expected_renderer_sha256=args.renderer_sha256,
            expected_assets_sha256=args.assets_sha256,
        )
        return 0
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
