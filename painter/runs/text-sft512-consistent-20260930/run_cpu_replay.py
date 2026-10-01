#!/usr/bin/env python3
"""Run a bounded, hash-locked Linux replay of saved painting programs.

The runner is deliberately separate from model evaluation.  It invokes only
the pinned local renderer, with seed 0 and a 180-second render deadline.  It
does not contact an API, start a browser on the calling host, or overwrite a
previous replay archive.  It accepts ordinary target-hash replays and the
explicit ``first_turn_recovery`` input schema, whose results have no target
comparison.  Run it on the Linux node as root; each renderer process is
dropped to the configured unprivileged user.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable


SCHEMA = "text-sft512-cpu-replay-results-v1"
INPUT_SCHEMA = "text-sft512-cpu-replay-inputs-v1"
RECOVERY_INPUT_SCHEMA = "text-paint-first-turn-recovery-inputs-v1"
SEED = 0
RENDER_TIMEOUT_SECONDS = 180
MIN_WORKERS = 2
MAX_WORKERS = 4
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ReplayError(RuntimeError):
    """A replay preflight or artifact validation failure."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value.lower()):
        raise ReplayError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _safe_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ReplayError(f"{label} must be a relative path")
    candidate = root / relative
    if candidate.is_symlink():
        raise ReplayError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReplayError(f"{label} escapes the replay input root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is missing")
    return candidate


def _renderer_identity(renderer: Path) -> tuple[str, dict[str, str]]:
    if renderer.is_symlink() or not renderer.is_file():
        raise ReplayError("pinned renderer source is missing or symlinked")
    assets_dir = renderer.parent / "assets"
    actual = {
        path.name: sha256(path)
        for path in sorted(assets_dir.glob("*.js"))
        if path.is_file() and not path.is_symlink()
    }
    if not actual:
        raise ReplayError("pinned renderer has no local JavaScript assets")
    return sha256(renderer), actual


def load_replay_inputs(manifest_path: Path, renderer: Path) -> dict[str, Any]:
    """Validate every portable input hash before any renderer is launched."""

    raw_manifest_path = manifest_path.expanduser()
    if raw_manifest_path.is_symlink() or not raw_manifest_path.is_file():
        raise ReplayError("replay input manifest is missing or symlinked")
    # Resolve the containing temporary/archive root after checking the manifest
    # entry itself. macOS commonly exposes /tmp through a symlink.
    manifest_path = raw_manifest_path.resolve()
    root = manifest_path.parent
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayError("replay input manifest is not valid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema") not in {INPUT_SCHEMA, RECOVERY_INPUT_SCHEMA}:
        raise ReplayError("unexpected replay input manifest schema")
    recovery_mode = raw.get("schema") == RECOVERY_INPUT_SCHEMA
    if recovery_mode and raw.get("replay_mode") != "first_turn_recovery":
        raise ReplayError("first-turn recovery input manifest has the wrong replay mode")
    if raw.get("seed") != SEED:
        raise ReplayError("replay input seed is not the pinned seed 0")
    expected_renderer = raw.get("renderer")
    if not isinstance(expected_renderer, dict):
        raise ReplayError("replay input renderer identity is missing")
    raw_renderer = renderer.expanduser()
    if raw_renderer.is_symlink():
        raise ReplayError("pinned renderer source is missing or symlinked")
    renderer_sha, actual_assets = _renderer_identity(raw_renderer.absolute())
    if renderer_sha != _hash_value(expected_renderer.get("renderer_sha256"), "renderer_sha256"):
        raise ReplayError("pinned renderer source hash mismatch")
    expected_assets = expected_renderer.get("assets_sha256")
    if not isinstance(expected_assets, dict) or not expected_assets:
        raise ReplayError("pinned renderer asset hashes are missing")
    normalized_assets = {str(name): _hash_value(value, f"asset {name}") for name, value in expected_assets.items()}
    if actual_assets != normalized_assets:
        raise ReplayError("pinned renderer asset hash set mismatch")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ReplayError("replay input manifest has no cases")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ReplayError("replay case is not an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not SAFE_ID.fullmatch(case_id) or case_id in seen:
            raise ReplayError("replay case id is missing, unsafe, or duplicated")
        seen.add(case_id)
        source = _safe_file(root, case.get("source_file"), f"source for {case_id}")
        expected_source = _hash_value(case.get("source_sha256"), f"source_sha256 for {case_id}")
        actual_source = sha256(source)
        if actual_source != expected_source:
            raise ReplayError(f"source hash mismatch for {case_id}")
        target_value = case.get("target_png_sha256")
        if target_value is None and recovery_mode:
            target = None
        else:
            target = _hash_value(target_value, f"target_png_sha256 for {case_id}")
        if case.get("seed") != SEED:
            raise ReplayError(f"seed mismatch for {case_id}")
        normalized.append({
            "id": case_id,
            "source": source,
            "source_relative": source.relative_to(root).as_posix(),
            "source_sha256": actual_source,
            "target_png_sha256": target,
            "family": case.get("family") if isinstance(case.get("family"), str) else None,
            "source_turn": case.get("source_turn"),
            "original_receipt_sha256": case.get("original_receipt_sha256"),
        })
    return {
        "path": manifest_path,
        "root": root,
        "manifest_sha256": sha256(manifest_path),
        "schema": raw["schema"],
        "replay_mode": "first_turn_recovery" if recovery_mode else "target_replay",
        "protocol_version": raw.get("protocol_version"),
        "seed": SEED,
        "renderer_sha256": renderer_sha,
        "assets_sha256": normalized_assets,
        "cases": normalized,
    }


def build_command(
    *,
    source: Path,
    output: Path,
    renderer: Path,
    renderer_python: Path,
    run_as_user: str | None,
) -> list[str]:
    command = [
        str(renderer_python), str(renderer), str(source),
        "--output", str(output), "--seed", str(SEED),
        "--timeout", str(RENDER_TIMEOUT_SECONDS), "--backend", "swiftshader",
    ]
    if run_as_user:
        if not SAFE_ID.fullmatch(run_as_user):
            raise ReplayError("run-as-user must be a simple local account name")
        command = ["/usr/sbin/runuser", "-u", run_as_user, "--", *command]
    return command


def invoke_renderer(command: list[str], *, browser_path: Path | None = None) -> tuple[int | None, str | None, float]:
    """Invoke one renderer process with a hard outer bound and no log capture."""

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
    }
    if browser_path is not None:
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_path.expanduser().resolve())
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        return None, f"runner_{type(exc).__name__}", time.monotonic() - started
    try:
        returncode = process.wait(timeout=RENDER_TIMEOUT_SECONDS + 30)
        return returncode, None, time.monotonic() - started
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return None, "runner_timeout", time.monotonic() - started


def _portable_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Keep renderer evidence while dropping absolute/private fields."""

    allowed = (
        "valid", "finished", "timed_out", "error_code", "errors", "seed",
        "canvas_contract", "source_sha256", "png_sha256", "renderer_sha256",
        "chromium_version",
        "assets", "requested_angle_backend", "telemetry_enabled",
        "network_enabled", "browser_sandbox_enabled", "elapsed_seconds",
        "painting_seconds", "phase_seconds", "dimensions", "capture_method",
        "blocked_requests",
    )
    result: dict[str, Any] = {}
    for key in allowed:
        if key not in receipt:
            continue
        value = receipt[key]
        if isinstance(value, (str, bool, int, float)) or value is None:
            result[key] = value
        elif isinstance(value, list):
            result[key] = [item for item in value[:32] if isinstance(item, (str, int, float, bool))]
        elif isinstance(value, dict):
            result[key] = {str(k): v for k, v in list(value.items())[:32] if isinstance(v, (str, int, float, bool, type(None)))}
    return result


def _read_renderer_receipt(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if path.is_symlink():
        return None, "symlinked_renderer_receipt"
    if not path.is_file():
        return None, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        portable = {"valid": False, "error_code": "malformed_renderer_receipt"}
        path.write_text(json.dumps(portable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return portable, "malformed_renderer_receipt"
    if not isinstance(raw, dict):
        portable = {"valid": False, "error_code": "malformed_renderer_receipt"}
        path.write_text(json.dumps(portable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return portable, "malformed_renderer_receipt"
    portable = _portable_receipt(raw)
    path.write_text(json.dumps(portable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return portable, None


def _case_result(
    case: dict[str, Any],
    case_dir: Path,
    returncode: int | None,
    runner_error: str | None,
    elapsed: float,
) -> dict[str, Any]:
    png = case_dir / "painting.png"
    receipt_path = case_dir / "painting.json"
    receipt, receipt_error = _read_renderer_receipt(receipt_path)
    png_hash = sha256(png) if png.is_file() and not png.is_symlink() else None
    valid = receipt is not None and receipt.get("valid") is True
    receipt_hash = receipt.get("png_sha256") if receipt else None
    reasons: list[str] = []
    if returncode != 0:
        reasons.append("renderer_nonzero_exit")
    if runner_error:
        reasons.append(runner_error)
    if receipt_error:
        reasons.append(receipt_error)
    if not valid:
        reasons.append("renderer_invalid")
    if png_hash is None:
        reasons.append("missing_png")
    if valid and receipt_hash != png_hash:
        reasons.append("renderer_png_hash_mismatch")
    comparison = "unavailable"
    status = "failed"
    if not reasons:
        if case["target_png_sha256"] is None:
            comparison = "not-applicable"
            status = "replayed"
        else:
            comparison = "recovered-identical" if png_hash == case["target_png_sha256"] else "new-replay"
            status = "verified"
    replay_receipt = {
        "schema": "text-sft512-cpu-replay-receipt-v1",
        "id": case["id"],
        "seed": SEED,
        "timeout_seconds": RENDER_TIMEOUT_SECONDS,
        "source_sha256": case["source_sha256"],
        "target_png_sha256": case["target_png_sha256"],
        "replay_mode": "first_turn_recovery" if case["target_png_sha256"] is None else "target_replay",
        "original_receipt_sha256": case.get("original_receipt_sha256"),
        "replay_png_sha256": png_hash,
        "status": status,
        "comparison": comparison,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "failure_reasons": reasons,
    }
    replay_receipt_path = case_dir / "replay-receipt.json"
    replay_receipt_path.write_text(json.dumps(replay_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "id": case["id"],
        "family": case.get("family"),
        "source_turn": case.get("source_turn"),
        "source_sha256": case["source_sha256"],
        "target_png_sha256": case["target_png_sha256"],
        "replay_mode": "first_turn_recovery" if case["target_png_sha256"] is None else "target_replay",
        "original_receipt_sha256": case.get("original_receipt_sha256"),
        "replay_png_sha256": png_hash,
        "comparison": comparison,
        "status": status,
        "elapsed_seconds": round(elapsed, 3),
        "failure_reasons": reasons,
        "program_file": f"cases/{case['id']}/program.js",
        "png_file": f"cases/{case['id']}/painting.png" if png_hash else None,
        "renderer_receipt_file": (
            f"cases/{case['id']}/painting.json"
            if receipt_path.is_file() and not receipt_path.is_symlink()
            else None
        ),
        "replay_receipt_file": f"cases/{case['id']}/replay-receipt.json",
    }


def _prepare_case_dir(case_dir: Path, run_as_user: str | None) -> None:
    """Make a root-created case directory writable by the renderer account."""

    if not run_as_user:
        return
    try:
        account = pwd.getpwnam(run_as_user)
    except KeyError as exc:
        raise ReplayError(f"renderer account {run_as_user!r} does not exist") from exc
    current_uid = os.geteuid()
    if current_uid not in (0, account.pw_uid):
        raise ReplayError("run-as-user requires root or the same local account")
    if current_uid == 0:
        os.chown(case_dir, account.pw_uid, account.pw_gid)
    # The parent archive and cases directory are created with traversal mode in
    # run_replay; keep case directories traversable after ownership changes.
    case_dir.chmod(0o755)


def run_replay(
    *,
    manifest_path: Path,
    renderer: Path,
    renderer_python: Path,
    output_dir: Path,
    workers: int = MIN_WORKERS,
    browser_path: Path | None = None,
    run_as_user: str | None = "painter",
    platform_name: str | None = None,
    invoke: Callable[[list[str], Path | None], tuple[int | None, str | None, float]] | None = None,
) -> dict[str, Any]:
    platform_name = platform_name or sys.platform
    if platform_name != "linux":
        raise ReplayError("CPU replay renders only on Linux; refusing to render on this host")
    if isinstance(workers, bool) or not isinstance(workers, int) or not MIN_WORKERS <= workers <= MAX_WORKERS:
        raise ReplayError(f"workers must be between {MIN_WORKERS} and {MAX_WORKERS}")
    inputs = load_replay_inputs(manifest_path, renderer)
    renderer = renderer.expanduser().absolute()
    if renderer.is_symlink():
        raise ReplayError("pinned renderer source is missing or symlinked")
    # A venv's bin/python is commonly a symlink. Resolving it here would bypass
    # that venv's site-packages and browser dependencies.
    renderer_python = renderer_python.expanduser().absolute()
    if not renderer_python.is_file():
        raise ReplayError("renderer Python executable is missing")
    raw_output_dir = output_dir.expanduser()
    if raw_output_dir.is_symlink():
        raise ReplayError("replay output directory must not be a symlink")
    output_dir = raw_output_dir.absolute()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise ReplayError("refusing to overwrite an existing replay archive")
    else:
        output_dir.mkdir(parents=True)
    output_dir.chmod(0o755)
    cases_root = output_dir / "cases"
    cases_root.mkdir()
    cases_root.chmod(0o755)
    invoke_fn = invoke or (lambda command, browser: invoke_renderer(command, browser_path=browser))
    results: dict[str, dict[str, Any]] = {}

    def one(case: dict[str, Any]) -> dict[str, Any]:
        case_dir = cases_root / case["id"]
        case_dir.mkdir()
        _prepare_case_dir(case_dir, run_as_user)
        shutil.copyfile(case["source"], case_dir / "program.js")
        command = build_command(
            source=case_dir / "program.js",
            output=case_dir / "painting.png",
            renderer=renderer,
            renderer_python=renderer_python,
            run_as_user=run_as_user,
        )
        returncode, runner_error, elapsed = invoke_fn(command, browser_path)
        return _case_result(case, case_dir, returncode, runner_error, elapsed)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, case): case["id"] for case in inputs["cases"]}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                results[case_id] = future.result()
            except Exception as exc:
                # Preserve a portable failure receipt even for runner-side
                # exceptions; the other cases continue to completion.
                case = next(row for row in inputs["cases"] if row["id"] == case_id)
                case_dir = cases_root / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                results[case_id] = _case_result(case, case_dir, None, f"runner_{type(exc).__name__}", 0.0)
            print(
                json.dumps(
                    {
                        "case": case_id,
                        "status": results[case_id]["status"],
                        "comparison": results[case_id]["comparison"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    ordered = [results[case["id"]] for case in inputs["cases"]]
    summary = {
        "cases": len(ordered),
        "verified": sum(row["status"] == "verified" for row in ordered),
        "replayed": sum(row["status"] == "replayed" for row in ordered),
        "failed": sum(row["status"] == "failed" for row in ordered),
        "recovered_identical": sum(row["comparison"] == "recovered-identical" for row in ordered),
        "new_replay": sum(row["comparison"] == "new-replay" for row in ordered),
        "not_applicable_comparison": sum(row["comparison"] == "not-applicable" for row in ordered),
        "target_comparison_denominator": (
            "none; first-turn recovery inputs have no target PNG hashes"
            if inputs["replay_mode"] == "first_turn_recovery"
            else "verified replay PNGs only"
        ),
    }
    report = {
        "schema": SCHEMA,
        "input_manifest_sha256": inputs["manifest_sha256"],
        "protocol_version": inputs["protocol_version"],
        "seed": SEED,
        "timeout_seconds": RENDER_TIMEOUT_SECONDS,
        "workers": workers,
        "renderer": {
            "renderer_sha256": inputs["renderer_sha256"],
            "assets_sha256": inputs["assets_sha256"],
        },
        "replay_mode": inputs["replay_mode"],
        "summary": summary,
        "cases": ordered,
        "limitations": [
            (
                "This is a first-turn recovery replay of saved failed programs; no target PNG hashes were supplied."
                if inputs["replay_mode"] == "first_turn_recovery"
                else "Target canvases were absent locally; comparison uses recorded target PNG hashes."
            ),
            "No model, API, or remote service was contacted; the pinned local Chromium renderer executes each replay.",
            "Failed renderer receipts are retained per case and are excluded from comparison counts.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# CPU replay report",
        "",
        f"Mode: {inputs['replay_mode']} | Cases: {summary['cases']} | verified: {summary['verified']} | replayed: {summary['replayed']} | failed: {summary['failed']}",
        f"Recovered-identical: {summary['recovered_identical']} | new replay: {summary['new_replay']}",
        "",
        (
            "No target PNG hashes were supplied; this replay is not a three-turn evaluation."
            if inputs["replay_mode"] == "first_turn_recovery"
            else "Comparison uses recorded target PNG hashes; target PNG files were not available locally."
        ),
        "",
        "| Case | Status | Comparison | Replay PNG SHA-256 |",
        "|---|---|---|---|",
    ]
    for row in ordered:
        lines.append(f"| {row['id']} | {row['status']} | {row['comparison']} | {row['replay_png_sha256'] or ''} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-path", type=Path)
    parser.add_argument("--run-as-user", default="painter")
    parser.add_argument("--workers", type=int, default=MIN_WORKERS)
    args = parser.parse_args(argv)
    report = run_replay(
        manifest_path=args.manifest,
        renderer=args.renderer,
        renderer_python=args.renderer_python,
        output_dir=args.output,
        workers=args.workers,
        browser_path=args.browser_path,
        run_as_user=args.run_as_user,
    )
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
