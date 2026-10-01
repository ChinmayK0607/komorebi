#!/usr/bin/env python3
"""Bounded Linux profiling for saved first-turn painting programs.

Each instrumented case runs in a separate process group.  The worker preserves
the frozen renderer's automatic p5 startup HTML and emits brush start/end
events through Playwright console events.  The parent owns the hard deadline,
kills the complete worker/browser group on timeout, and writes a partial
receipt from the journal.  ``--mode both`` also runs the untouched renderer on
two valid controls and compares PNG hashes with the instrumented baseline.

This path is explicitly opt-in with ``--execute`` and refuses non-Linux hosts.
It does not generate model programs, call a judge, or access the network.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "watercolour-render-profile-v2"
PROFILE_VERSION = "original-auto-startup-process-watchdog-v1"
DEFAULT_CASES = ("t600-309", "t600-323", "t600-342", "t600-542", "t600-352")
DEFAULT_CONTROL_CASES = ("t600-342", "t600-542")
CANVAS = (600, 600)
WORKER = Path(__file__).with_name("profile_worker.py")


class ProfileError(ValueError):
    """Raised for an unsafe or mismatched profile request."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PROFILE_RUNNER_SHA256 = sha256(Path(__file__))


def resource_snapshot() -> dict[str, Any]:
    """Capture non-secret run-level CPU/runtime metadata for auditability."""

    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            affinity = None
    cgroup: dict[str, str] = {}
    for name in ("cpu.max", "cpu.weight", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        try:
            cgroup[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            cgroup[name] = "unavailable"
    return {
        "platform": sys.platform,
        "platform_release": os.uname().release if hasattr(os, "uname") else "unavailable",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "sched_getaffinity": affinity,
        "cgroup": cgroup,
    }


def static_brush_metrics(source: str) -> dict[str, Any]:
    """Return source-only call/geometry descriptors, never runtime claims."""

    calls = Counter(
        match.group(1)
        for match in re.finditer(r"\bbrush\s*\.\s*([A-Za-z_$][\w$]*)\s*\(", source)
    )
    starts = [match.end() for match in re.finditer(r"\bbrush\s*\.\s*polygon\s*\(", source)]
    literal_count = unresolved_count = point_count = 0
    area = bbox_area = 0.0
    for start in starts:
        argument = _balanced_array(source, start)
        if argument is None:
            unresolved_count += 1
            continue
        try:
            points = ast.literal_eval(argument)
        except (SyntaxError, ValueError):
            unresolved_count += 1
            continue
        if not _numeric_points(points):
            unresolved_count += 1
            continue
        literal_count += 1
        point_count += len(points)
        area += _polygon_area(points)
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        bbox_area += (max(xs) - min(xs)) * (max(ys) - min(ys))
    return {
        "kind": "static_source_counts",
        "source_bytes": len(source.encode("utf-8")),
        "source_lines": source.count("\n") + 1,
        "call_counts": dict(sorted(calls.items())),
        "fill_calls": sum(calls[name] for name in ("fill", "fillBleed", "fillTexture")),
        "stroke_calls": sum(calls[name] for name in ("line", "stroke", "flowLine")),
        "polygon_calls": calls["polygon"],
        "polygon_helper_calls": len(re.findall(r"\b(?:poly|disc)\s*\(", source)),
        "loop_keywords": len(re.findall(r"\b(?:for|while|do)\b", source)),
        "literal_polygon_count": literal_count,
        "unresolved_polygon_count": unresolved_count,
        "literal_point_count": point_count,
        "literal_polygon_area_px2": round(area, 3),
        "literal_polygon_bbox_area_px2": round(bbox_area, 3),
    }


def _balanced_array(source: str, start: int) -> str | None:
    index = start
    while index < len(source) and source[index].isspace():
        index += 1
    if index >= len(source) or source[index] != "[":
        return None
    opening, depth, quote, escaped = index, 0, None, False
    while index < len(source):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
        index += 1
    return None


def _numeric_points(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 3
        and all(
            isinstance(point, list)
            and len(point) >= 2
            and all(isinstance(number, (int, float)) for number in point[:2])
            for point in value
        )
    )


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(
        sum(
            float(points[index][0]) * float(points[(index + 1) % len(points)][1])
            - float(points[(index + 1) % len(points)][0]) * float(points[index][1])
            for index in range(len(points))
        )
        / 2.0
    )


def load_profile_inputs(manifest_path: Path, runtime_root: Path, case_ids: Iterable[str]) -> list[dict[str, Any]]:
    manifest_path, runtime_root = Path(manifest_path), Path(runtime_root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"Cannot read replay manifest: {error}") from error
    if manifest.get("schema") != "text-paint-first-turn-recovery-inputs-v1":
        raise ProfileError("Profile requires the first-turn recovery input manifest")
    renderer = manifest.get("renderer") or {}
    renderer_path = runtime_root / "renderer.py"
    expected_renderer = renderer.get("renderer_sha256")
    if not renderer_path.is_file() or not expected_renderer or sha256(renderer_path) != expected_renderer:
        raise ProfileError("Runtime renderer hash does not match the saved replay identity")
    for name, expected in sorted((renderer.get("assets_sha256") or {}).items()):
        asset = runtime_root / "assets" / name
        if not asset.is_file() or sha256(asset) != expected:
            raise ProfileError(f"Runtime asset hash mismatch for {name}")
    wanted = list(dict.fromkeys(case_ids))
    by_id = {str(case["id"]): case for case in manifest.get("cases", [])}
    missing = [case_id for case_id in wanted if case_id not in by_id]
    if missing:
        raise ProfileError("Unknown profile case(s): " + ", ".join(missing))
    programs = manifest_path.parent / "programs"
    records = []
    for case_id in wanted:
        case = by_id[case_id]
        source = programs / f"{case_id}.js"
        if not source.is_file() or (case.get("source_sha256") and sha256(source) != case["source_sha256"]):
            raise ProfileError(f"Saved program hash mismatch or missing for {case_id}")
        records.append({
            "id": case_id,
            "source": source,
            "source_sha256": sha256(source),
            "seed": int(case.get("seed", manifest.get("seed", 0))),
            "historical_receipt": case.get("original_receipt_sha256"),
        })
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def journal_summary(path: Path) -> dict[str, Any]:
    """Summarize durable events, including a call whose end was never seen."""

    events: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    completed: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)
        kind = event.get("event")
        if kind == "brush_start":
            active = event
        elif kind == "brush_end":
            completed[str(event.get("name"))] += 1
            durations[str(event.get("name"))] += float(event.get("duration_ms", 0.0))
            if active and active.get("id") == event.get("id"):
                active = None
    return {
        "event_count": len(events),
        "brush_completed_counts": dict(sorted(completed.items())),
        "brush_duration_ms": {name: round(value, 6) for name, value in sorted(durations.items())},
        "active_brush_call_at_last_event": active,
        "last_event": events[-1] if events else None,
        "events": events,
    }


def _minimal_env(scratch: Path, browsers_path: Path | None) -> dict[str, str]:
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(scratch), "TMPDIR": str(scratch), "LANG": "C.UTF-8"}
    if browsers_path:
        environment["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path.absolute())
    elif os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        environment["PLAYWRIGHT_BROWSERS_PATH"] = str(Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]).absolute())
    return environment


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_watchdog(command: Sequence[str], receipt: Path, journal: Path, log: Path, timeout: float, env: Mapping[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        process = subprocess.Popen(list(command), stdout=stream, stderr=subprocess.STDOUT, env=dict(env), start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # The renderer/worker normally exits before this call.  Killing
            # the owned process group anyway prevents a descendant Chromium
            # process from surviving a failed or partially failed worker.
            _kill_group(process)
    elapsed = time.monotonic() - started
    summary = journal_summary(journal)
    if receipt.is_file():
        try:
            result = json.loads(receipt.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
    else:
        result = {}
    result.update({"watchdog_elapsed_seconds": round(elapsed, 6), "watchdog_timed_out": timed_out, "returncode": process.returncode, "journal": summary})
    if timed_out:
        result.update({"valid": False, "finished": False, "timed_out": True, "error_code": "profile_timeout", "error": f"External profile deadline exceeded ({timeout}s)"})
    elif not receipt.is_file():
        result.update({"valid": False, "finished": False, "error_code": "profile_worker_failure", "error": f"Worker exited {process.returncode} without a receipt"})
    _write_json(receipt, result)
    return result


def run_profile_case(record: Mapping[str, Any], runtime: Path, output_root: Path, timeout: float, browsers_path: Path | None) -> dict[str, Any]:
    case_id = str(record["id"])
    output = output_root / f"{case_id}.png"
    receipt = output_root / f"{case_id}.json"
    journal = output_root / f"{case_id}.journal.jsonl"
    log = output_root / f"{case_id}.worker.log"
    if output.exists() or receipt.exists() or journal.exists():
        raise ProfileError(f"Refusing to overwrite profile output for {case_id}")
    with tempfile.TemporaryDirectory(prefix=f"profile-{case_id}-") as scratch:
        command = [sys.executable, str(WORKER), "--runtime-root", str(runtime), "--source", str(record["source"]), "--output", str(output), "--receipt", str(receipt), "--journal", str(journal), "--seed", str(record["seed"]), "--timeout", str(timeout), "--backend", "swiftshader"]
        result = run_watchdog(command, receipt, journal, log, timeout, _minimal_env(Path(scratch), browsers_path))
    result.update({"schema": SCHEMA, "profile_version": PROFILE_VERSION, "profile_runner_sha256": PROFILE_RUNNER_SHA256, "id": case_id, "source_sha256": record["source_sha256"], "seed": record["seed"], "original_receipt_sha256": record.get("historical_receipt"), "brush": static_brush_metrics(Path(record["source"]).read_text(encoding="utf-8")), "renderer_sha256": sha256(runtime / "renderer.py"), "assets_sha256": {name: sha256(runtime / "assets" / name) for name in ("p5.min.js", "p5.brush.js")}, "network_enabled": False, "canvas_contract": list(CANVAS)})
    _write_json(receipt, result)
    return result


def run_profile(records: Sequence[Mapping[str, Any]], runtime: Path, output_root: Path, timeout: float, browsers_path: Path | None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": SCHEMA, "profile_version": PROFILE_VERSION, "profile_runner_sha256": PROFILE_RUNNER_SHA256, "mode": "instrumented_baseline", "status": "running", "case_ids": [str(record["id"]) for record in records], "resource_snapshot": resource_snapshot(), "renderer_sha256": sha256(runtime / "renderer.py"), "assets_sha256": {name: sha256(runtime / "assets" / name) for name in ("p5.min.js", "p5.brush.js")}, "results": []}
    _write_json(output_root / "profile.json", report)
    try:
        for record in records:
            result = run_profile_case(record, runtime, output_root, timeout, browsers_path)
            report["results"].append(result)
            _write_json(output_root / "profile.json", report)
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        _write_json(output_root / "profile.json", report)
        raise
    report.update({"status": "complete", "valid_count": sum(bool(result.get("valid")) for result in report["results"]), "timed_out_count": sum(bool(result.get("timed_out")) for result in report["results"])})
    _write_json(output_root / "profile.json", report)
    return report


def control_command(record: Mapping[str, Any], runtime: Path, renderer_python: Path, output: Path, timeout: float) -> list[str]:
    # Invoke the frozen worker directly.  Calling the renderer's parent mode
    # would create a second process group outside this watchdog's ownership.
    return [str(renderer_python), str(runtime / "renderer.py"), "--worker", str(record["source"]), "--output", str(output), "--seed", str(record["seed"]), "--timeout", str(timeout), "--backend", "swiftshader"]


def run_controls(records: Sequence[Mapping[str, Any]], runtime: Path, renderer_python: Path, output_root: Path, timeout: float, browsers_path: Path | None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": SCHEMA, "profile_version": PROFILE_VERSION, "mode": "direct_frozen_worker_control", "status": "running", "case_ids": [str(record["id"]) for record in records], "resource_snapshot": resource_snapshot(), "results": []}
    _write_json(output_root / "control.json", report)
    try:
        for record in records:
            case_id = str(record["id"])
            output = output_root / f"{case_id}.png"
            receipt = output.with_suffix(".json")
            journal = output_root / f"{case_id}.control.journal.jsonl"
            log = output_root / f"{case_id}.control.worker.log"
            if output.exists() or receipt.exists():
                raise ProfileError(f"Refusing to overwrite control output for {case_id}")
            with tempfile.TemporaryDirectory(prefix=f"control-{case_id}-") as scratch:
                result = run_watchdog(control_command(record, runtime, renderer_python, output, timeout), receipt, journal, log, timeout, _minimal_env(Path(scratch), browsers_path))
            if output.is_file():
                result["png_sha256"] = sha256(output)
            result.update({"id": case_id, "control_mode": "direct_frozen_worker", "source_sha256": record["source_sha256"], "seed": record["seed"], "renderer_sha256": sha256(runtime / "renderer.py"), "assets_sha256": {name: sha256(runtime / "assets" / name) for name in ("p5.min.js", "p5.brush.js")}, "resource_snapshot": resource_snapshot()})
            _write_json(receipt, result)
            report["results"].append(result)
            _write_json(output_root / "control.json", report)
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        _write_json(output_root / "control.json", report)
        raise
    report["status"] = "complete"
    _write_json(output_root / "control.json", report)
    return report


def compare_control(profile: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    controls = {str(row["id"]): row for row in control.get("results", [])}
    rows = []
    for row in profile.get("results", []):
        case_id = str(row["id"])
        reference = controls.get(case_id, {})
        rows.append({"id": case_id, "profile_valid": bool(row.get("valid")), "control_valid": bool(reference.get("valid")), "png_sha_equal": bool(row.get("png_sha256") and reference.get("png_sha256") and row["png_sha256"] == reference["png_sha256"]), "profile_elapsed_seconds": row.get("watchdog_elapsed_seconds"), "control_elapsed_seconds": reference.get("watchdog_elapsed_seconds")})
    return {"schema": "watercolour-render-profile-comparison-v2", "same_renderer_identity": profile.get("renderer_sha256") == control.get("renderer_sha256"), "cases": rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--control-case", dest="control_cases", action="append")
    parser.add_argument("--mode", choices=["baseline", "control", "both", "pooled"], default="both")
    parser.add_argument("--renderer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--browsers-path", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize Linux browser execution")
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("The profile runner is Linux-only; use Codex Cloud or the Linux node")
    if args.mode == "pooled":
        parser.error("pooled mode is deferred until it has an external per-case watchdog")
    if not args.execute:
        parser.error("Pass --execute to opt in to the finite browser profile")
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be between 1 and 300 seconds")
    cases = args.cases or list(DEFAULT_CASES)
    control_cases = args.control_cases or list(DEFAULT_CONTROL_CASES)
    try:
        records = load_profile_inputs(args.manifest, args.runtime_root, cases)
        control_records = load_profile_inputs(args.manifest, args.runtime_root, control_cases)
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {"schema": SCHEMA, "profile_version": PROFILE_VERSION, "profile_runner_sha256": PROFILE_RUNNER_SHA256, "manifest": str(args.manifest), "runtime_root": str(args.runtime_root), "case_ids": cases, "control_case_ids": control_cases, "network_enabled": False, "execution_host": "linux-required", "resource_snapshot": resource_snapshot()}
        if args.mode in {"baseline", "both"}:
            summary["baseline"] = run_profile(records, args.runtime_root, args.output_root / "baseline", args.timeout, args.browsers_path)
        if args.mode in {"control", "both"}:
            summary["control"] = run_controls(control_records, args.runtime_root, args.renderer_python, args.output_root / "control", args.timeout, args.browsers_path)
        if args.mode == "both":
            summary["comparison"] = compare_control(summary["baseline"], summary["control"])
        summary["status"] = "complete"
        _write_json(args.output_root / "profile-summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ProfileError as error:
        parser.error(str(error))
    except Exception as error:
        parser.error(f"profile failed: {type(error).__name__}: {error}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
