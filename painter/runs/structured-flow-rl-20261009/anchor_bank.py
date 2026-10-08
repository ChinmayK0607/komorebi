#!/usr/bin/env python3
"""Prepare lazy pilot rows and materialize their real frozen SFT512 anchors.

The first node preflight needs a manifest before the owned inference server is
started, while the actual frozen opponent PNGs can only be generated after
that server has loaded ``sft-step512``.  This module bridges those two phases:
``prepare_lazy`` writes an explicit, non-admitted manifest; ``materialize``
generates 32 train and 16 heldout anchors through the existing native frozen
opponent protocol, writes the required provenance, and atomically replaces the
lazy rows with the verified pilot manifests.

It never fabricates a PNG or treats a renderer/model failure as an anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import tempfile
import time
from typing import Any, Mapping


RUN = Path(__file__).resolve().parent
CURRICULUM = RUN / "prepare_curriculum.py"
_spec = importlib.util.spec_from_file_location("structured_flow_curriculum", CURRICULUM)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"curriculum module is unavailable: {CURRICULUM}")
curriculum = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = curriculum
_spec.loader.exec_module(curriculum)

CHECKPOINT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
ALIAS = "sft-step512"
PROTOCOL = "native-painting-rl-v2"
RENDER_PROTOCOL = "native-painting-v1"
GENERATION_PROTOCOL = "structured-flow-anchor-generation-v2"
RESUME_PROTOCOL = "structured-flow-anchor-resume-v1"
DECODING_CONFIG = "{}"
MAX_TURNS = 3
MAX_COMPLETION_TOKENS = 8192
CONTEXT_LENGTH = 32768
MAX_GENERATION_WORKERS = 8
MAX_ANCHOR_RESAMPLES = 1


def _browser_root(run_root: Path) -> Path:
    """Resolve the verified browser installation for native evaluation."""
    configured = (
        os.environ.get("PAINTER_BROWSER_ROOT")
        or os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        or str(Path(run_root) / "browsers")
    )
    path = Path(configured).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"native evaluation browser root is missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_anchor_seed(source_group: str, reference_sha256: str, retry_index: int = 0) -> int:
    if not isinstance(retry_index, int) or retry_index < 0:
        raise ValueError("retry_index must be a non-negative integer")
    material = (
        f"20261009:{source_group}:{reference_sha256}"
        if retry_index == 0
        else f"20261009:retry-{retry_index}:{source_group}:{reference_sha256}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def write_private_log(path: Path, value: str) -> None:
    encoded = str(value or "").encode("utf-8")
    if len(encoded) > 64 * 1024:
        encoded = encoded[:64 * 1024] + b"\n...[truncated]\n"
    path.write_bytes(encoded)
    path.chmod(0o600)


def _read_trace_bundle(attempt: Path) -> tuple[list[Mapping[str, Any]], list[Any], list[Path]] | None:
    """Read a native attempt's trace stream once for all case lookups."""
    try:
        from painter.native_painting_report import read_traces
    except ImportError:
        try:
            from native_painting_report import read_traces  # type: ignore
        except ImportError:
            return None
    return read_traces(attempt)


def _terminal_metadata(
    trace_bundle: tuple[list[Mapping[str, Any]], list[Any], list[Path]] | None,
    case_id: str,
) -> dict[str, Any]:
    """Record the native episode's terminal action without changing its policy."""
    if trace_bundle is None:
        return {"status": "unavailable", "terminal_canvas_policy": "native_current_canvas_at_episode_end"}
    traces, _parse_errors, _trace_files = trace_bundle
    matching_trace: Mapping[str, Any] | None = None
    for trace in traces:
        info = trace.get("info") if isinstance(trace.get("info"), Mapping) else {}
        task = trace.get("task") if isinstance(trace.get("task"), Mapping) else {}
        data = task.get("data") if isinstance(task.get("data"), Mapping) else {}
        identifiers = {
            str(value)
            for value in (
                info.get("task_id"), info.get("case_id"), info.get("source_group"),
                data.get("task_id"), data.get("case_id"), data.get("name"),
            )
            if isinstance(value, str) and value
        }
        if case_id not in identifiers:
            continue
        # Keep the same last-match semantics as _native_eval_canvas when a
        # trace stream contains a retried episode for the same case id.
        matching_trace = trace
    if matching_trace is not None:
        trace = matching_trace
        info = trace.get("info") if isinstance(trace.get("info"), Mapping) else {}
        turns = info.get("turns") if isinstance(info.get("turns"), list) else []
        last = turns[-1] if turns and isinstance(turns[-1], Mapping) else {}
        valid_canvas_turns = [
            turn for turn in turns
            if isinstance(turn, Mapping)
            and bool(turn.get("valid"))
            and (turn.get("canvas") or turn.get("final_canvas"))
        ]
        last_valid = valid_canvas_turns[-1] if valid_canvas_turns else {}
        invalid_turns = [
            {
                "turn": turn.get("turn"),
                "action": turn.get("action"),
                "failure_kind": turn.get("failure_kind"),
                "valid": turn.get("valid"),
            }
            for turn in turns
            if isinstance(turn, Mapping) and turn.get("valid") is False
        ]
        return {
            "status": "verified",
            "terminal_canvas_policy": "native_current_canvas_at_episode_end",
            "final_canvas_valid": bool(info.get("final_canvas_valid")),
            "operational_cutoff": bool(info.get("operational_cutoff")),
            "terminal_canvas_reference": info.get("final_canvas"),
            "last_turn": last.get("turn"),
            "last_action": last.get("action"),
            "last_action_valid": last.get("valid"),
            "last_failure_kind": last.get("failure_kind"),
            "last_valid_canvas_turn": last_valid.get("turn"),
            "last_valid_canvas_reference": last_valid.get("canvas") or last_valid.get("final_canvas"),
            "invalid_turns": invalid_turns,
        }
    return {"status": "not_found", "terminal_canvas_policy": "native_current_canvas_at_episode_end"}


def _trace_workspace_root(
    trace_bundle: tuple[list[Mapping[str, Any]], list[Any], list[Path]] | None,
    attempt: Path,
) -> Path:
    """Map absolute node paths in a collected trace back into its batch root."""
    if trace_bundle is not None:
        traces, _parse_errors, _trace_files = trace_bundle
        for trace in traces:
            info = trace.get("info") if isinstance(trace.get("info"), Mapping) else {}
            candidates: list[Any] = [info.get("final_canvas")]
            turns = info.get("turns") if isinstance(info.get("turns"), list) else []
            candidates.extend(
                turn.get("canvas") or turn.get("final_canvas")
                for turn in turns
                if isinstance(turn, Mapping)
            )
            for value in candidates:
                if not isinstance(value, str) or not value.startswith("/"):
                    continue
                path = Path(value)
                parts = path.parts
                for marker in ("eval-output", "evaluation-rollouts"):
                    if marker not in parts:
                        continue
                    index = parts.index(marker)
                    relative = Path(*parts[index:])
                    if (attempt / relative).is_file():
                        return Path(*parts[:index])
    return Path(os.environ.get("PAINTER_FROZEN_OPPONENT_ROOT") or Path(__file__).resolve().parents[3]).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _proposal_rows(path: Path, split: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != curriculum.SCHEMA or value.get("split") != split:
        raise ValueError(f"{path} is not the structured-flow {split} proposal")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no rows")
    return [dict(row) for row in rows]


def _pilot_rows(train_proposal: Path, heldout_proposal: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = _proposal_rows(train_proposal, "train")
    heldout = _proposal_rows(heldout_proposal, "heldout")
    plan = curriculum.build_pilot_plan(train, heldout)
    train_by_id = {str(row["id"]): row for row in train}
    heldout_by_id = {str(row["id"]): row for row in heldout}
    selected_train = [train_by_id[ident] for ident in plan["train_ids"]]
    selected_heldout = [heldout_by_id[ident] for ident in plan["heldout_ids"]]
    return selected_train, selected_heldout


def _lazy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.update(
        {
            "max_turns": MAX_TURNS,
            "anchor_status": "pending_runtime_generation",
            "admission_status": "waiting_for_node_frozen_anchor",
            "lazy_opponent": {
                "model_alias": ALIAS,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "protocol_version": PROTOCOL,
                "render_protocol": RENDER_PROTOCOL,
                "decoding_config": DECODING_CONFIG,
                "base_seed": 20261009,
                "max_turns": MAX_TURNS,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "context_length": CONTEXT_LENGTH,
                "generator": "structured-flow.anchor_bank:_batch_native_generation",
            },
        }
    )
    value.pop("opponent", None)
    value.pop("opponent_sha256", None)
    return value


def prepare_lazy(*, train_proposal: Path, heldout_proposal: Path, train_output: Path, heldout_output: Path) -> dict[str, Any]:
    train_rows, heldout_rows = _pilot_rows(train_proposal.resolve(), heldout_proposal.resolve())
    train_doc = {
        "schema": "structured-flow-lazy-pilot-manifest-v1",
        "run_id": curriculum.RUN_ID,
        "split": "train",
        "status": "pending_runtime_frozen_anchors",
        "native_rl_compatible": False,
        "rows": [_lazy_row(row) for row in train_rows],
    }
    heldout_cases = [_lazy_row(row) for row in heldout_rows]
    heldout_doc = {
        "schema": "structured-flow-lazy-pilot-manifest-v1",
        "run_id": curriculum.RUN_ID,
        "split": "heldout",
        "status": "pending_runtime_frozen_anchors",
        "native_rl_compatible": False,
        "rows": heldout_cases,
        "cases": heldout_cases,
    }
    _write_json(train_output, train_doc)
    _write_json(heldout_output, heldout_doc)
    return {"status": train_doc["status"], "train_rows": len(train_rows), "heldout_cases": len(heldout_rows)}

def _is_final(doc: Mapping[str, Any], expected: int) -> bool:
    rows = doc.get("rows")
    return (
        isinstance(rows, list)
        and len(rows) == expected
        and all(isinstance(row, dict) and isinstance(row.get("opponent"), str) and row.get("opponent_sha256") for row in rows)
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") or value.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"manifest has no rows/cases: {path}")
    return [dict(row) for row in rows]


def _batch_native_generation(
    rows: list[Mapping[str, Any]],
    *,
    anchor_root: Path,
    receipt_root: Path,
    proposal_root: Path,
    run_root: Path,
    workers: int,
    retry_index: int = 0,
    failed_cases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate all pilot anchors through one native VF evaluator process.

    The generic frozen-opponent helper builds a one-case manifest and defaults
    its reference kind to image. Structured proposals are text briefs, so this
    run-owned bridge writes one explicit text manifest and evaluates the whole
    48-case batch against the already served sft-step512 alias. The evaluator
    owns request/render concurrency; this process only validates and copies its
    hash-bound terminal canvases.
    """
    try:
        from painter.native_painting_eval import config_text
    except ImportError:
        from native_painting_eval import config_text  # type: ignore
    try:
        from painter.frozen_opponent import _native_eval_canvas
    except ImportError:
        from frozen_opponent import _native_eval_canvas  # type: ignore

    renderer = os.environ.get("PAINTER_FROZEN_OPPONENT_RENDERER") or os.environ.get("PAINTER_RENDERER_PATH")
    renderer_python = os.environ.get("PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON") or os.environ.get("PAINTER_RENDERER_PYTHON_PATH")
    endpoint = os.environ.get("PAINTER_FROZEN_OPPONENT_ENDPOINT", "http://127.0.0.1:8100/v1")
    prime_root = Path(os.environ.get("PAINTER_FROZEN_OPPONENT_PRIME_ROOT") or os.environ.get("PRIME_ROOT", run_root / "prime-rl")).resolve()
    eval_bin = Path(os.environ.get("PAINTER_FROZEN_OPPONENT_EVAL") or prime_root / ".venv/bin/eval").resolve()
    workspace_root = Path(os.environ.get("PAINTER_FROZEN_OPPONENT_ROOT") or Path(__file__).resolve().parents[3]).resolve()
    if not renderer or not renderer_python:
        raise RuntimeError("frozen anchor batch renderer paths are not configured")
    renderer_path = Path(renderer).resolve()
    renderer_python_path = Path(renderer_python).absolute()
    browser_root = _browser_root(run_root)
    if not renderer_path.is_file() or not renderer_python_path.is_file():
        raise RuntimeError("frozen anchor batch renderer runtime is missing")
    if not eval_bin.is_file() or not prime_root.is_dir():
        raise RuntimeError("frozen anchor batch native eval runtime is missing")

    if not isinstance(retry_index, int) or not 0 <= retry_index <= MAX_ANCHOR_RESAMPLES:
        raise ValueError(f"retry_index must be in [0,{MAX_ANCHOR_RESAMPLES}]")
    batch_seed = 20261009 + retry_index
    attempt = anchor_root / "native-generation" / f"batch-{os.getpid()}-{time.time_ns()}-retry-{retry_index}"
    attempt.mkdir(parents=True, exist_ok=True)
    manifest = attempt / "native-eval-manifest.json"
    normalized = []
    for row in sorted(rows, key=lambda item: str(item["id"])):
        reference = Path(str(row["reference"]))
        if not reference.is_absolute():
            reference = proposal_root / reference
        reference = reference.resolve()
        if reference.is_symlink() or not reference.is_file():
            raise FileNotFoundError(f"reference prompt is missing: {reference}")
        reference_sha = sha256_file(reference)
        if reference_sha != row.get("reference_sha256"):
            raise ValueError(f"reference prompt hash mismatch: {row.get('id')}")
        normalized.append({
            "id": str(row["id"]),
            "source_group": str(row["source_group"]),
            "family": str(row.get("family", "structured-flow")),
            "reference": str(reference),
            "reference_sha256": reference_sha,
            "reference_kind": "text",
            "max_turns": MAX_TURNS,
            "case_seed": deterministic_anchor_seed(str(row["source_group"]), reference_sha, retry_index),
            "retry_index": retry_index,
            "renderer": str(renderer_path),
            "renderer_python": str(renderer_python_path),
        })
    manifest.write_text(json.dumps({"cases": normalized}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path = attempt / "native-eval.toml"
    generated = config_text(
        taskset_id="native_painting",
        root=attempt,
        manifest=manifest,
        policy_label="frozen-sft512-anchor-batch",
        model=ALIAS,
        output_dir=attempt / "eval-output",
        count=len(normalized),
        max_tokens=MAX_COMPLETION_TOKENS,
        max_turns=MAX_TURNS,
        context_length=CONTEXT_LENGTH,
        max_concurrent=workers,
        renderer=renderer_path,
        renderer_python=renderer_python_path,
    )
    default_endpoint = 'base_url = "http://127.0.0.1:8000/v1"'
    if default_endpoint not in generated:
        raise RuntimeError("native anchor batch config has no local endpoint")
    generated = generated.replace(default_endpoint, f"base_url = {json.dumps(endpoint)}", 1)
    generated = generated.replace("seed = 918200", f"seed = {batch_seed}", 1)
    config_path.write_text(generated, encoding="utf-8")
    environment = os.environ.copy()
    # Frozen SFT anchors are the fixed opponent target. Do not apply the
    # structured-flow policy hook to this 48-case batch; its task IDs are not
    # the 32/16 RL/eval manifest IDs and must remain generic native prompts.
    environment["PAINTER_STRUCTURED_FLOW_POLICY"] = "0"
    environment.pop("PAINTER_PRIME_ROLE", None)
    environment.pop("PAINTER_RL_MANIFEST", None)
    environment.pop("PAINTER_EVAL_MANIFEST", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(workspace_root), str(RUN), environment.get("PYTHONPATH", "")) if item
    )
    # NativePaintingTaskset resolves Playwright's executable from the process
    # environment. Without this explicit binding it derives ``attempt/browsers``
    # beside the temporary native-eval manifest and misses the installed node
    # browser, even though the renderer smoke test already passed.
    environment["PAINTER_BROWSER_ROOT"] = str(browser_root)
    environment["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_root)
    environment["PAINTER_ANCHOR_REQUIRE_NO_LOOP"] = "1"
    environment.setdefault("WANDB_MODE", "disabled")
    command = [str(eval_bin), "@", str(config_path), "--no-serve", "--no-push"]
    try:
        completed = subprocess.run(
            command,
            cwd=prime_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("frozen anchor batch native eval timed out") from exc
    write_private_log(attempt / "native-eval.stdout.log", completed.stdout or "")
    write_private_log(attempt / "native-eval.stderr.log", completed.stderr or "")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native anchor batch failed").strip()[-2000:]
        raise RuntimeError(f"frozen anchor batch native eval failed: {detail}")

    trace_bundle = _read_trace_bundle(attempt)
    provenance: dict[str, dict[str, Any]] = {}
    for row in normalized:
        case_id = row["id"]
        if trace_bundle is None:
            produced = _native_eval_canvas(attempt, case_id, workspace_root)
        else:
            produced = _native_eval_canvas(attempt, case_id, workspace_root, trace_bundle=trace_bundle)
        if isinstance(produced, Mapping):
            status = produced.get("status")
            reason = str(produced.get("reason") or status or "invalid native model output")
            failure = {
                "schema": "structured-flow-anchor-failure-v1",
                "generation_protocol": GENERATION_PROTOCOL,
                "status": str(status or "invalid_model_output"),
                "case_id": case_id,
                "source_group": row["source_group"],
                "reference_sha256": row["reference_sha256"],
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "model_alias": ALIAS,
                "batch_seed": batch_seed,
                "case_seed": row["case_seed"],
                "retry_index": retry_index,
                "batch_manifest": str(manifest.relative_to(anchor_root)),
                "attempt_root": str(attempt.relative_to(anchor_root)),
                "reason": reason,
            }
            _write_json(receipt_root / f"{case_id}.retry-{retry_index}.json", failure)
            if str(status) == "invalid_model_output" and failed_cases is not None:
                failed_cases[case_id] = failure
                continue
            raise RuntimeError(f"frozen anchor generation failed for {case_id}: {reason}")
        source = Path(produced).resolve()
        if source.is_symlink() or not source.is_file() or source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"frozen anchor output is not a regular PNG: {case_id}")
        target = anchor_root / f"{case_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        png_sha = sha256_file(target)
        terminal_metadata = _terminal_metadata(trace_bundle, case_id)
        receipt = {
            "schema": "structured-flow-anchor-generation-v1",
            "generation_protocol": GENERATION_PROTOCOL,
            "status": "valid",
            "case_id": case_id,
            "source_group": row["source_group"],
            "reference_sha256": row["reference_sha256"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_alias": ALIAS,
            "protocol_version": PROTOCOL,
            "render_protocol": RENDER_PROTOCOL,
            "batch_seed": batch_seed,
            "case_seed": row["case_seed"],
            "retry_index": retry_index,
            "seed_binding": "registered_per_case_and_batch",
            "batch_manifest": str(manifest.relative_to(anchor_root)),
            "png_sha256": png_sha,
            "terminal_metadata": terminal_metadata,
        }
        receipt_path = receipt_root / f"{case_id}.json"
        _write_json(receipt_path, receipt)
        provenance[case_id] = {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "reference_sha256": row["reference_sha256"],
            "source_group": row["source_group"],
            "png_sha256": png_sha,
            "valid": True,
            "generation_receipt": str(receipt_path.relative_to(anchor_root)),
            "generation_protocol": GENERATION_PROTOCOL,
            "terminal_metadata": terminal_metadata,
        }
    return provenance


def _resume_existing_batches(
    rows: list[Mapping[str, Any]],
    *,
    anchor_root: Path,
    receipt_root: Path,
    run_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Recover hash-verified finals from earlier paid native batches.

    A failed materialization can leave many completed native traces behind.
    This scans only run-owned batch manifests/configs that still bind the
    current text references, frozen alias, renderer paths, and bounded seed
    protocol. It copies only the trace-reported final PNG after rechecking its
    bytes; a prior canvas or an unbound image glob is never admitted.
    """
    renderer = os.environ.get("PAINTER_FROZEN_OPPONENT_RENDERER") or os.environ.get("PAINTER_RENDERER_PATH")
    renderer_python = os.environ.get("PAINTER_FROZEN_OPPONENT_RENDERER_PYTHON") or os.environ.get("PAINTER_RENDERER_PYTHON_PATH")
    if not renderer or not renderer_python:
        return {}, {}
    renderer_path = Path(renderer).resolve()
    renderer_python_path = Path(renderer_python).absolute()
    try:
        from painter.frozen_opponent import _native_eval_canvas
    except ImportError:
        from frozen_opponent import _native_eval_canvas  # type: ignore
    generation_root = anchor_root / "native-generation"
    manifests = sorted(
        generation_root.glob("batch-*/native-eval-manifest.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    row_by_id = {str(row["id"]): row for row in rows}
    provenance: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        attempt = manifest.parent
        try:
            attempt.relative_to(anchor_root)
            config_path = attempt / "native-eval.toml"
            config_text_value = config_path.read_text(encoding="utf-8")
            model_match = re.search(r"(?m)^model\s*=\s*[\"']([^\"']+)[\"']", config_text_value)
            seed_match = re.search(r"(?m)^seed\s*=\s*(\d+)\s*$", config_text_value)
            if model_match is None or model_match.group(1) != ALIAS or seed_match is None:
                continue
            batch_seed = int(seed_match.group(1))
            retry_index = batch_seed - 20261009
            if not 0 <= retry_index <= MAX_ANCHOR_RESAMPLES:
                continue
            document = json.loads(manifest.read_text(encoding="utf-8"))
            cases = document.get("cases")
            if not isinstance(cases, list):
                continue
            cases_by_id = {str(case.get("id")): case for case in cases if isinstance(case, Mapping)}
            trace_bundle = _read_trace_bundle(attempt)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for case_id, row in row_by_id.items():
            if case_id in provenance:
                continue
            case = cases_by_id.get(case_id)
            if not isinstance(case, Mapping):
                continue
            try:
                case_max_turns = int(case.get("max_turns", -1))
            except (TypeError, ValueError):
                continue
            if (
                case.get("source_group") != row.get("source_group")
                or case.get("reference_sha256") != row.get("reference_sha256")
                or case.get("reference_kind") != "text"
                or case_max_turns != MAX_TURNS
                or (case.get("retry_index") is not None and case.get("retry_index") != retry_index)
                or Path(str(case.get("renderer", ""))).resolve() != renderer_path
                or Path(str(case.get("renderer_python", ""))).absolute() != renderer_python_path
            ):
                continue
            try:
                expected_case_seed = deterministic_anchor_seed(
                    str(row["source_group"]), str(row["reference_sha256"]), retry_index
                )
                if int(case.get("case_seed", -1)) != expected_case_seed:
                    continue
            except (TypeError, ValueError):
                continue
            try:
                workspace_root = _trace_workspace_root(trace_bundle, attempt)
                if trace_bundle is None:
                    produced = _native_eval_canvas(attempt, case_id, workspace_root)
                else:
                    produced = _native_eval_canvas(attempt, case_id, workspace_root, trace_bundle=trace_bundle)
            except Exception as exc:
                # A trace parser/service error is preserved as unavailable and
                # is never converted into model-invalid retry evidence.
                failures.setdefault(case_id, {
                    "schema": "structured-flow-anchor-failure-v1",
                    "generation_protocol": GENERATION_PROTOCOL,
                    "status": "unavailable",
                    "case_id": case_id,
                    "retry_index": retry_index,
                    "batch_seed": batch_seed,
                    "attempt_root": str(attempt.relative_to(anchor_root)),
                    "reason": f"resume trace unavailable: {type(exc).__name__}: {exc}",
                })
                continue
            if isinstance(produced, Mapping):
                failures.setdefault(case_id, {
                    "schema": "structured-flow-anchor-failure-v1",
                    "generation_protocol": GENERATION_PROTOCOL,
                    "status": str(produced.get("status") or "invalid_model_output"),
                    "case_id": case_id,
                    "source_group": row["source_group"],
                    "reference_sha256": row["reference_sha256"],
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "model_alias": ALIAS,
                    "batch_seed": batch_seed,
                    "case_seed": case.get("case_seed", deterministic_anchor_seed(str(row["source_group"]), str(row["reference_sha256"]), retry_index)),
                    "retry_index": retry_index,
                    "batch_manifest": str(manifest.relative_to(anchor_root)),
                    "attempt_root": str(attempt.relative_to(anchor_root)),
                    "reason": str(produced.get("reason") or produced.get("status") or "invalid native model output"),
                })
                continue
            source = Path(produced).resolve()
            if source.is_symlink() or not source.is_file() or source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                failures.setdefault(case_id, {
                    "schema": "structured-flow-anchor-failure-v1",
                    "generation_protocol": GENERATION_PROTOCOL,
                    "status": "unavailable",
                    "case_id": case_id,
                    "retry_index": retry_index,
                    "batch_seed": batch_seed,
                    "attempt_root": str(attempt.relative_to(anchor_root)),
                    "reason": "resume trace final canvas is not a regular PNG",
                })
                continue
            target = anchor_root / f"{case_id}.png"
            temporary = target.with_name(target.name + f".resume-{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
            png_sha = sha256_file(target)
            receipt_path = receipt_root / f"{case_id}.resume-{retry_index}.json"
            terminal_metadata = _terminal_metadata(trace_bundle, case_id)
            _write_json(receipt_path, {
                "schema": "structured-flow-anchor-generation-v1",
                "generation_protocol": GENERATION_PROTOCOL,
                "resume_protocol": RESUME_PROTOCOL,
                "status": "valid",
                "generation_mode": "resumed_paid_batch",
                "case_id": case_id,
                "source_group": row["source_group"],
                "reference_sha256": row["reference_sha256"],
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "model_alias": ALIAS,
                "batch_seed": batch_seed,
                "case_seed": case.get("case_seed", deterministic_anchor_seed(str(row["source_group"]), str(row["reference_sha256"]), retry_index)),
                "retry_index": retry_index,
                "source_attempt": str(attempt.relative_to(anchor_root)),
                "batch_manifest": str(manifest.relative_to(anchor_root)),
                "png_sha256": png_sha,
                "resume_validation": "manifest_config_trace_png_hash",
                "terminal_metadata": terminal_metadata,
            })
            provenance[case_id] = {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "reference_sha256": row["reference_sha256"],
                "source_group": row["source_group"],
                "png_sha256": png_sha,
                "valid": True,
                "generation_receipt": str(receipt_path.relative_to(anchor_root)),
                "generation_mode": "resumed_paid_batch",
                "generation_protocol": GENERATION_PROTOCOL,
                "resume_protocol": RESUME_PROTOCOL,
                "source_attempt": str(attempt.relative_to(anchor_root)),
                "terminal_metadata": terminal_metadata,
            }
            failures.pop(case_id, None)
    return provenance, failures


def _batch_with_bounded_resample(
    rows: list[Mapping[str, Any]],
    *,
    anchor_root: Path,
    receipt_root: Path,
    proposal_root: Path,
    run_root: Path,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Admit valid anchors and retry each model-invalid case once.

    Native traces and failure receipts from the first attempt remain under
    ``native-generation``. A retry gets a new deterministic case seed and a
    new evaluator root; no previous PNG is used as a fallback. Infrastructure
    failures still raise immediately so a broken evaluator cannot be mistaken
    for a valid frozen opponent.
    """
    provenance, resumed_failures = _resume_existing_batches(
        rows,
        anchor_root=anchor_root,
        receipt_root=receipt_root,
        run_root=run_root,
    )
    unavailable = {
        case_id: record
        for case_id, record in resumed_failures.items()
        if record.get("status") != "invalid_model_output"
    }
    if unavailable:
        _write_json(receipt_root / "resume-failures.json", {
            "schema": "structured-flow-anchor-resume-failures-v1",
            "failures": unavailable,
        })
        failed_ids = ", ".join(sorted(unavailable))
        raise RuntimeError(f"existing native anchor traces are unavailable for {failed_ids}; refusing to classify infrastructure as model-invalid")

    history: dict[str, dict[str, dict[str, Any]]] = {}

    def remember(record: Mapping[str, Any]) -> None:
        case_id = str(record["case_id"])
        key = "initial" if int(record.get("retry_index", 0)) == 0 else "retry"
        history.setdefault(case_id, {})[key] = dict(record)

    for record in resumed_failures.values():
        remember(record)

    initial_rows = [
        row for row in rows
        if str(row["id"]) not in provenance and str(row["id"]) not in resumed_failures
    ]
    initial_failures: dict[str, dict[str, Any]] = {}
    if initial_rows:
        provenance.update(_batch_native_generation(
            initial_rows,
            anchor_root=anchor_root,
            receipt_root=receipt_root,
            proposal_root=proposal_root,
            run_root=run_root,
            workers=min(workers, len(initial_rows)),
            retry_index=0,
            failed_cases=initial_failures,
        ))
        for record in initial_failures.values():
            remember(record)

    retry_rows = [
        row for row in rows
        if str(row["id"]) not in provenance
        and str(row["id"]) in history
        and int(max(
            history[str(row["id"])].values(),
            key=lambda record: int(record.get("retry_index", 0)),
        ).get("retry_index", 0)) < MAX_ANCHOR_RESAMPLES
    ]
    final_failures: dict[str, dict[str, Any]] = {}
    if retry_rows:
        provenance.update(_batch_native_generation(
            retry_rows,
            anchor_root=anchor_root,
            receipt_root=receipt_root,
            proposal_root=proposal_root,
            run_root=run_root,
            workers=min(workers, len(retry_rows)),
            retry_index=1,
            failed_cases=final_failures,
        ))
        for record in final_failures.values():
            remember(record)
    failures = {
        case_id: records
        for case_id, records in history.items()
        if case_id not in provenance
    }
    return provenance, failures


def materialize(*, train_manifest: Path, heldout_manifest: Path, anchor_root: Path, output_dir: Path, run_root: Path) -> dict[str, Any]:
    train_manifest = train_manifest.resolve()
    heldout_manifest = heldout_manifest.resolve()
    anchor_root = anchor_root.resolve()
    output_dir = output_dir.resolve()
    run_root = run_root.resolve()
    train_doc = json.loads(train_manifest.read_text(encoding="utf-8"))
    heldout_doc = json.loads(heldout_manifest.read_text(encoding="utf-8"))
    train_rows = _rows(train_manifest)
    heldout_rows = _rows(heldout_manifest)
    if _is_final(train_doc, 32) and _is_final(heldout_doc, 16):
        return {"status": "already_materialized", "train_rows": 32, "heldout_cases": 16}
    if len(train_rows) != 32 or len(heldout_rows) != 16:
        raise ValueError("lazy pilot must contain exactly 32 train and 16 heldout rows")
    anchor_root.mkdir(parents=True, exist_ok=True)
    receipt_root = anchor_root / "native-generation"
    receipt_root.mkdir(parents=True, exist_ok=True)
    selected = sorted([*train_rows, *heldout_rows], key=lambda row: str(row["id"]))
    proposal_root = run_root / "curriculum"
    try:
        workers = int(os.environ.get("PAINTER_ANCHOR_GENERATION_WORKERS", str(MAX_GENERATION_WORKERS)))
    except ValueError as exc:
        raise ValueError("PAINTER_ANCHOR_GENERATION_WORKERS must be an integer") from exc
    if not 1 <= workers <= MAX_GENERATION_WORKERS:
        raise ValueError(f"PAINTER_ANCHOR_GENERATION_WORKERS must be in [1,{MAX_GENERATION_WORKERS}]")
    # One native VF process owns the 48-case request/render batch. Its
    # max-concurrent setting is the bounded worker count; only model-invalid
    # cases get one smaller resample batch after the first attempt.
    provenance, failures = _batch_with_bounded_resample(
        selected,
        anchor_root=anchor_root,
        receipt_root=receipt_root,
        proposal_root=proposal_root,
        run_root=run_root,
        workers=workers,
    )
    _write_json(anchor_root / curriculum.ANCHOR_PROVENANCE_FILE, {
        "schema": "structured-flow-anchor-provenance-v1",
        "run_id": curriculum.RUN_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "generation_protocol": GENERATION_PROTOCOL,
        "resume_protocol": RESUME_PROTOCOL,
        "anchors": provenance,
    })
    if failures:
        failure_path = anchor_root / "native-generation" / "anchor-failures.json"
        _write_json(failure_path, {
            "schema": "structured-flow-anchor-failures-v1",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "max_resamples": MAX_ANCHOR_RESAMPLES,
            "failures": failures,
        })
        failed_ids = ", ".join(sorted(failures))
        raise RuntimeError(f"frozen anchor generation left invalid model outputs after one resample: {failed_ids}")

    train_proposal = proposal_root / "train-proposal.json"
    heldout_proposal = proposal_root / "heldout-proposal.json"
    if not train_proposal.is_file() or not heldout_proposal.is_file():
        raise FileNotFoundError("train-proposal.json and heldout-proposal.json are required on the node")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + "-", dir=str(output_dir.parent)))
    try:
        import materialize as _materialize  # type: ignore
    except ImportError:
        spec = importlib.util.spec_from_file_location("structured_flow_materialize", RUN / "materialize.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("materialize.py unavailable")
        _materialize = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _materialize
        spec.loader.exec_module(_materialize)
    _materialize.materialize_pilot(
        train_proposal=train_proposal,
        heldout_proposal=heldout_proposal,
        anchor_root=anchor_root,
        output_root=staging,
        frozen_sft512_checkpoint_sha256=CHECKPOINT_SHA256,
        flow_policy_enabled=os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1",
    )
    final_train = staging / "pilot-train-manifest.json"
    final_eval = staging / "matched16-manifest.json"
    shutil.copyfile(final_train, train_manifest.with_name(train_manifest.name + ".final.tmp"))
    Path(train_manifest.with_name(train_manifest.name + ".final.tmp")).replace(train_manifest)
    shutil.copyfile(final_eval, heldout_manifest.with_name(heldout_manifest.name + ".final.tmp"))
    Path(heldout_manifest.with_name(heldout_manifest.name + ".final.tmp")).replace(heldout_manifest)
    # Preserve the official named artifacts alongside the replaced config path.
    for name in ("pilot-train-manifest.json", "pilot-heldout-manifest.json", "matched16-manifest.json", "pilot-manifest-receipt.json", "pilot-materialization-receipt.json"):
        source = staging / name
        if source.is_file():
            shutil.copyfile(source, run_root / "curriculum" / name)
    return {"status": "materialized", "train_rows": 32, "heldout_cases": 16, "anchor_provenance": str(anchor_root / curriculum.ANCHOR_PROVENANCE_FILE)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lazy = sub.add_parser("prepare-lazy")
    lazy.add_argument("--train-proposal", type=Path, required=True)
    lazy.add_argument("--heldout-proposal", type=Path, required=True)
    lazy.add_argument("--train-output", type=Path, required=True)
    lazy.add_argument("--heldout-output", type=Path, required=True)
    mat = sub.add_parser("materialize")
    mat.add_argument("--train-manifest", type=Path, required=True)
    mat.add_argument("--heldout-manifest", type=Path, required=True)
    mat.add_argument("--anchor-root", type=Path, required=True)
    mat.add_argument("--output-dir", type=Path, required=True)
    mat.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-lazy":
        value = prepare_lazy(train_proposal=args.train_proposal, heldout_proposal=args.heldout_proposal, train_output=args.train_output, heldout_output=args.heldout_output)
    else:
        value = materialize(train_manifest=args.train_manifest, heldout_manifest=args.heldout_manifest, anchor_root=args.anchor_root, output_dir=args.output_dir, run_root=args.run_root)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
