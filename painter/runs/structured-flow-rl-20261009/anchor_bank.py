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
DECODING_CONFIG = "{}"
MAX_TURNS = 3
MAX_COMPLETION_TOKENS = 8192
CONTEXT_LENGTH = 32768
MAX_GENERATION_WORKERS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_anchor_seed(source_group: str, reference_sha256: str) -> int:
    material = f"20261009:{source_group}:{reference_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def write_private_log(path: Path, value: str) -> None:
    encoded = str(value or "").encode("utf-8")
    if len(encoded) > 64 * 1024:
        encoded = encoded[:64 * 1024] + b"\n...[truncated]\n"
    path.write_bytes(encoded)
    path.chmod(0o600)


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
    if not renderer_path.is_file() or not renderer_python_path.is_file():
        raise RuntimeError("frozen anchor batch renderer runtime is missing")
    if not eval_bin.is_file() or not prime_root.is_dir():
        raise RuntimeError("frozen anchor batch native eval runtime is missing")

    attempt = anchor_root / "native-generation" / f"batch-{os.getpid()}-{time.time_ns()}"
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
            "case_seed": deterministic_anchor_seed(str(row["source_group"]), reference_sha),
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
    generated = generated.replace("seed = 918200", "seed = 20261009", 1)
    config_path.write_text(generated, encoding="utf-8")
    environment = os.environ.copy()
    # Frozen SFT anchors are the fixed opponent target. Do not apply the
    # structured-flow policy hook to this 48-case batch; its task IDs are not
    # the 32/16 RL/eval manifest IDs and must remain generic native prompts.
    environment["PAINTER_STRUCTURED_FLOW_POLICY"] = "0"
    environment.pop("PAINTER_PRIME_ROLE", None)
    environment.pop("PAINTER_RL_MANIFEST", None)
    environment.pop("PAINTER_EVAL_MANIFEST", None)
    environment["PYTHONPATH"] = str(workspace_root) + os.pathsep + environment.get("PYTHONPATH", "")
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

    provenance: dict[str, dict[str, Any]] = {}
    for row in normalized:
        case_id = row["id"]
        produced = _native_eval_canvas(attempt, case_id, workspace_root)
        if isinstance(produced, Mapping):
            status = produced.get("status")
            raise RuntimeError(f"frozen anchor generation failed for {case_id}: {produced.get('reason') or status}")
        source = Path(produced).resolve()
        if source.is_symlink() or not source.is_file() or source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"frozen anchor output is not a regular PNG: {case_id}")
        target = anchor_root / f"{case_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        png_sha = sha256_file(target)
        receipt = {
            "schema": "structured-flow-anchor-generation-v1",
            "status": "valid",
            "case_id": case_id,
            "source_group": row["source_group"],
            "reference_sha256": row["reference_sha256"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_alias": ALIAS,
            "protocol_version": PROTOCOL,
            "render_protocol": RENDER_PROTOCOL,
            "batch_seed": 20261009,
            "case_seed": row["case_seed"],
            "seed_binding": "registered_per_case_and_batch",
            "batch_manifest": str(manifest.relative_to(anchor_root)),
            "png_sha256": png_sha,
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
        }
    return provenance


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
    # max-concurrent setting is the bounded worker count; no per-case
    # evaluator processes are spawned here.
    provenance = _batch_native_generation(
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
        "anchors": provenance,
    })

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
