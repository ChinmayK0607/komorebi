#!/usr/bin/env python3
"""Package saved first-turn failures for a renderer-only Linux replay.

The failed evaluation produced one ``program.js`` and one invalid
``painting.json`` per case before the renderer worker could import Playwright.
This helper copies those exact programs, verifies their recorded source hashes,
and records the original renderer identity.  It never invokes JavaScript,
Playwright, a model, or a remote service.  The resulting manifest deliberately
has no target PNG hashes: a later replay checks renderer execution only and is
not presented as the original three-turn evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "text-paint-first-turn-recovery-inputs-v1"
SEED = 0
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value.lower()):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _safe_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe evaluation task id: {value!r}")
    return value


def _portable_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Keep stable failure identity while dropping absolute node paths."""

    keys = (
        "valid", "error_code", "seed", "canvas_contract", "renderer_sha256",
        "assets", "requested_angle_backend", "telemetry_enabled",
        "network_enabled", "browser_sandbox_enabled",
    )
    return {key: receipt[key] for key in keys if key in receipt}


def _renderer_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    renderer_sha = _hash(receipt.get("renderer_sha256"), "renderer_sha256")
    assets = receipt.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("renderer asset hashes are missing from a failed receipt")
    normalized = {
        str(name): _hash(value, f"asset {name}")
        for name, value in assets.items()
    }
    return {"renderer_sha256": renderer_sha, "assets_sha256": normalized}


def build_manifest(
    *,
    evaluation_root: Path,
    output_dir: Path,
    trace: Path | None = None,
    renderer: Path | None = None,
    original_turn_cap: int = 3,
) -> dict[str, Any]:
    evaluation_root = evaluation_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not evaluation_root.is_dir():
        raise FileNotFoundError(f"evaluation artifact root is missing: {evaluation_root}")
    if isinstance(original_turn_cap, bool) or original_turn_cap < 1:
        raise ValueError("original_turn_cap must be a positive integer")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    programs_dir = output_dir / "programs"
    receipts_dir = output_dir / "receipts"
    programs_dir.mkdir(exist_ok=True)
    receipts_dir.mkdir(exist_ok=True)

    receipts = sorted(evaluation_root.rglob("painting.json"))
    if not receipts:
        raise ValueError(f"evaluation artifact root contains no painting receipts: {evaluation_root}")

    identity: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    failures: dict[str, int] = {}
    for receipt_path in receipts:
        if receipt_path.is_symlink():
            raise ValueError(f"failed receipt must not be a symlink: {receipt_path}")
        if receipt_path.parent.name != "turn-1":
            raise ValueError(f"first-turn recovery requires turn-1 receipts: {receipt_path}")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid failed receipt: {receipt_path}") from exc
        if not isinstance(receipt, dict):
            raise ValueError(f"failed receipt is not an object: {receipt_path}")
        if receipt.get("valid") is not False:
            raise ValueError(f"expected an invalid first-turn receipt: {receipt_path}")
        if receipt.get("seed") != SEED:
            raise ValueError(f"unexpected renderer seed in {receipt_path}")
        renderer_identity = _renderer_identity(receipt)
        if identity is None:
            identity = renderer_identity
        elif renderer_identity != identity:
            raise ValueError(f"renderer identity differs in {receipt_path}")
        task_id = _safe_id(receipt_path.parents[2].name)
        if task_id in seen:
            raise ValueError(f"duplicate task id in failed receipts: {task_id}")
        seen.add(task_id)
        source_path = receipt_path.with_name("program.js")
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"saved program is missing for {task_id}: {source_path}")
        expected_source = _hash(receipt.get("source_sha256"), f"source_sha256 for {task_id}")
        actual_source = sha256(source_path)
        if actual_source != expected_source:
            raise ValueError(f"saved program hash mismatch for {task_id}")
        source_name = f"{task_id}.js"
        receipt_name = f"{task_id}.json"
        destination = programs_dir / source_name
        destination_receipt = receipts_dir / receipt_name
        destination.write_bytes(source_path.read_bytes())
        destination_receipt.write_text(
            json.dumps(_portable_receipt(receipt), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        error_code = str(receipt.get("error_code") or "unknown")
        failures[error_code] = failures.get(error_code, 0) + 1
        rows.append({
            "id": task_id,
            "source_group": task_id,
            "seed": SEED,
            "source_file": f"programs/{source_name}",
            "source_sha256": actual_source,
            "source_turn": 1,
            "original_receipt_file": f"receipts/{receipt_name}",
            "original_receipt_sha256": sha256(receipt_path),
            "original_error_code": error_code,
            "original_eval_turn_cap": original_turn_cap,
        })

    if renderer is not None:
        renderer = renderer.expanduser().resolve()
        if not renderer.is_file() or renderer.is_symlink():
            raise FileNotFoundError(f"renderer source is missing or symlinked: {renderer}")
        actual_renderer = sha256(renderer)
        if identity is None or actual_renderer != identity["renderer_sha256"]:
            raise ValueError("supplied renderer source hash does not match failed receipts")
        assets_dir = renderer.parent / "assets"
        actual_assets = {
            path.name: sha256(path)
            for path in sorted(assets_dir.glob("*.js"))
            if path.is_file() and not path.is_symlink()
        }
        if actual_assets != identity["assets_sha256"]:
            raise ValueError("supplied renderer asset hashes do not match failed receipts")

    if identity is None:  # receipts is non-empty, retained for type narrowing
        raise ValueError("failed receipts have no renderer identity")
    trace_hash = None
    if trace is not None:
        trace = trace.expanduser().resolve()
        if not trace.is_file():
            raise FileNotFoundError(f"evaluation trace is missing: {trace}")
        trace_hash = sha256(trace)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "replay_mode": "first_turn_recovery",
        "protocol_version": "native-painting-eval-v3",
        "seed": SEED,
        "original_eval_turn_cap": original_turn_cap,
        "renderer": identity,
        "source_trace_sha256": trace_hash,
        "summary": {
            "receipt_count": len(receipts),
            "replay_cases": len(rows),
            "failure_counts": failures,
        },
        "cases": rows,
        "limitations": [
            "Programs are copied from first-turn failed evaluation attempts; no model generation is replayed.",
            "No target PNG hashes are supplied, so replay results are not a three-turn evaluation or visual comparison.",
            "Renderer identity is taken from the original receipts and must be matched by the later Linux runtime.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--original-turn-cap", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        evaluation_root=args.evaluation_root,
        output_dir=args.output,
        trace=args.trace,
        renderer=args.renderer,
        original_turn_cap=args.original_turn_cap,
    )
    print(json.dumps({"output": str(args.output), "summary": manifest["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
