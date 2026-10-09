#!/usr/bin/env python3
"""Run-local strict SFT warmup for the structured-flow RL arm.

The controller can use :func:`validate_inputs` and :func:`build_config` without
Prime, torch, a model, or a renderer.  ``audit`` and ``train`` are node-only:
they import the reviewed mixed-SFT source, run the real multimodal processor,
and then launch the pinned one-GPU trainer.  This file never creates a target
painting or a demonstration program; every admitted target must already have a
reviewed rendered PNG and a hash-bound receipt.
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
import subprocess
import sys
from typing import Any, Mapping, Sequence
import tomllib


RUN_ID = "structured-flow-rl-20261009"
WARMUP_SCHEMA = "structured-flow-sft-warmup-v1"
PRIME_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_ID = "Qwen/Qwen3.8-27B"
FLASH12_REVISION = "1227a2427a8ae16014d2fdca4d1adb6855d3fa1c"
FLASH12_PREFIX = "rl-judge-matched-20261004-flash/step_12"
FLASH12_SHA256 = "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TENSORS = 992
# The approved subset may be as small as 15/2: five train families with
# three rich rows each and one held-out family with two diagnostic rows. The
# four-update schedule consumes 12 unique train rows, balanced by flow state.
MIN_TRAIN_ROWS = 15
MAX_TRAIN_ROWS = 32
MIN_DEV_ROWS = 2
MAX_DEV_ROWS = 8
MIN_TRAIN_FAMILIES = 5
MIN_DEV_FAMILIES = 1
MAX_STEPS = 4
BATCH_SIZE = 3
MICRO_BATCH_SIZE = 1
LEARNING_RATE = 1e-6
SEQ_LEN = 16384
DEFAULT_WARMUP_GPU = "1"
FLOW_STATES = ("construction", "revision", "finish")
WARMUP_STATE_PATTERN = (
    ("construction", "revision", "finish"),
    ("construction", "revision", "finish"),
    ("construction", "revision", "finish"),
    ("construction", "revision", "finish"),
)
# Rotating the state pools keeps the 12-row pilot from dropping the last
# approved family when each state has five source rows.  The offsets are part
# of the schedule receipt and are deterministic across retries.
WARMUP_STATE_POOL_OFFSETS = {"construction": 0, "revision": 1, "finish": 2}
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
FLOW_STATE_ALIASES = {
    "construction": "construction", "block_in": "construction", "block-in": "construction", "initial": "construction",
    "revision": "revision", "improved": "revision", "improved_final": "revision", "edit": "revision",
    "finish": "finish", "finished": "finish", "finish_state": "finish", "terminal": "finish",
}
# The pinned mixed-SFT loader reserves these names for its own ordered
# schedule.  This warmup deliberately uses its supported legacy path: the
# rows are a finite, already-ordered four-batch painting stream, while the
# flow curriculum is recorded under warmup_* fields in our own receipt.
NATIVE_TASK_FAMILY = "painting"
SCHEDULE_RESERVED_FIELDS = frozenset({"scheduled_step", "scheduled_slot", "schedule_group"})
WARMUP_SCHEDULE_SCHEMA = "structured-flow-warmup-schedule-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(value)
    return rows


def _text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"]


def _image_parts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [part for part in content if isinstance(part, dict) and (part.get("type") in {"image", "image_url"} or "image_url" in part)]


def _resolve_file(raw: Any, base: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_approval(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") not in {"reviewed", "approved", "flash_decision"}:
        raise ValueError("warmup approval manifest is not reviewed/approved")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("warmup approval manifest has no candidates")
    by_id = {str(candidate.get("id")): candidate for candidate in candidates if isinstance(candidate, Mapping)}
    return {"path": str(path), "sha256": sha256_file(path), "by_id": by_id}, sha256_file(path)


def _approval_target(row: Mapping[str, Any], approval: Mapping[str, Any] | None) -> tuple[bool, str | None]:
    if approval is None:
        return False, None
    candidate = approval["by_id"].get(str(row.get("source_group")))
    if not isinstance(candidate, Mapping) or not candidate.get("render_valid") or candidate.get("decision") not in {"approved", "flash_decision"} or candidate.get("improved_reviewed") is not True:
        raise ValueError(f"{row.get('id', '<row>')}: source candidate lacks an approved visual decision")
    stage = str(row.get("flow_stage") or row.get("flow_state") or row.get("state") or "")
    expected = candidate.get("initial_png_sha256") if stage in {"block_in", "construction", "initial"} else candidate.get("final_png_sha256")
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        raise ValueError(f"{row.get('id', '<row>')}: approval manifest has no stage PNG hash")
    return True, expected


def _target_evidence(row: Mapping[str, Any], base: Path, approval: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Require an approved render receipt and a current PNG hash."""
    render = row.get("target_render") or row.get("render") or row.get("target")
    if not isinstance(render, Mapping):
        render = {}
    status = row.get("target_status") or row.get("target_admission") or row.get("admission")
    status = status or render.get("status") or render.get("admission")
    approval_ok, approval_hash = _approval_target(row, approval)
    approved = approval_ok or status in {
        "approved", "human_approved", "flash_decision", "render_approved", "visual_review_approved",
    } or render.get("approved") is True or render.get("valid") is True and row.get("target_approved") is True
    if not approved:
        raise ValueError(f"{row.get('id', '<row>')}: target is not explicitly approved")
    raw_path = (
        row.get("target_png") or row.get("target_path") or row.get("render_png")
        or render.get("png") or render.get("path") or render.get("file")
    )
    target = _resolve_file(raw_path, base)
    if target is None or target.is_symlink() or not target.is_file():
        raise ValueError(f"{row.get('id', '<row>')}: approved target PNG is missing")
    expected = row.get("target_sha256") or row.get("render_sha256") or render.get("png_sha256") or render.get("sha256")
    if approval_hash is not None:
        if expected is not None and expected != approval_hash:
            raise ValueError(f"{row.get('id', '<row>')}: row render hash disagrees with approval manifest")
        expected = approval_hash
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        raise ValueError(f"{row.get('id', '<row>')}: approved target hash is missing")
    actual = sha256_file(target)
    if actual != expected:
        raise ValueError(f"{row.get('id', '<row>')}: approved target PNG hash mismatch")
    receipt = row.get("target_receipt") or render.get("receipt")
    if receipt is None and render.get("receipt_path"):
        receipt_path = _resolve_file(render.get("receipt_path"), base)
        if receipt_path and receipt_path.is_file() and not receipt_path.is_symlink():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt is not None and not isinstance(receipt, (Mapping, str)):
        raise ValueError(f"{row.get('id', '<row>')}: target receipt is malformed")
    if isinstance(receipt, str):
        receipt_path = _resolve_file(receipt, base)
        if receipt_path is None or not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError(f"{row.get('id', '<row>')}: target receipt is missing")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if isinstance(receipt, Mapping):
        if receipt.get("verified") is False or receipt.get("valid") is False:
            raise ValueError(f"{row.get('id', '<row>')}: target receipt is invalid")
        receipt_hash = receipt.get("png_sha256") or receipt.get("target_sha256") or receipt.get("sha256")
        if receipt_hash is not None and receipt_hash != actual:
            raise ValueError(f"{row.get('id', '<row>')}: target receipt hash mismatch")
    return {"path": str(target), "sha256": actual, "status": status or "approved"}


def _prior_canvas_evidence(row: Mapping[str, Any], images: Sequence[Mapping[str, Any]], base: Path, flow_state: str) -> dict[str, Any]:
    prior = row.get("prior_canvas")
    if prior is None and not images and flow_state == "construction":
        return {"path": None, "sha256": None, "available": False}
    if isinstance(prior, Mapping):
        raw_path = prior.get("path") or prior.get("file")
        path = _resolve_file(raw_path, base)
        expected = prior.get("sha256")
        if path is None or path.is_symlink() or not path.is_file() or not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise ValueError(f"{row.get('id', '<row>')}: prior canvas path/hash is required")
        if sha256_file(path) != expected:
            raise ValueError(f"{row.get('id', '<row>')}: prior canvas hash mismatch")
        return {"path": str(path), "sha256": expected}
    for image in images:
        path = _resolve_file(image.get("path") or image.get("file"), base)
        expected = image.get("sha256") or image.get("image_sha256")
        if path and path.is_file() and not path.is_symlink() and isinstance(expected, str) and HEX64.fullmatch(expected):
            if sha256_file(path) != expected:
                raise ValueError(f"{row.get('id', '<row>')}: prior image hash mismatch")
            return {"path": str(path), "sha256": expected}
    if flow_state == "construction":
        return {"path": None, "sha256": None, "available": False}
    raise ValueError(f"{row.get('id', '<row>')}: a hash-bound prior canvas image is required")


def _canonical_flow_state(row: Mapping[str, Any]) -> str | None:
    raw_state = row.get("flow_state") or row.get("state") or row.get("example_kind") or row.get("flow_stage")
    return FLOW_STATE_ALIASES.get(str(raw_state).strip().lower())


def validate_row(row: Mapping[str, Any], split: str, base: Path, approval: Mapping[str, Any] | None = None) -> dict[str, Any]:
    ident = row.get("id")
    if not isinstance(ident, str) or not ident.strip():
        raise ValueError("every warmup row needs a nonempty id")
    source_group = row.get("source_group")
    if not isinstance(source_group, str) or not source_group.strip():
        raise ValueError(f"{ident}: source_group is required")
    if row.get("split") not in {split, "validation" if split == "dev" else None, None}:
        raise ValueError(f"{ident}: split does not match {split}")
    flow_state = _canonical_flow_state(row)
    if flow_state is None:
        raise ValueError(f"{ident}: flow_state must be construction, revision, or finish")
    messages = row.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages if isinstance(m, Mapping)] != ["system", "user", "assistant"]:
        raise ValueError(f"{ident}: strict SFT requires exactly system/user/assistant messages")
    if any(not isinstance(message, Mapping) for message in messages):
        raise ValueError(f"{ident}: malformed message")
    if not _text_parts(messages[0].get("content")) or not _text_parts(messages[1].get("content")):
        raise ValueError(f"{ident}: system and user text are required")
    if not _text_parts(messages[2].get("content")):
        raise ValueError(f"{ident}: assistant target text is required")
    images = _image_parts(messages[1].get("content"))
    if not images and flow_state != "construction":
        raise ValueError(f"{ident}: user turn must include the prior canvas image")
    target = _target_evidence(row, base, approval)
    prior = _prior_canvas_evidence(row, images, base, flow_state)
    return {
        "id": ident,
        "source_group": source_group,
        "split": split,
        "target": target,
        "prior_canvas": prior,
        "assistant_text_sha256": sha256_text("".join(_text_parts(messages[2].get("content")))),
        "flow_state": flow_state,
    }


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def resolve_warmup_gpu(environ: Mapping[str, str] | None = None) -> str:
    """Select one physical ordinal without inheriting an anchor GPU by accident."""
    environ = os.environ if environ is None else environ
    requested = environ.get("PAINTER_WARMUP_GPU", DEFAULT_WARMUP_GPU).strip()
    if requested == "inherit":
        requested = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not re.fullmatch(r"[01]", requested):
        raise ValueError("PAINTER_WARMUP_GPU must be ordinal 0 or 1, or explicit 'inherit' from one visible ordinal")
    if requested == "inherit" or "," in requested:
        raise ValueError("warmup requires exactly one visible GPU ordinal")
    return requested


def validate_inputs(train_path: Path, dev_path: Path, approval_path: Path | None = None) -> dict[str, Any]:
    """Validate the bounded, actually admitted flow-agent pool before any processor import."""
    train_path = train_path.expanduser().resolve()
    dev_path = dev_path.expanduser().resolve()
    train = _read_jsonl(train_path)
    dev = _read_jsonl(dev_path)
    approval, approval_sha = _load_approval(approval_path)
    if not MIN_TRAIN_ROWS <= len(train) <= MAX_TRAIN_ROWS or not MIN_DEV_ROWS <= len(dev) <= MAX_DEV_ROWS:
        raise ValueError(f"warmup requires {MIN_TRAIN_ROWS}..{MAX_TRAIN_ROWS} train and {MIN_DEV_ROWS}..{MAX_DEV_ROWS} dev rows after approval")
    seen: set[str] = set()
    groups: dict[str, set[str]] = {"train": set(), "dev": set()}
    records: dict[str, list[dict[str, Any]]] = {}
    for split, path, rows in (("train", train_path, train), ("dev", dev_path, dev)):
        records[split] = []
        for row in rows:
            check = validate_row(row, split, path.parent, approval)
            if check["id"] in seen:
                raise ValueError(f"duplicate warmup row id: {check['id']}")
            other_split = "dev" if split == "train" else "train"
            if check["source_group"] in groups[other_split]:
                raise ValueError(f"source_group leaks across warmup splits: {check['source_group']}")
            seen.add(check["id"])
            groups[split].add(check["source_group"])
            records[split].append(check)
    if len(groups["train"]) < MIN_TRAIN_FAMILIES or len(groups["dev"]) < MIN_DEV_FAMILIES:
        raise ValueError(
            f"warmup admission needs at least {MIN_TRAIN_FAMILIES} train and "
            f"{MIN_DEV_FAMILIES} dev source groups"
        )
    return {
        "schema": WARMUP_SCHEMA,
        "run_id": RUN_ID,
        "status": "inputs_approved_rendered_targets",
        "counts": {"train": len(train), "dev": len(dev)},
        "source_files": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "dev": {"path": str(dev_path), "sha256": sha256_file(dev_path)},
        },
        "approval_manifest": approval and {"path": approval["path"], "sha256": approval_sha},
        "source_groups": {split: sorted(groups[split]) for split in ("train", "dev")},
        "flow_state_counts": {
            split: {state: sum(record["flow_state"] == state for record in records[split]) for state in FLOW_STATES}
            for split in ("train", "dev")
        },
        "rows": records,
    }


def build_schedule(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Materialize four B3 steps that mix construction, revision, and finish.

    The admitted train rows remain in the input receipt. The 12 emitted rows
    are explicit unique schedule draws, so the warmup consumes each flow state
    evenly without inventing rows for censored candidates.
    """
    rows = list(rows)
    pools = {}
    for state in FLOW_STATES:
        ordered = sorted((dict(row) for row in rows if _canonical_flow_state(row) == state), key=lambda row: str(row["id"]))
        if ordered:
            offset = WARMUP_STATE_POOL_OFFSETS[state] % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
        pools[state] = ordered
    required = {state: sum(pattern.count(state) for pattern in WARMUP_STATE_PATTERN) for state in FLOW_STATES}
    missing = {state: required[state] - len(pools[state]) for state in FLOW_STATES if len(pools[state]) < required[state]}
    if missing:
        raise ValueError(f"warmup train pool is missing flow-state coverage: {missing}")
    cursors = {state: 0 for state in FLOW_STATES}
    emitted: list[dict[str, Any]] = []
    for step, pattern in enumerate(WARMUP_STATE_PATTERN, 1):
        for slot, state in enumerate(pattern):
            source = dict(pools[state][cursors[state]])
            cursors[state] += 1
            source_id = str(source["id"])
            source.update({
                "id": f"{source_id}__warmup_step_{step:02d}_slot_{slot:02d}",
                "source_example_id": source_id, "warmup_step": step, "warmup_slot": slot,
                "warmup_flow_state": state, "task_family": NATIVE_TASK_FAMILY,
            })
            emitted.append(source)
    if len(emitted) != MAX_STEPS * BATCH_SIZE or len({row["id"] for row in emitted}) != len(emitted):
        raise AssertionError("warmup schedule must contain four complete unique optimizer batches of three rows")
    return emitted


def build_schedule_receipt(rows: Sequence[Mapping[str, Any]], train_sha256: str) -> dict[str, Any]:
    """Describe the bounded legacy stream without activating PRIME scheduling.

    ``scheduled_data.py`` has a separate painting/retention schedule contract
    which would misclassify flow examples (especially finish examples) if we
    projected them into retention families.  The caller therefore keeps every
    row in the native ``painting`` family and records the four ordered batches
    in this run-owned receipt.  The pinned loader sees an ordinary ordered
    dataset and returns ``None`` from ``guard_runtime``.
    """
    rows = list(rows)
    if len(rows) != MAX_STEPS * BATCH_SIZE:
        raise ValueError("warmup schedule receipt requires exactly twelve emitted rows")
    expected: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_seen: set[str] = set()
    flow_counts = {state: 0 for state in FLOW_STATES}
    for index, row in enumerate(rows):
        step, slot = divmod(index, BATCH_SIZE)
        expected_state = WARMUP_STATE_PATTERN[step][slot]
        ident = row.get("id")
        source = row.get("source_example_id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("warmup schedule receipt requires unique emission IDs")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{ident}: schedule source identity is required")
        if source in source_seen:
            raise ValueError(f"{ident}: warmup schedule repeats a source identity")
        if row.get("task_family") != NATIVE_TASK_FAMILY:
            raise ValueError(f"{ident}: legacy warmup rows must use task_family=painting")
        if any(field in row for field in SCHEDULE_RESERVED_FIELDS):
            raise ValueError(f"{ident}: pinned schedule fields must not enter the legacy stream")
        if (row.get("warmup_step"), row.get("warmup_slot"), row.get("warmup_flow_state")) != (step + 1, slot, expected_state):
            raise ValueError(f"{ident}: warmup flow order differs from the bounded pattern")
        if _canonical_flow_state(row) != expected_state:
            raise ValueError(f"{ident}: warmup flow_state was not preserved")
        seen.add(ident)
        source_seen.add(source)
        flow_counts[expected_state] += 1
        expected.append({
            "id": ident, "source_example_id": source, "warmup_step": step + 1,
            "warmup_slot": slot, "warmup_flow_state": expected_state,
        })
    if not isinstance(train_sha256, str) or not HEX64.fullmatch(train_sha256):
        raise ValueError("warmup schedule receipt requires the train JSONL SHA-256")
    return {
        "schema": WARMUP_SCHEDULE_SCHEMA,
        "mode": "legacy_ordered_painting",
        "task_family": NATIVE_TASK_FAMILY,
        "batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "row_count": len(rows),
        "flow_state_pattern": [list(pattern) for pattern in WARMUP_STATE_PATTERN],
        "flow_state_pool_offsets": dict(WARMUP_STATE_POOL_OFFSETS),
        "flow_state_counts": flow_counts,
        "train_sha256": train_sha256,
        "emissions": expected,
        "pinned_loader": "scheduled_data.validate_returns_none_without_reserved_schedule_fields",
    }


def validate_warmup_schedule(data_dir: Path, receipt_path: Path, native_schedule_path: Path | None = None) -> dict[str, Any]:
    """Validate the own bounded schedule and, when supplied, the pinned API.

    This intentionally calls the real reviewed ``scheduled_data.validate``
    implementation when a materialized source is available.  A ``None``
    result is required: it proves the rows remain on the pinned legacy path
    rather than silently entering the native painting/retention scheduler.
    """
    data_dir = data_dir.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != WARMUP_SCHEDULE_SCHEMA:
        raise ValueError("warmup schedule receipt schema is not recognized")
    train_path = data_dir / "train.jsonl"
    rows = _read_jsonl(train_path)
    raw = train_path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if actual_sha != receipt.get("train_sha256"):
        raise ValueError("warmup schedule train hash differs from its receipt")
    if data_dir.joinpath("schedule.json").exists():
        raise ValueError("legacy warmup data must not contain a native schedule.json")
    expected = build_schedule_receipt(rows, actual_sha)
    if expected != receipt:
        raise ValueError("warmup schedule receipt does not match the emitted train stream")
    native_result = None
    if native_schedule_path is not None:
        spec = importlib.util.spec_from_file_location("structured_flow_pinned_scheduled_data", native_schedule_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"missing pinned schedule validator: {native_schedule_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native_result = module.validate(data_dir)
        if native_result is not None:
            raise ValueError("warmup rows unexpectedly entered the pinned ordered schedule")
    return {"status": "validated", "train_sha256": actual_sha, "native_schedule": native_result, "rows": len(rows)}


def build_config(model_dir: Path, data_dir: Path, run_name: str = "structured-flow-sft-warmup") -> dict[str, Any]:
    """Return the native Prime config; no Prime import is needed to inspect it."""
    if not RUN_NAMESPACE.fullmatch(run_name):
        raise ValueError("run_name is not a safe Prime/HF namespace")
    data = {
        "type": "sft", "name": str(data_dir.resolve()), "batch_size": BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE, "seq_len": SEQ_LEN, "num_workers": 1,
        "seed": 918210, "loss_mask": {"system": False, "user": False, "assistant": True, "tool": False},
    }
    model = {
        "name": str(model_dir.resolve()), "impl": "custom", "compile": "None", "attn": "auto",
        "optimization_dtype": "bfloat16", "reduce_dtype": "bfloat16",
        "vlm": {"vision_encoder_attr": "model.visual", "language_model_attr": "model.language_model", "freeze_vision_encoder": True},
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "target_modules": list(TARGET_MODULES)},
        "ac": {"freq": 1}, "ac_offloading": "None", "optim_cpu_offload": False,
    }
    return {
        "max_steps": MAX_STEPS, "run": {"name": run_name},
        "deployment": {"gpus_per_node": 1, "num_train_gpus": 1},
        "model": model, "renderer": {"name": "qwen3.5", "enable_thinking": False},
        "data": {**data, "splits": ["train"], "shuffle": False},
        "val": {"interval": MAX_STEPS, "eval_on_start": True, "data": {**data, "splits": ["validation"], "shuffle": False}},
        "optim": {"type": "adamw", "lr": LEARNING_RATE, "betas1": 0.9, "betas2": 0.999, "weight_decay": 0.01, "max_norm": 1.0},
        "scheduler": {"type": "constant"}, "ckpt": {"interval": MAX_STEPS, "keep_last": MAX_STEPS},
        "monitors": {"file": {}},
    }


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def _toml_lines(value: Mapping[str, Any], prefix: str = "") -> list[str]:
    lines: list[str] = []
    scalars = [(key, item) for key, item in value.items() if not isinstance(item, Mapping)]
    for key, item in scalars:
        lines.append(f"{key} = {_toml_scalar(item)}")
    for key, item in value.items():
        if isinstance(item, Mapping):
            section = f"{prefix}.{key}" if prefix else key
            lines.extend(["", f"[{section}]"])
            lines.extend(_toml_lines(item, section))
    return lines


def write_toml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_toml_lines(value)).lstrip() + "\n", encoding="utf-8")


def resolve_adapter_path(raw: Path) -> Path:
    path = raw.expanduser()
    if path.is_file() and path.name not in {"adapter_model.safetensors", "adapter_config.json"}:
        text = path.read_text(encoding="utf-8").strip().splitlines()
        if len(text) != 1:
            raise ValueError(f"adapter pointer must contain exactly one path: {path}")
        path = Path(text[0]).expanduser()
    return path.resolve()


def verify_parent_adapter(path: Path, *, expected_sha256: str = FLASH12_SHA256) -> dict[str, Any]:
    path = resolve_adapter_path(path)
    weights = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    if path.is_symlink() or not path.is_dir() or not weights.is_file() or not config_path.is_file():
        raise ValueError(f"Flash12 adapter directory is incomplete: {path}")
    actual = sha256_file(weights)
    if actual != expected_sha256:
        raise ValueError("warmup parent adapter bytes do not match the pinned Flash12 SHA-256")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if metadata.get("peft_type") != "LORA" or metadata.get("r") != LORA_RANK or metadata.get("lora_alpha") != LORA_ALPHA:
        raise ValueError("warmup parent adapter rank/alpha is not the pinned rank-16 LoRA contract")
    if set(metadata.get("target_modules", [])) != set(TARGET_MODULES):
        raise ValueError("warmup parent adapter target modules differ from the pinned profile")
    return {"path": str(path), "sha256": actual, "revision": FLASH12_REVISION, "prefix": FLASH12_PREFIX,
            "base_model": metadata.get("base_model_name_or_path", MODEL_ID), "rank": LORA_RANK, "alpha": LORA_ALPHA}


def verify_prime_pin(prime_root: Path, *, expected: str = PRIME_COMMIT) -> str:
    prime_root = prime_root.expanduser().resolve()
    try:
        actual = subprocess.check_output(["git", "-C", str(prime_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        marker = prime_root / "prime-commit.txt"
        actual = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if not actual:
            raise ValueError("cannot verify the required Prime commit") from exc
    if actual != expected:
        raise ValueError(f"Prime revision mismatch: expected {expected}, got {actual}")
    return actual


def _training_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("id", "source_group", "split", "brief", "flow_guidance", "objective", "messages")
    return {key: row[key] for key in keys if key in row}


def _processor_mask_modes() -> tuple[tuple[str, str | None], ...]:
    """Return both the ordinary and actual training mask environments."""
    return (("baseline", None), ("capacity_prefill", "1"))


def _mask_observation(
    mask_mode: str,
    total: int,
    masks: Sequence[bool],
    learned: Sequence[int],
    expected: Sequence[int],
    seq_len: int,
) -> dict[str, Any]:
    """Classify one mask; only the runtime capacity branch is a gate.

    The baseline branch is retained as evidence because the native renderer
    may leave disabled-thinking/generation-wrapper tokens visible there.  The
    training environment always enables the capacity-prefill branch, whose
    exact target and no-truncation checks are mandatory.
    """
    supervised = sum(bool(mask) for mask in masks)
    exact = list(learned) == list(expected)
    observation = {
        "status": "diagnostic" if mask_mode == "baseline" else "required_pass",
        "exact_target_match": exact,
        "total_tokens": total,
        "assistant_tokens": len(learned),
        "expected_assistant_tokens": len(expected),
        "target_token_delta": len(learned) - len(expected),
        "supervised_tokens": supervised,
        "extra_supervised_tokens": max(0, supervised - len(expected)),
        "fits": total <= seq_len,
    }
    if mask_mode != "capacity_prefill":
        return observation
    if total > seq_len:
        raise ValueError(f"capacity_prefill sample has {total} tokens exceeding {seq_len}; refusing truncation")
    if not exact:
        raise ValueError("capacity_prefill assistant target/mask/tokenizer parity mismatch")
    if not 0 < supervised < total:
        raise ValueError("capacity_prefill has an invalid assistant-only loss mask")
    return observation


def processor_audit(config_path: Path, output: Path, brush_root: Path) -> dict[str, Any]:
    """Run the reviewed processor and prove exact assistant target masking."""
    if sys.platform != "linux":
        raise RuntimeError("processor audit runs on the Linux training node")
    brush_root = brush_root.resolve()
    mixed = brush_root / "integrations/speedpainting/mixed_sft"
    prime_sft = brush_root / "integrations/prime/watercolour_sft"
    sys.path[:0] = [str(mixed), str(prime_sft)]
    from datasets import Dataset
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.trainer.model import setup_processor, setup_tokenizer
    from renderers.base import create_renderer

    source = mixed / "train.py"
    spec = importlib.util.spec_from_file_location("structured_flow_warmup_strict_sft", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing reviewed strict SFT source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolved = SFTConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    tokenizer = setup_tokenizer(resolved.tokenizer)
    renderer = create_renderer(tokenizer, resolved.renderer)
    renderer._processor = setup_processor(resolved.model)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    result: list[dict[str, Any]] = []
    dataset_dir = Path(resolved.data.name)
    paths = {"train": dataset_dir / "train.jsonl", "validation": dataset_dir / "validation.jsonl"}
    previous_capacity = os.environ.get("CAPACITY_STATE_COMPLETE")
    try:
        for mask_mode, capacity_flag in _processor_mask_modes():
            if capacity_flag is None:
                os.environ.pop("CAPACITY_STATE_COMPLETE", None)
            else:
                os.environ["CAPACITY_STATE_COMPLETE"] = capacity_flag
            for split, path in paths.items():
                raw_rows = _read_jsonl(path)
                if not raw_rows:
                    raise ValueError(f"{split} is empty: {path}")
                wrapper = module.StrictSFTDataset(Dataset.from_list([_training_projection(raw_rows[0])]), renderer,
                                                  seq_len=2**62, multimodal=True,
                                                  loss_mask_config=resolved.data.loss_mask)
                for raw in raw_rows:
                    row = _training_projection(raw)
                    wrapper.dataset = Dataset.from_list([row])
                    try:
                        sample = wrapper._process(row)
                    except Exception as exc:
                        if mask_mode == "capacity_prefill":
                            raise
                        result.append({"id": raw["id"], "split": split, "source_group": raw.get("source_group"),
                                       "mask_mode": mask_mode, "capacity_state_complete": False,
                                       "status": "diagnostic_processor_error", "error_type": type(exc).__name__,
                                       "total_tokens": 0, "assistant_tokens": 0, "expected_assistant_tokens": 0,
                                       "target_token_delta": 0, "supervised_tokens": 0, "extra_supervised_tokens": 0,
                                       "fits": False, "exact_target_match": False, "images": 0})
                        continue
                    if sample is None:
                        if mask_mode == "capacity_prefill":
                            raise ValueError(f"{raw.get('id')}: no trainable assistant tokens")
                        result.append({"id": raw["id"], "split": split, "source_group": raw.get("source_group"),
                                       "mask_mode": mask_mode, "capacity_state_complete": False,
                                       "status": "diagnostic_no_trainable_tokens", "total_tokens": 0,
                                       "assistant_tokens": 0, "expected_assistant_tokens": 0,
                                       "target_token_delta": 0, "supervised_tokens": 0,
                                       "extra_supervised_tokens": 0, "fits": False,
                                       "exact_target_match": False, "images": len(_image_parts(row["messages"][1]["content"]))})
                        continue
                    total = len(sample["input_ids"])
                    masks = list(sample["loss_mask"])
                    learned = [token for token, mask in zip(sample["target_ids"], masks) if mask]
                    target = "".join(_text_parts(row["messages"][-1]["content"]))
                    expected = tokenizer.encode(target, add_special_tokens=False) + [eos]
                    observation = _mask_observation(mask_mode, total, masks, learned, expected, resolved.data.seq_len)
                    result.append({"id": raw["id"], "split": split, "source_group": raw.get("source_group"),
                                   "mask_mode": mask_mode, "capacity_state_complete": capacity_flag == "1", **observation,
                                   "mask_sha256": sha256_bytes(bytes(int(bool(mask)) for mask in masks)),
                                   "target_sha256": sha256_text(target),
                                   "images": len(_image_parts(row["messages"][1]["content"]))})
    finally:
        if previous_capacity is None:
            os.environ.pop("CAPACITY_STATE_COMPLETE", None)
        else:
            os.environ["CAPACITY_STATE_COMPLETE"] = previous_capacity
    mode_reports = {
        mode: {
            "capacity_state_complete": flag == "1",
            "train_rows": sum(item["split"] == "train" and item["mask_mode"] == mode for item in result),
            "validation_rows": sum(item["split"] == "validation" and item["mask_mode"] == mode for item in result),
            "exact_target_matches": sum(item.get("exact_target_match") is True and item["mask_mode"] == mode for item in result),
        }
        for mode, flag in _processor_mask_modes()
    }
    report = {"schema": "structured-flow-sft-mask-audit-v1", "status": "passed",
              "seq_len": resolved.data.seq_len, "mask": "assistant_only_exact_target_plus_im_end",
              "mask_modes": mode_reports, "runtime_capacity_state_complete": True,
              "no_truncation": True, "processor_class": type(renderer._processor).__name__,
              "train_rows": len(_read_jsonl(paths["train"])),
              "validation_rows": len(_read_jsonl(paths["validation"])),
              "max_tokens": max((r["total_tokens"] for r in result), default=0), "examples": result,
              "config_sha256": sha256_file(config_path),
              "dataset_sha256": {split: sha256_file(path) for split, path in paths.items()}}
    _write_json(output, report)
    return report


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty warmup directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parent = verify_parent_adapter(args.parent_adapter)
    receipt = validate_inputs(args.train, args.dev, args.approval_manifest)
    train_rows = _read_jsonl(args.train.resolve())
    dev_rows = _read_jsonl(args.dev.resolve())
    scheduled_rows = build_schedule(train_rows)
    data_dir = output / "data"
    data_dir.mkdir()
    def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
        raw = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
        path.write_bytes(raw)
        return sha256_bytes(raw)
    source_train_sha = write_rows(data_dir / "source-train.jsonl", train_rows)
    train_sha = write_rows(data_dir / "train.jsonl", scheduled_rows)
    dev_sha = write_rows(data_dir / "dev.jsonl", dev_rows)
    validation_sha = write_rows(data_dir / "validation.jsonl", dev_rows)
    schedule = build_schedule_receipt(scheduled_rows, train_sha)
    _write_json(output / "schedule.json", schedule)
    validate_warmup_schedule(data_dir, output / "schedule.json")
    config = build_config(args.model_dir, data_dir, args.run_name)
    write_toml(output / "warmup.toml", config)
    _write_json(output / "config.json", config)
    _write_json(output / "resolved.setup.json", {
        "schema": "structured-flow-sft-initialization-v1", "initialization_mode": "trained_adapter",
        "init_adapter": parent["path"], "parent_adapter_sha256": parent["sha256"],
        "parent_adapter_revision": parent["revision"], "parent_adapter_prefix": parent["prefix"],
        "optimizer": "fresh", "native_resume": False, "before_first_forward": True,
        "prime_commit": PRIME_COMMIT, "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
        "max_steps": MAX_STEPS, "batch_size": BATCH_SIZE, "micro_batch_size": MICRO_BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "rank": LORA_RANK, "alpha": LORA_ALPHA,
    })
    receipt.update({"parent_adapter": parent, "config": {"path": str(output / "warmup.toml"), "sha256": sha256_file(output / "warmup.toml")},
                    "data": {"source_train_sha256": source_train_sha, "train_sha256": train_sha,
                             "dev_sha256": dev_sha, "validation_sha256": validation_sha,
                             "source_train_rows": len(train_rows), "scheduled_train_rows": len(scheduled_rows), "dev_rows": len(dev_rows),
                             "flow_state_pattern": [list(pattern) for pattern in WARMUP_STATE_PATTERN],
                             "flow_state_pool_offsets": dict(WARMUP_STATE_POOL_OFFSETS),
                             "schedule": {"path": str(output / "schedule.json"), "sha256": sha256_file(output / "schedule.json"),
                                          "schema": WARMUP_SCHEDULE_SCHEMA, "mode": schedule["mode"]},
                             "validation_alias": "dev", "input_train": str(args.train.resolve()), "input_dev": str(args.dev.resolve())},
                    "training": {"max_steps": MAX_STEPS, "batch_size": BATCH_SIZE, "micro_batch_size": MICRO_BATCH_SIZE,
                                 "gpus_per_node": 1, "num_train_gpus": 1, "learning_rate": LEARNING_RATE,
                                 "fresh_optimizer": True, "rank": LORA_RANK, "tensor_contract": LORA_TENSORS},
                    "prime_commit": PRIME_COMMIT, "status": "prepared_requires_node_processor_audit"})
    _write_json(output / "warmup-input-receipt.json", receipt)
    return receipt


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source, target_is_directory=source.is_dir())


def materialize_source_layout(source_root: Path, output: Path) -> Path:
    """Make a read-only-looking runtime layout, including packaged direct dirs."""
    source_root = source_root.expanduser().resolve()
    full_mixed = source_root / "integrations/speedpainting/mixed_sft"
    full_prime = source_root / "integrations/prime/watercolour_sft"
    packaged = False
    if not full_mixed.is_dir() or not full_prime.is_dir():
        packaged_mixed = source_root / "mixed_sft"
        packaged_prime = source_root / "watercolour_sft"
        if packaged_mixed.is_dir() and packaged_prime.is_dir():
            full_mixed, full_prime = packaged_mixed, packaged_prime
            packaged = True
        else:
            raise FileNotFoundError("warmup source must be full brush-rl or sft_source/{mixed_sft,watercolour_sft}")
    source_manifest = source_root / "manifest.json"
    if packaged and not source_manifest.is_file():
        raise FileNotFoundError("packaged warmup source manifest is missing")
    if packaged:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != "structured-flow-sft-source-manifest-v1" or manifest.get("prime_commit") != PRIME_COMMIT:
            raise ValueError("packaged warmup source manifest is not bound to the pinned Prime commit")
        for relative, expected in manifest.get("files", {}).items():
            path = source_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"packaged warmup helper hash mismatch: {relative}")
    for required in ("train.py", "scheduled_data.py", "group_validation.py", "update_audit.py"):
        if not (full_mixed / required).is_file():
            raise FileNotFoundError(f"reviewed mixed-SFT helper is missing: {required}")
    for required in ("init_adapter.py", "export_adapter.py", "model_profiles.py"):
        if not (full_prime / required).is_file():
            raise FileNotFoundError(f"reviewed watercolour helper is missing: {required}")
    runtime = output / "runtime-source"
    if runtime.exists():
        raise FileExistsError(f"refusing to replace existing warmup runtime source: {runtime}")
    runtime.mkdir(parents=True)
    mixed_dest = runtime / "integrations/speedpainting/mixed_sft"
    prime_dest = runtime / "integrations/prime/watercolour_sft"
    for source in full_mixed.iterdir():
        if source.name == "__pycache__":
            continue
        if packaged and source.name == "train.py":
            mixed_dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, mixed_dest / source.name)
        else:
            _link_or_copy(source, mixed_dest / source.name)
    for source in full_prime.iterdir():
        if source.name == "__pycache__":
            continue
        _link_or_copy(source, prime_dest / source.name)
    scripts = source_root / "scripts"
    if not scripts.is_dir() and packaged and source_root.parent.joinpath("scripts").is_dir():
        scripts = source_root.parent / "scripts"
    if not (scripts / "upload_checkpoints.py").is_file():
        raise FileNotFoundError("reviewed scripts/upload_checkpoints.py is required for public checkpoint verification")
    _link_or_copy(scripts, runtime / "scripts")
    if packaged:
        _link_or_copy(source_manifest, runtime / "manifest.json")
    return runtime


def source_identity(source_root: Path) -> dict[str, Any]:
    """Record the reviewed helper hashes without recording credentials or weights."""
    source_root = source_root.resolve()
    relative_files = [
        "integrations/speedpainting/mixed_sft/train.py",
        "integrations/speedpainting/mixed_sft/scheduled_data.py",
        "integrations/speedpainting/mixed_sft/group_validation.py",
        "integrations/speedpainting/mixed_sft/update_audit.py",
        "integrations/prime/watercolour_sft/init_adapter.py",
        "integrations/prime/watercolour_sft/export_adapter.py",
        "integrations/prime/watercolour_sft/model_profiles.py",
        "scripts/upload_checkpoints.py",
    ]
    values = {relative: sha256_file(source_root / relative) for relative in relative_files}
    manifest = source_root / "manifest.json"
    return {"source_root": str(source_root), "prime_commit": PRIME_COMMIT, "files": values,
            "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None}


def build_train_command(runtime_root: Path, config_path: Path, python: str = sys.executable) -> list[str]:
    train = runtime_root / "integrations/speedpainting/mixed_sft/train.py"
    return [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1", str(train), "@", str(config_path)]


def train(args: argparse.Namespace) -> dict[str, Any]:
    if sys.platform != "linux":
        raise RuntimeError("warmup training is Linux-node only")
    prepared = args.prepared.expanduser().resolve()
    input_receipt = json.loads((prepared / "warmup-input-receipt.json").read_text(encoding="utf-8"))
    if input_receipt.get("status") != "prepared_requires_node_processor_audit":
        raise ValueError("warmup input receipt is not a prepared warmup")
    parent = verify_parent_adapter(args.parent_adapter)
    config_value = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    if config_value.get("run", {}).get("name") != args.run_name:
        raise ValueError("warmup run name differs between prepared config and node launch")
    if parent["sha256"] != input_receipt.get("parent_adapter", {}).get("sha256"):
        raise ValueError("warmup parent adapter differs from the prepared Flash12 parent")
    verify_prime_pin(args.prime_root)
    runtime = materialize_source_layout(args.source_root, prepared)
    validate_warmup_schedule(
        prepared / "data", prepared / "schedule.json",
        runtime / "integrations/speedpainting/mixed_sft/scheduled_data.py",
    )
    _write_json(prepared / "source-receipt.json", source_identity(runtime))
    setup = runtime / "runs" / f"{args.run_name}-setup"
    setup.mkdir(parents=True)
    shutil.copyfile(prepared / "warmup.toml", setup / "resolved.toml")
    shutil.copyfile(prepared / "resolved.setup.json", setup / "resolved.setup.json")
    audit_path = prepared / "sft-mask-audit.json"
    audit_config_path = prepared / "warmup.toml"
    processor_audit(audit_config_path, audit_path, runtime)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    token_path = Path(os.environ.get("HF_TOKEN_PATH", "")).expanduser()
    if not token_path.is_file():
        raise ValueError("HF_TOKEN_PATH must point to the node credential file; it is never copied to receipts")
    repo = os.environ.get("HF_CHECKPOINT_REPO", "")
    upload_run = os.environ.get("HF_CHECKPOINT_RUN", "")
    if not repo or not RUN_NAMESPACE.fullmatch(upload_run):
        raise ValueError("HF_CHECKPOINT_REPO and a unique HF_CHECKPOINT_RUN are required for public checkpoint uploads")
    warmup_gpu = resolve_warmup_gpu()
    env = os.environ.copy()
    env.update({"BRUSH_ROOT": str(runtime), "PRIME_ROOT": str(args.prime_root.resolve()),
                "PRL_OUTPUT_DIR": str(output), "FRESH_BASE": "0", "WATERCOLOUR_INIT_ADAPTER": parent["path"],
                "CAPACITY_STATE_COMPLETE": "1", "ADAPTER_ONLY_CHECKPOINTS": "1",
                "MIXED_SFT_TRACE": str(output / "batch-shapes.jsonl"), "BRUSH_MODEL_PROFILE": "qwen38_27b",
                "HF_CHECKPOINT_REPO": repo, "HF_CHECKPOINT_RUN": upload_run, "CUDA_VISIBLE_DEVICES": warmup_gpu,
                "OMP_NUM_THREADS": "8", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"})
    existing_pythonpath = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in (
        runtime / "integrations/speedpainting/mixed_sft", runtime / "integrations/prime/watercolour_sft", runtime / "scripts",
        *[Path(item) for item in existing_pythonpath],
    ))
    command = build_train_command(runtime, setup / "resolved.toml", args.python)
    subprocess.run(command, cwd=args.prime_root.resolve(), env=env, check=True)
    _write_json(prepared / "warmup-training-receipt.json", {
        "schema": "structured-flow-sft-training-receipt-v1", "status": "training_finished",
        "gpu_ordinal": warmup_gpu, "prime_commit": PRIME_COMMIT, "hf_checkpoint_run": upload_run,
        "parent_adapter_sha256": parent["sha256"], "mask_audit": str(audit_path), "command": command,
    })
    return {"status": "training_finished", "output_dir": str(output), "audit": str(audit_path), "command": command,
            "parent_adapter": parent, "prime_commit": PRIME_COMMIT, "hf_checkpoint_run": upload_run,
            "warmup_gpu": warmup_gpu}


def _checkpoint_dirs(root: Path) -> list[Path]:
    roots = [root / "artifacts/adapters", root / "train-output/artifacts/adapters"]
    found = [path for base in roots if base.is_dir() for path in base.glob("step_[0-9]*")]
    if not found:
        found = [path for path in root.glob("**/artifacts/adapters/step_[0-9]*") if path.is_dir()]
    return sorted({path.resolve() for path in found}, key=lambda path: int(path.name.split("_", 1)[1]))


def validate_checkpoint(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"step_[0-9]+", path.name):
        raise ValueError(f"invalid adapter checkpoint directory: {path}")
    weight = path / "adapter_model.safetensors"
    validation_path = path / "validation.json"
    upload_path = path / "hf-upload.json"
    if not all(file.is_file() and not file.is_symlink() for file in (weight, validation_path, upload_path)) or not (path / ".finished").is_file():
        raise ValueError(f"checkpoint is not a verified public export: {path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    upload = json.loads(upload_path.read_text(encoding="utf-8"))
    actual = sha256_file(weight)
    if validation.get("finite") is not True or validation.get("sha256") != actual:
        raise ValueError(f"checkpoint validation hash/finite flag failed: {path}")
    if upload.get("verified") is not True or upload.get("private") is not False:
        raise ValueError(f"checkpoint upload is not verified public: {path}")
    return {"step": int(path.name.split("_", 1)[1]), "path": str(path), "sha256": actual,
            "upload": {"repo_id": upload.get("repo_id"), "commit": upload.get("commit"), "url": upload.get("url"), "verified": True, "private": False}}


def validate_optimizer_audit(path: Path) -> dict[str, Any]:
    """Require four finite native optimizer updates with changed tensors."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"optimizer update audit is missing: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != MAX_STEPS:
        raise ValueError(f"expected {MAX_STEPS} optimizer audit records, got {len(records)}")
    for expected_step, record in enumerate(records, 1):
        if record.get("completed_updates") != expected_step or record.get("all_finite") is not True:
            raise ValueError("optimizer audit does not confirm four finite ordered updates")
        if int(record.get("changed_elements", 0)) <= 0 or int(record.get("changed_tensors", 0)) <= 0:
            raise ValueError("optimizer audit reports an update with no changed adapter tensors")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "updates": MAX_STEPS,
            "all_finite": True, "changed_elements": [int(record["changed_elements"]) for record in records]}


def find_optimizer_audit(root: Path) -> Path:
    candidates = sorted({path.resolve() for path in root.glob("**/optimizer-update-audit.jsonl") if path.is_file() and not path.is_symlink()})
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one optimizer update audit under {root}, got {len(candidates)}")
    return candidates[0]


def write_final_receipt(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    input_receipt = json.loads((output / "warmup-input-receipt.json").read_text(encoding="utf-8"))
    parent = verify_parent_adapter(args.parent_adapter)
    if parent["sha256"] != input_receipt.get("parent_adapter", {}).get("sha256"):
        raise ValueError("final receipt parent does not match prepared Flash12 parent")
    checkpoints = [validate_checkpoint(path) for path in _checkpoint_dirs(args.checkpoint_root.expanduser().resolve())]
    if [item["step"] for item in checkpoints] != [MAX_STEPS]:
        raise ValueError(f"expected exactly one public checkpoint at step {MAX_STEPS}, got {[item['step'] for item in checkpoints]}")
    optimizer_audit_path = find_optimizer_audit(args.checkpoint_root.expanduser().resolve())
    optimizer_audit = validate_optimizer_audit(optimizer_audit_path)
    training_receipt_path = output / "warmup-training-receipt.json"
    if training_receipt_path.is_symlink() or not training_receipt_path.is_file():
        raise ValueError("warmup training receipt is missing")
    training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    if training_receipt.get("status") != "training_finished" or training_receipt.get("gpu_ordinal") not in {"0", "1"}:
        raise ValueError("warmup training receipt does not confirm the bounded GPU run")
    adapter_path = Path(checkpoints[-1]["path"])
    selected_sha = checkpoints[-1]["sha256"]
    if selected_sha == parent["sha256"]:
        raise ValueError("warmup adapter bytes are unchanged from the pinned Flash12 parent")
    pointer = output / "warmup-adapter-path.txt"
    pointer.write_text(str(adapter_path) + "\n", encoding="utf-8")
    receipt = {"schema": "structured-flow-sft-warmup-receipt-v1", "run_id": RUN_ID, "status": "verified_public_adapter_ready_for_rl",
               "parent_adapter": parent, "prime_commit": PRIME_COMMIT, "training": input_receipt["training"],
               "data": input_receipt["data"], "mask_audit": {"path": str(output / "sft-mask-audit.json"), "sha256": sha256_file(output / "sft-mask-audit.json")},
               "source": json.loads((output / "source-receipt.json").read_text(encoding="utf-8")),
               "training_receipt": {"path": str(training_receipt_path), "sha256": sha256_file(training_receipt_path), "gpu_ordinal": training_receipt["gpu_ordinal"]},
               "checkpoints": checkpoints, "optimizer_audit": optimizer_audit, "optimizer_updates_verified": MAX_STEPS,
               "adapter_changed_from_parent": {"changed": True, "parent_sha256": parent["sha256"], "warmup_sha256": selected_sha},
               "selected_adapter_path": str(adapter_path), "selected_adapter_sha256": selected_sha,
               "adapter_path_file": str(pointer), "public_uploads": True}
    _write_json(output / "warmup-receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--train", type=Path, required=True); p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--parent-adapter", type=Path, required=True); p.add_argument("--run-name", default="structured-flow-sft-warmup")
    p.add_argument("--approval-manifest", type=Path)
    p.set_defaults(handler=prepare)
    p = commands.add_parser("audit")
    p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--brush-root", type=Path, required=True)
    p.set_defaults(handler=lambda a: processor_audit(a.config.resolve(), a.output.resolve(), a.brush_root.resolve()))
    p = commands.add_parser("train")
    p.add_argument("--prepared", type=Path, required=True); p.add_argument("--source-root", type=Path, required=True); p.add_argument("--prime-root", type=Path, required=True)
    p.add_argument("--parent-adapter", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--run-name", default="structured-flow-sft-warmup"); p.add_argument("--python", default=sys.executable)
    p.set_defaults(handler=train)
    p = commands.add_parser("receipt")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--checkpoint-root", type=Path, required=True); p.add_argument("--parent-adapter", type=Path, required=True)
    p.set_defaults(handler=write_final_receipt)
    args = parser.parse_args(argv)
    value = args.handler(args)
    print(json.dumps({key: value[key] for key in ("status", "counts", "train_rows", "dev_rows", "selected_adapter_sha256") if key in value}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
