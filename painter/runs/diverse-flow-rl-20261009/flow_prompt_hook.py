#!/usr/bin/env python3
"""Inject the opt-in Diverse-flow hint into native RL task prompts.

The shared native taskset deliberately has no run-specific casebrief field.
This hook is loaded only by this run's ``sitecustomize.py`` and only when
``PAINTER_STRUCTURED_FLOW_POLICY=1``.  It preserves the native user brief and
adds the hash-verified casebrief system section to ``Task.data.system_prompt``
before task serialization, so the hint reaches both the training orchestrator
and verifier environment without changing shared painter code.
"""

from __future__ import annotations

import hashlib
import json
import os
import copy
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "Diverse-flow-prompt-hook-v1"
_INSTALLED = False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_casebrief(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"casebrief is not a regular file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.startswith("[SYSTEM]\n") or "\n\n[USER]\n" not in text:
        raise ValueError(f"casebrief markers are malformed: {path}")
    system, user = text[len("[SYSTEM]\n"):].split("\n\n[USER]\n", 1)
    user = user.rstrip("\n")
    if not system.strip() or not user.strip():
        raise ValueError(f"casebrief sections must be nonempty: {path}")
    return system.strip(), user.strip()


def _manifest_rows(env_name: str) -> tuple[dict[str, dict[str, Any]], str]:
    raw_manifest = os.environ.get(env_name)
    if not raw_manifest and env_name == "PAINTER_EVAL_MANIFEST":
        # The lightweight hook contract has only one fake taskset. On the
        # node, evaluate.sh always exports the separate matched manifest.
        raw_manifest = os.environ.get("PAINTER_RL_MANIFEST")
    manifest = Path(raw_manifest or "").resolve()
    if not manifest.is_file():
        raise ValueError(f"Diverse-flow manifest is missing: {manifest}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw.get("rows") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("Diverse-flow manifest does not contain rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Diverse-flow manifest contains a malformed row")
        ident = str(row.get("id") or row.get("task_id") or row.get("case_id") or "")
        if not ident or ident in result:
            raise ValueError(f"Diverse-flow manifest has duplicate/missing ID: {ident!r}")
        result[ident] = row
    return result, _sha(manifest)


def _brief_from_reference(task: Any) -> str:
    data = task.data
    reference = Path(str(data.reference))
    return reference.read_text(encoding="utf-8").strip()


def _validated_model_copy(value: Any, updates: Mapping[str, Any]) -> Any:
    """Return a validated copy of a Pydantic model when one is supplied.

    Prime's current ``TaskData`` and ``Task`` models are frozen.  Assigning a
    field in place works with the lightweight contract fake but raises during
    a real taskset load.  Revalidate the complete payload so the copy keeps
    the native model type and all immutable metadata.
    """

    model_dump = getattr(value, "model_dump", None)
    model_validate = getattr(type(value), "model_validate", None)
    if callable(model_dump) and callable(model_validate):
        payload = model_dump(mode="python")
        if not isinstance(payload, dict):
            raise TypeError(f"cannot copy {type(value).__name__}: model_dump is not an object")
        payload.update(updates)
        return model_validate(payload)

    # Keep compatibility with the pinned v1-style model API used by older
    # Prime images while preferring validated parsing over raw ``copy``.
    model_dict = getattr(value, "dict", None)
    parse_obj = getattr(type(value), "parse_obj", None)
    if callable(model_dict) and callable(parse_obj):
        payload = model_dict()
        if not isinstance(payload, dict):
            raise TypeError(f"cannot copy {type(value).__name__}: dict is not an object")
        payload.update(updates)
        return parse_obj(payload)

    copier = getattr(value, "model_copy", None) or getattr(value, "copy", None)
    if callable(copier):
        return copier(update=dict(updates))
    # The offline contract uses a SimpleNamespace task data object.  Preserve
    # that compatibility without mutating the original; real frozen Prime
    # models take one of the validated branches above.
    clone = copy.copy(value)
    try:
        for name, updated in updates.items():
            setattr(clone, name, updated)
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"task data does not support an immutable validated copy: {type(value).__name__}") from exc
    return clone


def _task_with_data(task: Any, data: Any) -> Any:
    """Return ``task`` with ``data`` replaced without mutating a frozen task."""

    model_copy = getattr(task, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"data": data})
    copier = getattr(task, "copy", None)
    if callable(copier):
        return copier(update={"data": data})
    clone = copy.copy(task)
    try:
        clone.data = data
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"task does not support an immutable data replacement: {type(task).__name__}") from exc
    return clone


def _patch_task(task: Any, row: Mapping[str, Any]) -> Any:
    data = task.data
    casebrief_value = row.get("casebrief")
    if not isinstance(casebrief_value, str) or not casebrief_value:
        raise ValueError(f"flow policy is enabled but {data.task_id} has no casebrief")
    casebrief = Path(casebrief_value).resolve()
    if _sha(casebrief) != row.get("casebrief_sha256"):
        raise ValueError(f"casebrief hash mismatch for {data.task_id}")
    system_hint, user_brief = _parse_casebrief(casebrief)
    if user_brief != _brief_from_reference(task):
        raise ValueError(f"casebrief user brief differs from native reference for {data.task_id}")
    marker = "\n[STRUCTURED_FLOW_POLICY]\n"
    if marker.strip() in str(data.system_prompt):
        return task
    patched_data = _validated_model_copy(
        data,
        {"system_prompt": str(data.system_prompt) + marker + system_hint},
    )
    return _task_with_data(task, patched_data)


def install() -> dict[str, Any]:
    global _INSTALLED
    if _INSTALLED:
        return {"schema": SCHEMA, "installed": True, "already_installed": True}
    if os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") != "1":
        return {"schema": SCHEMA, "installed": False, "enabled": False}
    train_rows, train_manifest_sha = _manifest_rows("PAINTER_RL_MANIFEST")
    eval_rows, eval_manifest_sha = _manifest_rows("PAINTER_EVAL_MANIFEST")

    def patch_taskset(taskset_cls: Any, rows: Mapping[str, Mapping[str, Any]]):
        original = taskset_cls.load

        def load(self):
            tasks = list(original(self))
            patched = []
            for task in tasks:
                ident = str(task.data.task_id)
                if ident not in rows:
                    raise ValueError(f"Diverse-flow task is absent from the manifest hook map: {ident}")
                patched.append(_patch_task(task, rows[ident]))
            return patched

        taskset_cls.load = load

    try:
        from painter.native_painting_rl import NativePaintingRLTaskset
    except ImportError:
        from native_painting_rl import NativePaintingRLTaskset  # type: ignore
    patch_taskset(NativePaintingRLTaskset, train_rows)
    patched_eval = False
    try:
        from painter.native_painting import NativePaintingTaskset
    except ImportError:
        try:
            from native_painting import NativePaintingTaskset  # type: ignore
        except ImportError:
            NativePaintingTaskset = None  # type: ignore
    if NativePaintingTaskset is not None:
        patch_taskset(NativePaintingTaskset, eval_rows)
        patched_eval = True

    # Keep a single explicit receipt for both policy and matched-baseline
    # paths. The eval taskset is optional in the lightweight contract test but
    # is present on the Linux Prime node.
    receipt_path_value = os.environ.get("PAINTER_FLOW_HOOK_RECEIPT")
    if receipt_path_value:
        path = Path(receipt_path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": SCHEMA,
            "installed": True,
            "enabled": True,
            "rl_manifest_sha256": train_manifest_sha,
            "eval_manifest_sha256": eval_manifest_sha,
            "rl_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "evaluation_taskset_patched": patched_eval,
        }, sort_keys=True) + "\n")
    _INSTALLED = True
    return {"schema": SCHEMA, "installed": True, "enabled": True, "rows": len(train_rows), "eval_rows": len(eval_rows), "evaluation_taskset_patched": patched_eval}


__all__ = ["install"]
