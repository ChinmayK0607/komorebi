"""Prepare deterministic fresh/correction curriculum orders from a review manifest.

The helper is pure data preparation.  It does not render, call a model, or
decide visual quality.  Fresh tasks are one task per source group; correction
tasks retain reviewed anchors but are scheduled round-robin by family and
source group so repeated anchors cannot occupy the early prefix.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "painter.curriculum-order.v1"
MODES = ("fresh", "correction")
DEFAULT_FRESH_LIMIT = 24
DEFAULT_CORRECTION_LIMIT = 8
DEFAULT_FIRST_N = 12
DEFAULT_MAX_TURNS = 3


def contract_snapshot() -> dict[str, Any]:
    """Return the exact run-local contract identity embedded in a plan."""
    path = Path(__file__).with_name("contract.py")
    if not path.is_file():
        raise FileNotFoundError(f"run contract is missing: {path}")
    spec = importlib.util.spec_from_file_location("curriculum_repair_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load run contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "path": path.name,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity": module.identity(),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("review manifest must be a JSON object")
    reviewed = (
        value.get("reviewed") is True
        or value.get("review_status") == "reviewed"
        or value.get("status") in {"reviewed", "approved", "candidate_approved"}
    )
    if not reviewed:
        raise ValueError("review manifest must explicitly set reviewed=true or review_status=reviewed")
    rows = value.get("tasks") or value.get("training_states") or value.get("train") or value.get("correction") or value.get("repair")
    if not isinstance(rows, list) or not rows:
        raise ValueError("review manifest must contain a non-empty supported task list")
    return value


def _raw_modes(row: dict[str, Any], default_mode: str | None = None) -> list[str]:
    value = row.get("modes", row.get("mode"))
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    if row.get("fresh") is True:
        values.append("fresh")
    if row.get("correction") is True:
        values.append("correction")
    if "both" in values:
        values = [item for item in values if item != "both"] + list(MODES)
    if not values and default_mode is not None:
        values = [default_mode]
    result = sorted(set(values), key=MODES.index) if values else []
    if not result or any(item not in MODES for item in result):
        raise ValueError(f"row {row.get('id', '<unknown>')} must declare fresh/correction mode")
    return result


def _anchor_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = row.get("anchors")
    if anchors is None:
        return [row]
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(f"row {row.get('id', '<unknown>')} has an invalid anchors list")
    expanded: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        if isinstance(anchor, dict):
            item = copy.deepcopy(row)
            item.pop("anchors", None)
            item.update(anchor)
            item.setdefault("anchor_id", anchor.get("id", f"anchor-{index}"))
        elif isinstance(anchor, str):
            item = copy.deepcopy(row)
            item.pop("anchors", None)
            item["anchor_id"] = anchor
            item["anchor"] = anchor
        else:
            raise ValueError(f"row {row.get('id', '<unknown>')} has a non-object anchor")
        expanded.append(item)
    return expanded


def _normalize(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "tasks" in manifest:
        sources = [(manifest["tasks"], None)]
    elif "training_states" in manifest:
        sources = [(manifest["training_states"], None)]
    else:
        sources = []
        if "train" in manifest:
            sources.append((manifest["train"], "fresh"))
        if "correction" in manifest:
            sources.append((manifest["correction"], "correction"))
        if "repair" in manifest:
            sources.append((manifest["repair"], "correction"))
    if not sources:
        raise ValueError("review manifest has no supported task rows")
    normalized: list[dict[str, Any]] = []
    global_index = 0
    for rows, default_mode in sources:
        if not isinstance(rows, list):
            raise ValueError("review manifest task source must be a list")
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError(f"manifest row {global_index} is not an object")
            raw = copy.deepcopy(raw)
            if default_mode == "correction" and raw.get("repair_id"):
                # Candidate-manifest repair rows name the reviewed intermediate
                # and complete target directly.  The complete target is the
                # reference; the intermediate is both the correction start and
                # the real pairwise anchor.  The reference is never reused as
                # the RL opponent.
                raw.setdefault("id", raw["repair_id"])
                raw.setdefault("reference", raw.get("target_png"))
                raw.setdefault("initial_canvas", raw.get("input_png"))
                raw.setdefault("initial_program", raw.get("input_program"))
                raw.setdefault("anchor", raw.get("input_png"))
            task_id = raw.get("id") or raw.get("task_id")
            group = raw.get("source_group")
            family = raw.get("family")
            reference = raw.get("reference") or raw.get("reference_png")
            if not all(isinstance(value, str) and value for value in (task_id, group, family, reference)):
                raise ValueError(f"manifest row {global_index} lacks id/source_group/family/reference")
            modes = _raw_modes(raw, default_mode=default_mode)
            expanded = _anchor_rows(raw)
            for anchor_index, item in enumerate(expanded):
                item.setdefault("id", task_id)
                item["task_id"] = task_id
                item["source_group"] = group
                item["family"] = family
                item["reference"] = reference
                item["_input_index"] = global_index
                item["_anchor_index"] = anchor_index
                item["_modes"] = modes
                item.setdefault("anchor_id", item.get("anchor") or task_id)
                normalized.append(item)
            global_index += 1
    return normalized


def _rows_for_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if mode in row["_modes"]]
    if mode == "fresh":
        # A fresh reference task has no opponent/anchor start.  Collapse any
        # repeated anchor records to one deterministic task per source group.
        by_group: dict[str, dict[str, Any]] = {}
        for row in selected:
            by_group.setdefault(row["source_group"], row)
        selected = list(by_group.values())
    else:
        seen: set[tuple[str, str]] = set()
        unique: list[dict[str, Any]] = []
        for row in selected:
            key = (row["source_group"], str(row["anchor_id"]))
            if key in seen:
                continue
            seen.add(key)
            if not isinstance(row.get("initial_canvas") or row.get("canvas"), str):
                raise ValueError(f"correction row {row['task_id']} lacks an initial canvas")
            if not isinstance(row.get("initial_program") or row.get("program"), str):
                raise ValueError(f"correction row {row['task_id']} lacks an initial program")
            unique.append(row)
        selected = unique
    if not selected:
        raise ValueError(f"review manifest has no {mode} rows")
    return selected


def _sort_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["family"]),
        str(row["source_group"]),
        int(row["_input_index"]),
        int(row["_anchor_index"]),
        str(row.get("anchor_id", "")),
    )


def order_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin families, then groups, then reviewed anchors."""
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[row["family"]][row["source_group"]].append(row)
    for family in buckets:
        for group in buckets[family]:
            buckets[family][group].sort(key=_sort_key)
    families = sorted(buckets)
    emitted: dict[tuple[str, str], int] = defaultdict(int)
    ordered: list[dict[str, Any]] = []
    while True:
        made_progress = False
        for family in families:
            groups = [group for group, values in buckets[family].items() if values]
            if not groups:
                continue
            group = min(groups, key=lambda item: (emitted[(family, item)], item))
            row = buckets[family][group].pop(0)
            emitted[(family, group)] += 1
            ordered.append(row)
            made_progress = True
        if not made_progress:
            return ordered


def _public_row(row: dict[str, Any], mode: str, order: int) -> dict[str, Any]:
    value = {
        key: copy.deepcopy(item)
        for key, item in row.items()
        if not key.startswith("_") and key not in {"modes", "fresh", "correction"}
    }
    value["mode"] = mode
    value["order"] = order
    value["task_id"] = row["task_id"]
    value["source_group"] = row["source_group"]
    value["family"] = row["family"]
    value["reference"] = row["reference"]
    return value


def coverage(rows: list[dict[str, Any]], first_n: int) -> dict[str, Any]:
    prefix = rows[:first_n]
    prefixes = []
    for count in range(1, min(first_n, len(rows)) + 1):
        if count <= 12 or count == min(first_n, len(rows)):
            subset = rows[:count]
            prefixes.append({
                "n": count,
                "groups": sorted({row["source_group"] for row in subset}),
                "families": sorted({row["family"] for row in subset}),
            })
    return {
        "first_n": first_n,
        "first_n_groups": sorted({row["source_group"] for row in prefix}),
        "first_n_families": sorted({row["family"] for row in prefix}),
        "prefixes": prefixes,
    }


def prepare_mode(
    manifest: dict[str, Any],
    mode: str,
    *,
    limit: int | None,
    first_n: int = DEFAULT_FIRST_N,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if first_n < 1:
        raise ValueError("first_n must be positive")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    rows = _rows_for_mode(_normalize(manifest), mode)
    ordered = order_rows(rows)
    if limit is None:
        limit = len(ordered)
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(ordered) < limit:
        raise ValueError(f"{mode} curriculum needs {limit} rows but reviewed manifest provides {len(ordered)}")
    selected = ordered[:limit]
    tasks = [_public_row(row, mode, index) for index, row in enumerate(selected)]
    return {
        "schema": SCHEMA,
        "source_manifest_schema": manifest.get("schema"),
        "contract_snapshot": contract_snapshot(),
        "mode": mode,
        "max_turns": max_turns,
        "policy": {
            "family_round_robin": True,
            "source_group_round_robin": True,
            "fresh_one_task_per_source_group": mode == "fresh",
            "repeated_anchor_protection": True,
            "selection_limit": limit,
        },
        "counts": {
            "tasks": len(tasks),
            "families": len({row["family"] for row in tasks}),
            "source_groups": len({row["source_group"] for row in tasks}),
        },
        "coverage": coverage(selected, first_n),
        "tasks": tasks,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "correction", "both"), default="both")
    parser.add_argument("--fresh-limit", type=int, default=DEFAULT_FRESH_LIMIT)
    parser.add_argument("--correction-limit", type=int, default=DEFAULT_CORRECTION_LIMIT)
    parser.add_argument("--first-n", type=int, default=DEFAULT_FIRST_N)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    args = parser.parse_args(argv)
    manifest = _load_manifest(args.manifest)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    modes = MODES if args.mode == "both" else (args.mode,)
    limits = {"fresh": args.fresh_limit, "correction": args.correction_limit}
    for mode in modes:
        prepared = prepare_mode(
            manifest,
            mode,
            limit=limits[mode],
            first_n=args.first_n,
            max_turns=args.max_turns,
        )
        prepared["source_manifest_sha256"] = manifest_sha256
        (args.output_dir / f"{mode}-curriculum.json").write_text(
            json.dumps(prepared, indent=2, sort_keys=True) + "\n"
        )
        outputs[mode] = prepared
    if args.mode == "both":
        combined = {
            "schema": SCHEMA,
            "source_manifest_schema": manifest.get("schema"),
            "contract_snapshot": contract_snapshot(),
            "source_manifest_sha256": manifest_sha256,
            "first_cycle": {"fresh": args.fresh_limit, "correction": args.correction_limit},
            "max_turns": args.max_turns,
            "modes": outputs,
        }
        (args.output_dir / "curriculum-plan.json").write_text(
            json.dumps(combined, indent=2, sort_keys=True) + "\n"
        )
    return outputs


if __name__ == "__main__":
    main()
