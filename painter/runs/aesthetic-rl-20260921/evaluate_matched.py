"""Judge-free matched evaluation for four held-out v2 painting tasks.

The model and renderer are deliberately reached only through the existing
Linux ``rollout.run`` harness.  This script records terminal canvas outcomes;
it never calls the aesthetic judge or adds a reward.  Baseline and candidate
runs use the same task subset, seed, turn cap, token cap, and temperature.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import functools
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


P = Path(__file__).resolve().parent
HELDOUT_GROUPS = (
    "v2-box-007",
    "v2-flower-007",
    "v2-bus-007",
    "v2-vase-007",
)
EVAL_SEED = 918200
TEMPERATURE = 1.0
MAX_TURNS = 3
MAX_TOKENS = 8192
LABELS = frozenset({"simplebaseline", "candidate"})

# The node bundle flattens the historical inputs directory, so ``rollout`` is
# available there.  The local checkout keeps it in the historical run; tests
# can replace this binding with a mock without importing model dependencies.
run: Callable[..., dict[str, Any]] | None = None


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def resolve_input(value: str) -> Path:
    """Resolve a relative input from the current bundle or historical inputs."""

    path = Path(value)
    if path.is_absolute():
        return path
    local = P / path
    if local.exists():
        return local
    historical = P.parent / "revision-rl-20260918" / "inputs" / path
    return historical


def _load_runner() -> Callable[..., dict[str, Any]]:
    global run
    if run is not None:
        return run
    try:
        from rollout import run as rollout_run
    except ModuleNotFoundError:
        historical = P.parent / "revision-rl-20260918" / "inputs"
        if not historical.is_dir():
            raise
        sys.path.insert(0, str(historical))
        try:
            from rollout import run as rollout_run
        finally:
            sys.path.remove(str(historical))
    run = rollout_run
    return rollout_run


def _force_temperature(runner: Callable[..., dict[str, Any]]) -> None:
    """Force the historical client call to the matched 1.0 temperature."""

    module = sys.modules.get(runner.__module__)
    if module is None or not hasattr(module, "completion"):
        return
    current = module.completion
    if getattr(current, "_matched_eval_temperature", None) == TEMPERATURE:
        return

    @functools.wraps(current)
    def completion(*args: Any, **kwargs: Any) -> Any:
        kwargs["temperature"] = TEMPERATURE
        return current(*args, **kwargs)

    completion._matched_eval_temperature = TEMPERATURE
    module.completion = completion


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc


def _task_subset(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("evaluation tasks must be a JSON list")
    selected = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("kind") == "paint"
        and row.get("source_group") in HELDOUT_GROUPS
    ]
    by_group = {row.get("source_group"): row for row in selected}
    if set(by_group) != set(HELDOUT_GROUPS) or len(selected) != len(HELDOUT_GROUPS):
        raise ValueError(
            "evaluation tasks must contain exactly one paint task for each held-out group: "
            + ", ".join(HELDOUT_GROUPS)
        )
    required = {"id", "reference", "source_group", "kind"}
    for row in selected:
        if not required <= set(row):
            raise ValueError(f"held-out paint task is missing fields: {row.get('id')}")
    return [by_group[group] for group in HELDOUT_GROUPS]


def _safe_label(value: str) -> str:
    if value not in LABELS:
        raise ValueError("label must be simplebaseline or candidate")
    return value


def _api_version(base_url: str) -> str:
    path = base_url.rstrip("/").rsplit("/", 1)[-1]
    return path if re.fullmatch(r"v[0-9]+(?:[.-][A-Za-z0-9_-]+)?", path) else "unspecified"


def _evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a JSON object")
    value = copy.deepcopy(config)
    limits = value.setdefault("operational_limits", {})
    if not isinstance(limits, dict):
        raise ValueError("operational_limits must be an object")
    limits["max_turns"] = MAX_TURNS
    limits["max_tokens_per_response"] = MAX_TOKENS
    return value


def _metadata(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "painter.matched-evaluation.v1",
        "label": args.label,
        "model": args.model,
        "policy_version": args.policy_version,
        "api_version": args.api_version or _api_version(args.base_url),
        "base_url": args.base_url,
        "seed": EVAL_SEED,
        "temperature": TEMPERATURE,
        "max_turns": MAX_TURNS,
        "max_tokens": MAX_TOKENS,
        "heldout_groups": list(HELDOUT_GROUPS),
        "config": config,
        "judge_reward": False,
    }


def _same_metadata(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = (
        "schema", "label", "model", "policy_version", "api_version", "base_url",
        "seed", "temperature", "max_turns", "max_tokens", "heldout_groups", "config",
        "judge_reward",
    )
    return all(existing.get(key) == current.get(key) for key in keys)


def _one(
    task: dict[str, Any],
    out: Path,
    endpoint: str,
    model: str,
    renderer: Path,
    renderer_python: Path,
    config: dict[str, Any],
    runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    folder = out / task["id"]
    summary = folder / "summary.json"
    if summary.exists():
        value = _load_json(summary)
        if not isinstance(value, dict):
            raise ValueError(f"existing rollout summary is not an object: {summary}")
        return dict(task=task, **value)
    result = runner(
        P / task["reference"],
        folder,
        endpoint,
        model,
        renderer,
        renderer_python,
        config,
        seed=EVAL_SEED,
        initial_canvas=None,
        initial_program=None,
    )
    if not isinstance(result, dict):
        raise ValueError("rollout runner returned a non-object summary")
    return dict(task=task, **result)


def _failure(row: dict[str, Any]) -> bool:
    return bool(row.get("error")) or row.get("valid_canvas") is not True


def main(argv: list[str] | None = None) -> dict[str, Any]:
    if sys.platform != "linux":
        raise SystemExit("Matched model/render evaluation runs on the Linux node only.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Served API model ID; never used as an output path")
    parser.add_argument("--label", required=True, choices=sorted(LABELS))
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--tasks", default="evaluation-tasks.json")
    parser.add_argument("--config", default="campaign.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--output", default="matched-evaluation")
    parser.add_argument("--renderer", default="vendor/integrations/watercolour/renderer.py")
    parser.add_argument("--renderer-python", default="renderer-env/bin/python")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    args.label = _safe_label(args.label)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    tasks = _task_subset(_load_json(resolve_input(args.tasks)))
    config = _evaluation_config(_load_json(resolve_input(args.config)))
    metadata = _metadata(args, config)
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = P / output_root
    out = (output_root / args.label).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.json"
    if metadata_path.exists() and not _same_metadata(_load_json(metadata_path), metadata):
        raise ValueError(f"existing evaluation metadata differs: {metadata_path}")
    atomic(metadata_path, metadata)

    lock = (out / "producer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    runner = _load_runner()
    _force_temperature(runner)
    renderer = resolve_input(args.renderer)
    renderer_python = resolve_input(args.renderer_python)
    rows: list[dict[str, Any]] = []
    expected = len(tasks)

    def one(task: dict[str, Any]) -> dict[str, Any]:
        try:
            return _one(task, out, args.base_url, args.model, renderer, renderer_python, config, runner)
        except Exception as exc:
            return {
                "task": task,
                "valid_canvas": False,
                "finished": False,
                "failure": f"{type(exc).__name__}: {exc}",
                "error": f"{type(exc).__name__}: {exc}",
            }

    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, expected)) as pool:
            futures = {pool.submit(one, task): task for task in tasks}
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: row["task"]["source_group"])
                atomic(out / "progress.json", {
                    **metadata,
                    "completed": len(rows),
                    "expected": expected,
                    "failures": sum(_failure(row) for row in rows),
                    "valid_canvas": sum(row.get("valid_canvas") is True for row in rows),
                    "finished": sum(row.get("finished") is True for row in rows),
                })
    finally:
        lock.close()

    summary = {
        **metadata,
        "completed": len(rows),
        "expected": expected,
        "failures": sum(_failure(row) for row in rows),
        "valid_canvas": sum(row.get("valid_canvas") is True for row in rows),
        "finished": sum(row.get("finished") is True for row in rows),
        "results": rows,
    }
    atomic(out / "summary.json", summary)
    print(json.dumps({
        "label": args.label,
        "model": args.model,
        "completed": summary["completed"],
        "expected": summary["expected"],
        "failures": summary["failures"],
        "output": str(out),
    }, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
