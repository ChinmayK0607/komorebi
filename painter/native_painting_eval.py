"""Prepare a reproducible current-pin ``vf eval`` config.

This helper validates the reviewed reference manifest and writes only a small
TOML config. It does not import a model, render an image, or start an eval.
The generated command runs the installed ``NativePaintingEnv`` for one policy;
invoke it twice with different served LoRA model ids and the same config values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path


DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CONTEXT = 32768
DEFAULT_ROLLOUT_TIMEOUT = 1200
DEFAULT_SEED = 918200
DEFAULT_MAX_CONCURRENT = 4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_evaluation_artifacts(artifact_root: Path) -> dict[str, int | str]:
    """Verify that every valid renderer receipt has its hash-bound PNG.

    Prime can preserve a ``painting.json`` receipt when the corresponding
    canvas is absent from a collected run.  Such a receipt is useful failure
    evidence, but it must not count as a completed image for the generation
    receipt.  Keep this audit local to the explicit evaluation artifact root;
    it never follows paths recorded inside JSON.
    """
    raw_root = artifact_root.expanduser()
    if raw_root.is_symlink():
        raise ValueError("evaluation artifact root must not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation artifact root is missing: {root}")
    receipts = sorted(root.rglob("painting.json"))
    if not receipts:
        raise ValueError(f"evaluation artifact root contains no painting receipts: {root}")

    valid_receipts = 0
    valid_with_png = 0
    invalid_receipts = 0
    missing_png = 0
    hash_mismatch = 0
    for receipt_path in receipts:
        if receipt_path.is_symlink():
            raise ValueError("evaluation artifact receipt must not be a symlink")
        try:
            receipt_path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("evaluation artifact receipt resolves outside its root") from exc
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid painting receipt: {receipt_path}") from exc
        if not isinstance(receipt, dict):
            raise ValueError(f"painting receipt is not an object: {receipt_path}")
        png_path = receipt_path.with_name("painting.png")
        if png_path.is_symlink():
            raise ValueError("evaluation artifact PNG must not be a symlink")
        try:
            png_path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("evaluation artifact PNG resolves outside its root") from exc
        if receipt.get("valid") is not True:
            invalid_receipts += 1
            continue
        valid_receipts += 1
        if not png_path.is_file():
            missing_png += 1
            continue
        expected = receipt.get("png_sha256")
        if not isinstance(expected, str) or _sha(png_path) != expected:
            hash_mismatch += 1
            continue
        valid_with_png += 1

    if missing_png or hash_mismatch:
        raise ValueError(
            "valid painting receipts are missing hash-verified PNGs: "
            f"missing_png={missing_png}, hash_mismatch={hash_mismatch}"
        )
    if valid_receipts == 0:
        raise ValueError("evaluation produced no valid painting receipt")
    return {
        "receipt_count": len(receipts),
        "valid_receipts": valid_receipts,
        "valid_receipts_with_png": valid_with_png,
        "invalid_receipts": invalid_receipts,
        "missing_png": missing_png,
        "hash_mismatch": hash_mismatch,
    }


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_manifest(root: Path, manifest: Path, expected_count: int | None = None) -> dict:
    if expected_count is not None and (isinstance(expected_count, bool) or expected_count < 1):
        raise ValueError("expected_count must be a positive integer when supplied")
    if manifest.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    else:
        raw = json.loads(manifest.read_text())
        rows = raw if isinstance(raw, list) else raw.get("cases") or raw.get("eval") or raw.get("tasks") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest must contain a non-empty list of cases, found {len(rows) if isinstance(rows, list) else 'non-list'}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} manifest cases, found {len(rows)}")
    groups: set[str] = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest cases must be objects")
        case_id = row.get("id") or row.get("task_id") or row.get("case_id")
        group = row.get("source_group") or row.get("group") or case_id
        ref = row.get("reference_png") or row.get("reference")
        if not all(isinstance(value, str) and value for value in (case_id, group, ref)):
            raise ValueError("each case needs id, source_group/group, and reference_png/reference")
        if group in groups:
            raise ValueError(f"duplicate source group: {group}")
        groups.add(group)
        path = Path(ref) if Path(ref).is_absolute() else manifest.parent / ref
        if not path.is_file():
            path = root / ref
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha(path)
        expected = row.get("reference_sha256")
        if expected is None and isinstance(row.get("sha256"), dict):
            expected = row["sha256"].get("png")
        elif expected is None:
            expected = row.get("sha256")
        if expected and expected != digest:
            raise ValueError(f"reference hash mismatch for {case_id}: {digest}")
        normalized.append({"id": case_id, "source_group": group, "reference": str(path), "sha256": digest})
    return {"count": len(normalized), "groups": sorted(groups), "cases": normalized}


def config_text(
    *,
    taskset_id: str,
    root: Path,
    manifest: Path,
    policy_label: str,
    model: str,
    output_dir: Path,
    count: int = 12,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_turns: int = DEFAULT_MAX_TURNS,
    rollout_timeout: int = DEFAULT_ROLLOUT_TIMEOUT,
    context_length: int = DEFAULT_CONTEXT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    renderer: Path | None = None,
    renderer_python: Path | None = None,
) -> str:
    _positive_int("count", count)
    _positive_int("max_tokens", max_tokens)
    _positive_int("max_turns", max_turns)
    _positive_int("rollout_timeout", rollout_timeout)
    _positive_int("context_length", context_length)
    _positive_int("max_concurrent", max_concurrent)
    resolved_output_dir = output_dir.expanduser().resolve()
    evaluation_output_root = resolved_output_dir / "evaluation-rollouts"
    return f'''# Generated by painter/native_painting_eval.py; no credentials or weights.
model = {json.dumps(model)}
num_tasks = {count}
num_rollouts = 1
max_concurrent = {max_concurrent}
push = false
output_dir = {json.dumps(str(resolved_output_dir))}

[client]
base_url = "http://127.0.0.1:8000/v1"
api_key_var = "PAINTER_LOCAL_API_KEY"

[sampling]
temperature = 1.0
max_tokens = {max_tokens}
seed = {DEFAULT_SEED}
extra_body = {{ chat_template_kwargs = {{ enable_thinking = false }} }}

[env.taskset]
id = {json.dumps(taskset_id)}
root = {json.dumps(str(root))}
manifest = {json.dumps(str(manifest))}
expected_count = {count}
max_turns = {max_turns}
policy_label = {json.dumps(policy_label)}
output_root = {json.dumps(str(evaluation_output_root))}
{f'renderer = {json.dumps(str(renderer.expanduser().resolve()))}' if renderer else ''}
{f'renderer_python = {json.dumps(str(renderer_python.expanduser().resolve()))}' if renderer_python else ''}

[env.agent]
max_turns = {max_turns}
# Deliberately omit env.agent.max_output_tokens: current-pin treats that as a
# cumulative per-rollout cap. sampling.max_tokens above is per request.

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"

[env.agent.timeout]
rollout = {rollout_timeout}

# The serving endpoint must expose at least this context length.
# native_painting_eval does not add an unknown Prime config key for it.
# context_length = {context_length}
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--taskset-id", default="native_painting")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--rollout-timeout", type=int, default=DEFAULT_ROLLOUT_TIMEOUT)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--renderer-python", type=Path)
    args = parser.parse_args(argv)
    audit = validate_manifest(args.root, args.manifest, expected_count=args.expected_count)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(config_text(
        taskset_id=args.taskset_id,
        root=args.root,
        manifest=args.manifest,
        policy_label=args.policy_label,
        model=args.model,
        output_dir=args.output_dir,
        count=audit["count"],
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        rollout_timeout=args.rollout_timeout,
        context_length=args.context_length,
        max_concurrent=args.max_concurrent,
        renderer=args.renderer,
        renderer_python=args.renderer_python,
    ))
    command = (
        f"cd {shlex.quote(str(args.root / 'prime-rl'))} && "
        f"PYTHONPATH={shlex.quote(str(args.root))} uv run eval @ {shlex.quote(str(args.config))} "
        "--no-serve --no-push"
    )
    print(json.dumps({
        "config": str(args.config),
        "cases": audit["count"],
        "groups": audit["groups"],
        "protocol": {"seed": DEFAULT_SEED, "temperature": 1.0, "max_turns": args.max_turns, "max_tokens": args.max_tokens, "context_length": args.context_length, "rollout_timeout": args.rollout_timeout, "max_concurrent": args.max_concurrent, "thinking": False},
        "command": command,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
