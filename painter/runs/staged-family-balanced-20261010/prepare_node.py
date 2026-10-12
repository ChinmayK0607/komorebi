#!/usr/bin/env python3
"""Fail-closed, finite preparation for a staged Linux node.

The command only verifies already-materialized local inputs and writes a
node-private config/receipt.  It never downloads, invokes CUDA, contacts a
service, starts Prime, or creates a background process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
SPEC = importlib.util.spec_from_file_location("staged_prepare_config", RUN_ROOT / "prepare_config.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("prepare_config.py is unavailable")
CONFIG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONFIG
SPEC.loader.exec_module(CONFIG)
RESOURCE_SPEC = importlib.util.spec_from_file_location("staged_resource_preflight", RUN_ROOT / "resource_preflight.py")
if RESOURCE_SPEC is None or RESOURCE_SPEC.loader is None:
    raise RuntimeError("resource_preflight.py is unavailable")
RESOURCE = importlib.util.module_from_spec(RESOURCE_SPEC)
sys.modules[RESOURCE_SPEC.name] = RESOURCE
RESOURCE_SPEC.loader.exec_module(RESOURCE)


RUN_ID = CONFIG.RUN_ID
PRIME_COMMIT = CONFIG.PRIME_COMMIT
BASE_MODEL_REVISION = CONFIG.BASE_MODEL_REVISION
POLICY_SHA256 = CONFIG.POLICY["sha256"]
FROZEN_SFT_SHA256 = CONFIG.FROZEN_SFT["sha256"]
RENDERER_SHA256 = "dc6b9d837ed7be382784a6c84de1b64911506b17a19b499ce054c0c572e75794"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_linux(system_name: str | None = None) -> None:
    actual = system_name or platform.system()
    if actual != "Linux":
        raise RuntimeError(f"staged Prime preparation requires Linux; found {actual}")


def _directory(path: Path, description: str, *, create: bool = False) -> Path:
    path = path.expanduser()
    if path.exists() and path.is_symlink():
        raise ValueError(f"{description} cannot be a symlink: {path}")
    if create:
        if path.exists() and not path.is_dir():
            raise ValueError(f"{description} is not a directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"{description} is not a directory: {path}")
    return path.resolve(strict=True)


def _file(path: Path, description: str, *, executable: bool = False) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is not a regular file: {path}")
    if executable and (path.stat().st_mode & 0o111) == 0:
        raise ValueError(f"{description} is not executable: {path}")
    return path.resolve(strict=True)


def _hf_snapshot_file(path: Path, description: str) -> Path:
    """Accept a normal HF snapshot link only when it stays in its cache."""

    path = path.expanduser().absolute()
    if not path.exists() and not path.is_symlink():
        raise ValueError(f"{description} is missing: {path}")
    if not path.is_symlink():
        if not path.is_file():
            raise ValueError(f"{description} is not a regular file: {path}")
        return path
    cache_root: Path | None = None
    current = path.parent
    while current != current.parent:
        if current.name == "snapshots":
            cache_root = current.parent.absolute()
            break
        current = current.parent
    if cache_root is None or not (cache_root / "blobs").is_dir():
        raise ValueError(f"{description} is not below a trusted Hugging Face snapshot cache: {path}")
    try:
        path.relative_to(cache_root)
    except ValueError as exc:
        raise ValueError(f"{description} escapes its Hugging Face cache root: {path}") from exc
    resolved_cache_root = cache_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_cache_root)
    except ValueError as exc:
        raise ValueError(f"{description} symlink escapes its Hugging Face cache root: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{description} symlink does not resolve to a regular file: {path}")
    return path


def _runtime_executable(path: Path, description: str, runtime_root: Path) -> Path:
    """Accept only a normal venv interpreter link and preserve that link."""

    path = path.expanduser().absolute()
    runtime_root = runtime_root.expanduser().absolute()
    try:
        path.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(f"{description} is outside its runtime root: {path}") from exc
    if not path.exists() and not path.is_symlink():
        raise ValueError(f"{description} is missing: {path}")
    target = path.resolve(strict=True)
    if not target.is_file() or (target.stat().st_mode & 0o111) == 0:
        raise ValueError(f"{description} does not resolve to an executable file: {path}")
    if path.is_symlink():
        allowed = [runtime_root.resolve(strict=True)]
        allowed_targets: list[Path] = []
        pyvenv = runtime_root / "pyvenv.cfg"
        if pyvenv.is_file() and not pyvenv.is_symlink():
            for line in pyvenv.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("home ="):
                    home = Path(line.split("=", 1)[1].strip()).expanduser()
                    if home.is_absolute():
                        allowed.append(home.resolve(strict=True))
                elif line.strip().lower().startswith("executable ="):
                    executable = Path(line.split("=", 1)[1].strip()).expanduser()
                    if executable.is_absolute():
                        allowed_targets.append(executable.resolve(strict=True))
        if target not in allowed_targets and not any(target == root or root in target.parents for root in allowed):
            raise ValueError(f"{description} resolves outside its trusted interpreter roots: {target}")
        current = target
        while True:
            if current.stat().st_mode & 0o022:
                raise ValueError(f"{description} target is writable by group/other: {current}")
            if current.parent == current:
                break
            current = current.parent
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_revision(prime_root: Path) -> str:
    """Read the checkout identity without fetching or mutating Git state."""

    try:
        result = subprocess.run(
            ["git", "-C", str(prime_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot read pinned Prime revision: {exc}") from exc
    revision = result.stdout.strip()
    if revision != PRIME_COMMIT:
        raise RuntimeError(f"Prime checkout is {revision!r}, expected {PRIME_COMMIT}")
    return revision


def _validate_protocol(source_root: Path) -> dict[str, Any]:
    protocol_path = _file(source_root / "PROTOCOL.json", "staged protocol")
    config_path = _file(source_root / "config.json", "staged config")
    curriculum_path = _file(source_root / "curriculum-manifest.json", "staged curriculum")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema") != "staged-family-balanced-protocol-v1":
        raise ValueError("staged protocol schema is incorrect")
    if protocol.get("run_id") != RUN_ID:
        raise ValueError("staged protocol run ID is incorrect")
    source_ids = protocol.get("source_ids")
    if not isinstance(source_ids, dict) or source_ids.get("prime_revision") != PRIME_COMMIT:
        raise ValueError("staged protocol Prime revision is not pinned")
    config_sha = ((protocol.get("config") or {}).get("sha256"))
    manifest_sha = ((protocol.get("curriculum") or {}).get("manifest_sha256"))
    if config_sha != sha256_file(config_path):
        raise ValueError("staged config hash does not match PROTOCOL.json")
    if manifest_sha != sha256_file(curriculum_path):
        raise ValueError("staged curriculum hash does not match PROTOCOL.json")
    return {
        "path": protocol_path.name,
        "sha256": sha256_file(protocol_path),
        "config_sha256": config_sha,
        "curriculum_sha256": manifest_sha,
    }


def _validate_adapter(path: Path, expected_sha256: str, description: str) -> dict[str, Any]:
    path = _directory(path, description)
    weights = _hf_snapshot_file(path / "adapter_model.safetensors", f"{description} weights")
    config_path = _hf_snapshot_file(path / "adapter_config.json", f"{description} config")
    actual = sha256_file(weights)
    if actual != expected_sha256:
        raise ValueError(f"{description} SHA-256 is {actual}, expected {expected_sha256}")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    expected_modules = set(CONFIG.TARGET_MODULES)
    modules = metadata.get("target_modules")
    if metadata.get("r") != 16 or metadata.get("lora_alpha") != 32 or set(modules or []) != expected_modules:
        raise ValueError(f"{description} LoRA rank/alpha/target modules do not match the pinned profile")
    return {
        "path": str(path),
        "weights_sha256": actual,
        "config_sha256": sha256_file(config_path),
    }


def _validate_materialized(materialized_root: Path) -> dict[str, Any]:
    materialized_root = _directory(materialized_root, "materialized root")
    receipt_path = _file(materialized_root / "materialization-receipt.json", "materialization receipt")
    train_path = _file(materialized_root / "train-manifest.json", "materialized train manifest")
    heldout_path = _file(materialized_root / "heldout-manifest.json", "materialized heldout manifest")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("status") != "materialized_with_verified_frozen_sft512_anchors":
        raise ValueError("materialization receipt is not an admitted verified receipt")
    counts = receipt.get("counts")
    if receipt.get("run_id") != RUN_ID or not isinstance(counts, dict) or counts.get("train_rows") != 96 or counts.get("heldout_rows") != 32:
        raise ValueError("materialized receipt has the wrong staged identity or row counts")
    if receipt.get("checkpoint_sha256") != FROZEN_SFT_SHA256:
        raise ValueError("materialized receipt has the wrong frozen SFT checkpoint")
    outputs = receipt.get("output_manifests")
    if not isinstance(outputs, dict):
        raise ValueError("materialized receipt lacks output manifest hashes")
    if outputs.get("train", {}).get("sha256") != sha256_file(train_path) or outputs.get("heldout", {}).get("sha256") != sha256_file(heldout_path):
        raise ValueError("materialized manifest bytes do not match the receipt")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    rows = train.get("rows") if isinstance(train, dict) else None
    cases = heldout.get("rows") if isinstance(heldout, dict) else None
    if not isinstance(rows, list) or len(rows) != 96 or len({row.get("id") for row in rows}) != 96:
        raise ValueError("materialized train manifest must contain 96 unique rows")
    if not isinstance(cases, list) or len(cases) != 32 or len({row.get("id") for row in cases}) != 32:
        raise ValueError("materialized heldout manifest must contain 32 unique rows")
    for split, split_rows in (("train", rows), ("heldout", cases)):
        for row in split_rows:
            if not isinstance(row, dict) or row.get("split") != split:
                raise ValueError(f"materialized {split} row split is malformed")
            for key in ("reference", "opponent", "anchor_provenance_file"):
                value = row.get(key)
                if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ValueError(f"materialized {split} row has an unsafe {key} path")
                candidate = materialized_root / value
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError(f"materialized {split} row asset is missing or symlinked: {candidate}")
            if row.get("reference_kind") != "text" or row.get("frozen_sft512_checkpoint_sha256") != FROZEN_SFT_SHA256:
                raise ValueError(f"materialized {split} row has incompatible native identity")
            if sha256_file(materialized_root / row["reference"]) != row.get("reference_sha256"):
                raise ValueError(f"materialized {split} reference hash mismatch: {row.get('id')}")
            if sha256_file(materialized_root / row["opponent"]) != row.get("opponent_sha256"):
                raise ValueError(f"materialized {split} opponent hash mismatch: {row.get('id')}")
    return {
        "root": str(materialized_root),
        "receipt_sha256": sha256_file(receipt_path),
        "train_manifest_sha256": sha256_file(train_path),
        "heldout_manifest_sha256": sha256_file(heldout_path),
        "train_rows": 96,
        "heldout_rows": 32,
    }


def expose_materialized_runtime(materialized_root: Path) -> Path:
    """Make validated materialized inputs readable by the painter sandbox.

    This function is intentionally limited to the materialized tree. It
    rejects symlinks first, then grants directory traversal and regular-file
    read access without changing any model, adapter, source, or credential
    path on the node.
    """

    root = materialized_root.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"materialized root is not a regular directory: {root}")
    paths = [root, *sorted(root.rglob("*"), key=lambda path: (len(path.parts), str(path)))]
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"materialized runtime contains a symlink: {path}")
        if not path.exists():
            raise ValueError(f"materialized runtime entry disappeared: {path}")
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    return root


def prepare_node(
    *,
    source_root: Path,
    node_root: Path,
    materialized_root: Path,
    prime_root: Path,
    model_dir: Path,
    policy_adapter: Path,
    frozen_adapter: Path,
    renderer: Path,
    renderer_python: Path,
    renderer_root: Path,
    repo_root: Path = REPO_ROOT,
    gpus_per_node: int = 4,
    num_train_gpus: int = 2,
    num_infer_gpus: int = 2,
    trainer_memory_gib: int | None = None,
) -> dict[str, Any]:
    require_linux()
    source_root = _directory(source_root, "staged source root")
    repo_root = _directory(repo_root, "repository root")
    node_root = Path(node_root).expanduser().absolute()
    if node_root.exists():
        if node_root.is_symlink() or not node_root.is_dir():
            raise ValueError("node root must be a non-symlink directory")
        if any(node_root.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty node root: {node_root}")
    else:
        node_root.mkdir(parents=True, exist_ok=False)
    prime_root = _directory(prime_root, "Prime root")
    renderer_root = _directory(renderer_root, "renderer root")
    prime_python = _runtime_executable(prime_root / ".venv/bin/python", "pinned Prime Python", prime_root / ".venv")
    revision = _git_revision(prime_root)
    protocol_receipt = _validate_protocol(source_root)
    materialized = _validate_materialized(materialized_root)
    expose_materialized_runtime(Path(materialized["root"]))
    model_dir = _directory(model_dir, "base model directory")
    if model_dir.name != BASE_MODEL_REVISION:
        raise ValueError(f"base model directory must end in {BASE_MODEL_REVISION}")
    if not (model_dir / "config.json").is_file():
        raise ValueError("base model config.json is missing")
    policy = _validate_adapter(policy_adapter, POLICY_SHA256, "Flash12 policy initializer")
    frozen = _validate_adapter(frozen_adapter, FROZEN_SFT_SHA256, "frozen SFT512 opponent")
    resource_preflight = RESOURCE.inspect_resource_hooks(
        prime_root, expected_world_size=4, trainer_memory_gib=trainer_memory_gib,
    )
    renderer = _file(renderer, "renderer")
    try:
        renderer.absolute().relative_to(renderer_root.absolute())
    except ValueError as exc:
        raise ValueError("renderer is outside the verified renderer root") from exc
    if sha256_file(renderer) != RENDERER_SHA256:
        raise ValueError("renderer does not match the pinned renderer SHA-256")
    renderer_python = _runtime_executable(renderer_python, "renderer Python", renderer_python.parent.parent)
    try:
        renderer_python.absolute().relative_to(renderer_root.absolute())
    except ValueError as exc:
        raise ValueError("renderer Python is outside the verified renderer root") from exc

    config_path = node_root / "native.toml"
    config_receipt_path = node_root / "config-receipt.json"
    config = CONFIG.build_config(
        model=model_dir,
        node_root=node_root,
        materialized_root=Path(materialized["root"]),
        repo_root=repo_root,
        renderer=renderer,
        renderer_python=renderer_python,
        gpus_per_node=gpus_per_node,
        num_train_gpus=num_train_gpus,
        num_infer_gpus=num_infer_gpus,
    )
    config_receipt = CONFIG.write_config(
        output=config_path,
        receipt_output=config_receipt_path,
        config=config,
        node_root=node_root,
        materialized_root=Path(materialized["root"]),
    )
    receipt = {
        "schema": "staged-family-balanced-node-preparation-v1",
        "status": "prepared_no_process_started",
        "run_id": RUN_ID,
        "prime_revision": revision,
        "prime_root": str(prime_root),
        "prime_python": str(prime_python),
        "source_root": str(source_root),
        "node_root": str(node_root),
        "materialized": materialized,
        "protocol": protocol_receipt,
        "model": {"path": str(model_dir), "revision": BASE_MODEL_REVISION, "config_sha256": sha256_file(model_dir / "config.json")},
        "policy_initializer": policy,
        "frozen_sft": frozen,
        "renderer": {"root": str(renderer_root), "path": str(renderer), "sha256": RENDERER_SHA256, "python": str(renderer_python)},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path), "receipt_sha256": sha256_file(config_receipt_path)},
        "config_contract": {"batch_size": 24, "group_size": 4, "constant_trainer_batch_size": True, "optimizer_updates": 20, "checkpoint_interval": 4, "train_rows": 96, "heldout_rows": 32},
        "resource_preflight": resource_preflight,
        "trainer_memory_gib": trainer_memory_gib,
        "prime_validation": "deferred_to_canary",
    }
    _atomic_json(node_root / "prepare-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--prime-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--policy-adapter", type=Path, required=True)
    parser.add_argument("--frozen-adapter", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--renderer-python", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-train-gpus", type=int, default=2)
    parser.add_argument("--num-infer-gpus", type=int, default=2)
    parser.add_argument("--trainer-memory-gib", type=int)
    args = parser.parse_args(argv)
    receipt = prepare_node(**vars(args))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
