#!/usr/bin/env python3
"""Preserve a stopped revision-RL run without copying weights into evidence.

The script is intentionally node-oriented. It archives only allowlisted run
evidence, uploads discovered nonzero LoRA adapter steps to an existing public
HF model repository, and verifies the uploaded files before reporting success.
It never starts training, rents/releases a node, connects over SSH, or deletes
source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/root/painter")
DEFAULT_RUN = "revision-rl-luna-20260920"
HF_REPO = "CK0607/qwen3.8-27b-brush-painting"
ARCHIVE_SUFFIXES = {".tar", ".gz", ".tgz", ".zip", ".7z"}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".dcp"}
WEIGHT_DIR_PARTS = {
    "weights",
    "optimizer",
    "optimizer-state",
    "optimizer_state",
    "native-checkpoint",
    "native_checkpoint",
    "full-checkpoint",
    "full_checkpoint",
    "checkpoints",
}
NATIVE_HANDOFF_BIN_RE = re.compile(r"^training-handoffs/step_\d+/rank_\d+\.bin$")
HANDOFF_STEP_JSON_RE = re.compile(r"^training-handoffs/step_\d+/[^/]+\.json$")
HANDOFF_ROOT_JSON_NAMES = frozenset({
    "training-handoff.json",
    "training-state.json",
    "training-preflight.json",
    "training-ready.json",
})
SENSITIVE_NAME_PARTS = {
    "token",
    "tokens",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "private-key",
    "private_key",
    "known_hosts",
    "authorized_keys",
}
STEP_RE = re.compile(r"^step[_-](\d+)$", re.IGNORECASE)
TOP_JSON_RE = re.compile(
    r"^(?:experiment|config|setup|supervisor|status|metrics|audit)(?:[-_.].*)?\.json$",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|BEGIN [^-\r\n]+ PRIVATE KEY|"
    rb"Authorization\s*:\s*Bearer\s+\S+|"
    rb"(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{16,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceFile:
    source: Path
    arcname: str


@dataclass(frozen=True)
class AdapterStep:
    step: int
    directory: Path
    weights: Path
    config: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def has_sensitive_name(path: Path, allow_blinded_reward_key: bool = False) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    lower_name = path.name.lower()
    if allow_blinded_reward_key and lower_name == "private-key.json" and "reward-batches" in lower_parts:
        return False
    return bool(lower_parts & SENSITIVE_NAME_PARTS) or any(
        marker in lower_name for marker in SENSITIVE_NAME_PARTS
    ) or any(
        marker in lower_name for marker in ("hf-token", "hf_token", "api-key", "api_key")
    )


def has_sensitive_content(path: Path, allow_blinded_reward_key: bool = False) -> bool:
    if allow_blinded_reward_key and path.name.lower() == "private-key.json" and "reward-batches" in {part.lower() for part in path.parts}:
        return False
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return False
    try:
        with path.open("rb") as handle:
            overlap = b""
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    return False
                probe = overlap + block
                if CREDENTIAL_RE.search(probe):
                    return True
                overlap = probe[-256:]
    except (OSError, UnicodeError):
        return True


def excluded_evidence_file(
    path: Path,
    output_dir: Path,
    allow_blinded_reward_key: bool = False,
    allow_native_transport: bool = False,
) -> bool:
    if path.is_symlink() or not path.is_file():
        return True
    if is_within(path, output_dir):
        return True
    if has_sensitive_name(path, allow_blinded_reward_key) or has_sensitive_content(path, allow_blinded_reward_key):
        return True
    if path.suffix.lower() in ARCHIVE_SUFFIXES or path.name.lower().endswith(".tar.gz"):
        return True
    if path.suffix.lower() in WEIGHT_SUFFIXES and not allow_native_transport:
        return True
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & WEIGHT_DIR_PARTS:
        return True
    return False


def add_evidence(
    result: dict[str, EvidenceFile],
    source: Path,
    arcname: str,
    output_dir: Path,
    allow_blinded_reward_key: bool = False,
    allow_native_transport: bool = False,
) -> None:
    if source.is_symlink():
        return
    source = source.resolve()
    arcname = str(Path(arcname))
    allow_native_transport = allow_native_transport and bool(NATIVE_HANDOFF_BIN_RE.fullmatch(arcname))
    if excluded_evidence_file(source, output_dir, allow_blinded_reward_key, allow_native_transport):
        return
    if arcname.startswith("../") or "/../" in arcname or arcname.startswith("/"):
        raise ValueError(f"unsafe archive path: {arcname}")
    result.setdefault(arcname, EvidenceFile(source=source, arcname=arcname))


def collect_tree(
    result: dict[str, EvidenceFile],
    base: Path,
    arc_prefix: str,
    output_dir: Path,
    allow_blinded_reward_key: bool = False,
) -> None:
    if not base.exists() or base.is_symlink():
        return
    if base.is_file():
        add_evidence(result, base, arc_prefix, output_dir, allow_blinded_reward_key)
        return
    for current, directories, files in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if not (current_path / name).is_symlink()
            and name not in {"__pycache__", ".cache", ".git"}
            and name.lower() not in WEIGHT_DIR_PARTS
        )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(base)
            add_evidence(result, path, str(Path(arc_prefix) / relative), output_dir, allow_blinded_reward_key)


def collect_training_handoffs(
    result: dict[str, EvidenceFile],
    root: Path,
    output_dir: Path,
) -> None:
    """Retain only native rank payloads and direct per-step JSON summaries."""
    base = root / "training-handoffs"
    if not base.is_dir() or base.is_symlink():
        return
    for step_dir in sorted(base.iterdir()):
        if (not step_dir.is_dir() or step_dir.is_symlink()
                or not re.fullmatch(r"step_\d+", step_dir.name)):
            continue
        for path in sorted(step_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                continue
            arcname = str(path.relative_to(root))
            if NATIVE_HANDOFF_BIN_RE.fullmatch(arcname):
                add_evidence(result, path, arcname, output_dir, allow_native_transport=True)
            elif HANDOFF_STEP_JSON_RE.fullmatch(arcname):
                add_evidence(result, path, arcname, output_dir)


def collect_evidence(root: Path, run_name: str, output_dir: Path) -> dict[str, EvidenceFile]:
    files: dict[str, EvidenceFile] = {}

    # Current source at the node root and the run's sibling Python/shell files.
    for path in sorted(root.iterdir() if root.exists() else []):
        if path.is_file() and (path.suffix.lower() in {".py", ".sh", ".js", ".cjs"} or TOP_JSON_RE.match(path.name)):
            add_evidence(files, path, path.name, output_dir)
    run_source = Path(__file__).resolve().parent
    if is_within(run_source, root):
        for path in sorted(run_source.iterdir()):
            if path.is_file() and path.suffix.lower() in {".py", ".sh", ".js", ".cjs"}:
                add_evidence(files, path, str(path.relative_to(root.resolve())), output_dir)

    for directory in ("logs", "rl-rollouts", "judge-queue", "evaluation", "rl-states"):
        collect_tree(files, root / directory, directory, output_dir)
    collect_tree(files, root / "reward-batches", "reward-batches", output_dir, allow_blinded_reward_key=True)
    collect_training_handoffs(files, root, output_dir)
    for name in sorted(HANDOFF_ROOT_JSON_NAMES):
        add_evidence(files, root / name, name, output_dir)

    async_root = root / "async-output" / run_name
    # The live layout has both `configs/attempt_N` and root-level metrics or
    # chunks. The same file filter excludes weights, native checkpoints,
    # optimizer state, duplicate archives, and credential-bearing files.
    collect_tree(files, async_root, str(Path("async-output") / run_name), output_dir)
    for name in (
        "rl-states.json",
        "evaluation-manifest.json",
        "evaluation-campaign.json",
        "judge-provenance-audit.json",
        "async-config-receipt.json",
        "metrics.jsonl",
    ):
        add_evidence(files, root / name, name, output_dir)
    return files


def discover_adapters(root: Path, run_name: str, selected_steps: set[int] | None) -> tuple[list[AdapterStep], list[dict[str, str]]]:
    """Find step_N LoRA directories without descending into model/checkpoint state."""
    search_roots = [root / "async-output" / run_name, root / "broadcasts", root / "adapters", root / "artifacts"]
    found: dict[int, list[AdapterStep]] = {}
    skip_dirs = {
        ".git",
        ".cache",
        "__pycache__",
        "node_modules",
        "runtime",
        "renderer-env",
        "prime-rl",
        "weights",
        "optimizer",
        "optimizer-state",
        "native-checkpoint",
        "native_checkpoint",
        "checkpoints",
    }
    seen_roots: set[Path] = set()
    for search_root in search_roots:
        search_root = search_root.resolve()
        if search_root in seen_roots or not search_root.exists():
            continue
        seen_roots.add(search_root)
        for current, directories, _files in os.walk(search_root, topdown=True, followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(
                name
                for name in directories
                if not (current_path / name).is_symlink() and name.lower() not in skip_dirs
            )
            match = STEP_RE.match(current_path.name)
            if not match:
                continue
            step = int(match.group(1))
            weights = current_path / "adapter_model.safetensors"
            config = current_path / "adapter_config.json"
            if not (weights.is_file() and config.is_file()):
                continue
            if selected_steps is not None and step not in selected_steps:
                continue
            if selected_steps is None and step == 0:
                continue
            found.setdefault(step, []).append(AdapterStep(step, current_path, weights, config))

    chosen: list[AdapterStep] = []
    duplicates: list[dict[str, str]] = []
    for step in sorted(found):
        candidates = sorted(
            found[step],
            key=lambda item: (
                0 if "broadcast" in {part.lower() for part in item.directory.parts} else 1,
                len(item.directory.parts),
                str(item.directory),
            ),
        )
        chosen.append(candidates[0])
        for duplicate in candidates[1:]:
            duplicates.append({"step": str(step), "skipped": str(duplicate.directory), "kept": str(candidates[0].directory)})
    return chosen, duplicates


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_archive(
    root: Path,
    output_dir: Path,
    run_name: str,
    files: dict[str, EvidenceFile],
    adapters: list[AdapterStep],
    duplicates: list[dict[str, str]],
) -> tuple[Path, Path, dict[str, object]]:
    reason = (
        "DO_NOT_USE: run stopped after invalid judge provenance. These adapter files are preserved for audit only; "
        "they are not evidence of learning, must not be promoted, and must not be used for further training."
    )
    marker = {
        "status": "DO_NOT_USE",
        "reason": reason,
        "run_name": run_name,
        "judge_provenance": "invalid",
        "quality_improvement_validated": False,
        "source_steps": [adapter.step for adapter in adapters],
    }
    marker_path = output_dir / "INVALID_RUN.json"
    write_json(marker_path, marker)
    # The marker is created inside the output directory, which is otherwise
    # excluded from source discovery; include this generated quarantine record
    # explicitly without opening the output tree to recursive collection.
    files["INVALID_RUN.json"] = EvidenceFile(source=marker_path.resolve(), arcname="INVALID_RUN.json")
    for adapter in adapters:
        add_evidence(
            files,
            adapter.config,
            str(Path("adapter-configs") / f"step_{adapter.step}" / "adapter_config.json"),
            output_dir,
        )

    manifest_files = []
    for arcname in sorted(files):
        source = files[arcname].source
        manifest_files.append({"path": arcname, "bytes": source.stat().st_size, "sha256": sha256_file(source)})
    manifest = {
        "schema": 1,
        "run_name": run_name,
        "contains_weights": False,
        "contains_raw_hf_logs": False,
        "files": manifest_files,
        "duplicate_adapter_steps_skipped": duplicates,
    }
    manifest_path = output_dir / "evidence-manifest.json"
    write_json(manifest_path, manifest)

    archive = output_dir / f"{run_name}-invalid-judge-provenance-evidence.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")
    with tarfile.open(partial, mode="w:gz") as handle:
        for entry in manifest_files:
            handle.add(files[entry["path"]].source, arcname=entry["path"], recursive=False)
        handle.add(manifest_path, arcname="evidence-manifest.json", recursive=False)
    partial.replace(archive)
    receipt = {
        "schema": 1,
        "run_name": run_name,
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        "manifest": {"path": manifest_path.name, "files": len(manifest_files), "contains_weights": False},
        "adapter_steps": [adapter.step for adapter in adapters],
        "duplicate_adapter_steps_skipped": duplicates,
        "status": "local_evidence_ready",
    }
    receipt_path = output_dir / "preservation-receipt.json"
    write_json(receipt_path, receipt)
    return archive, receipt_path, marker


def read_token() -> tuple[str, Path]:
    token_path_value = os.environ.get("HF_TOKEN_PATH")
    if not token_path_value:
        raise RuntimeError("HF_TOKEN_PATH is required for the authorized HF upload")
    token_path = Path(token_path_value).expanduser()
    if not token_path.is_file():
        raise RuntimeError("HF_TOKEN_PATH does not point to a readable private file")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("HF_TOKEN_PATH is empty")
    return token, token_path


def upload_and_verify(
    adapters: list[AdapterStep], marker: dict[str, object], run_name: str, repo_id: str
) -> tuple[list[dict[str, object]], str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - node dependency
        raise RuntimeError("huggingface_hub is required on the node") from exc

    token, _token_path = read_token()
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", token=token)
    if getattr(info, "private", False):
        raise RuntimeError("authorized checkpoint repository is not public")

    expected: list[dict[str, object]] = []
    for adapter in adapters:
        if has_sensitive_name(adapter.config) or has_sensitive_content(adapter.config):
            raise RuntimeError(f"adapter config appears credential-bearing: {adapter.config.name}")
        prefix = f"{run_name}/invalid-judge-provenance/step_{adapter.step}"
        entries = (
            (adapter.weights, f"{prefix}/adapter_model.safetensors"),
            (adapter.config, f"{prefix}/adapter_config.json"),
        )
        step_marker = dict(marker)
        step_marker["step"] = adapter.step
        step_marker["adapter_source"] = str(adapter.directory)
        step_marker["adapter_sha256"] = sha256_file(adapter.weights)
        marker_bytes = (json.dumps(step_marker, indent=2, sort_keys=True) + "\n").encode("utf-8")
        marker_path = f"{prefix}/INVALID_RUN.json"
        entries = (*entries, (marker_bytes, marker_path))
        for local, remote in entries:
            local_sha = sha256_file(local) if isinstance(local, Path) else sha256_bytes(local)
            if isinstance(local, Path):
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remote,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                    commit_message=f"Quarantine invalid-judge RL adapter step {adapter.step}",
                )
                size = local.stat().st_size
            else:
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remote,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                    commit_message=f"Quarantine invalid-judge RL metadata step {adapter.step}",
                )
                size = len(local)
            expected.append({"path": remote, "sha256": local_sha, "bytes": size, "step": adapter.step})

    final_info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True, token=token)
    siblings = {getattr(item, "rfilename", ""): item for item in getattr(final_info, "siblings", [])}
    verified: list[dict[str, object]] = []
    small_limit = 16 * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="hf-verify-") as cache_dir:
        for entry in expected:
            remote = siblings.get(str(entry["path"]))
            if remote is None:
                raise RuntimeError(f"HF upload missing from remote tree: {entry['path']}")
            lfs = getattr(remote, "lfs", None)
            remote_lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if remote_lfs_sha:
                if remote_lfs_sha != entry["sha256"]:
                    raise RuntimeError(f"remote LFS hash mismatch: {entry['path']}")
                method = "remote_lfs_sha256"
            elif int(entry["bytes"]) <= small_limit:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=str(entry["path"]),
                    revision=getattr(final_info, "sha", None),
                    repo_type="model",
                    token=token,
                    cache_dir=cache_dir,
                    force_download=True,
                )
                if sha256_file(Path(downloaded)) != entry["sha256"]:
                    raise RuntimeError(f"remote small-file hash mismatch: {entry['path']}")
                method = "download_sha256"
            else:
                raise RuntimeError(f"no remote LFS hash for large upload: {entry['path']}")
            verified.append({**entry, "verification": method})
    hf_commit_sha = getattr(final_info, "sha", None)
    if not hf_commit_sha:
        raise RuntimeError("HF repository did not return an immutable commit sha")
    return verified, hf_commit_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="node painter root (default: /root/painter)")
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, help="local node preservation directory")
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--include-step", action="append", type=int, help="upload exactly this step; repeatable")
    parser.add_argument("--required-step", type=int, default=1, help="fail unless this step is found (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="discover and report without writing or uploading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"root does not exist: {root}")
    output_dir = (args.output or (root / "preservation" / f"{args.run_name}-invalid-judge-provenance")).expanduser().resolve()
    if is_within(output_dir, root) and output_dir == root:
        raise RuntimeError("output directory cannot be the root")
    selected_steps = set(args.include_step) if args.include_step else None
    adapters, duplicates = discover_adapters(root, args.run_name, selected_steps)
    if args.required_step not in {adapter.step for adapter in adapters}:
        raise RuntimeError(f"required adapter step_{args.required_step} was not found")
    files = collect_evidence(root, args.run_name, output_dir)
    summary = {
        "run_name": args.run_name,
        "root": str(root),
        "evidence_files": len(files),
        "adapter_steps": [adapter.step for adapter in adapters],
        "duplicate_adapter_steps_skipped": duplicates,
        "contains_weights": False,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform != "linux":
        raise RuntimeError("preservation must run on the Linux node")
    archive, receipt_path, marker = build_archive(root, output_dir, args.run_name, files, adapters, duplicates)
    try:
        verified, hf_commit_sha = upload_and_verify(adapters, marker, args.run_name, args.repo_id)
    except Exception as exc:
        failed = {
            "status": "hf_upload_unverified",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "archive": {"path": archive.name, "sha256": sha256_file(archive), "bytes": archive.stat().st_size},
            "adapter_steps": [adapter.step for adapter in adapters],
            "contains_weights": False,
        }
        write_json(receipt_path, failed)
        raise
    success = {
        "status": "preserved_and_hf_verified",
        "repo_id": args.repo_id,
        "archive": {"path": archive.name, "sha256": sha256_file(archive), "bytes": archive.stat().st_size},
        "adapter_steps": [adapter.step for adapter in adapters],
        "hf_files": verified,
        "hf_commit_sha": hf_commit_sha,
        "contains_weights": False,
        "raw_logs_uploaded_to_hf": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(receipt_path, success)
    print(json.dumps(success, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"preservation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
