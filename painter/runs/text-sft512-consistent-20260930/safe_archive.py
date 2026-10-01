#!/usr/bin/env python3
"""Screen, package, and optionally publish one finite run's evidence.

The archive contains evidence and receipts only. Model weights, private
directories, provider connection material, caches, vendored runtimes, and
credentials are rejected or skipped before packaging. Upload verification
downloads the public object anonymously and compares its SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
from typing import Any


REPO_ID = "CK0607/komorebi-painter-experiments"
RUN_NAME = os.environ.get("PAINTER_RUN_NAME", "text-sft512-consistent-20260930")
TEXT_EXTENSIONS = frozenset({
    ".csv", ".html", ".js", ".json", ".jsonl", ".log", ".md", ".py", ".sh", ".toml", ".txt",
})
IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS
EXCLUDED_PARTS = frozenset({
    ".git", ".venv", "__pycache__", "cache", "credentials", "private",
    "archive-materialized", "local-evidence", "public-verify", "vendor", "weights",
})
EXCLUDED_NAMES = frozenset({
    "connection.json", "connection.txt", "known_hosts", "pod.json", "ssh_config",
})
WEIGHT_EXTENSIONS = frozenset({".bin", ".ckpt", ".gguf", ".pt", ".pth", ".safetensors"})
SECRET_PATTERNS = (
    # Match a value assignment, including JSON's quoted key/value form.  The
    # key name alone is harmless source/config documentation and must remain
    # archivable.
    re.compile(rb"AI_GATEWAY_API_KEY[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._-]{20,}"),
    re.compile(rb"Authorization[\"']?\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9._-]{20,}", re.I),
    re.compile(rb"(?:sk-or-v1|vck|hf)_[A-Za-z0-9_-]{20,}"),
    re.compile(rb"BEGIN (?:OPENSSH|RSA|EC|PRIVATE) KEY"),
)
SCREEN_SKIPS: list[dict[str, str]] = []


class ArchiveSafetyError(RuntimeError):
    """Evidence selection or publication would violate the archive contract."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _relative_is_excluded(relative: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in relative.parts) or relative.name in EXCLUDED_NAMES


def _screen_file(path: Path, relative: Path) -> bool:
    if _relative_is_excluded(relative) or path.is_symlink() or not path.is_file():
        return False
    if path.suffix.lower() in WEIGHT_EXTENSIONS:
        SCREEN_SKIPS.append({"path": relative.as_posix(), "reason": "weight_or_model_artifact"})
        return False
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    # Text evidence is small enough to scan before it enters the archive;
    # images are binary evidence and do not carry token-like text fields.
    if path.suffix.lower() in TEXT_EXTENSIONS:
        carry = b""
        try:
            with path.open("rb") as stream:
                for stream_block in iter(lambda: stream.read(1024 * 1024), b""):
                    content = carry + stream_block
                    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                        SCREEN_SKIPS.append({"path": relative.as_posix(), "reason": "credential_like_content"})
                        return False
                    # Retain only enough overlap to catch a token split at a
                    # chunk boundary; never hold a whole JSONL/log in memory.
                    carry = content[-2048:]
        except OSError:
            SCREEN_SKIPS.append({"path": relative.as_posix(), "reason": "unreadable"})
            return False
    return True


def select_latest_public_checkpoint(run_root: Path, *, run_name: str = RUN_NAME) -> dict[str, Any] | None:
    """Return the highest-step adapter whose upload receipt verifies bytes."""

    root = run_root / "async-output" / run_name / "artifacts" / "adapters"
    candidates: list[dict[str, Any]] = []
    if not root.is_dir():
        return None
    for receipt_path in root.glob("step_*/hf-upload.json"):
        match = re.fullmatch(r"step_([0-9]+)", receipt_path.parent.name)
        if match is None:
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weight = receipt_path.parent / "adapter_model.safetensors"
        expected = receipt.get("files", {}).get("adapter_model.safetensors", {}).get("sha256")
        if (receipt.get("verified") is not True or receipt.get("private") is not False or
                not weight.is_file() or not isinstance(expected, str) or digest(weight) != expected):
            continue
        candidates.append({
            "step": int(match.group(1)), "adapter": str(receipt_path.parent),
            "receipt": str(receipt_path), "adapter_sha256": expected,
            "public_commit": receipt.get("commit"), "public_path": receipt.get("path"),
        })
    return max(candidates, key=lambda row: row["step"]) if candidates else None


def select_files(run_root: Path, *, output: Path | None = None) -> dict[str, Path]:
    """Select all screened evidence below ``run_root`` without weights/secrets."""

    run_root = run_root.resolve(strict=True)
    output_resolved = output.resolve() if output is not None else None
    files: dict[str, Path] = {}
    SCREEN_SKIPS.clear()
    for path in sorted(run_root.rglob("*")):
        relative = path.relative_to(run_root)
        if output_resolved is not None:
            try:
                path.resolve().relative_to(output_resolved.parent)
            except ValueError:
                pass
            else:
                # Never recurse into the destination's previous archive or
                # verification directory when rerunning after a failure.
                if relative.parts and relative.parts[0] == output_resolved.parent.name:
                    continue
        if path.is_file() and path.name.endswith('.jsonl.zst'):
            try:
                import zstandard
                target = run_root / 'archive-materialized' / relative.with_suffix('')
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as source, target.open('wb') as plain:
                    zstandard.ZstdDecompressor().copy_stream(source, plain)
                if _screen_file(target, relative.with_suffix('')):
                    files[relative.with_suffix('').as_posix()] = target
            except (ImportError, OSError, ValueError) as exc:
                SCREEN_SKIPS.append({"path": relative.as_posix(), "reason": f"zstd_unavailable:{type(exc).__name__}"})
            continue
        if _screen_file(path, relative):
            files[relative.as_posix()] = path
    if not files:
        raise ArchiveSafetyError("no screened evidence files were found")
    return files


def package(files: dict[str, Path], output: Path) -> dict[str, Any]:
    """Create a weight-free tarball and an in-archive SHA manifest."""

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(files.items()):
            relative = Path(name)
            if _relative_is_excluded(relative) or path.is_symlink():
                raise ArchiveSafetyError(f"unsafe archive path: {name}")
            if path.suffix.lower() in WEIGHT_EXTENSIONS:
                raise ArchiveSafetyError(f"weight path reached package stage: {name}")
            rows.append({"path": name, "sha256": digest(path), "bytes": path.stat().st_size})
            archive.add(path, arcname=name, recursive=False)
        manifest = (json.dumps({
            "schema": "text-sft512-consistent-evidence-manifest-v1",
            "run_name": RUN_NAME,
            "files": rows,
            "screen_skips": list(SCREEN_SKIPS),
        }, indent=2, sort_keys=True) + "\n").encode()
        entry = tarfile.TarInfo("evidence-manifest.json")
        entry.size = len(manifest)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(manifest))
    return {"files": len(rows), "bytes": output.stat().st_size, "sha256": digest(output),
            "screen_skips": list(SCREEN_SKIPS)}


def _safe_remote_path(value: str) -> str:
    path = Path(value)
    if not value or value.startswith("/") or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise ArchiveSafetyError("remote archive path is unsafe")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise ArchiveSafetyError("remote archive path contains unsafe characters")
    return value


def _protected_token(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ArchiveSafetyError("HF token file must be owner-only and non-symlink")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ArchiveSafetyError("HF token file is empty")
    return token


def publish_public(*, archive: Path, report: dict[str, Any], token_file: Path,
                   repo_id: str, remote_path: str, verify_dir: Path) -> dict[str, Any]:
    """Upload one archive and verify its public bytes without a token."""

    remote_path = _safe_remote_path(remote_path)
    token = _protected_token(token_file)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - pinned node supplies this
        raise ArchiveSafetyError("huggingface_hub is unavailable") from exc
    api = HfApi(token=token)
    info = api.repo_info(repo_id, repo_type="dataset")
    if getattr(info, "private", False):
        raise ArchiveSafetyError("evidence dataset is private")
    commit = api.upload_file(path_or_fileobj=str(archive), path_in_repo=remote_path,
                             repo_id=repo_id, repo_type="dataset")
    verify_dir.mkdir(parents=True, exist_ok=True)
    public = Path(hf_hub_download(repo_id, filename=remote_path, repo_type="dataset",
                                  revision=commit.oid, token=False, force_download=True,
                                  local_dir=verify_dir))
    if digest(public) != report["sha256"]:
        raise ArchiveSafetyError("anonymous public archive hash mismatch")
    return {**report, "repo_id": repo_id, "path": remote_path,
            "commit": commit.oid, "private": False, "public_hash_verified": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--remote-path", default=f"runs/{RUN_NAME}/evidence.tar.gz")
    parser.add_argument("--verify-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files = select_files(args.run_root, output=args.output)
    report = package(files, args.output)
    if args.token_file is not None:
        verify_dir = args.verify_dir or args.output.parent / "public-verify"
        report = publish_public(archive=args.output, report=report,
                                token_file=args.token_file, repo_id=args.repo_id,
                                remote_path=args.remote_path, verify_dir=verify_dir)
        receipt = args.output.with_suffix(args.output.suffix + ".upload.json")
        temporary = receipt.with_suffix(receipt.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(receipt)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArchiveSafetyError, OSError, ValueError) as exc:
        print(f"safe archive: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
