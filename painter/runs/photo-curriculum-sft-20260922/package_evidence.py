"""Build a private, hash-verified evidence archive for the photo SFT run.

The archive is deliberately data-only: it includes run records, configs,
metrics, logs, native source identifiers, teacher records and rendered
outputs, while excluding weights, caches, environments, archives and
credential-like files.  It never uploads or releases anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


RUN_NAME = "photo-curriculum-sft-20260922"
SOURCE_FILES = (
    "CONTRACT.md",
    "README.md",
    "contract.py",
    "native_painting.py",
    "native_painting_renderer.py",
    "native_painting_eval.py",
    "native_painting_report.py",
    "native_painting_rl.py",
    "foundation-lock.json",
    "pyproject.toml",
)

DENY_DIRS = {
    ".git",
    ".cache",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "renderer-env",
    "inference-env",
    "runtime",
    "cache",
    "caches",
    "huggingface",
    "hf_home",
    "checkpoint",
    "checkpoints",
    "private",
    "credentials",
    "secrets",
    "evidence-private",
}
DENY_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".distcp",
    ".npz",
    ".npy",
    ".tar",
    ".gz",
    ".tgz",
    ".zip",
    ".7z",
}
DENY_NAMES = {
    ".env",
    "env.sh",
    "credentials.json",
    "token.json",
    "private-key.json",
    "id_rsa",
    "known_hosts",
}
DENY_NAME_PARTS = ("secret", "credential", "private-key", "api_key", "apikey", "access_token")
SAFE_METRIC_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".txt", ".md", ".toml"}
TEXT_SUFFIXES = SAFE_METRIC_SUFFIXES | {".py", ".sh", ".cjs", ".js", ".html"}
_TEXT_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)(https?://)[^/\s:@]+:[^@\s]+@"),
    re.compile(r"(?i)\b(?:hf|sk)[_-][A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)((?:api[_-]?key|authorization|password|secret|credential|access[_-]?token|proxy)\s*[:=]\s*)[^\s,}\]]+"),
)


def sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    exact = {
        "api_key",
        "api_key_var",
        "authorization",
        "password",
        "secret",
        "credential",
        "endpoint",
        "proxy",
        "base_url",
        "access_token",
        "token",
        "token_path",
        "hf_token",
        "hf_token_path",
    }
    return normalized in exact or normalized.endswith(("_api_key", "_token", "_endpoint", "_proxy"))


def redact_json(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        result: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str) and sensitive_key(key):
                result[key] = "<redacted>"
                changed = True
            else:
                result[key], child_changed = redact_json(child)
                changed = changed or child_changed
        return result, changed
    if isinstance(value, list):
        result = []
        changed = False
        for child in value:
            redacted, child_changed = redact_json(child)
            result.append(redacted)
            changed = changed or child_changed
        return result, changed
    if isinstance(value, str):
        return redact_text(value)
    return value, False


def redact_text(text: str) -> tuple[str, bool]:
    changed = False
    for pattern in _TEXT_CREDENTIAL_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            changed = True
            if match.lastindex and match.lastindex >= 1:
                return f"{match.group(1)}<redacted>"
            return "<redacted>"

        text = pattern.sub(replace, text)
    return text, changed


def stage_content(source: Path) -> tuple[bytes, bool]:
    raw = source.read_bytes()
    if source.suffix.lower() not in TEXT_SUFFIXES:
        return raw, False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw, False
    if source.suffix.lower() == ".json":
        try:
            redacted, changed = redact_json(json.loads(text))
            return (json.dumps(redacted, indent=2, sort_keys=True) + "\n").encode(), changed
        except (json.JSONDecodeError, TypeError):
            pass
    elif source.suffix.lower() == ".jsonl":
        lines: list[str] = []
        changed = False
        parsed_any = False
        for line in text.splitlines(keepends=True):
            if not line.strip():
                lines.append(line)
                continue
            try:
                redacted, line_changed = redact_json(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                lines.append(line)
                continue
            parsed_any = True
            newline = "\n" if line.endswith("\n") else ""
            lines.append(json.dumps(redacted, sort_keys=True) + newline)
            changed = changed or line_changed
        if parsed_any:
            return "".join(lines).encode(), changed
    redacted_text, changed = redact_text(text)
    return redacted_text.encode(), changed


def credential_hit_count(text: str) -> int:
    count = 0
    for pattern in _TEXT_CREDENTIAL_PATTERNS:
        for match in pattern.finditer(text):
            if "<redacted>" not in match.group(0):
                count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(path: Path, *, max_bytes: int) -> bool:
    """Return whether a source file can enter the private archive."""
    if not path.is_file() or path.is_symlink():
        return False
    lower_name = path.name.lower()
    if lower_name in DENY_NAMES or any(part in lower_name for part in DENY_NAME_PARTS):
        return False
    if path.suffix.lower() in DENY_SUFFIXES:
        return False
    # Ignore the filesystem's POSIX root and first mount component when
    # checking directory names: macOS temporary sources live under /private,
    # while a run-local ``private`` directory must still be excluded.
    parts = path.parts[2:] if path.is_absolute() else path.parts
    if any(part.lower() in DENY_DIRS for part in parts):
        return False
    if path.stat().st_size > max_bytes:
        raise ValueError(f"refusing oversized evidence file ({path.stat().st_size} bytes): {path}")
    return True


def walk_files(root: Path, *, max_bytes: int) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_symlink():
        raise ValueError(f"refusing symlink source root: {root}")
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirs[:] = sorted(d for d in dirs if d not in DENY_DIRS and not (current_path / d).is_symlink())
        for name in sorted(names):
            path = current_path / name
            if safe_member(path, max_bytes=max_bytes):
                yield path


def archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member name: {name}")
    return path.as_posix()


def add_file(files: dict[str, Path], name: str, source: Path, *, max_bytes: int) -> None:
    name = archive_name(name)
    if name in files:
        raise ValueError(f"duplicate archive member: {name}")
    if not safe_member(source, max_bytes=max_bytes):
        raise ValueError(f"unsafe evidence source: {source}")
    files[name] = source


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def summarize_gpu_telemetry(paths: Iterable[tuple[str, Path]]) -> dict[str, Any]:
    """Summarize each telemetry file independently, never blending windows."""
    summaries: dict[str, Any] = {}
    for source_name, path in paths:
        if path.suffix.lower() != ".csv" or "telemetr" not in path.name.lower():
            continue
        samples: dict[str, list[tuple[str, float]]] = defaultdict(list)
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = [field or "" for field in (reader.fieldnames or [])]
                gpu_field = next((f for f in fields if "index" in f.lower()), None)
                util_field = next((f for f in fields if "utilization" in f.lower() and "gpu" in f.lower()), None)
                time_field = next((f for f in fields if "timestamp" in f.lower() or f.lower() == "time"), None)
                if not gpu_field or not util_field:
                    continue
                for row in reader:
                    gpu = row.get(gpu_field, "").strip() or "unknown"
                    util = parse_number(row.get(util_field))
                    if util is not None:
                        samples[gpu].append((row.get(time_field, "").strip() if time_field else "", util))
            if samples:
                per_gpu: dict[str, Any] = {}
                for gpu, values in sorted(samples.items()):
                    utilizations = [value for _, value in values]
                    times = [stamp for stamp, _ in values if stamp]
                    per_gpu[gpu] = {
                        "samples": len(values),
                        "window_start": min(times) if times else None,
                        "window_end": max(times) if times else None,
                        "utilization_gpu_percent": {
                            "mean": round(sum(utilizations) / len(utilizations), 3),
                            "min": min(utilizations),
                            "max": max(utilizations),
                        },
                    }
                summaries[source_name] = {"gpus": per_gpu, "denominator": "telemetry samples per GPU"}
        except (OSError, UnicodeError, csv.Error):
            continue
    summary: dict[str, Any] = {
        "schema": "photo-curriculum-sft-gpu-telemetry-summary.v1",
        "sources": summaries,
    }
    return summary


def add_source_files(files: dict[str, Path], root: Path, *, max_bytes: int) -> list[dict[str, Any]]:
    source_ids: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            source_ids.append({"path": relative, "status": "missing"})
            continue
        add_file(files, f"source/painter/{relative}", path, max_bytes=max_bytes)
        source_ids.append({"path": relative, "status": "included", "sha256": sha256(path), "bytes": path.stat().st_size})
    return source_ids


def add_explicit_sources(files: dict[str, Path], sources: Iterable[Path], *, max_bytes: int) -> list[dict[str, Any]]:
    source_ids: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        source = source.resolve()
        if source.is_dir():
            candidates = list(walk_files(source, max_bytes=max_bytes))
        else:
            candidates = [source]
        if not candidates:
            raise ValueError(f"explicit source has no safe files: {source}")
        for candidate in candidates:
            relative = candidate.relative_to(source) if source.is_dir() else Path(candidate.name)
            name = f"source/external/{index:03d}/{relative.as_posix()}"
            add_file(files, name, candidate, max_bytes=max_bytes)
            source_ids.append({"path": str(candidate), "archive_path": name, "sha256": sha256(candidate), "bytes": candidate.stat().st_size})
    return source_ids


def build(*, root: Path, run_root: Path, output: Path, sources: tuple[Path, ...], max_bytes: int) -> dict[str, Any]:
    root = root.resolve()
    run_root = run_root.resolve()
    output = output.resolve()
    if not run_root.is_dir():
        raise ValueError(f"run root is not a directory: {run_root}")
    if output.exists() and output.is_symlink():
        raise ValueError(f"refusing symlink output: {output}")
    files: dict[str, Path] = {}
    for path in walk_files(run_root, max_bytes=max_bytes):
        add_file(files, f"run/{path.relative_to(run_root).as_posix()}", path, max_bytes=max_bytes)
    source_ids = add_source_files(files, root, max_bytes=max_bytes)
    source_ids.extend(add_explicit_sources(files, sources, max_bytes=max_bytes))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{RUN_NAME}-evidence-") as temporary:
        stage = Path(temporary)
        staged: dict[str, Path] = {}
        manifest: dict[str, dict[str, Any]] = {}
        redacted_files: list[str] = []
        for name, source in sorted(files.items()):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content, redacted = stage_content(source)
            target.write_bytes(content)
            staged[name] = target
            manifest[name] = {
                "source_bytes": source.stat().st_size,
                "source_sha256": sha256(source),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "redacted": redacted,
            }
            if redacted:
                redacted_files.append(name)

        telemetry = summarize_gpu_telemetry(files.items())
        summary_path = stage / "gpu-telemetry-summary.json"
        summary_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True) + "\n")
        staged["gpu-telemetry-summary.json"] = summary_path
        manifest["gpu-telemetry-summary.json"] = {
            "source_bytes": None,
            "source_sha256": None,
            "bytes": summary_path.stat().st_size,
            "sha256": sha256(summary_path),
            "redacted": False,
        }

        source_ids_path = stage / "source-ids.json"
        source_ids_path.write_text(
            json.dumps({"schema": "photo-curriculum-sft-source-ids.v1", "files": source_ids}, indent=2, sort_keys=True) + "\n"
        )
        staged["source-ids.json"] = source_ids_path
        manifest["source-ids.json"] = {
            "source_bytes": None,
            "source_sha256": None,
            "bytes": source_ids_path.stat().st_size,
            "sha256": sha256(source_ids_path),
            "redacted": False,
        }

        credential_scan_files = 0
        credential_hits_found: dict[str, int] = {}
        for name, path in sorted(staged.items()):
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            credential_scan_files += 1
            try:
                count = credential_hit_count(path.read_text())
                if count:
                    credential_hits_found[name] = count
            except UnicodeDecodeError:
                continue
        if credential_hits_found:
            detail = ", ".join(f"{name} ({count} matches)" for name, count in sorted(credential_hits_found.items()))
            raise ValueError("credential-like values remain after redaction in: " + detail)
        (stage / "evidence-manifest.json").write_text(json.dumps(dict(sorted(manifest.items())), indent=2, sort_keys=True) + "\n")
        manifest["evidence-manifest.json"] = {
            "source_bytes": None,
            "source_sha256": None,
            "bytes": (stage / "evidence-manifest.json").stat().st_size,
            "sha256": sha256(stage / "evidence-manifest.json"),
            "redacted": False,
        }
        archive = output.with_suffix(output.suffix + ".partial")
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
        archive.replace(output)

    receipt = {
        "schema": "photo-curriculum-sft-private-evidence.v1",
        "run": RUN_NAME,
        "archive": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "contains_weights": False,
        "contains_credentials": "no_detected_credentials",
        "credential_scan": {"status": "passed", "text_files": credential_scan_files, "matches": 0},
        "redacted_files": sorted(redacted_files),
        "source_ids": source_ids,
        "files": len(staged) + 1,
        "source_bytes": sum(item["source_bytes"] for item in manifest.values() if item["source_bytes"] is not None),
        "archived_bytes": sum(item["bytes"] for item in manifest.values()),
        "gpu_telemetry_summary": "gpu-telemetry-summary.json",
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/painter"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/painter/runs") / RUN_NAME)
    parser.add_argument("--output", type=Path, required=True, help="private local .tar.gz destination")
    parser.add_argument("--source", type=Path, action="append", default=[], help="extra teacher/data source file or directory")
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024 * 1024, help="maximum admitted source file size; native checkpoint directories are always excluded")
    args = parser.parse_args()
    if args.max_file_bytes < 1:
        raise SystemExit("--max-file-bytes must be positive")
    print(json.dumps(build(root=args.root, run_root=args.run_root, output=args.output, sources=tuple(args.source), max_bytes=args.max_file_bytes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
