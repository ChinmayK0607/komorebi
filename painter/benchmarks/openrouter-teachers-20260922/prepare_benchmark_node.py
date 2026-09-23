#!/usr/bin/env python3
"""Create and upload the source bundle plus 40 reference images for the node.

The Git bundle contains tracked source/configuration only. Reference JPEGs are
ignored by Git and are uploaded as a separate hash-checked archive. No dotenv,
key, result, cache, or generated episode path is eligible for either artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_BUNDLE = HERE / "source.bundle"
DEFAULT_REFERENCES = HERE / "references.tar.gz"
REMOTE_ROOT = "/root/painter"


class PrepareError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def references() -> list[tuple[str, Path, str]]:
    manifest = json.loads((HERE / "refs.json").read_text(encoding="utf-8"))
    rows = manifest.get("references") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) != 40:
        raise PrepareError("refs.json must contain exactly 40 references")
    selected: list[tuple[str, Path, str]] = []
    for row in rows:
        image_name = row.get("image") if isinstance(row, dict) else None
        expected = row.get("sha256") if isinstance(row, dict) else None
        if not isinstance(image_name, str) or not image_name.startswith("references/"):
            raise PrepareError(f"unsafe reference path: {image_name!r}")
        relative = Path(image_name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in {".jpg", ".jpeg"}:
            raise PrepareError(f"unsafe reference path: {image_name}")
        source = (HERE / relative).resolve()
        if HERE not in source.parents or source.is_symlink() or not source.is_file():
            raise PrepareError(f"reference is missing or symlinked: {source}")
        if not isinstance(expected, str) or sha(source) != expected:
            raise PrepareError(f"reference hash mismatch: {source.name}")
        selected.append((image_name, source, expected))
    return selected


def create_bundle(path: Path) -> None:
    status = run(["git", "-C", str(REPO), "status", "--porcelain", "--", "painter/benchmarks/openrouter-teachers-20260922", "painter/vendor/integrations/watercolour"]).stdout
    if status.strip():
        raise PrepareError("commit benchmark and renderer source before creating the Git bundle")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "-C", str(REPO), "bundle", "create", str(path), "--all"])
    run(["git", "-C", str(REPO), "bundle", "verify", str(path)])


def create_references(path: Path, selected: list[tuple[str, Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-references-") as temp:
        manifest = Path(temp) / "references-manifest.json"
        manifest.write_text(json.dumps({
            "schema": "painter.ai-gateway-reference-archive.v1",
            "count": len(selected),
            "files": [{"path": name, "sha256": expected} for name, _, expected in selected],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tarfile.open(path, "w:gz") as archive:
            for name, source, _ in selected:
                info = tarfile.TarInfo(name)
                info.size = source.stat().st_size
                info.mode = 0o644
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
            info = tarfile.TarInfo("references-manifest.json")
            info.size = manifest.stat().st_size
            info.mode = 0o644
            with manifest.open("rb") as stream:
                archive.addfile(info, stream)


def upload(target: str, bundle: Path, refs: Path, setup: Path, remote_root: str) -> None:
    run(["lium", "exec", target, f"mkdir -p {remote_root}"])
    run(["lium", "scp", target, str(bundle), f"{remote_root}/source.bundle"])
    run(["lium", "scp", target, str(refs), f"{remote_root}/references.tar.gz"])
    run(["lium", "scp", target, str(setup), f"{remote_root}/setup_benchmark_node.sh"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", default="painter-photo-rl")
    parser.add_argument("--bundle-output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--references-output", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.remote_root.startswith("/") or any(char.isspace() for char in args.remote_root) or any(char in args.remote_root for char in "'\";$`"):
            raise PrepareError("--remote-root must be an absolute shell-safe path")
        selected = references()
        if args.dry_run:
            dirty = run(["git", "-C", str(REPO), "status", "--porcelain", "--", "painter/benchmarks/openrouter-teachers-20260922", "painter/vendor/integrations/watercolour"]).stdout.strip()
            print(json.dumps({
                "status": "dry_run", "pod": args.pod, "reference_count": len(selected),
                "credentials_included": False, "source_dirty": bool(dirty),
                "next": "commit reviewed source before creating/uploading the bundle",
            }, indent=2, sort_keys=True))
            return 0
        setup = HERE / "setup_benchmark_node.sh"
        if setup.is_symlink() or not setup.is_file():
            raise PrepareError("setup_benchmark_node.sh is missing or symlinked")
        create_bundle(args.bundle_output)
        create_references(args.references_output, selected)
        report: dict[str, Any] = {
            "status": "prepared" if args.dry_run else "uploaded",
            "pod": args.pod,
            "bundle": str(args.bundle_output),
            "bundle_sha256": sha(args.bundle_output),
            "references": str(args.references_output),
            "references_sha256": sha(args.references_output),
            "reference_count": len(selected),
            "credentials_included": False,
        }
        if not args.dry_run:
            upload(args.pod, args.bundle_output, args.references_output, setup, args.remote_root)
            report["remote_files"] = [f"{args.remote_root}/source.bundle", f"{args.remote_root}/references.tar.gz", f"{args.remote_root}/setup_benchmark_node.sh"]
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Next: lium ssh {args.pod}", file=sys.stderr)
        print(f"Then run: bash {args.remote_root}/setup_benchmark_node.sh {args.remote_root}", file=sys.stderr)
        return 0
    except (PrepareError, OSError, subprocess.CalledProcessError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"benchmark package error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
