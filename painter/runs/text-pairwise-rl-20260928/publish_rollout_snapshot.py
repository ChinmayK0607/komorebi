#!/usr/bin/env python3
"""Publish a finite, screened live rollout snapshot before node expiry.

Only old, stable render/code/receipt files are copied; active files are skipped.
This is raw generated evidence, not a matched evaluation or training result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download

from archive_evidence import IMAGE_EXT, REPO_ID, TEXT_EXT, digest, package


def snapshot(run: Path, step: int, output: Path, remote_path: str,
             token_file: Path | None, min_age_seconds: float,
             dry_run: bool = False) -> dict:
    if step < 1 or min_age_seconds < 60:
        raise ValueError("step must be positive and age at least 60 seconds")
    if remote_path.startswith("/") or ".." in Path(remote_path).parts:
        raise ValueError("unsafe remote path")
    if not dry_run and (token_file is None or token_file.is_symlink()
                        or not token_file.is_file()
                        or token_file.stat().st_mode & 0o077):
        raise PermissionError("token file must be an owner-only regular file")
    checkpoint = run / f"async-output/text-pairwise-continue-20260929/artifacts/adapters/step_{step}/hf-upload.json"
    receipt = json.loads(checkpoint.read_text())
    if receipt.get("private") is not False or receipt.get("verified") is not True:
        raise ValueError("step checkpoint must be public and hash-verified")
    root = run / "rl-rollouts-continued"
    cutoff = time.time() - min_age_seconds
    selected = [path for path in root.rglob("*") if path.is_file()
                and path.suffix.lower() in TEXT_EXT | IMAGE_EXT
                and path.stat().st_mtime <= cutoff]
    if not selected:
        raise ValueError("no stable rollout evidence found")
    selected.sort()
    with tempfile.TemporaryDirectory(prefix="painter-rollout-snapshot-", dir=output.parent) as tmp:
        temp = Path(tmp)
        files = {}
        skipped = 0
        for path in selected:
            name = path.relative_to(run).as_posix()
            before = path.stat()
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                target.unlink()
                skipped += 1
                continue
            if path.suffix.lower() in {".json", ".jsonl"}:
                try:
                    content = target.read_text()
                    if path.suffix.lower() == ".json":
                        json.loads(content)
                    else:
                        for line in content.splitlines():
                            if line.strip():
                                json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    target.unlink()
                    skipped += 1
                    continue
            if path.suffix.lower() == ".png" and not target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                target.unlink()
                skipped += 1
                continue
            files[name] = target
        meta = {"schema": "painter.live-rollout-snapshot.v1", "step_checkpoint": step,
                "checkpoint_commit": receipt["commit"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "minimum_file_age_seconds": min_age_seconds,
                "selected_files": len(files), "unstable_or_invalid_files_skipped": skipped,
                "interpretation": "Raw saved generated rollouts, including speculative work; not matched evaluation."}
        (temp / "snapshot.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        files["snapshot.json"] = temp / "snapshot.json"
        report = package(files, output)
    if dry_run:
        return {**report, **meta, "public_hash_verified": False}
    assert token_file is not None
    api = HfApi(token=token_file.read_text().strip())
    if api.repo_info(REPO_ID, repo_type="dataset").private:
        raise ValueError("public dataset required")
    commit = api.upload_file(path_or_fileobj=str(output), path_in_repo=remote_path,
                             repo_id=REPO_ID, repo_type="dataset",
                             commit_message=f"Preserve text-paint rollout snapshot through step {step}")
    public = Path(hf_hub_download(REPO_ID, filename=remote_path, repo_type="dataset",
                                  revision=commit.oid, token=False, force_download=True,
                                  local_dir=output.parent / "public-snapshot-verify"))
    if digest(public) != report["sha256"]:
        raise ValueError("anonymous public hash mismatch")
    result = {**report, **meta, "repo_id": REPO_ID, "path": remote_path,
              "commit": commit.oid, "public_hash_verified": True}
    output.with_suffix(".upload.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--min-age-seconds", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = snapshot(args.run_root, args.step, args.output, args.remote_path,
                      args.token_file, args.min_age_seconds, args.dry_run)
    summary_keys = ("files", "bytes", "sha256", "selected_files",
                    "unstable_or_invalid_files_skipped", "public_hash_verified")
    print(json.dumps({**{key: result[key] for key in summary_keys},
                      **({"commit": result["commit"]} if "commit" in result else {})},
                     sort_keys=True))


if __name__ == "__main__":
    main()
