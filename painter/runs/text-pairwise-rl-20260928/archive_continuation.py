"""Screen, publish, and anonymously verify the finite RL-continuation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_url
import requests

from archive_evidence import REPO_ID, _add_dir, digest, package


RUN_NAME = "text-pairwise-continue-20260929"


def retry_rate_limit(action):
    for attempt in range(6):
        try:
            return action()
        except Exception as exc:
            response = getattr(exc, "response", None)
            if getattr(response, "status_code", None) != 429 or attempt == 5:
                raise
            time.sleep(min(60, 2 ** (attempt + 2)))


def public_file_info(remote_path: str):
    info = retry_rate_limit(lambda: HfApi(token=False).repo_info(
        REPO_ID, repo_type="dataset", files_metadata=True))
    if info.private:
        raise ValueError("public evidence repository required")
    existing = next((item for item in info.siblings
                     if item.rfilename == remote_path), None)
    return info, existing


def verify_anonymous(remote_path: str, revision: str, expected_sha: str,
                     expected_bytes: int) -> None:
    url = hf_hub_url(REPO_ID, filename=remote_path, repo_type="dataset",
                     revision=revision)
    for attempt in range(3):
        try:
            remote_hash = hashlib.sha256()
            remote_bytes = 0
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                for block in response.iter_content(1024 * 1024):
                    remote_hash.update(block)
                    remote_bytes += len(block)
            if remote_bytes != expected_bytes or remote_hash.hexdigest() != expected_sha:
                raise ValueError("anonymous public archive hash mismatch")
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def ensure_stream(run: Path) -> Path:
    target = run / "evidence/continued-stream.jsonl"
    if target.is_file():
        return target
    source = run / f"async-output/{RUN_NAME}/monitors/file/traces/stream/00000.jsonl"
    compressed = source.with_suffix(source.suffix + ".zst")
    if not source.is_file() and not compressed.is_file():
        raise FileNotFoundError("native Prime trace stream missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copyfile(source, target)
    else:
        import zstandard  # type: ignore[import-not-found]
        with compressed.open("rb") as source_file, target.open("wb") as out:
            zstandard.ZstdDecompressor().copy_stream(source_file, out)
    if not target.stat().st_size:
        raise ValueError("native Prime trace stream is empty")
    return target


def select(run: Path, evaluation: str, step: int) -> dict[str, Path]:
    if not evaluation.startswith("text-pairwise-continue-") or "/" in evaluation or ".." in evaluation:
        raise ValueError("unsafe evaluation name")
    files: dict[str, Path] = {}
    required = [
        "CONTINUATION_20260929.md", "CONTINUATION_FAILURE_AUDIT_20260929.md",
        "native-continued.toml",
        "plot_reward_curve.py", "profile_continuation.py", "profile_js_preflight.py",
        "profile_completion_failures.py", "publish_rollout_snapshot.py",
        "build_rollout_audit.py",
        "verify_adapter_delta.py",
        "rl-config-receipt.json", "continued-step16-download.json",
        "continued-anchors/reuse-receipt.json",
        "continued-anchors/train-manifest.json",
        "continued-anchors/validation-manifest.json",
        "continued-anchors/reserve-manifest.json",
        "evidence/continued-stream.jsonl",
        "evidence/continued-evaluated-step.txt",
        "evidence/continued-evaluation-scope.txt",
        "evidence/profile-continuation.json",
        "evidence/completion-failure-profile.json",
        "evidence/js-preflight-sample16.json",
        "evidence/adapter-delta.json",
        "validation/scaled-sft-control/validation-summary.json",
        "prior-step16/validation/scaled-step-16-all/comparison/summary.json",
        f"async-output/{RUN_NAME}/monitors/file/metrics.jsonl",
        f"async-output/{RUN_NAME}/logs/attempt_1/orchestrator.log",
        f"async-output/{RUN_NAME}/logs/attempt_1/trainer.log",
        f"async-output/{RUN_NAME}/artifacts/adapters/step_{step}/hf-upload.json",
        f"validation/{evaluation}/comparison/summary.json",
    ]
    for name in required:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(name)
        files[name] = path
    for path in sorted((run / f"async-output/{RUN_NAME}/artifacts/adapters").glob("step_*/hf-upload.json")):
        files[path.relative_to(run).as_posix()] = path
    for folder in (
        "continued-anchors/prompts", "rl-rollouts-continued",
        f"validation/{evaluation}", f"async-output/{RUN_NAME}/monitors",
        f"async-output/{RUN_NAME}/logs", "evidence/continued-gallery",
        "evidence/continued-canvases", "evidence/reward-curve",
        "evidence/rollout-audit",
        "evidence/live-step16-curve",
    ):
        _add_dir(files, run, folder)
    for path in sorted((run / "logs").glob("gpu-telemetry*.csv")):
        files[path.relative_to(run).as_posix()] = path
    return {name: path for name, path in files.items() if "cache" not in Path(name).parts}


def publish(run: Path, evaluation: str, step: int, output: Path,
            remote_path: str, token_file: Path) -> dict:
    if remote_path.startswith("/") or ".." in Path(remote_path).parts:
        raise ValueError("unsafe public repository path")
    if token_file.is_symlink() or not token_file.is_file() or token_file.stat().st_mode & 0o077:
        raise PermissionError("token file must be an owner-only regular file")
    ensure_stream(run)
    report = package(select(run, evaluation, step), output)
    public_info, existing = public_file_info(remote_path)
    if existing is not None and (existing.lfs is None or existing.lfs.sha256 != report["sha256"]):
        raise ValueError("remote evidence path already contains a different archive")
    if existing is None:
        api = HfApi(token=token_file.read_text().strip())
        commit = retry_rate_limit(lambda: api.upload_file(
            path_or_fileobj=str(output), path_in_repo=remote_path,
            repo_id=REPO_ID, repo_type="dataset",
            commit_message=f"Add text-paint continuation step {step} evidence"))
        revision = commit.oid
    else:
        revision = public_info.sha
    verify_anonymous(remote_path, revision, report["sha256"], report["bytes"])
    receipt = {**report, "repo_id": REPO_ID, "path": remote_path,
               "commit": revision, "step": step, "public_hash_verified": True}
    output.with_suffix(".upload.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.run_root, args.evaluation, args.step, args.output,
                             args.remote_path, args.token_file), sort_keys=True))


if __name__ == "__main__":
    main()
