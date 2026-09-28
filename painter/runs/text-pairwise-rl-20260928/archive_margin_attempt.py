"""Publish the finite margin-RL attempt and checked validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from archive_evidence import REPO_ID, _add_dir, digest, package


RUN_NAME = "text-pairwise-margin-workerfix-20260928"


def select(run: Path, eval_dirs: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    fixed = [
        "EXPERIMENT.md", "native-rl-margin-workerfix.toml",
        "rl-config-receipt.json", "anchors-frozen/train-manifest.json",
        "anchors-frozen/validation-manifest.json", "prepared/pool.json",
        f"async-output/{RUN_NAME}/monitors/file/traces/stream/00000.jsonl",
        f"async-output/{RUN_NAME}/monitors/file/metrics.jsonl",
        f"async-output/{RUN_NAME}/logs/attempt_1/orchestrator.log",
        f"async-output/{RUN_NAME}/logs/attempt_1/trainer.log",
        "validation/sft-step512-control/comparison-margin/summary.json",
    ]
    for name in fixed:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(name)
        files[name] = path
    for step in (4, 8, 12, 16, 20):
        name = f"async-output/{RUN_NAME}/artifacts/adapters/step_{step}/hf-upload.json"
        path = run / name
        if path.is_file():
            files[name] = path
    for folder in (
        "prepared/prompts", "anchors-frozen/anchors",
        "rl-rollouts-margin-workerfix",
        "validation/sft-step512-control/comparison-margin",
    ):
        _add_dir(files, run, folder)
    for eval_dir in eval_dirs:
        if not eval_dir.startswith(f"{RUN_NAME}-step-") or "/" in eval_dir:
            raise ValueError(f"unexpected evaluation directory: {eval_dir}")
        folder = f"validation/{eval_dir}/comparison"
        if not (run / folder / "summary.json").is_file():
            raise FileNotFoundError(folder + "/summary.json")
        _add_dir(files, run, folder)
        name = f"validation/{eval_dir}/cases.json"
        files[name] = run / name
    telemetry = run / "logs/gpu-telemetry-20260928T192953Z.csv"
    if not telemetry.is_file():
        raise FileNotFoundError(telemetry)
    files[telemetry.relative_to(run).as_posix()] = telemetry
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eval-dir", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if args.remote_path.startswith("/") or ".." in Path(args.remote_path).parts:
        raise ValueError("remote path must be a safe repository path")
    token = args.token_file
    if token.is_symlink() or not token.is_file() or token.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only and not a symlink")
    summary = package(select(args.run_root, args.eval_dir), args.output)
    api = HfApi(token=token.read_text().strip())
    if api.repo_info(REPO_ID, repo_type="dataset").private:
        raise ValueError("evidence dataset must be public")
    commit = api.upload_file(path_or_fileobj=str(args.output),
                             path_in_repo=args.remote_path, repo_id=REPO_ID,
                             repo_type="dataset")
    public = Path(hf_hub_download(REPO_ID, filename=args.remote_path,
                                  repo_type="dataset", revision=commit.oid,
                                  token=False, force_download=True,
                                  local_dir=args.output.parent / "public-verify"))
    if digest(public) != summary["sha256"]:
        raise ValueError("public evidence hash mismatch")
    receipt = {**summary, "repo_id": REPO_ID, "path": args.remote_path,
               "commit": commit.oid, "public_hash_verified": True}
    args.output.with_suffix(".upload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
