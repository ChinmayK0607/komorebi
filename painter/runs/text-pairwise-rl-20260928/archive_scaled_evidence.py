"""Publish the finite scaled text-RL evidence without weights or credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from archive_evidence import REPO_ID, _add_dir, digest, package


RUN_NAME = "text-pairwise-scaled-20260929"


def select(run: Path, eval_dirs: list[str]) -> dict[str, Path]:
    pointer = run / "prepared/scaled-anchors-path.txt"
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    anchors = Path(pointer.read_text().strip())
    if anchors.parent.resolve() != run.resolve() or anchors.name not in {
            "anchors-scaled", "anchors-scaled-first"}:
        raise ValueError("unexpected frozen anchor location")
    required = [
        "NEXT_SCALED_RUN.md", "native-scaled.toml", "rl-config-receipt.json",
        "prepared/scaled-run-pool/pool.json", "prepared/scaled-control-cases.json",
        "prepared/scaled-anchors-path.txt", "judge-canary-scaled/summary.json",
        "judge-canary-scaled/receipt.json", "evidence/stream-final.jsonl",
        f"{anchors.name}/train-manifest.json",
        f"{anchors.name}/validation-manifest.json",
        f"{anchors.name}/reserve-manifest.json",
        f"{anchors.name}/train-probe-manifest.json",
        f"{anchors.name}/anchor-audit.json",
        "validation/scaled-sft-control/comparison/summary.json",
        "validation/scaled-sft-control/validation-summary.json",
        "validation/scaled-sft-control/probe-summary.json",
        f"async-output/{RUN_NAME}/monitors/file/metrics.jsonl",
        f"async-output/{RUN_NAME}/logs/attempt_1/orchestrator.log",
        f"async-output/{RUN_NAME}/logs/attempt_1/trainer.log",
    ]
    files = {}
    for name in required:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = path
    receipts = sorted((run / f"async-output/{RUN_NAME}/artifacts/adapters").glob(
        "step_*/hf-upload.json"))
    if not receipts:
        raise ValueError("no verified public checkpoint receipt")
    for path in receipts:
        files[path.relative_to(run).as_posix()] = path
    for name in eval_dirs:
        if not name.startswith("scaled-") or "/" in name or ".." in name:
            raise ValueError(f"unsafe evaluation directory: {name}")
        path = run / "validation" / name / "comparison/summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
    for folder in (
        "prepared/scaled-run-pool/prompts", f"{anchors.name}/anchors",
        "baseline-scaled-184", "baseline-scaled-v3", "baseline-scaled",
        "judge-canary-scaled", "validation/scaled-sft-control",
        "rl-rollouts-scaled", "evidence/canvases", "evidence/gallery",
        f"async-output/{RUN_NAME}/monitors",
        f"async-output/{RUN_NAME}/logs",
    ):
        _add_dir(files, run, folder)
    for name in eval_dirs:
        _add_dir(files, run, "validation/" + name)
    telemetry = run / "evidence/gpu-telemetry.csv"
    if telemetry.is_file():
        files[telemetry.relative_to(run).as_posix()] = telemetry
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eval-dir", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if args.remote_path.startswith("/") or ".." in Path(args.remote_path).parts:
        raise ValueError("unsafe remote path")
    token = args.token_file
    if token.is_symlink() or not token.is_file() or token.stat().st_mode & 0o077:
        raise PermissionError("HF token file must be owner-only and not a symlink")
    report = package(select(args.run_root, args.eval_dir), args.output)
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
    if digest(public) != report["sha256"]:
        raise ValueError("public evidence hash mismatch")
    receipt = {**report, "repo_id": REPO_ID, "path": args.remote_path,
               "commit": commit.oid, "public_hash_verified": True}
    args.output.with_suffix(".upload.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
