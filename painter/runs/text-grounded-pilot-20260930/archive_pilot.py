"""Publish the finite grounded-pilot evidence after matched evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

OLD = Path(__file__).resolve().parents[1] / "text-pairwise-rl-20260928"
sys.path.insert(0, str(OLD))
from archive_evidence import REPO_ID, _add_dir, package  # noqa: E402
from archive_continuation import (  # noqa: E402
    public_file_info,
    retry_rate_limit,
    verify_anonymous,
)

RUN_NAME = "text-grounded-pilot-20260930"


def ensure_stream(run: Path) -> Path:
    target = run / "evidence/native-stream.jsonl"
    if target.is_file() and target.stat().st_size:
        return target
    source = run / f"async-output/{RUN_NAME}/monitors/file/traces/stream/00000.jsonl"
    compressed = source.with_suffix(source.suffix + ".zst")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copyfile(source, target)
    elif compressed.is_file():
        import zstandard

        with compressed.open("rb") as src, target.open("wb") as out:
            zstandard.ZstdDecompressor().copy_stream(src, out)
    else:
        raise FileNotFoundError("native trace stream missing")
    if not target.stat().st_size:
        raise ValueError("native trace stream is empty")
    return target


def select(run: Path, *, mid_step: int, final_step: int) -> dict[str, Path]:
    if not (0 < mid_step < final_step):
        raise ValueError("expected distinct positive midpoint and final steps")
    files: dict[str, Path] = {}
    required = [
        "EXPERIMENT.md", "prepare_node.sh", "run.sh", "evaluate.sh",
        "score_validation.py", "score_direct_pilot.py", "prepare_controls.py", "archive_pilot.py",
        "finalize_pilot.sh",
        "native.toml", "init-adapter.json", "initial-adapter-receipt.json",
        "rl-config-receipt.json", "balanced16-validation-manifest.json",
        "anchors-all/reuse-receipt.json",
        "anchors-all/grounded-pilot-train-manifest.json",
        "anchors-all/grounded-pilot-train-manifest-audit.json",
        "anchors-all/validation-manifest.json",
        "evidence/native-stream.jsonl",
        f"async-output/{RUN_NAME}/artifacts/initial-adapter-audit.json",
        f"async-output/{RUN_NAME}/monitors/file/metrics.jsonl",
        f"async-output/{RUN_NAME}/logs/attempt_1/orchestrator.log",
        f"async-output/{RUN_NAME}/logs/attempt_1/trainer.log",
        f"async-output/{RUN_NAME}/artifacts/adapters/step_{mid_step}/hf-upload.json",
        f"async-output/{RUN_NAME}/artifacts/adapters/step_{final_step}/hf-upload.json",
        f"validation/step-{mid_step}-balanced16/comparison/summary.json",
        f"validation/step-{final_step}-all/comparison/summary.json",
        "validation/direct-sft/summary.json",
        "validation/direct-step16/summary.json",
        "validation/direct-midpoint/summary.json",
        "evidence/midpoint-canvases/manifest.json",
        "evidence/final-canvases/manifest.json",
        "evidence/frozen-sft-summary.json",
        "evidence/frozen-step16-summary.json",
        "evidence/profile.json", "evidence/final-analysis.json",
    ]
    for name in required:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(name)
        files[name] = path
    judge_source = run.parents[1] / "text_pairwise_grounded_judge.py"
    if not judge_source.is_file():
        raise FileNotFoundError(judge_source)
    files["source/text_pairwise_grounded_judge.py"] = judge_source
    for folder in (
        "anchors-all/prompts", "anchors-all/anchors", "rl-rollouts-grounded",
        f"validation/step-{mid_step}-balanced16",
        f"validation/step-{final_step}-all",
        "validation/direct-sft", "validation/direct-step16",
        "validation/direct-midpoint",
        f"async-output/{RUN_NAME}/monitors",
        f"async-output/{RUN_NAME}/logs",
        "evidence/midpoint-canvases", "evidence/final-canvases",
        "evidence/frozen-control-images", "evidence/gallery",
        "evidence/reward-curve",
    ):
        _add_dir(files, run, folder)
    for path in sorted((run / f"async-output/{RUN_NAME}/artifacts/adapters").glob("step_*/hf-upload.json")):
        files[path.relative_to(run).as_posix()] = path
    for path in sorted((run / "logs").glob("gpu-telemetry-*.csv")):
        files[path.relative_to(run).as_posix()] = path
    for name in ("evidence/run-receipt.json", "evidence/adapter-delta.json"):
        path = run / name
        if path.is_file():
            files[name] = path
    return files


def publish(run: Path, *, mid_step: int, final_step: int, output: Path,
            remote_path: str, token_file: Path) -> dict:
    if (remote_path.startswith("/") or ".." in Path(remote_path).parts
            or token_file.is_symlink() or not token_file.is_file()
            or token_file.stat().st_mode & 0o077):
        raise ValueError("unsafe destination or token file")
    ensure_stream(run)
    result = package(select(run, mid_step=mid_step, final_step=final_step), output)
    info, existing = public_file_info(remote_path)
    if existing is not None:
        if existing.lfs is None or existing.lfs.sha256 != result["sha256"]:
            raise ValueError("remote path already contains different evidence")
        revision = info.sha
    else:
        api = HfApi(token=token_file.read_text().strip())
        commit = retry_rate_limit(lambda: api.upload_file(
            path_or_fileobj=str(output), path_in_repo=remote_path,
            repo_id=REPO_ID, repo_type="dataset",
            commit_message=f"Add grounded painter pilot step {final_step} evidence"))
        revision = commit.oid
    verify_anonymous(remote_path, revision, result["sha256"], result["bytes"])
    receipt = {**result, "repo_id": REPO_ID, "path": remote_path,
               "commit": revision, "mid_step": mid_step,
               "final_step": final_step, "public_hash_verified": True}
    output.with_suffix(".upload.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mid-step", type=int, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.run_root, mid_step=args.mid_step,
                             final_step=args.final_step, output=args.output,
                             remote_path=args.remote_path,
                             token_file=args.token_file), sort_keys=True))


if __name__ == "__main__":
    main()
