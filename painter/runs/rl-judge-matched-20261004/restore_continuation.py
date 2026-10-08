#!/usr/bin/env python3
"""Restore the pinned public inputs for the matched continuation.

This is deliberately a recovery helper, not a training entrypoint.  It
downloads the immutable baseline and Flash evidence archives anonymously,
checks their exact bytes, extracts only evidence files, and downloads the
public Flash step-12 adapter into the new arm root.  Archive members are
screened before extraction so an old source checkout, symlink, hardlink, or
path traversal cannot replace the newly prepared runtime/configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any, Iterable


DATASET_REPO = "CK0607/komorebi-painter-experiments"
DATASET_REVISION = "16ddc7694b0f447f88ea0bb7f9df1f52fb043e68"
BASELINE_ARCHIVE = {
    "remote_path": "runs/rl-judge-matched-20261004/baseline-evidence.tar.gz",
    "bytes": 23309572,
    "sha256": "6967e1304b8b51aeda5b73bb00b05365edf72e44ae92baffaabaf93327158d41",
}
FLASH_ARCHIVE = {
    "remote_path": "runs/rl-judge-matched-20261004-flash/evidence.tar.gz",
    "bytes": 206207624,
    "sha256": "88a872d20561519f9d3a151f171ae3dfa7544caeec1c57553a0def755c46911b",
}
MODEL_REPO = "CK0607/qwen3.8-27b-brush-painting"
MODEL_REVISION = "a9566118df43ec92568472e23b1552946b01bd15"
MODEL_PREFIX = "rl-judge-matched-20261004-flash/step_12"
MODEL_WEIGHT = {
    "bytes": 233590600,
    "sha256": "c70c47ec0d6d7172b3bc25c4f54178919cf7c079cd9652fff3bc76db9e1eca79",
}
BASELINE_INITIALIZER_SHA = "e5211706a23b46354aa269f2b9cfaaaad1b64eef1012b433b1887c149f5a927c"
MATCHED_CASE_COUNT = 16


class RecoveryError(RuntimeError):
    """The immutable continuation evidence failed validation."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"unreadable JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"receipt is not an object: {path}")
    return value


def _assert_new_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RecoveryError(f"{label} must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise RecoveryError(f"refusing to overwrite existing {label}: {path}")
    else:
        path.mkdir(parents=True)


def _assert_prepared_directory(path: Path, label: str) -> None:
    """Accept the materialized arm root while rejecting replacement paths."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RecoveryError(f"{label} must be a real directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _safe_member_target(destination: Path, member: tarfile.TarInfo) -> Path:
    # Tar names are POSIX paths even when produced on another platform.
    name = member.name.replace("\\", "/")
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RecoveryError(f"unsafe archive member path: {member.name!r}")
    if member.issym() or member.islnk():
        raise RecoveryError(f"archive links are not permitted: {member.name!r}")
    target = destination.joinpath(*relative.parts)
    try:
        target.resolve(strict=False).relative_to(destination.resolve())
    except ValueError as exc:
        raise RecoveryError(f"archive member escapes extraction root: {member.name!r}") from exc
    return target


def _extract_safe(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        targets: set[Path] = set()
        for member in members:
            target = _safe_member_target(destination, member)
            if target in targets or target.exists() or target.is_symlink():
                raise RecoveryError(f"archive would overwrite an existing path: {member.name!r}")
            targets.add(target)
        for member in members:
            target = _safe_member_target(destination, member)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=False)
                continue
            if not member.isfile():
                raise RecoveryError(f"unsupported archive member type: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RecoveryError(f"archive member has no data: {member.name!r}")
            with target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777 or 0o644)


def _download_dataset(path: str, target: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - pinned Linux node supplies it
        raise RecoveryError("huggingface_hub is unavailable for public recovery") from exc
    return Path(hf_hub_download(
        repo_id=DATASET_REPO,
        filename=path,
        repo_type="dataset",
        revision=DATASET_REVISION,
        token=False,
        force_download=True,
        local_dir=target,
    ))


def _download_adapter(target: Path) -> Path:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - pinned Linux node supplies it
        raise RecoveryError("huggingface_hub is unavailable for public recovery") from exc
    info = HfApi(token=False).model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    if getattr(info, "sha", None) != MODEL_REVISION or getattr(info, "private", None) is not False:
        raise RecoveryError("public Flash model metadata did not match the pinned revision")
    siblings = {getattr(item, "rfilename", ""): item for item in getattr(info, "siblings", ())}
    remote = siblings.get(MODEL_PREFIX + "/adapter_model.safetensors")
    if remote is None or getattr(getattr(remote, "lfs", None), "sha256", None) != MODEL_WEIGHT["sha256"]:
        raise RecoveryError("public Flash LFS metadata did not match the pinned adapter")
    snapshot = Path(snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[MODEL_PREFIX + "/*"],
        token=False,
    ))
    source = snapshot / MODEL_PREFIX
    weight = source / "adapter_model.safetensors"
    config = source / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise RecoveryError("public Flash adapter is incomplete")
    if weight.stat().st_size != MODEL_WEIGHT["bytes"] or digest(weight) != MODEL_WEIGHT["sha256"]:
        raise RecoveryError("downloaded Flash adapter does not match the pinned bytes")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        source_file = source / name
        # Hugging Face cache snapshots commonly expose regular downloaded
        # blobs through symlinks.  They are accepted here after the metadata
        # and byte checks above; tar archive links remain rejected by
        # _extract_safe because they can redirect extraction.
        if not source_file.is_file():
            raise RecoveryError(f"public adapter file is not readable: {source_file}")
        shutil.copyfile(source_file, target / name)
    return source


def _copy_selected(source: Path, destination: Path, relative: Iterable[str]) -> None:
    for name in relative:
        src = source / name
        if not src.is_file() or src.is_symlink():
            raise RecoveryError(f"trusted archive is missing a regular file: {name}")
        dst = destination / name
        if dst.exists() or dst.is_symlink():
            raise RecoveryError(f"refusing to overwrite restored path: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _copy_validation(source: Path, destination: Path) -> None:
    src_root = source / "validation"
    if not src_root.is_dir() or src_root.is_symlink():
        raise RecoveryError("baseline archive has no validation evidence")
    for src in sorted(src_root.rglob("*")):
        relative = src.relative_to(source)
        if src.is_symlink():
            raise RecoveryError(f"validation evidence contains a symlink: {relative}")
        if src.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
            continue
        if not src.is_file():
            raise RecoveryError(f"validation evidence contains an unsupported path: {relative}")
        dst = destination / relative
        if dst.exists() or dst.is_symlink():
            raise RecoveryError(f"refusing to overwrite restored validation path: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _validate_baseline(baseline_root: Path, shared_root: Path) -> list[str]:
    identity = _load(baseline_root / "baseline-identity.json")
    if (
        identity.get("mode") != "recovered_policy"
        or identity.get("display_checkpoint") != "fresh_rl16_initializer"
        or identity.get("trained") is not False
        or identity.get("optimizer_resume") is not False
        or identity.get("scope") != "balanced16"
        or identity.get("case_count") != MATCHED_CASE_COUNT
        or identity.get("seed") != 918210
        or identity.get("initializer_sha256") != BASELINE_INITIALIZER_SHA
    ):
        raise RecoveryError("restored baseline identity is not the pinned fresh RL16 baseline")
    generation = _load(baseline_root / "validation/step-1-balanced16/generation-receipt.json")
    if (
        generation.get("scope") != "balanced16"
        or generation.get("adapter_sha256") != BASELINE_INITIALIZER_SHA
        or generation.get("checkpoint_provenance") != "public_download"
        or generation.get("matched_control") != "sft-step512"
        or generation.get("max_turns") != 3
        or generation.get("max_tokens") != 8192
    ):
        raise RecoveryError("restored baseline generation receipt is not the pinned balanced16 run")
    actual = _load(baseline_root / "validation/step-1-balanced16/cases.json").get("cases")
    expected = _load(shared_root / "anchors-frozen/validation-matched16.json").get("cases")
    if not isinstance(actual, list) or not isinstance(expected, list):
        raise RecoveryError("baseline or shared matched16 cases are malformed")
    actual_ids = [row.get("id") for row in actual if isinstance(row, dict)]
    expected_ids = [row.get("id") for row in expected if isinstance(row, dict)]
    if len(actual_ids) != MATCHED_CASE_COUNT or actual_ids != expected_ids:
        raise RecoveryError("restored baseline case order does not match the new shared panel")
    return actual_ids


def _validate_flash_receipt(archive_root: Path) -> dict[str, Any]:
    receipt_path = archive_root / "async-output/rl-judge-matched-20261004-flash/artifacts/adapters/step_12/hf-upload.json"
    receipt = _load(receipt_path)
    files = receipt.get("files", {})
    weight = files.get("adapter_model.safetensors", {}) if isinstance(files, dict) else {}
    if (
        receipt.get("repo_id") != MODEL_REPO
        or receipt.get("private") is not False
        or receipt.get("verified") is not True
        or receipt.get("commit") != MODEL_REVISION
        or receipt.get("path") != MODEL_PREFIX
        or weight.get("sha256") != MODEL_WEIGHT["sha256"]
        or weight.get("bytes") != MODEL_WEIGHT["bytes"]
    ):
        raise RecoveryError("archived Flash step-12 receipt is not the pinned public checkpoint")
    state = _load(archive_root / "supervisor-state.json")
    if (
        state.get("status") not in {"stopped", "completed"}
        or state.get("completed_optimizer_steps") != 12
        or state.get("last_optimizer_step") != 12
        or state.get("max_updates") != 12
    ):
        raise RecoveryError("archived Flash supervision receipt does not prove twelve updates")
    return receipt


def restore(*, baseline_root: Path, flash_root: Path, shared_root: Path, state_root: Path,
            baseline_archive_file: Path | None = None, flash_archive_file: Path | None = None,
            adapter_source: Path | None = None) -> dict[str, Any]:
    _assert_new_directory(baseline_root, "baseline root")
    _assert_prepared_directory(flash_root, "Flash arm root")
    recovery_root = state_root / "continuation-recovery"
    _assert_new_directory(recovery_root, "continuation recovery root")
    downloads = recovery_root / "downloads"
    downloads.mkdir()
    baseline_archive = baseline_archive_file or _download_dataset(BASELINE_ARCHIVE["remote_path"], downloads / "baseline")
    flash_archive = flash_archive_file or _download_dataset(FLASH_ARCHIVE["remote_path"], downloads / "flash")
    for path, expected, label in (
        (baseline_archive, BASELINE_ARCHIVE, "baseline archive"),
        (flash_archive, FLASH_ARCHIVE, "Flash archive"),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RecoveryError(f"{label} bytes do not match the pinned public archive")
    with tempfile.TemporaryDirectory(prefix="baseline-extract-", dir=recovery_root) as temp:
        extracted = Path(temp)
        _extract_safe(baseline_archive, extracted)
        _copy_selected(extracted, baseline_root, ("baseline-identity.json",))
        _copy_validation(extracted, baseline_root)
    with tempfile.TemporaryDirectory(prefix="flash-extract-", dir=recovery_root) as temp:
        extracted = Path(temp)
        _extract_safe(flash_archive, extracted)
        trusted_receipt = _validate_flash_receipt(extracted)
        _copy_selected(extracted, flash_root, ("supervisor-state.json", "consistent-group-admission-activation.json"))
        (flash_root / "restored-supervision-receipt.json").write_text(json.dumps({
            "schema": "matched-flash-recovered-supervision-v1",
            "actual_optimizer_updates": 12,
            "training_invoked_during_continuation": False,
            "source_archive_sha256": FLASH_ARCHIVE["sha256"],
            "trusted_checkpoint_revision": MODEL_REVISION,
            "trusted_checkpoint_prefix": MODEL_PREFIX,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        destination = flash_root / "async-output/rl-judge-matched-20261004-flash/artifacts/adapters/step_12"
        adapter_source_resolved = _download_adapter(destination) if adapter_source is None else adapter_source
        if adapter_source is not None:
            weight = adapter_source / "adapter_model.safetensors"
            config = adapter_source / "adapter_config.json"
            if adapter_source.is_symlink() or not weight.is_file() or not config.is_file() or weight.is_symlink() or config.is_symlink():
                raise RecoveryError("local adapter source is not a regular pinned adapter")
            if weight.stat().st_size != MODEL_WEIGHT["bytes"] or digest(weight) != MODEL_WEIGHT["sha256"]:
                raise RecoveryError("local adapter source does not match the pinned Flash bytes")
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(weight, destination / "adapter_model.safetensors")
            shutil.copyfile(config, destination / "adapter_config.json")
        receipt = {**trusted_receipt,
                   "provenance_type": "public_download",
                   "public_metadata_verified": True,
                   "recovered_from_trusted_archive": True}
        (destination / "hf-upload.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "validation.json").write_text(json.dumps({
            "step": 12, "sha256": MODEL_WEIGHT["sha256"], "finite": True,
            "source": "recovered_public_checkpoint",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case_ids = _validate_baseline(baseline_root, shared_root)
    (baseline_root / "baseline-recovery-receipt.json").write_text(json.dumps({
        "schema": "matched-fresh-rl16-baseline-recovery-v1",
        "provenance_type": "public_archive_recovery",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "archive": dict(BASELINE_ARCHIVE),
        "case_ids": case_ids,
        "regenerated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (flash_root / "flash-recovery-receipt.json").write_text(json.dumps({
        "schema": "matched-flash-evaluation-recovery-v1",
        "provenance_type": "public_checkpoint_recovery",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "archive": dict(FLASH_ARCHIVE),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_prefix": MODEL_PREFIX,
        "weights": dict(MODEL_WEIGHT),
        "actual_optimizer_updates": 12,
        "training_invoked": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "matched-continuation-recovery-v1",
        "baseline": {**dict(BASELINE_ARCHIVE), "case_ids": case_ids},
        "flash": {**dict(FLASH_ARCHIVE), "actual_optimizer_updates": 12,
                   "adapter_sha256": MODEL_WEIGHT["sha256"], "adapter_bytes": MODEL_WEIGHT["bytes"]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--flash-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--baseline-archive-file", type=Path)
    parser.add_argument("--flash-archive-file", type=Path)
    parser.add_argument("--adapter-source", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(restore(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RecoveryError, ValueError) as exc:
        print(f"matched continuation recovery: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
