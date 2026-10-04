#!/usr/bin/env python3
"""Normalize and verify the pinned public initializer download receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(repo: str, revision: str) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - pinned node supplies this
        raise ValueError("huggingface_hub is unavailable for anonymous metadata verification") from exc
    return HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)


def normalize_download(
    *,
    download_receipt: Path,
    adapter: Path,
    output: Path,
    repo: str,
    revision: str,
    prefix: str,
    expected_sha256: str,
    public_proof: Path | None = None,
    metadata_lookup: Callable[[str, str], Any] = _metadata,
) -> dict[str, Any]:
    try:
        raw = json.loads(download_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("public adapter download receipt is unreadable") from exc
    if not isinstance(raw, dict) or raw.get("verified") is not True:
        raise ValueError("public adapter download receipt is not verified")
    if raw.get("private") is True:
        raise ValueError("public adapter download receipt identifies a private source")
    if raw.get("repo") != repo or raw.get("revision") != revision or raw.get("prefix") != prefix:
        raise ValueError("public adapter download identity does not match the pinned initializer")
    if raw.get("adapter_sha256") != expected_sha256:
        raise ValueError("public adapter download SHA does not match the pinned initializer")
    raw_path = raw.get("adapter_path")
    if not isinstance(raw_path, str) or Path(raw_path).resolve() != adapter.resolve():
        raise ValueError("public adapter download path does not match the selected adapter")

    weight = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weight.is_file() or not config.is_file():
        raise ValueError("public adapter download is incomplete")
    actual_sha256 = _sha256(weight)
    if actual_sha256 != expected_sha256:
        raise ValueError("downloaded public adapter bytes do not match the pinned SHA")

    if public_proof is not None:
        try:
            proof = json.loads(public_proof.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("anonymous public metadata proof is unreadable") from exc
        if (
            not isinstance(proof, dict)
            or proof.get("schema") != "baseline-anonymous-public-metadata-v1"
            or proof.get("repo_id") != repo
            or proof.get("revision") != revision
            or proof.get("prefix") != prefix
            or proof.get("private") is not False
            or proof.get("adapter_sha256") != expected_sha256
            or proof.get("anonymous_metadata_verified") is not True
            or proof.get("hf_upload_receipt_present") is not False
        ):
            raise ValueError("anonymous public metadata proof does not match the pinned initializer")
    else:
        info = metadata_lookup(repo, revision)
        if getattr(info, "sha", None) != revision or getattr(info, "private", None) is not False:
            raise ValueError("anonymous public model metadata does not match the pinned revision")
        siblings = {getattr(item, "rfilename", ""): item for item in getattr(info, "siblings", ())}
        remote = siblings.get(prefix + "/adapter_model.safetensors")
        remote_lfs = getattr(remote, "lfs", None)
        if remote is None or getattr(remote_lfs, "sha256", None) != expected_sha256:
            raise ValueError("anonymous public model metadata has the wrong adapter SHA")

    result = {
        "schema": "matched-public-checkpoint-receipt-v1",
        "provenance_type": "public_download",
        "repo": repo,
        "commit": revision,
        "prefix": prefix,
        "adapter_path": str(adapter),
        "verified": True,
        "private": False,
        "public_metadata_verified": True,
        "files": {
            "adapter_model.safetensors": {
                "sha256": actual_sha256,
                "bytes": weight.stat().st_size,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--public-proof", type=Path)
    args = parser.parse_args()
    print(json.dumps(normalize_download(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
