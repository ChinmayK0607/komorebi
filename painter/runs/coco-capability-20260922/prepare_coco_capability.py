"""Prepare the paired COCO capability check without running inference.

The source pool is the already imported 28-image COCO128 development split.
This command selects a deterministic, category- and object-count-diverse
subset, copies the source bytes, validates their hashes, and writes one native
painting manifest plus one config for each policy.  It does not render, load a
model, or contact an evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painter.native_painting_eval import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SEED,
    config_text,
    validate_manifest,
)


RUN_ID = "coco-capability-20260922"
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "painter/runs/scaled-curriculum-20260921/real-pool/reference-manifest.json"
)
DEFAULT_OUTPUT = REPO_ROOT / f"painter/runs/{RUN_ID}"
DEFAULT_EVAL_ROOT = Path("/root/painter")
SELECTION_SEED = RUN_ID
TARGET_COUNT = 12

BASE_SERVED_MODEL = "base"
SFT_SERVED_MODEL = "sft"
BASE_MODEL_REPO = "Qwen/Qwen3.8-27B"
SFT_POLICY_LABEL = SFT_SERVED_MODEL
SFT_DISPLAY_LABEL = "contract-quality-step24"
SFT_REPO = "CK0607/qwen3.8-27b-brush-painting"
SFT_REVISION = "b7103dd9923f19449ccf814f9b67bb2dcfed0909"
SFT_PREFIX = "contract-quality-20260918/step_24"
SFT_WEIGHTS_SHA256 = "d10eeb3b7b53fc539aa9d5e71eb98bcfe1c59d6a4878c7a69c93b8b09b59bcfa"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BASE_CONFIG_NAME = "base.toml"
SFT_CONFIG_NAME = "sft.toml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_bucket(count: int) -> str:
    if count == 1:
        return "one"
    if count <= 4:
        return "two_to_four"
    if count <= 9:
        return "five_to_nine"
    return "ten_plus"


def _categories(row: dict[str, Any]) -> set[str]:
    values = row.get("original_annotation_classes") or row.get("annotation_classes") or []
    return {str(value) for value in values if isinstance(value, str) and value}


def _family(row: dict[str, Any]) -> str:
    value = row.get("primary_family") or row.get("family")
    return str(value) if value else "unknown"


def _candidate_key(row: dict[str, Any], covered_categories: set[str], covered_buckets: set[str], covered_families: set[str]) -> tuple[int, str]:
    categories = _categories(row)
    bucket = _object_bucket(len(row.get("annotations") or []))
    family = _family(row)
    new_categories = categories - covered_categories
    new_bucket = bucket not in covered_buckets
    new_family = family not in covered_families
    score = 100 * len(new_categories) + 30 * int(new_bucket) + 10 * int(new_family)
    source_id = str(row.get("source_id") or row.get("id") or "")
    # The score is primary; source_id is an explicit ascending deterministic
    # tie-breaker rather than an accidental dependence on JSON order.
    return score, source_id


def select_cases(rows: list[dict[str, Any]], count: int = TARGET_COUNT) -> list[dict[str, Any]]:
    """Select dev-only cases with deterministic greedy coverage."""

    if isinstance(count, bool) or count < 1:
        raise ValueError("count must be positive")
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    if len(dev_rows) < count:
        raise ValueError(f"need {count} dev rows, found {len(dev_rows)}")
    dev_rows.sort(key=lambda row: str(row.get("source_id") or row.get("id") or ""))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered_categories: set[str] = set()
    covered_buckets: set[str] = set()
    covered_families: set[str] = set()
    for rank in range(1, count + 1):
        candidates = [
            row
            for row in dev_rows
            if str(row.get("source_id") or row.get("id") or "") not in selected_ids
        ]
        if not candidates:
            raise ValueError("selection count exceeds unique dev rows")
        # min() on the negated score and source id gives a stable ascending
        # source-id tie break while maximising the coverage score.
        row = min(
            candidates,
            key=lambda candidate: (
                -_candidate_key(candidate, covered_categories, covered_buckets, covered_families)[0],
                _candidate_key(candidate, covered_categories, covered_buckets, covered_families)[1],
            ),
        )
        categories = _categories(row)
        bucket = _object_bucket(len(row.get("annotations") or []))
        family = _family(row)
        new_categories = sorted(categories - covered_categories)
        new_bucket = bucket if bucket not in covered_buckets else None
        new_family = family not in covered_families
        score, _ = _candidate_key(row, covered_categories, covered_buckets, covered_families)
        selected_row = dict(row)
        selected_row["selection_rank"] = rank
        selected_row["selection_reason"] = (
            f"greedy rank {rank}: score={score}; "
            f"new_categories={','.join(new_categories) if new_categories else 'none'}; "
            f"new_object_count_bucket={new_bucket or 'none'}; "
            f"new_primary_family={'yes' if new_family else 'no'}"
        )
        selected.append(selected_row)
        selected_ids.add(str(row.get("source_id") or row.get("id") or ""))
        covered_categories.update(categories)
        covered_buckets.add(bucket)
        covered_families.add(family)
    return selected


def _resolve_source_image(source_manifest: Path, reference: str) -> Path:
    candidates = []
    path = Path(reference)
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((source_manifest.parent / path, REPO_ROOT / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"source reference not found: {reference!r}; tried {candidates}")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_source(source_manifest: Path) -> dict[str, Any]:
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("references"), list):
        raise ValueError("source manifest must be an object with a references list")
    rows = raw["references"]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("source references must be objects")
    return raw


def prepare(
    *,
    source_manifest: Path,
    output: Path,
    eval_root: Path = DEFAULT_EVAL_ROOT,
    count: int = TARGET_COUNT,
    force: bool = False,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output = output.resolve()
    source = _read_source(source_manifest)
    rows = source["references"]
    if source.get("split", {}).get("source_group_disjoint") is not True:
        raise ValueError("source split does not attest source-group disjointness")
    if source.get("split", {}).get("dev_count") != len([row for row in rows if row.get("split") == "dev"]):
        raise ValueError("source dev_count disagrees with rows")
    source_groups = [str(row.get("source_group")) for row in rows]
    if len(source_groups) != len(set(source_groups)):
        raise ValueError("source manifest has duplicate source groups")
    selected = select_cases(rows, count=count)
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"output is non-empty; pass --force to rebuild: {output}")
    output.mkdir(parents=True, exist_ok=True)
    references_dir = output / "references"
    references_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for row in selected:
        if row.get("split") != "dev":
            raise ValueError(f"selected non-dev row: {row.get('id')}")
        source_id = str(row.get("source_id") or row.get("id"))
        source_group = str(row.get("source_group") or "")
        source_image = _resolve_source_image(source_manifest, str(row.get("reference") or row.get("reference_path")))
        source_digest = sha256(source_image)
        expected_digest = row.get("image_sha256")
        if expected_digest and expected_digest != source_digest:
            raise ValueError(f"source hash mismatch for {source_id}: {source_digest}")
        target = references_dir / f"{source_id}{source_image.suffix.lower() or '.jpg'}"
        if target.exists() and sha256(target) != source_digest:
            raise ValueError(f"existing copied reference hash mismatch for {source_id}")
        if not target.exists():
            shutil.copyfile(source_image, target)
        if sha256(target) != source_digest:
            raise ValueError(f"copied reference hash mismatch for {source_id}")
        categories = sorted(_categories(row))
        object_count = len(row.get("annotations") or [])
        case = {
            "id": f"{RUN_ID}-{source_id}",
            "source_group": source_group,
            "split": "dev",
            "mode": "fresh",
            "reference": f"references/{target.name}",
            "reference_sha256": source_digest,
            "source_id": source_id,
            "source_reference": str(row.get("reference") or row.get("reference_path")),
            "source_reference_sha256": source_digest,
            "source_split": row.get("split"),
            "family": _family(row),
            "primary_family": _family(row),
            "member_families": row.get("member_families") or categories,
            "annotation_classes": categories,
            "original_annotation_classes": row.get("original_annotation_classes") or categories,
            "annotations": row.get("annotations") or [],
            "object_count": object_count,
            "object_count_bucket": _object_bucket(object_count),
            "selection_rank": row["selection_rank"],
            "selection_reason": row["selection_reason"],
            "source_image_url": row.get("source_image_url"),
            "provenance_urls": row.get("provenance_urls") or [],
            "dataset": row.get("dataset"),
            "dataset_attribution": row.get("dataset_attribution"),
            "dataset_license_documentation": row.get("dataset_license_documentation"),
            "per_image_license": row.get("per_image_license"),
            "per_image_license_status": row.get("per_image_license_status") or "unverified_not_provided_in_archive",
            "exposure_status": row.get("exposure_status"),
            "heldout_claim": False,
            "unseen_pretraining_claim": False,
            "teacher_claim": False,
            "dataset_role": "reference_only",
        }
        cases.append(case)

    if len({case["source_group"] for case in cases}) != count:
        raise ValueError("selected cases do not have unique source groups")
    selected_categories = sorted({category for case in cases for category in case["annotation_classes"]})
    selected_families = sorted({case["primary_family"] for case in cases})
    selected_buckets = sorted({case["object_count_bucket"] for case in cases})
    relative_run = output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else Path(f"painter/runs/{RUN_ID}")
    eval_root = Path(eval_root)
    eval_manifest = eval_root / relative_run / "eval-manifest.json"
    base_output = eval_root / relative_run / "base-eval"
    sft_output = eval_root / relative_run / "sft-step24-eval"
    manifest = {
        "schema": "painter.coco-capability-eval.v1",
        "run_id": RUN_ID,
        "status": "prepared_no_inference",
        "mode": "fresh",
        "cases": cases,
        "source_pool": {
            "manifest": _relative(source_manifest),
            "manifest_sha256": sha256(source_manifest),
            "dataset": source.get("source", {}).get("dataset"),
            "archive_url": source.get("source", {}).get("archive_url"),
            "archive_sha256": source.get("source", {}).get("archive_sha256"),
            "docs_url": source.get("source", {}).get("docs_url"),
            "dataset_attribution": source.get("source", {}).get("dataset_attribution"),
            "dataset_license_documentation": source.get("source", {}).get("dataset_license_documentation"),
            "per_image_license_status": source.get("source", {}).get("per_image_license_status"),
            "available_dev_count": len([row for row in rows if row.get("split") == "dev"]),
            "train_rows_excluded": len([row for row in rows if row.get("split") == "train"]),
        },
        "selection": {
            "seed": SELECTION_SEED,
            "source_split": "dev",
            "available_count": len([row for row in rows if row.get("split") == "dev"]),
            "selected_count": len(cases),
            "rule": "greedy score = 100*new_category_count + 30*new_object_count_bucket + 10*new_primary_family; source_id ascending tie-break",
            "covered_categories": selected_categories,
            "covered_primary_families": selected_families,
            "covered_object_count_buckets": selected_buckets,
            "object_count_distribution": dict(sorted(Counter(case["object_count_bucket"] for case in cases).items())),
        },
        "comparison_protocol": {
            "same_manifest": True,
            "same_seed": DEFAULT_SEED,
            "temperature": 1.0,
            "max_turns": DEFAULT_MAX_TURNS,
            "max_tokens_per_turn": DEFAULT_MAX_TOKENS,
            "context_tokens": DEFAULT_CONTEXT,
            "samples_per_reference": 1,
            "training": False,
            "judge_during_generation": False,
            "thinking": False,
            "base": {
                "label": BASE_SERVED_MODEL,
                "display_name": "unadapted base",
                "served_model": BASE_SERVED_MODEL,
                "base_model": BASE_MODEL_REPO,
                "revision": BASE_REVISION,
                "adapter": None,
            },
            "sft": {
                "label": SFT_POLICY_LABEL,
                "display_name": SFT_DISPLAY_LABEL,
                "served_model": SFT_SERVED_MODEL,
                "base_model": BASE_MODEL_REPO,
                "revision": SFT_REVISION,
                "adapter_repo": SFT_REPO,
                "adapter_prefix": SFT_PREFIX,
                "adapter_weights_sha256": SFT_WEIGHTS_SHA256,
            },
        },
        "evaluation_configs": {
            "base": {"path": BASE_CONFIG_NAME, "policy_label": BASE_SERVED_MODEL, "model": BASE_SERVED_MODEL},
            "sft": {"path": SFT_CONFIG_NAME, "policy_label": SFT_POLICY_LABEL, "model": SFT_SERVED_MODEL},
            "eval_root": str(eval_root),
        },
        "limitations": [
            "COCO128 is a reference-only development subset; no held-out or unseen-pretraining claim is made.",
            "Per-image license identity is unverified where the source archive provides no image-level license or creator record.",
            "A preparation manifest contains no model outputs or quality judgment.",
        ],
    }
    manifest_path = output / "eval-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Validate paths and hashes through the same native evaluator schema used at
    # launch time.  The evaluator only needs the local manifest for this check.
    validate_manifest(output, manifest_path, expected_count=count)
    (output / BASE_CONFIG_NAME).write_text(
        config_text(
            taskset_id="native_painting",
            root=eval_root,
            manifest=eval_manifest,
            policy_label=BASE_SERVED_MODEL,
            model=BASE_SERVED_MODEL,
            output_dir=base_output,
            count=count,
        )
        + "\n# Paired protocol requires the serving endpoint max-model-len/context length to be 32768.\n",
        encoding="utf-8",
    )
    (output / SFT_CONFIG_NAME).write_text(
        config_text(
            taskset_id="native_painting",
            root=eval_root,
            manifest=eval_manifest,
            policy_label=SFT_POLICY_LABEL,
            model=SFT_SERVED_MODEL,
            output_dir=sft_output,
            count=count,
        )
        + "\n# Paired protocol requires the serving endpoint max-model-len/context length to be 32768.\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    parser.add_argument("--force", action="store_true", help="rebuild an existing non-empty output directory")
    args = parser.parse_args(argv)
    manifest = prepare(
        source_manifest=args.source_manifest,
        output=args.output,
        eval_root=args.eval_root,
        count=args.count,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "manifest": str((args.output / "eval-manifest.json").resolve()),
                "count": len(manifest["cases"]),
                "covered_categories": len(manifest["selection"]["covered_categories"]),
                "covered_families": len(manifest["selection"]["covered_primary_families"]),
                "covered_object_count_buckets": manifest["selection"]["covered_object_count_buckets"],
                "configs": [BASE_CONFIG_NAME, SFT_CONFIG_NAME],
                "status": manifest["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
