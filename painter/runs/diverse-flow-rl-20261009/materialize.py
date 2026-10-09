#!/usr/bin/env python3
"""Admit the Diverse-flow train/evaluation manifests after verifying anchors.

This is metadata and file validation only.  It never renders, calls a model,
contacts a judge, downloads a checkpoint, or fills missing PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


RUN = Path(__file__).resolve().parent
PREPARER = RUN / "prepare_curriculum.py"
spec = importlib.util.spec_from_file_location("diverse_flow_curriculum", PREPARER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load curriculum preparer: {PREPARER}")
curriculum = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = curriculum
spec.loader.exec_module(curriculum)

CHECKPOINT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
FLOW_HINT = curriculum.FLOW_HINT
FULL_TRAIN_ROWS = 96
MIN_ADMITTED_TRAIN_ROWS = 88


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _casebrief(row: dict[str, Any], proposal_root: Path, output_root: Path, *, enabled: bool) -> None:
    # Keep the immutable native reference and its provenance hash unchanged;
    # the casebrief is an adjacent runtime hint consumed only when enabled.
    curriculum.write_casebrief(row, proposal_root, output_root, enabled=enabled)


def balanced_train_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Interleave the four tagged stages without claiming stage progression.

    Prime receives one ordinary manifest source in this run; no reviewed
    stage-aware sampler hook is available.  A deterministic round-robin order
    gives each contiguous manifest window mixed stage coverage while retaining
    the curriculum's immutable row identities and replay tags.
    """
    by_stage = {
        stage: sorted((row for row in rows if int(row.get("stage", 0)) == stage), key=lambda row: str(row["id"]))
        for stage in range(1, 5)
    }
    if any(len(items) != 24 for items in by_stage.values()):
        raise ValueError("balanced Diverse-flow order requires 24 rows in each stage")
    return [str(row["id"]) for index in range(24) for stage in range(1, 5) for row in by_stage[stage][index:index + 1]]


def materialize_pilot(
    *,
    train_proposal: Path,
    heldout_proposal: Path,
    anchor_root: Path,
    output_root: Path,
    frozen_sft512_checkpoint_sha256: str = CHECKPOINT_SHA256,
    flow_policy_enabled: bool = False,
    excluded_train_ids: set[str] | None = None,
    registered_train_ids: list[str] | None = None,
) -> dict[str, Any]:
    if frozen_sft512_checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("the pilot is bound to the immutable SFT512 checkpoint hash")
    train_proposal = train_proposal.expanduser().resolve()
    heldout_proposal = heldout_proposal.expanduser().resolve()
    anchor_root = anchor_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty materialization root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    excluded_train_ids = set(excluded_train_ids or ())
    train_rows = json.loads(train_proposal.read_text(encoding="utf-8")).get("rows", [])
    heldout_rows = json.loads(heldout_proposal.read_text(encoding="utf-8")).get("rows", [])
    expected_train_ids = balanced_train_ids(train_rows)
    expected_heldout_ids = [str(row["id"]) for row in heldout_rows]
    if len(expected_train_ids) != FULL_TRAIN_ROWS or len(expected_heldout_ids) != 32:
        raise ValueError("Diverse-flow proposals must contain exactly 96 train and 32 heldout rows")
    registered_train_ids = [str(ident) for ident in (registered_train_ids or expected_train_ids)]
    if len(registered_train_ids) != len(set(registered_train_ids)) or set(registered_train_ids) != set(expected_train_ids):
        raise ValueError("registered train IDs must equal the complete deterministic 96-row curriculum")
    if not excluded_train_ids <= set(registered_train_ids):
        raise ValueError("censored train IDs are outside the registered 96-row curriculum")
    admitted_train_ids = [ident for ident in registered_train_ids if ident not in excluded_train_ids]
    if not MIN_ADMITTED_TRAIN_ROWS <= len(admitted_train_ids) <= FULL_TRAIN_ROWS:
        raise ValueError(f"Diverse-flow requires {MIN_ADMITTED_TRAIN_ROWS}..{FULL_TRAIN_ROWS} admitted train anchors")
    # The curriculum module owns pilot selection and provenance checks.  Keep
    # that contract in one place so later additions (trace IDs, source groups,
    # or stage balancing) cannot weaken this wrapper.
    curriculum.materialize_pilot_manifests(
        train_proposal, heldout_proposal, output_root, anchor_root,
        frozen_sft512_checkpoint_sha256=frozen_sft512_checkpoint_sha256,
        admitted_train_ids=admitted_train_ids,
    )
    train_path = output_root / "pilot-train-manifest.json"
    heldout_path = output_root / "pilot-heldout-manifest.json"
    train_doc = json.loads(train_path.read_text(encoding="utf-8"))
    heldout_doc = json.loads(heldout_path.read_text(encoding="utf-8"))
    train = train_doc.get("rows")
    heldout = heldout_doc.get("cases") or heldout_doc.get("rows")
    if not isinstance(train, list) or not isinstance(heldout, list) or len(train) != len(admitted_train_ids) or len(heldout) != 32:
        raise ValueError("Diverse-flow materializer did not produce the registered 96/32 manifests")
    train_ids = {str(row.get("id")) for row in train}
    if [str(row.get("id")) for row in train] != admitted_train_ids:
        raise ValueError("curriculum pilot materializer changed the registered train order")
    if len(train_ids) != len(train) or not MIN_ADMITTED_TRAIN_ROWS <= len(train) <= FULL_TRAIN_ROWS:
        raise ValueError(f"materialized train rows must contain {MIN_ADMITTED_TRAIN_ROWS}..{FULL_TRAIN_ROWS} admitted rows")
    # The full proposal keeps six turns for later stages.  The bounded launch
    # is deliberately a three-turn pilot; set this after admission so the
    # native task loader cannot pick up the proposal default from a row.
    for row in [*train, *heldout]:
        row["max_turns"] = 3
    for row in train:
        _casebrief(row, train_proposal.parent, output_root, enabled=flow_policy_enabled)
    for row in heldout:
        _casebrief(row, heldout_proposal.parent, output_root, enabled=flow_policy_enabled)
    for row in [*train, *heldout]:
        opponent = Path(str(row["opponent"]))
        if opponent.is_symlink() or not opponent.is_file() or sha256_file(opponent) != row["opponent_sha256"]:
            raise ValueError(f"anchor verification failed for {row['id']}")
    train_doc["rows"] = train
    heldout_doc["rows"] = heldout
    heldout_doc["cases"] = heldout
    _write_json(train_path, train_doc)
    _write_json(heldout_path, heldout_doc)
    _write_json(output_root / "matched32-manifest.json", heldout_doc)
    receipt = {
        "schema": "Diverse-flow-materialization-v1",
        "run_id": curriculum.RUN_ID,
        "status": "prepared_with_verified_anchors",
        "train_rows": len(train),
        "original_train_rows": len(registered_train_ids),
        "excluded_train_ids": sorted(excluded_train_ids),
        "heldout_cases": len(heldout),
        "full_train_rows": 96,
        "full_heldout_rows": 32,
        "train_manifest_sha256": sha256_file(train_path),
        "heldout_manifest_sha256": sha256_file(output_root / "matched32-manifest.json"),
        "frozen_sft512_checkpoint_sha256": frozen_sft512_checkpoint_sha256,
        "flow_policy_enabled": flow_policy_enabled,
        "sampling_mode": "balanced_mixed_manifest",
        "stage_progression_claim": False,
        "anchors_verified": True,
        "limitations": [
            "The 96-row proposal and 32 heldout rows are admitted only after real frozen SFT512 PNGs and hashes are verified; bounded model-invalid train censoring is recorded in excluded_train_ids.",
            "This step did not render, run a model, judge an image, train, or contact a provider.",
        ],
    }
    _write_json(output_root / "pilot-materialization-receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-proposal", type=Path, required=True)
    parser.add_argument("--heldout-proposal", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-sft512-checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--enable-flow-policy", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(materialize_pilot(
        train_proposal=args.train_proposal,
        heldout_proposal=args.heldout_proposal,
        anchor_root=args.anchor_root,
        output_root=args.output_root,
        frozen_sft512_checkpoint_sha256=args.frozen_sft512_checkpoint_sha256,
        flow_policy_enabled=args.enable_flow_policy,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
