#!/usr/bin/env python3
"""Admit a small structured-flow pilot after verifying real anchor files.

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
spec = importlib.util.spec_from_file_location("structured_flow_curriculum", PREPARER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load curriculum preparer: {PREPARER}")
curriculum = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = curriculum
spec.loader.exec_module(curriculum)

CHECKPOINT_SHA256 = "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740"
FLOW_HINT = (
    "Plan before writing code: identify the subject hierarchy, placement, and "
    "largest shapes first; then make explicit canvas, palette, and brush "
    "choices. Revise by preserving correct structure and changing one control "
    "at a time. After observing the actual canvas, either return one meaningful "
    "complete painting revision or respond FINISHED when it satisfies the brief. "
    "For a paint action, return only executable painting code in the assistant turn."
)


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
    original = Path(str(row["reference"]))
    if not original.is_absolute():
        original = proposal_root / original
    original = original.resolve()
    if original.is_symlink() or not original.is_file():
        raise FileNotFoundError(f"prompt is missing: {original}")
    original_hash = sha256_file(original)
    if original_hash != row.get("reference_sha256"):
        raise ValueError(f"prompt hash mismatch for {row['id']}")
    body = original.read_text(encoding="utf-8").strip()
    hint = FLOW_HINT if enabled else "Structured-flow hint is disabled for this run."
    text = f"[SYSTEM]\n{hint}\n\n[USER]\n{body}\n"
    relative = Path("casebriefs") / str(row["split"]) / f"{row['id']}.txt"
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    row["source_prompt"] = str(original)
    row["source_prompt_sha256"] = original_hash
    # Keep the immutable native reference and its provenance hash unchanged;
    # the casebrief is an adjacent runtime hint consumed only when enabled.
    row["casebrief"] = str(target.resolve())
    row["casebrief_sha256"] = sha256_file(target)
    row["flow_policy_enabled"] = enabled
    row["system_hint_sha256"] = hashlib.sha256((hint + "\n").encode("utf-8")).hexdigest()


def materialize_pilot(
    *,
    train_proposal: Path,
    heldout_proposal: Path,
    anchor_root: Path,
    output_root: Path,
    frozen_sft512_checkpoint_sha256: str = CHECKPOINT_SHA256,
    flow_policy_enabled: bool = False,
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
    # The curriculum module owns pilot selection and provenance checks.  Keep
    # that contract in one place so later additions (trace IDs, source groups,
    # or stage balancing) cannot weaken this wrapper.
    curriculum.materialize_pilot_manifests(
        train_proposal, heldout_proposal, output_root, anchor_root,
        frozen_sft512_checkpoint_sha256=frozen_sft512_checkpoint_sha256,
    )
    train_path = output_root / "pilot-train-manifest.json"
    heldout_path = output_root / "pilot-heldout-manifest.json"
    train_doc = json.loads(train_path.read_text(encoding="utf-8"))
    heldout_doc = json.loads(heldout_path.read_text(encoding="utf-8"))
    train = train_doc.get("rows")
    heldout = heldout_doc.get("cases") or heldout_doc.get("rows")
    if not isinstance(train, list) or not isinstance(heldout, list) or len(train) != 32 or len(heldout) != 16:
        raise ValueError("curriculum pilot materializer did not produce the required 32/16 manifests")
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
    _write_json(output_root / "matched16-manifest.json", heldout_doc)
    receipt = {
        "schema": "structured-flow-pilot-materialization-v1",
        "run_id": curriculum.RUN_ID,
        "status": "prepared_with_verified_anchors",
        "train_rows": len(train),
        "heldout_cases": len(heldout),
        "full_train_rows": 240,
        "full_heldout_rows": 60,
        "train_manifest_sha256": sha256_file(train_path),
        "heldout_manifest_sha256": sha256_file(output_root / "matched16-manifest.json"),
        "frozen_sft512_checkpoint_sha256": frozen_sft512_checkpoint_sha256,
        "flow_policy_enabled": flow_policy_enabled,
        "anchors_verified": True,
        "limitations": [
            "Only 32 train and 16 heldout anchors are admitted for the bounded pilot.",
            "The remaining 208 train and 44 heldout rows remain prepared proposals until rendered and hash-verified.",
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
