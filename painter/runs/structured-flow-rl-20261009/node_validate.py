#!/usr/bin/env python3
"""Validate node-side structured-flow files and write its receipt."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(argv: list[str]) -> int:
    run, train_path, eval_path, policy_file, frozen_file = map(Path, argv[1:])
    train = json.loads(train_path.read_text())
    heldout = json.loads(eval_path.read_text())
    rows, cases = train.get("rows"), heldout.get("cases")
    receipt = json.loads((run / "rl-config-receipt.json").read_text())
    expected_train = receipt.get("curriculum", {}).get("pilot_rows")
    if not isinstance(expected_train, int) or not 24 <= expected_train <= 32:
        raise SystemExit("config receipt must bind an admitted train count in [24, 32]")
    if not isinstance(rows, list) or len(rows) != expected_train or len({row.get("id") for row in rows}) != expected_train:
        raise SystemExit(f"pilot train manifest must contain {expected_train} unique rows")
    if not isinstance(cases, list) or len(cases) != 16 or len({row.get("id") for row in cases}) != 16:
        raise SystemExit("matched evaluation manifest must contain 16 unique cases")
    for row in [*rows, *cases]:
        for key in ("reference", "opponent"):
            path = Path(str(row.get(key, "")))
            if not path.is_file() or path.is_symlink():
                raise SystemExit(f"manifest path is not a regular file: {row.get('id')} {key}")
    policy = Path(policy_file.read_text().strip())
    frozen = Path(frozen_file.read_text().strip())
    parent_policy_sha = os.environ["PAINTER_POLICY_SHA256"]
    active_policy_sha = os.environ.get("PAINTER_ACTIVE_POLICY_ADAPTER_SHA256", parent_policy_sha)
    actual_policy_sha = sha(policy / "adapter_model.safetensors")
    if actual_policy_sha != active_policy_sha:
        raise SystemExit("active policy adapter SHA mismatch")
    warmup = None
    active_receipt = run / "active-policy.json"
    if active_policy_sha != parent_policy_sha:
        if not active_receipt.is_file() or active_receipt.is_symlink():
            raise SystemExit("non-Flash active policy requires a verified warmup receipt")
        warmup = json.loads(active_receipt.read_text())
        if warmup.get("status") != "verified_warmup_adapter" or warmup.get("adapter_sha256") != active_policy_sha:
            raise SystemExit("active policy receipt does not bind the warmup adapter bytes")
    if sha(frozen / "adapter_model.safetensors") != os.environ["PAINTER_FROZEN_SFT_SHA256"]:
        raise SystemExit("frozen SFT512 SHA mismatch")
    if policy.resolve() == frozen.resolve():
        raise SystemExit("policy initializer and frozen SFT adapter must be distinct")
    config = run / "native.toml"
    value = {
        "schema": "structured-flow-rl-preparation-v1",
        "run_name": run.name,
        "profile": "structured-flow-rl-v1",
        "base_model_revision": receipt.get("base_model_revision"),
        "prime_revision": receipt.get("prime_revision"),
        "policy_initializer": receipt.get("policy_initializer"),
        "frozen_sft": receipt.get("frozen_sft"),
        "policy_adapter_sha256": actual_policy_sha,
        "parent_policy_adapter_sha256": parent_policy_sha,
        "frozen_sft_adapter_sha256": sha(frozen / "adapter_model.safetensors"),
        "optimizer_resume": False,
        "train_manifest_sha256": sha(train_path),
        "config_sha256": sha(config),
        "pilot_train_rows": expected_train,
        "matched_heldout_cases": 16,
        "checkpoint_steps": [4, 8],
        "structured_flow_policy_enabled": os.environ.get("PAINTER_STRUCTURED_FLOW_POLICY") == "1",
        "code_supervision_enabled": os.environ.get("PAINTER_CODE_SUPERVISION_ENABLED") == "1",
        "anchor_bank_status": f"pilot_only; admitted_{expected_train}_of_32_train; remaining_208_train_and_44_heldout_pending_render",
        "judge": {"model": os.environ["PAINTER_JUDGE_MODEL"], "effort": os.environ["PAINTER_JUDGE_EFFORT"], "format": os.environ["PAINTER_JUDGE_FORMAT"], "new_reward_bonus": False},
    }
    admission = run / "curriculum/anchor-admission.json"
    if admission.is_file() and not admission.is_symlink():
        value["anchor_admission_receipt"] = {
            "path": str(admission),
            "sha256": sha(admission),
            "admitted_train_rows": json.loads(admission.read_text()).get("admitted_train_count"),
            "excluded_train_ids": json.loads(admission.read_text()).get("excluded_train_ids", []),
        }
    (run / "preparation-receipt.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    # The reused evidence/finalization helpers use their historical manifest
    # locations.  Keep small byte-identical aliases so those helpers consume
    # this run's admitted train/16 heldout rows without importing the old run identity.
    legacy = run / "anchors-frozen"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "train-manifest.json").write_bytes(train_path.read_bytes())
    (legacy / "validation-manifest.json").write_bytes(eval_path.read_bytes())
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "matched16-manifest.json").write_bytes(eval_path.read_bytes())
    identity = {
        "schema": "structured-flow-source-identity-v1",
        "run_name": run.name,
        "model_revision": value["base_model_revision"],
        "initializer_revision": value["policy_initializer"]["revision"],
        "initializer_prefix": value["policy_initializer"]["prefix"],
        "initializer_sha256": value["policy_initializer"]["sha256"],
        "policy_model_revision": value["policy_initializer"]["revision"],
        "frozen_sft_model_revision": value["frozen_sft"]["revision"],
        "frozen_sft_initializer": value["frozen_sft"],
        "prime_revision": value["prime_revision"],
        "protocol": os.environ["PAINTER_JUDGE_PROTOCOL"],
    }
    if warmup is not None:
        identity["warmup"] = warmup
    identity_path = run / "source-identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text())
        if any(existing.get(key) != item for key, item in identity.items() if key in existing):
            raise SystemExit("existing source identity conflicts with structured pins")
        existing.update({key: item for key, item in identity.items() if key not in {"config_sha256", "manifest_sha256", "foundation_lock_sha256"}})
        identity_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    else:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
