#!/usr/bin/env python3
"""Bind the prepared config receipt to the active warmup adapter.

The structured config records the public Flash12 initializer.  After the
four-update warmup, Prime's generic launcher must validate the actual warmup
adapter bytes passed as ``WATERCOLOUR_INIT_ADAPTER``.  This small node-side
normalizer keeps both identities in one receipt and supplies the generic
launcher-compatible top-level fields without changing the prepared TOML.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


FROZEN_ANCHOR = {
    "alias": "sft-step512",
    "revision": "adf687e5b321514eda7be6eab050434fdc8e44ff",
    "prefix": "photo-curriculum-sft-20260922-balanced/step_512",
    "checkpoint_sha256": "bea72b066986faa1b92b9dc539c710093181ee916b99323e2b2498f318ac1740",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config receipt must contain a JSON object")
    return value


def normalize(
    receipt_path: Path,
    *,
    active_sha256: str,
    parent_sha256: str,
    parent_revision: str,
    parent_prefix: str,
    mode: str,
    active_policy_json: Path | None = None,
) -> dict:
    if len(active_sha256) != 64 or any(c not in "0123456789abcdef" for c in active_sha256):
        raise ValueError("active adapter SHA-256 is malformed")
    if len(parent_sha256) != 64 or any(c not in "0123456789abcdef" for c in parent_sha256):
        raise ValueError("parent adapter SHA-256 is malformed")
    if not parent_revision or not parent_prefix:
        raise ValueError("parent adapter identity is incomplete")
    value = _load(receipt_path)
    initializer = value.get("policy_initializer")
    if not isinstance(initializer, dict):
        raise ValueError("config receipt has no policy_initializer identity")
    expected = {
        "revision": parent_revision,
        "prefix": parent_prefix,
        "sha256": parent_sha256,
    }
    if any(initializer.get(key) != wanted for key, wanted in expected.items()):
        raise ValueError("config receipt policy_initializer does not match pinned Flash12 parent")
    frozen = value.get("frozen_sft")
    if not isinstance(frozen, dict):
        raise ValueError("config receipt has no frozen_sft identity")
    for key in ("alias", "revision", "prefix"):
        if frozen.get(key) != FROZEN_ANCHOR[key]:
            raise ValueError(f"config receipt frozen_sft {key} does not match pinned step-512 anchor")
    if frozen.get("sha256") != FROZEN_ANCHOR["checkpoint_sha256"]:
        raise ValueError("config receipt frozen_sft sha256 does not match pinned step-512 anchor")
    if active_policy_json is not None:
        active = _load(active_policy_json)
        if active.get("status") != "verified_warmup_adapter":
            raise ValueError("active policy receipt is not a verified warmup adapter")
        if active.get("adapter_sha256") != active_sha256:
            raise ValueError("active policy receipt does not match active adapter bytes")
    value["mode"] = mode
    value["adapter_revision"] = parent_revision
    value["adapter_prefix"] = parent_prefix
    # The generic Prime launcher validates this field against the bytes it
    # loads.  Keep the immutable parent identity under policy_initializer.
    value["adapter_sha256"] = active_sha256
    # photo-curriculum's generic launcher consumes this exact compatibility
    # shape. Keep the source receipt under frozen_sft and derive the checked
    # fields only after validating every pinned value above.
    value["frozen_anchor"] = {**FROZEN_ANCHOR, "immutable": True}
    value["active_policy"] = {
        "sha256": active_sha256,
        "parent_sha256": parent_sha256,
        "source": "verified_warmup_adapter" if active_sha256 != parent_sha256 else "public_flash12_initializer",
    }
    temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--active-sha256", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--parent-prefix", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--active-policy-json", type=Path)
    args = parser.parse_args(argv)
    normalize(
        args.receipt,
        active_sha256=args.active_sha256,
        parent_sha256=args.parent_sha256,
        parent_revision=args.parent_revision,
        parent_prefix=args.parent_prefix,
        mode=args.mode,
        active_policy_json=args.active_policy_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
