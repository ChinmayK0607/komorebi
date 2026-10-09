#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys


def main(argv: list[str]) -> int:
    receipt_path, pointer_path, revision, prefix, expected = argv[1:]
    value = json.loads(Path(receipt_path).read_text())
    if value.get("verified") is not True or value.get("revision") != revision or value.get("prefix") != prefix or value.get("adapter_sha256") != expected:
        raise SystemExit("public adapter receipt does not match the pinned binding")
    path = Path(str(value.get("adapter_path", ""))).resolve()
    weight = path / "adapter_model.safetensors"
    if not weight.is_file():
        raise SystemExit("verified public adapter is incomplete")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit("verified public adapter bytes do not match pinned SHA-256")
    target = Path(pointer_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(path) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
