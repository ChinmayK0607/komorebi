"""Package screened canary turns, traces, gallery, and receipts for public HF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from archive_evidence import IMAGE_EXT, TEXT_EXT, package


POLICIES = ("sft", "rl", "mix025", "mix050")


def add_tree(files: dict[str, Path], root: Path, prefix: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in {"cache", "private", "credentials"}
               for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_EXT | IMAGE_EXT:
            files[f"{prefix}/{relative.as_posix()}"] = path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files: dict[str, Path] = {}
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    files["protocol.md"] = args.protocol
    add_tree(files, args.inputs, "inputs")
    add_tree(files, args.gallery, "gallery")
    add_tree(files, args.eval_root / "logs", "canary/logs")
    for policy in POLICIES:
        add_tree(files, args.eval_root / policy, f"eval/{policy}")
        add_tree(files, args.rollouts / policy, f"rollouts/{policy}")
    for name in ("summary.json", "mix025/merge_receipt.json",
                 "mix050/merge_receipt.json", "mix025/adapter_config.json",
                 "mix050/adapter_config.json"):
        path = args.eval_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"canary/{name}"] = path
    receipt = package(files, args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
