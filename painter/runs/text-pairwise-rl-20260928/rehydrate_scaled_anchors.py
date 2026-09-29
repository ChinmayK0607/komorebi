"""Reuse hash-verified public SFT canvases for a larger text-RL curriculum.

This creates fresh absolute paths and manifests on the Linux node. It does not
render, judge, train, or modify the immutable public evidence archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


EVIDENCE_SHA256 = "ba4252251a54f9eeab16cff7cbf45939ee37e57730a8fa9191fe9121f235ccd3"
POOL_SHA256 = "494e4d9d76313faaae12fc7676d355c5041d138886d9ae1375f8f537fb8bcea6"
OLD_TRAIN_SHA256 = "9a4cd3795cf3cf1e22d1d00eda8399e50bb699ecaaae683edb0d0b1262fa896d"
ANCHOR_ROOT = "anchors-scaled"
POOL_ROOT = "prepared/scaled-run-pool"
ORDER_SEED = "text-scaled-anchor-20260929"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    if not member.isfile() or member.name != name:
        raise ValueError(f"unsafe evidence member: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"missing evidence member: {name}")
    return stream.read()


def rehydrate(archive_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if sha(archive_path) != EVIDENCE_SHA256:
        raise ValueError("source evidence archive hash mismatch")
    output.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            pool_raw = member_bytes(archive, f"{POOL_ROOT}/pool.json")
            if hashlib.sha256(pool_raw).hexdigest() != POOL_SHA256:
                raise ValueError("184-brief pool manifest hash mismatch")
            pool = json.loads(pool_raw)
            audit = json.loads(member_bytes(archive, f"{ANCHOR_ROOT}/anchor-audit.json"))
            old_raw = member_bytes(archive, f"{ANCHOR_ROOT}/train-manifest.json")
            if hashlib.sha256(old_raw).hexdigest() != OLD_TRAIN_SHA256:
                raise ValueError("old 68-row train manifest hash mismatch")
            old_ids = {row["id"] for row in json.loads(old_raw)["rows"]}
            decisions = {row["id"]: row for row in audit["decisions"]}
            cases = pool["cases"]
            if len(cases) != 184 or len(decisions) != 184 or len(old_ids) != 68:
                raise ValueError("source curriculum count mismatch")
            prompt_dir, anchor_dir = output / "prompts", output / "anchors"
            prompt_dir.mkdir()
            anchor_dir.mkdir()
            valid: list[dict] = []
            excluded: list[dict] = []
            for case in cases:
                ident = case["id"]
                if ident not in decisions or not ident.startswith("t600-"):
                    raise ValueError(f"case/audit mismatch: {ident}")
                prompt = member_bytes(archive, f"{POOL_ROOT}/{case['reference']}")
                if hashlib.sha256(prompt).hexdigest() != case["reference_sha256"]:
                    raise ValueError(f"brief hash mismatch: {ident}")
                prompt_path = prompt_dir / f"{ident}.txt"
                prompt_path.write_bytes(prompt)
                decision = decisions[ident]
                if decision["reason"] != "valid":
                    excluded.append({"id": ident, "split": case["split"],
                                     "reason": decision["reason"]})
                    continue
                canvas = member_bytes(archive, f"{ANCHOR_ROOT}/anchors/{ident}.png")
                expected = decision["anchor_sha256"]
                if hashlib.sha256(canvas).hexdigest() != expected:
                    raise ValueError(f"SFT anchor hash mismatch: {ident}")
                canvas_path = anchor_dir / f"{ident}.png"
                canvas_path.write_bytes(canvas)
                valid.append({**case, "reference": str(prompt_path.resolve()),
                              "opponent": str(canvas_path.resolve()),
                              "opponent_sha256": expected,
                              "opponent_source_group": case["source_group"],
                              "mode": "fresh", "max_turns": 3})
        train = [row for row in valid if row["split"] == "train"]
        simple = [row for row in train if row["family"] == "simple"]
        intermediate = [row for row in train if row["family"] == "intermediate"]
        complex_rows = [row for row in train if row["family"] == "complex"]
        key = lambda row: hashlib.sha256(f"{ORDER_SEED}:{row['id']}".encode()).hexdigest()
        new_complex = sorted((row for row in complex_rows if row["id"] not in old_ids), key=key)
        old_complex = sorted((row for row in complex_rows if row["id"] in old_ids), key=key)
        ordered = simple + intermediate + new_complex + old_complex
        validation = [row for row in valid if row["split"] == "validation"]
        reserve = [row for row in valid if row["split"] == "reserve"]
        if (len(ordered), len(simple), len(intermediate), len(complex_rows),
                len(validation), len(reserve), len(new_complex)) != (119, 24, 16, 79, 47, 16, 31):
            raise ValueError("reused source counts differ from reviewed 184-brief pool")
        if not old_ids <= {row["id"] for row in ordered}:
            raise ValueError("an old training anchor disappeared")
        files = {"train-manifest.json": {"rows": ordered},
                 "validation-manifest.json": {"cases": validation},
                 "reserve-manifest.json": {"cases": reserve}}
        hashes = {}
        for name, payload in files.items():
            path = output / name
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            hashes[name] = sha(path)
        receipt = {"schema": "text-paint-scale2-reused-anchors-v1",
                   "source_archive_sha256": EVIDENCE_SHA256,
                   "source_pool_sha256": POOL_SHA256,
                   "source_train_sha256": OLD_TRAIN_SHA256,
                   "anchor_count": len(valid), "train_count": len(ordered),
                   "new_complex_count": len(new_complex),
                   "validation_count": len(validation), "reserve_count": len(reserve),
                   "excluded": excluded, "manifest_sha256": hashes,
                   "curriculum": "simple, intermediate, unseen complex, previously used complex"}
        (output / "reuse-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    except BaseException:
        # A failed verification must never leave an apparently usable manifest.
        for item in sorted(output.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        output.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = rehydrate(args.archive, args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
