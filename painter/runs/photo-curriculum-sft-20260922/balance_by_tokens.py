"""Build a finite admission schedule from actual audited assistant tokens."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


DATA_FIELDS = (
    "id", "source_group", "split", "family", "source_kind", "example_kind",
    "complete_target", "reference_sha256", "target_sha256", "current_canvas_sha256",
    "current_program_sha256", "messages", "source_example_id", "admission_index",
    "admission_repeat", "admission_bucket",
)


def training_projection(row: dict) -> dict:
    """Remove heterogeneous nested provenance before Prime/Arrow schema load."""
    return {key: row[key] for key in DATA_FIELDS if key in row}


def build(candidates: Path, audit_path: Path, output: Path, steps: int, batch_size: int,
          targets: dict[str, float]) -> dict:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if any(value < 0 for value in targets.values()) or abs(sum(targets.values()) - 1.0) > 1e-9:
        raise ValueError("token-mass target fractions must sum to one")
    active_targets = {bucket: fraction for bucket, fraction in targets.items() if fraction > 0}
    if not active_targets:
        raise ValueError("at least one token-mass target must be positive")
    candidate_rows = {row["id"]: row for row in read_jsonl(candidates)}
    audit = json.loads(audit_path.read_text())
    measured = {row["id"]: row for row in audit["examples"] if row["split"] == "train"}
    if set(candidate_rows) != set(measured):
        raise ValueError("candidate rows and processor audit IDs differ")
    pools: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for ident, row in candidate_rows.items():
        kind = str(row.get("source_kind", "diverse_foundation"))
        bucket = "real" if kind == "real_photo_teacher" else "retention" if kind == "retention" else "foundation"
        pools[bucket].append((row, measured[ident]))
    missing = {bucket: len(pools[bucket]) for bucket in active_targets if not pools[bucket]}
    if missing:
        raise ValueError(f"missing token-balance pool: {missing}")
    # Shuffle each pool once with a fixed seed, then walk it round-robin.  This
    # keeps repeated admissions explicit while preventing a long first row from
    # monopolizing a bucket whenever its token deficit is largest.
    for bucket, pool in pools.items():
        pool.sort(key=lambda item: (str(item[0].get("source_group", "")), item[0]["id"]))
        random.Random(92022 + sum(ord(char) for char in bucket)).shuffle(pool)
    count = steps * batch_size
    desired_total = sum(int(measured[ident]["assistant_tokens"]) for ident in candidate_rows) * (count / max(1, len(candidate_rows)))
    desired = {bucket: desired_total * fraction for bucket, fraction in active_targets.items()}
    selected_mass = {bucket: 0 for bucket in active_targets}
    cursors = {bucket: 0 for bucket in active_targets}
    selected = []
    source_exposures = defaultdict(int)
    def admit(admission: int, bucket: str, pair: tuple[dict, dict]) -> None:
        row, metrics = pair
        copied = dict(row)
        original_id = row["id"]
        source_exposures[original_id] += 1
        copied["id"] = f"{original_id}__admit_{admission:06d}"
        copied["source_example_id"] = original_id
        copied["admission_index"] = admission
        copied["admission_repeat"] = source_exposures[original_id] > 1
        copied["admission_bucket"] = bucket
        selected.append(copied)
        selected_mass[bucket] += metrics_value(metrics)

    # With the substantial schedule (4096 slots for roughly 3k candidates),
    # expose every active candidate once before token-mass repeats.  This is a
    # coverage guarantee, not a claim that repeats are new references.
    eligible_count = sum(len(pools[bucket]) for bucket in active_targets)
    eligible_ids = {row["id"] for bucket in active_targets for row, _ in pools[bucket]}
    coverage_required = count >= eligible_count
    if coverage_required:
        for bucket in active_targets:
            for pair in pools[bucket]:
                admit(len(selected), bucket, pair)

    for admission in range(len(selected), count):
        # Choose the pool with the largest normalized token deficit.  The
        # candidate nearest to that deficit limits overshoot and preserves
        # actual measured token mass instead of example-count weighting.
        deficits = {bucket: desired[bucket] - selected_mass[bucket] for bucket in active_targets}
        bucket = max(active_targets, key=lambda key: deficits[key] / max(desired[key], 1.0))
        pool = pools[bucket]
        row, metrics = pool[cursors[bucket]]
        cursors[bucket] = (cursors[bucket] + 1) % len(pool)
        admit(admission, bucket, (row, metrics))

    # Keep runtime shuffle disabled so distributed resumption has a stable
    # admission order, but avoid long contiguous source buckets in the file.
    # Reassign IDs/indexes after permutation so the receipt describes the
    # actual order consumed by Prime rather than the construction order.
    order = list(range(len(selected)))
    random.Random(92022 + len(selected)).shuffle(order)
    shuffled = []
    for final_index, old_index in enumerate(order):
        copied = dict(selected[old_index])
        copied["id"] = f"{copied['source_example_id']}__admit_{final_index:06d}"
        copied["admission_index"] = final_index
        shuffled.append(copied)
    selected = shuffled
    candidate_train_sha256 = __import__("hashlib").sha256(candidates.read_bytes()).hexdigest()
    train_hash = _write_jsonl(output, [training_projection(row) for row in selected])
    validation_source = candidates.with_name("candidates-validation.jsonl")
    validation_hash = None
    if validation_source.is_file():
        validation_destination = output.with_name("validation.jsonl")
        validation_hash = _write_jsonl(validation_destination, [training_projection(row) for row in read_jsonl(validation_source)])
    actual_total = sum(selected_mass.values())
    receipt = {
        "schema": 1, "status": "balanced_schedule_ready", "steps": steps,
        "batch_size": batch_size, "row_count": len(selected), "finite": True,
        "requested_token_mass": targets,
        "actual_token_mass": {bucket: selected_mass.get(bucket, 0) / actual_total for bucket in targets},
        "assistant_tokens_by_bucket": selected_mass,
        "unique_source_examples": len(source_exposures),
        "repeated_source_examples": sum(n > 1 for n in source_exposures.values()),
        "coverage_required": coverage_required,
        "coverage_satisfied": len(source_exposures) == eligible_count if coverage_required else None,
        "uncovered_source_examples": sorted(eligible_ids - set(source_exposures)),
        "final_exposure_count_by_bucket": {
            bucket: sum(1 for row in selected if row.get("admission_bucket") == bucket)
            for bucket in targets
        },
        "train_sha256": train_hash,
        "candidate_train_sha256": candidate_train_sha256,
        "candidate_validation_sha256": __import__("hashlib").sha256(validation_source.read_bytes()).hexdigest() if validation_source.is_file() else None,
        "validation_sha256": validation_hash,
        "note": "Admission repeats are explicit; source-group and unique-reference counts are not inflated.",
    }
    output.with_name("balanced-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def metrics_value(row: dict) -> int:
    value = row.get("assistant_tokens")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid assistant token audit: {row}")
    return value


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    raw = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--real-fraction", type=float, default=0.40)
    p.add_argument("--prior-fraction", type=float, default=0.0,
                   help="deprecated alias folded into foundation-fraction")
    p.add_argument("--foundation-fraction", type=float, default=0.50)
    p.add_argument("--retention-fraction", type=float, default=0.10)
    args = p.parse_args()
    receipt = build(args.candidates.resolve(), args.audit.resolve(), args.output.resolve(), args.steps, args.batch_size,
                    {"real": args.real_fraction,
                     "foundation": args.foundation_fraction + args.prior_fraction,
                     "retention": args.retention_fraction})
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
