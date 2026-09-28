"""Score fixed matched paintings with durable per-view Gateway receipts.

Each completed orientation is written before the next paid call. Re-running
this command verifies the input image and prompt hashes, then skips those
completed calls. A timeout leaves the remaining views pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality-oss-rl-20260927"))
from gateway_judge import MODEL, PROMPT, digest, judge_view, load_key  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checked_image(gallery: Path, name: str | None, expected: str | None) -> Path | None:
    if name is None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("unsafe gallery image name")
    path = gallery / name
    if not path.is_file() or path.is_symlink() or digest(path) != expected:
        raise ValueError(f"missing or changed gallery image: {name}")
    return path


def checked_view(path: Path, *, reference: Path, a: Path, b: Path) -> dict:
    view = json.loads(path.read_text())
    hashes = {"reference": digest(reference), "A": digest(a), "B": digest(b)}
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    if (view.get("image_sha256") != hashes or view.get("prompt_sha256") != prompt_sha
            or view.get("requested_model") != MODEL):
        raise ValueError(f"cached judge view does not match fixed inputs: {path.name}")
    if view.get("label", {}).get("winner") not in ("A", "B", "tie", "uncertain"):
        raise ValueError(f"cached judge label is invalid: {path.name}")
    return view


def score_row(row: dict, *, gallery: Path, state: Path, key: str,
              timeout: float) -> dict:
    case = row["id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case):
        raise ValueError("unsafe case ID")
    reference = next(gallery.glob(f"{case}-reference.*"), None)
    if reference is None or reference.is_symlink() or digest(reference) != row["reference_sha256"]:
        raise ValueError(f"missing or changed reference: {case}")
    sft_data = row["sft-step512"]
    rl_data = row["rl-step-4"]
    a = checked_image(gallery, sft_data["final"], sft_data["final_sha256"])
    b = checked_image(gallery, rl_data["final"], rl_data["final_sha256"])
    result = {"id": case, "split": row["split"]}
    if a is None and b is None:
        return {**result, "status": "both_invalid", "score_sft": None, "score_rl": None}
    if a is None:
        return {**result, "status": "sft_invalid", "score_sft": 0.0, "score_rl": 1.0}
    if b is None:
        return {**result, "status": "rl_invalid", "score_sft": 1.0, "score_rl": 0.0}

    views = []
    for index, (left, right) in enumerate(((a, b), (b, a)), start=1):
        receipt = state / case / f"view-{index}.json"
        if receipt.is_file():
            view = checked_view(receipt, reference=reference, a=left, b=right)
        else:
            view = judge_view(key=key, reference=reference, a=left, b=right,
                              model=MODEL, prompt=PROMPT, timeout=timeout)
            checked_view_data = {"image_sha256": {"reference": digest(reference),
                                                  "A": digest(left), "B": digest(right)},
                                 "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()}
            if any(view.get(k) != v for k, v in checked_view_data.items()):
                raise ValueError("judge response hash mismatch")
            view["requested_model"] = MODEL
            write_json(receipt, view)
            print(json.dumps({"case": case, "view": index, "status": "saved",
                              "elapsed_seconds": view["elapsed_seconds"],
                              "cost": (view.get("usage") or {}).get("cost")}), flush=True)
        views.append(view)
    winners = [view["label"]["winner"] for view in views]
    if "uncertain" in winners:
        score = None
    else:
        score = ({"A": 1.0, "B": 0.0, "tie": 0.5}[winners[0]] +
                 {"A": 0.0, "B": 1.0, "tie": 0.5}[winners[1]]) / 2
    pair = {"score_A": score, "status": "scored" if score is not None else "uncertain",
            "views": views, "image_sha256": {"reference": digest(reference),
                                            "A": digest(a), "B": digest(b)}}
    return {**result, "status": pair["status"], "score_sft": score,
            "score_rl": None if score is None else 1 - score, "judge": pair}


def summarize(rows: list[dict]) -> dict:
    by_split = {}
    for split in ("train", "heldout"):
        subset = [row for row in rows if row["split"] == split]
        by_split[split] = {
            "n": len(subset),
            "sft_wins": sum(row["score_sft"] == 1 for row in subset),
            "rl_wins": sum(row["score_rl"] == 1 for row in subset),
            "ties": sum(row["score_sft"] == .5 for row in subset),
            "censored": sum(row["score_sft"] is None for row in subset),
            "sft_invalid": sum(row["status"] == "sft_invalid" for row in subset),
            "rl_invalid": sum(row["status"] == "rl_invalid" for row in subset),
        }
    return {"schema": "matched-mixed-gateway-judge-v1", "step": 4,
            "sft_wins": sum(row["score_sft"] == 1 for row in rows),
            "rl_wins": sum(row["score_rl"] == 1 for row in rows),
            "ties": sum(row["score_sft"] == .5 for row in rows),
            "censored": sum(row["score_sft"] is None for row in rows),
            "by_split": by_split, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=360)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 30 <= args.timeout <= 900:
        raise ValueError("invalid worker count or timeout")
    source = json.loads((args.gallery / "report.json").read_text())
    rows = source["rows"]
    if source.get("step") != 4 or len(rows) != 10 or {r["split"] for r in rows} != {"heldout", "train"}:
        raise ValueError("expected ten matched step-four rows")
    key = load_key(args.key_file)
    results: dict[str, dict] = {}
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_row, row, gallery=args.gallery,
                               state=args.state, key=key, timeout=args.timeout): row["id"]
                   for row in rows}
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case] = future.result()
            except Exception as exc:
                errors.append({"case": case, "error_type": type(exc).__name__})
    if errors:
        print(json.dumps({"pending": errors, "completed": len(results)}), flush=True)
        raise SystemExit(1)
    summary = summarize([results[row["id"]] for row in rows])
    write_json(args.output, summary)
    print(json.dumps({"by_split": summary["by_split"],
                      "sft_wins": summary["sft_wins"],
                      "rl_wins": summary["rl_wins"], "ties": summary["ties"],
                      "censored": summary["censored"]}), flush=True)


if __name__ == "__main__":
    main()
