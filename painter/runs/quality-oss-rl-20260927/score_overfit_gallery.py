"""Score matched final paintings with the validated two-orientation Gateway judge."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gateway_judge import digest, judge_pair, load_key


def score_row(row: dict, *, gallery: Path, step: int, key: str) -> dict:
    """A valid canvas beats a missing canvas without charging for an API call."""

    case = row["id"]
    reference = next(gallery.glob(f"{case}-reference.*"), None)
    if reference is None or digest(reference) != row["reference_sha256"]:
        raise ValueError(f"missing or changed reference for {case}")
    names = {"sft": row["sft-step512"]["final"],
             "rl": row[f"rl-step-{step}"]["final"]}
    for policy, name in names.items():
        if name is None:
            continue
        image = gallery / name
        expected = row["sft-step512" if policy == "sft" else f"rl-step-{step}"]["final_sha256"]
        if not image.is_file() or digest(image) != expected:
            raise ValueError(f"missing or changed {policy} canvas for {case}")
    a, b = names["sft"], names["rl"]
    if not a and not b:
        return {"id": case, "status": "both_invalid", "score_sft": None,
                "score_rl": None}
    if not a:
        return {"id": case, "status": "sft_invalid", "score_sft": 0.0,
                "score_rl": 1.0}
    if not b:
        return {"id": case, "status": "rl_invalid", "score_sft": 1.0,
                "score_rl": 0.0}
    result = judge_pair(key=key, reference=reference,
                        a=gallery / a, b=gallery / b)
    return {"id": case, "status": result["status"],
            "score_sft": result["score_A"], "score_rl":
            None if result["score_A"] is None else 1-result["score_A"],
            "judge": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gallery = json.loads((args.gallery / "report.json").read_text())
    if len(gallery["rows"]) != 4:
        raise ValueError("expected four matched references")
    step = gallery["step"]
    key = load_key(args.key_file)

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: score_row(row, gallery=args.gallery,
                                                   step=step, key=key), gallery["rows"]))
    summary = {"schema": "matched-overfit-gateway-judge-v2", "step": step,
               "sft_wins": sum(row.get("score_sft") == 1 for row in rows),
               "rl_wins": sum(row.get("score_rl") == 1 for row in rows),
               "ties": sum(row.get("score_sft") == .5 for row in rows),
               "censored": sum(row.get("score_sft") is None for row in rows),
               "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("step", "sft_wins", "rl_wins", "ties", "censored")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
