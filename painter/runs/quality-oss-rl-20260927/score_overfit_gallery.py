"""Score matched final paintings with the validated two-orientation Gateway judge."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gateway_judge import judge_pair, load_key


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

    def score(row: dict) -> dict:
        case = row["id"]
        a = row["sft-step512"]["final"]
        b = row[f"rl-step-{step}"]["final"]
        if not a or not b:
            return {"id": case, "status": "censored_missing_canvas", "score_sft": None}
        reference = next(args.gallery.glob(f"{case}-reference.*"), None)
        if reference is None:
            return {"id": case, "status": "censored_missing_reference", "score_sft": None}
        result = judge_pair(key=key, reference=reference,
                            a=args.gallery / a, b=args.gallery / b)
        return {"id": case, "status": result["status"],
                "score_sft": result["score_A"], "score_rl":
                None if result["score_A"] is None else 1-result["score_A"],
                "judge": result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(score, gallery["rows"]))
    summary = {"schema": "matched-overfit-gateway-judge-v1", "step": step,
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
