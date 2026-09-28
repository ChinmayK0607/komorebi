"""Finite, resumable MiMo judge audit against the user's category labels.

This calls Vercel for judgments only; all images are checked against their
frozen manifest hashes before any paid request. Credentials never enter output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from gateway_judge import PROMPT, SUBJECT_FIRST_PROMPT, digest, judge_view, load_key


PROMPTS = {"default": PROMPT, "subject_first": SUBJECT_FIRST_PROMPT}
SIDES = {"ab": ("A", "B"), "ba": ("B", "A")}


def cases_for_audit(manifest: dict, labels: dict, images: Path) -> list[dict]:
    if labels.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("human export and image manifest differ")
    by_id = {case["id"]: case for case in manifest["cases"]}
    chosen = []
    for label in labels["reviewed_preferences"]:
        case = by_id[label["case_id"]]
        if label["verdict"] not in {"A", "B", "tie"}:
            continue
        paths = {}
        for side in ("reference", "A", "B"):
            image = case["reference"] if side == "reference" else case["candidates"][side]
            path = images / Path(image["path"]).name
            if not path.is_file() or digest(path) != image["sha256"]:
                raise ValueError(f"missing or changed image for {case['id']} {side}")
            paths[side] = path
        for side in ("A", "B"):
            if label["candidate_sha256"][side] != case["candidates"][side]["sha256"]:
                raise ValueError(f"human export candidate mismatch: {case['id']} {side}")
        if label["reference_sha256"] != case["reference"]["sha256"]:
            raise ValueError(f"human export reference mismatch: {case['id']}")
        chosen.append({"id": case["id"], "category": label["category"],
                       "human": label["verdict"], "paths": paths})
    if len({case["id"] for case in chosen}) != len(chosen):
        raise ValueError("duplicate labeled case")
    return chosen


def comparable_winner(view: dict, orientation: str) -> str:
    winner = view["label"]["winner"]
    if winner in {"tie", "uncertain"}:
        return winner
    return SIDES[orientation][0 if winner == "A" else 1]


def summarize(cases: list[dict], receipts: Path) -> dict:
    rows = []
    for case in cases:
        for prompt_name in PROMPTS:
            views = {}
            for orientation in SIDES:
                path = receipts / f"{case['id']}-{prompt_name}-{orientation}.json"
                views[orientation] = json.loads(path.read_text()) if path.exists() else None
            winners = [comparable_winner(v, o) for o, v in views.items() if v and "label" in v]
            if len(winners) != 2 or "uncertain" in winners:
                predicted = "uncertain"
            elif winners[0] == winners[1]:
                predicted = winners[0]
            else:
                predicted = "split"
            rows.append({"id": case["id"], "category": case["category"],
                         "human": case["human"], "prompt": prompt_name,
                         "orientation_winners": winners, "predicted": predicted,
                         "exact_agreement": predicted == case["human"]})
    by_prompt = {}
    for name in PROMPTS:
        subset = [row for row in rows if row["prompt"] == name]
        by_category = {}
        for category in sorted({row["category"] for row in subset}):
            relevant = [row for row in subset if row["category"] == category]
            by_category[category] = {"n": len(relevant),
                                      "exact_agreement": sum(x["exact_agreement"] for x in relevant)}
        by_prompt[name] = {"n": len(subset),
                           "exact_agreement": sum(row["exact_agreement"] for row in subset),
                           "orientation_split": sum(row["predicted"] == "split" for row in subset),
                           "uncertain": sum(row["predicted"] == "uncertain" for row in subset),
                           "by_category": by_category}
    return {"schema": "human-category-judge-calibration-v1", "by_prompt": by_prompt,
            "rows": rows, "scope": "19 labeled category preferences from golden-categories-v1; not beauty gold"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels.read_text())
    cases = cases_for_audit(manifest, labels, args.images)
    jobs = [(case, prompt, orientation) for case in cases
            for prompt in PROMPTS for orientation in SIDES]
    if args.dry_run:
        print(json.dumps({"labeled_cases": len(cases), "orientation_calls": len(jobs),
                          "categories": sorted({case["category"] for case in cases})}))
        return
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers must be between 1 and 12")
    key = load_key(args.key_file)
    receipts = args.outdir / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)

    def run(job: tuple[dict, str, str]) -> tuple[str, str | None]:
        case, prompt_name, orientation = job
        path = receipts / f"{case['id']}-{prompt_name}-{orientation}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            left, right = SIDES[orientation]
            expected_hashes = {"reference": digest(case["paths"]["reference"]),
                               "A": digest(case["paths"][left]),
                               "B": digest(case["paths"][right])}
            if (saved.get("prompt_sha256") == hashlib.sha256(PROMPTS[prompt_name].encode()).hexdigest()
                    and saved.get("image_sha256") == expected_hashes):
                return path.name, None
            raise ValueError(f"existing receipt has different prompt or images: {path.name}")
        left, right = SIDES[orientation]
        for attempt in range(2):
            try:
                result = judge_view(key=key, reference=case["paths"]["reference"],
                                    a=case["paths"][left], b=case["paths"][right],
                                    prompt=PROMPTS[prompt_name])
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                tmp.replace(path)
                return path.name, None
            except (TimeoutError, ValueError, RuntimeError) as exc:
                if attempt:
                    return path.name, type(exc).__name__
                time.sleep(2)
        raise AssertionError("unreachable")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            name, error = future.result()
            if error:
                failures.append({"receipt": name, "error_type": error})
            print(f"{index}/{len(jobs)} {name} {'failed:'+error if error else 'saved'}", flush=True)
    summary = summarize(cases, receipts)
    summary["manifest_sha256"] = manifest["manifest_sha256"]
    summary["failures"] = failures
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["by_prompt"], sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
