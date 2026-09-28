"""Finite subject-fidelity calibration on an existing matched painting gallery.

The default judge labels are read from the prior result; only the alternative
prompt makes new API calls. Each result is checkpointed individually so an
interrupted comparison resumes without paying for finished cases again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gateway_judge import MODEL, SUBJECT_FIRST_PROMPT, digest, judge_pair, load_key


def calibrate(*, gallery: Path, existing: Path, outdir: Path, key_file: Path,
              limit: int | None = None) -> dict:
    report = json.loads((gallery / "report.json").read_text())
    baseline = json.loads(existing.read_text())
    old = {row["id"]: row for row in baseline["rows"]}
    outdir.mkdir(parents=True, exist_ok=True)
    key = load_key(key_file)
    records = []
    for row in report["rows"][:limit]:
        case = row["id"]
        reference = next(gallery.glob(f"{case}-reference.*"))
        a_name = row["sft-step512"]["final"]
        b_name = row[f"rl-step-{report['step']}"]["final"]
        if not a_name or not b_name:
            records.append({"id": case, "status": "missing_canvas"})
            continue
        a, b = gallery / a_name, gallery / b_name
        hashes = {"reference": digest(reference), "A": digest(a), "B": digest(b)}
        prior = old[case]
        if prior.get("judge", {}).get("image_sha256") != hashes:
            raise ValueError(f"baseline judge image identity differs for {case}")
        path = outdir / f"{case}.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("image_sha256") != hashes or saved.get("prompt") != "subject-first-v1":
                raise ValueError(f"saved calibration identity differs for {case}")
        else:
            result = judge_pair(key=key, reference=reference, a=a, b=b,
                                model=MODEL, prompt=SUBJECT_FIRST_PROMPT)
            saved = {"id": case, "prompt": "subject-first-v1", "image_sha256": hashes,
                     "baseline_score_sft": prior.get("score_sft"),
                     "subject_score_sft": result["score_A"], "judge": result}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
            tmp.replace(path)
        records.append(saved)
    summary = {"schema": "painter-subject-judge-calibration-v1",
               "model": MODEL, "cases": len(records), "rows": records}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = calibrate(**vars(args))
    print(json.dumps({"cases": summary["cases"], "outdir": str(args.outdir)}))


if __name__ == "__main__":
    main()
