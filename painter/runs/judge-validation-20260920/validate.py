"""Validate three blinded Luna shards and preserve uncertainty as masks."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import tempfile
from pathlib import Path

JUDGES = ("judge-01", "judge-02", "judge-03")
JUDGMENT_KEYS = {"id", "winner", "confidence", "acceptable_A", "acceptable_B",
                 "reason", "observations", "reviewed_images"}
PNG = b"\x89PNG\r\n\x1a\n"


def _write(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_path(root: Path, judge: str) -> Path | None:
    found = [p for p in (root / "labels" / f"{judge}.json",
                         root / "labels" / f"{judge}-labels.json",
                         root / f"{judge}-labels.json") if p.exists()]
    if len(found) > 1:
        raise ValueError(f"multiple label files for {judge}: {found}")
    return found[0] if found else None


def _flip(side: str) -> str:
    return {"A": "B", "B": "A"}[side]


def _canonical(label: dict, orientation: str) -> tuple[str, dict[str, str]]:
    winner = label["winner"]
    if winner in {"A", "B"} and orientation == "ba":
        winner = _flip(winner)
    acceptable = {"A": label["acceptable_A"], "B": label["acceptable_B"]}
    if orientation == "ba":
        acceptable = {"A": acceptable["B"], "B": acceptable["A"]}
    return winner, acceptable


def _parse_label_file(path: Path, ids: set[str]) -> tuple[dict, list[str]]:
    errors = []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"{path}: invalid JSON ({exc})"]
    if not isinstance(doc, dict):
        return {}, [f"{path}: top level must be an object"]
    if set(doc) != {"judge_model", "reasoning_effort", "judgments"}:
        errors.append(f"{path}: top-level schema mismatch")
    if doc.get("judge_model") != "gpt-5.6-luna":
        errors.append(f"{path}: judge_model must be gpt-5.6-luna")
    if doc.get("reasoning_effort") != "low":
        errors.append(f"{path}: reasoning_effort must be low")
    judgments = doc.get("judgments")
    if not isinstance(judgments, list):
        return doc, errors + [f"{path}: judgments must be a list"]
    seen = set()
    for row in judgments:
        if not isinstance(row, dict) or set(row) != JUDGMENT_KEYS:
            errors.append(f"{path}: judgment schema mismatch")
            continue
        ident = row.get("id")
        if ident not in ids:
            errors.append(f"{path}: unexpected judgment id {ident!r}")
        if ident in seen:
            errors.append(f"{path}: duplicate judgment id {ident!r}")
        seen.add(ident)
        if row.get("winner") not in {"A", "B", "tie", "uncertain"}:
            errors.append(f"{path}: invalid winner for {ident}")
        if row.get("confidence") not in {"high", "medium", "low"}:
            errors.append(f"{path}: invalid confidence for {ident}")
        for side in ("A", "B"):
            if row.get(f"acceptable_{side}") not in {"yes", "no", "uncertain"}:
                errors.append(f"{path}: acceptable_{side} must be yes/no/uncertain for {ident}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            errors.append(f"{path}: empty reason for {ident}")
        observations = row.get("observations")
        if not isinstance(observations, dict) or set(observations) != {"reference", "A", "B"}:
            errors.append(f"{path}: observations schema mismatch for {ident}")
        elif any(not isinstance(observations[x], str) or not observations[x].strip()
                 for x in ("reference", "A", "B")):
            errors.append(f"{path}: every observation must be non-empty for {ident}")
        if not isinstance(row.get("reviewed_images"), list):
            errors.append(f"{path}: reviewed_images must be a list for {ident}")
    return doc, errors


def validate(root: str | Path, write: bool = True) -> dict:
    root = Path(root).resolve()
    errors, missing, bound_paths = [], [], set()
    try:
        private = json.loads((root / "artifacts" / "private.json").read_text(encoding="utf-8"))
    except Exception as exc:
        result = {"status": "fail", "overall_pass": False, "errors": [f"private key: {exc}"]}
        if write: _write(root / "validation.json", result)
        return result
    pairs = private.get("pairs", {})
    controls = private.get("controls", {})
    if not isinstance(pairs, dict) or not isinstance(controls, dict):
        errors.append("private key pairs/controls must be objects")
        pairs, controls = {}, {}
    # Public view IDs are intentionally unrelated to the selection IDs.
    view_info, expected_by_judge = {}, collections.defaultdict(set)
    for pair_id, private_pair in pairs.items():
        for public_id, view in private_pair.get("views", {}).items():
            if public_id in view_info:
                errors.append(f"duplicate private view id {public_id}")
            view_info[public_id] = {"pair_id": pair_id, "pair": private_pair,
                                    "view": view}
            if view.get("judge") in JUDGES:
                expected_by_judge[view["judge"]].add(public_id)
            else:
                errors.append(f"{public_id}: invalid assigned judge")
    manifests, manifest_ids = {}, set()
    manifest_map = {}
    for judge in JUDGES:
        path = root / f"{judge}.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {exc}"); doc = []
        if not isinstance(doc, list):
            errors.append(f"{path}: manifest must be a list"); doc = []
        local, seen = {}, set()
        for item in doc:
            if not isinstance(item, dict) or set(item) != {"id", "reference", "A", "B"}:
                errors.append(f"{path}: manifest item schema mismatch"); continue
            ident = item["id"]
            if ident in seen: errors.append(f"{path}: duplicate id {ident}")
            if ident in manifest_ids: errors.append(f"{path}: view appears in multiple shards {ident}")
            seen.add(ident); local[ident] = item; manifest_ids.add(ident)
            for side in ("reference", "A", "B"):
                image = Path(item[side]) if isinstance(item[side], str) else Path("<invalid>")
                if not image.is_absolute() or image.suffix.lower() != ".png" or not image.is_file():
                    errors.append(f"{path}: invalid image path for {ident}/{side}"); continue
                if image.read_bytes()[:8] != PNG: errors.append(f"{path}: non-PNG image {image}")
                if str(image) in bound_paths: errors.append(f"{path}: duplicate image path {image}")
                bound_paths.add(str(image))
            if len({item.get("reference"), item.get("A"), item.get("B")}) != 3:
                errors.append(f"{path}: duplicate image within {ident}")
        manifests[judge], manifest_map[judge] = doc, local
        if not local: missing.append(judge + ":manifest")
    expected_ids = set(view_info)
    for judge in JUDGES:
        if set(manifest_map[judge]) != expected_by_judge[judge]:
            errors.append(f"{judge}: manifest ids do not match private key")
    if len(manifest_ids) != len(view_info):
        errors.append("manifests do not bind every private view exactly once")
    for public_id, info in view_info.items():
        private_pair, view = info["pair"], info["view"]
        judge, item = view.get("judge"), manifest_map.get(view.get("judge"), {}).get(public_id)
        files, hashes = view.get("files", {}), view.get("hashes", {})
        orientation = view.get("orientation")
        if orientation not in {"ab", "ba"} or set(files) != {"reference", "A", "B"}:
            errors.append(f"{public_id}: private view schema mismatch"); continue
        if not item:
            continue
        for side in ("reference", "A", "B"):
            if item[side] != files.get(side): errors.append(f"{public_id}/{side}: path not bound")
            image = Path(item[side])
            if image.is_file() and hashes.get(side) != _sha(image): errors.append(f"{public_id}/{side}: hash mismatch")
            canonical = side if side == "reference" or orientation == "ab" else _flip(side)
            if image.is_file() and private_pair.get("source_hashes", {}).get(canonical) != _sha(image):
                errors.append(f"{public_id}/{side}: source hash mismatch")
    labels, records = {}, collections.defaultdict(list)
    for judge in JUDGES:
        try:
            path = _label_path(root, judge)
        except ValueError as exc:
            errors.append(str(exc)); path = None
        if not path:
            missing.append(judge + ":labels"); continue
        doc, parse_errors = _parse_label_file(path, expected_by_judge[judge])
        errors.extend(parse_errors)
        rows = doc.get("judgments", []) if isinstance(doc, dict) else []
        labels[judge] = {r["id"]: r for r in rows if isinstance(r, dict)
                         and set(r) == JUDGMENT_KEYS and r.get("id") in manifest_map[judge]
                         and r.get("winner") in {"A", "B", "tie", "uncertain"}
                         and r.get("confidence") in {"high", "medium", "low"}
                         and r.get("acceptable_A") in {"yes", "no", "uncertain"}
                         and r.get("acceptable_B") in {"yes", "no", "uncertain"}
                         and isinstance(r.get("observations"), dict)
                         and set(r["observations"]) == {"reference", "A", "B"}}
        for ident in expected_by_judge[judge]:
            if ident not in labels[judge]:
                missing.append(f"{judge}:{ident}")
            else:
                row = labels[judge][ident]
                expected_images = [manifest_map[judge][ident][x] for x in ("reference", "A", "B")]
                if row.get("reviewed_images") != expected_images:
                    errors.append(f"{judge}:{ident}: reviewed_images do not match manifest")
                info = view_info[ident]
                records[info["pair_id"]].append({"judge": judge, "id": ident,
                    "label": row, "orientation": info["view"]["orientation"]})

    masks = collections.Counter()
    for judge in JUDGES:
        masks["missing_label"] += len(expected_by_judge[judge] - set(labels.get(judge, {})))
    orientation = {"compared": 0, "agree": 0, "disagree": 0, "masked": 0}
    acceptability = {"compared": 0, "agree": 0, "disagree": 0, "masked": 0}
    by_kind = {kind: {"orientation": {"compared": 0, "agree": 0, "disagree": 0, "masked": 0},
                      "acceptability": {"compared": 0, "agree": 0, "disagree": 0, "masked": 0}}
               for kind in ("controls", "real_pairs")}
    control_checks, real_votes = [], collections.Counter()
    real_pair_details = []
    for ident in pairs:
        rows = records.get(ident, [])
        kind = "controls" if ident in controls else "real_pairs"
        kind_orientation, kind_acceptability = by_kind[kind]["orientation"], by_kind[kind]["acceptability"]
        by_orientation = collections.defaultdict(list)
        for record in rows:
            label, ori = record["label"], record["orientation"]
            if label.get("winner") == "uncertain": masks["uncertain_winner"] += 1
            if label.get("confidence") == "low": masks["low_confidence"] += 1
            if label.get("acceptable_A") == "uncertain" or label.get("acceptable_B") == "uncertain":
                masks["uncertain_acceptability"] += 1
            by_orientation[ori].append(record)
        for left in by_orientation["ab"]:
            for right in by_orientation["ba"]:
                l, r = left["label"], right["label"]
                if l.get("confidence") == "low" or r.get("confidence") == "low" or \
                   l.get("winner") == "uncertain" or r.get("winner") == "uncertain":
                    orientation["masked"] += 1; kind_orientation["masked"] += 1
                else:
                    orientation["compared"] += 1; kind_orientation["compared"] += 1
                    if _canonical(l, "ab")[0] == _canonical(r, "ba")[0]:
                        orientation["agree"] += 1; kind_orientation["agree"] += 1
                    else:
                        orientation["disagree"] += 1; kind_orientation["disagree"] += 1
                if l.get("confidence") == "low" or r.get("confidence") == "low" or \
                   l.get("acceptable_A") == "uncertain" or l.get("acceptable_B") == "uncertain" or \
                   r.get("acceptable_A") == "uncertain" or r.get("acceptable_B") == "uncertain":
                    acceptability["masked"] += 1; kind_acceptability["masked"] += 1
                else:
                    acceptability["compared"] += 1; kind_acceptability["compared"] += 1
                    if _canonical(l, "ab")[1] == _canonical(r, "ba")[1]:
                        acceptability["agree"] += 1; kind_acceptability["agree"] += 1
                    else:
                        acceptability["disagree"] += 1; kind_acceptability["disagree"] += 1
        if ident in controls:
            expected = controls[ident].get("expected_winner")
            checks = []
            for record in rows:
                label = record["label"]
                winner = _canonical(label, record["orientation"])[0] if label.get("winner") != "uncertain" else None
                checks.append(label.get("confidence") != "low" and winner == expected)
            control_checks.append(all(checks) and len(checks) == 2)
        else:
            votes = [_canonical(r["label"], r["orientation"])[0] for r in rows
                     if r["label"].get("winner") in {"A", "B", "tie"} and r["label"].get("confidence") != "low"]
            description = votes[0] if votes and len(set(votes)) == 1 else "disagree" if votes else "masked"
            real_votes[description] += 1
            real_pair_details.append({"id": ident, "description": description, "votes": votes})
    all_labels = not missing and all(len(labels.get(j, {})) == len(expected_by_judge[j]) for j in JUDGES)
    controls_correct = all(control_checks) and len(control_checks) == len(controls)
    files_bound = not any("manifest" in e or "path" in e or "hash" in e or "image" in e for e in errors)
    status = "pass" if all_labels and not errors and controls_correct else "incomplete" if missing else "fail"
    result = {"status": status, "overall_pass": status == "pass", "all_shards_complete": all_labels,
              "all_controls_correct": controls_correct, "all_files_bound": files_bound,
              "errors": errors, "missing": missing, "masked_reasons": dict(masks),
              "orientation_agreement": orientation, "acceptability_agreement": acceptability,
              "agreement_by_kind": by_kind,
              "real_pair_vote_descriptions": dict(real_votes), "real_pair_details": real_pair_details,
              "latency": latency_summary(root), "pairs": len(pairs), "views": len(expected_ids),
              "controls": len(controls)}
    if write: _write(root / "validation.json", result)
    return result

def latency_summary(root: Path) -> dict | None:
    values, invalid = [], 0
    for path in sorted((root / "labels").glob("*.latency.json")) if (root / "labels").is_dir() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            raw = value.values() if isinstance(value, dict) else value
            for item in raw:
                number = float(item)
                if number < 0 or not math.isfinite(number): invalid += 1
                else: values.append(number)
        except (ValueError, TypeError, json.JSONDecodeError):
            invalid += 1
    return {"count": len(values), "invalid": invalid,
            "mean_ms": sum(values) / len(values) if values else None} if values or invalid else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))
