"""Prepare and run a frozen-image v1/v2 grounded-judge calibration.

Preparation reads only the existing gallery HTML.  Running sends the frozen
brief and image paths to the existing local Gateway judge, records each
two-order receipt, and produces a reviewable HTML report.  Provider failures
are censored; they never become a numerical zero.

The output directory is intentionally caller supplied.  It is normally an
ignored ``local-evidence/judge-alignment-20260930`` directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
import statistics
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote

try:
    from painter import text_pairwise_grounded_v2_judge as judge
    from painter import text_pairwise_grounded_judge as old_judge
    from painter.text_pairwise_margin_judge import MODEL
except ImportError:  # Running the file directly from the painter checkout.
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from painter import text_pairwise_grounded_v2_judge as judge
    from painter import text_pairwise_grounded_judge as old_judge
    from painter.text_pairwise_margin_judge import MODEL


SCHEMA = "judge-alignment-input-v1"
RECEIPT_SCHEMA = "judge-alignment-receipt-v1"
SUMMARY_SCHEMA = "judge-alignment-summary-v1"
PANEL_IDS = (
    "t600-370", "t600-375", "t600-413", "t600-466", "t600-503",
    "t600-516", "t600-542", "t600-560", "t600-591",
)
ROLE_LABELS = ("frozen SFT", "SFT", "RL-step-16", "pilot-step-24")
ROLE_TOKEN = {
    "frozen SFT": "frozen-sft",
    "SFT": "sft",
    "RL-step-16": "rl-step-16",
    "pilot-step-24": "pilot-step-24",
}
FAMILY_ORDER = ("simple", "intermediate", "complex")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# These notes are deliberately metadata outside the model request.  Only the
# three explicit preferences contribute to agreement counts.  The qualitative
# notes remain visible in the report without becoming forced labels.
HUMAN_FEEDBACK: dict[str, dict[str, Any]] = {
    "t600-370": {"note": "step-24 was judged undeservedly high; inspect the faint/wireframe subject."},
    "t600-375": {"note": "step-24 looked good in qualitative review."},
    "t600-413": {
        "note": "step-16 should receive higher reward than step-24.",
        "explicit_preferences": [{"preferred_role": "RL-step-16", "other_role": "pilot-step-24"}],
    },
    "t600-466": {"note": "step-24 looked good in qualitative review."},
    "t600-503": {"note": "step-24 has a faint figure; retain this as qualitative context only."},
    "t600-516": {
        "note": "Only the frozen anchor looked good in qualitative review.",
        "explicit_preferences": [
            {"preferred_role": "frozen SFT", "other_role": "SFT"},
            {"preferred_role": "frozen SFT", "other_role": "RL-step-16"},
            {"preferred_role": "frozen SFT", "other_role": "pilot-step-24"},
        ],
    },
    "t600-542": {"note": "step-24 looked good; it has extra needles. Qualitative context only."},
    "t600-560": {"note": "step-24's subject was absent in qualitative review."},
    "t600-591": {
        "note": "step-16 should receive higher reward than step-24.",
        "explicit_preferences": [{"preferred_role": "RL-step-16", "other_role": "pilot-step-24"}],
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _safe_id(value: str) -> None:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe case or receipt identifier: {value!r}")


class _GalleryParser(HTMLParser):
    """Parse the deliberately simple gallery without third-party HTML libs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cases: list[dict[str, Any]] = []
        self.case: dict[str, Any] | None = None
        self.cell: dict[str, Any] | None = None
        self.mode: str | None = None
        self.text: list[str] = []
        self.strong_text: list[str] = []
        self.cell_depth = 0

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        return set((dict(attrs).get("class") or "").split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        classes = self._classes(attrs)
        if tag == "section":
            if self.case is not None:
                raise ValueError("nested gallery section")
            self.case = {"id": None, "family": None, "brief": None, "images": {}}
        elif tag == "h2" and self.case is not None:
            self.mode = "h2"
            self.text = []
        elif tag == "p" and self.case is not None:
            self.mode = "brief" if "brief" in classes else "cell_text"
            self.text = []
        elif tag == "strong" and self.cell is not None:
            self.mode = "strong"
            self.strong_text = []
        elif tag == "div" and "cell" in classes and self.case is not None:
            if self.cell is not None:
                raise ValueError("nested gallery cell")
            self.cell = {"label": None, "src": None, "status": None, "old_score": None}
            self.cell_depth = 1
        elif tag == "div" and self.cell is not None:
            self.cell_depth += 1
        elif tag == "img" and self.cell is not None:
            self.cell["src"] = attrs_map.get("src")

    def handle_data(self, data: str) -> None:
        if self.mode in {"h2", "brief", "cell_text"}:
            self.text.append(data)
        elif self.mode == "strong":
            self.strong_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "strong" and self.cell is not None and self.mode == "strong":
            self.cell["label"] = "".join(self.strong_text).strip()
            self.mode = "cell_text"
        elif tag == "h2" and self.case is not None and self.mode == "h2":
            title = "".join(self.text).strip()
            ident, separator, family = title.partition("·")
            self.case["id"] = ident.strip()
            self.case["family"] = family.strip() if separator else None
            self.mode = None
        elif tag == "p" and self.case is not None:
            if self.mode == "brief":
                self.case["brief"] = "".join(self.text).strip()
            elif self.mode == "cell_text" and self.cell is not None:
                value = " ".join("".join(self.text).split())
                score = re.search(r"\bscore\s+(-?(?:\d+(?:\.\d*)?|\.\d+))\b", value)
                if score:
                    self.cell["old_score"] = float(score.group(1))
                if "invalid" in value.lower():
                    self.cell["status"] = "invalid"
                elif "valid" in value.lower():
                    self.cell["status"] = "valid"
                elif "fixed anchor" in value.lower():
                    self.cell["status"] = "fixed_anchor"
            self.mode = None
        elif tag == "div" and self.cell is not None:
            self.cell_depth -= 1
            if self.cell_depth == 0:
                # A missing cell contains a nested ``div.missing`` and has no
                # image.  Drop it cleanly; a later valid cell must still parse.
                if self.cell.get("src") is not None:
                    label = self.cell.get("label")
                    if not label:
                        label = self._label_from_alt(self.cell.get("src"))
                    self.cell["label"] = label
                    if self.case is not None:
                        self.case["images"][label] = {
                            "src": self.cell["src"],
                            "status": self.cell.get("status"),
                            "old_score": self.cell.get("old_score"),
                        }
                self.cell = None
        elif tag == "section" and self.case is not None:
            self._finish_case()

    @staticmethod
    def _label_from_alt(src: str | None) -> str:
        name = Path(unquote(src or "")).name
        stem = name.removesuffix(Path(name).suffix)
        for ident in ("RL-step-16", "pilot-step-24", "frozen SFT", "SFT"):
            marker = "-" + ident
            if marker in stem:
                return ident
        return stem.rsplit("-", 1)[-1]

    def _finish_case(self) -> None:
        case = self.case
        self.case = None
        self.mode = None
        self.text = []
        if not case or not case.get("id"):
            raise ValueError("gallery section lacks a case ID")
        ident = str(case["id"])
        _safe_id(ident)
        if not case.get("family") or not case.get("brief"):
            raise ValueError(f"gallery case lacks family or brief: {ident}")
        self.cases.append(case)


def _resolve_image(gallery: Path, src: str) -> tuple[str, Path, str | None]:
    relative = Path(unquote(src.replace("\\", "/")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"gallery image escapes gallery directory: {src!r}")
    path = (gallery.parent / relative).resolve()
    root = gallery.parent.resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"gallery image escapes gallery directory: {src!r}")
    if path.is_symlink() or not path.is_file():
        return relative.as_posix(), path, None
    return relative.as_posix(), path, sha256_file(path)


def parse_gallery(gallery: Path) -> list[dict[str, Any]]:
    """Freeze case text, image hashes, paths, and historical gallery scores."""

    if gallery.is_symlink() or not gallery.is_file():
        raise FileNotFoundError(gallery)
    parser = _GalleryParser()
    parser.feed(gallery.read_text(encoding="utf-8"))
    parser.close()
    seen: set[str] = set()
    frozen: list[dict[str, Any]] = []
    for case in parser.cases:
        ident = case["id"]
        if ident in seen:
            raise ValueError(f"duplicate gallery case: {ident}")
        seen.add(ident)
        images: dict[str, dict[str, Any]] = {}
        for label, row in case["images"].items():
            if label not in ROLE_LABELS:
                continue
            relative, path, digest = _resolve_image(gallery, row["src"])
            images[label] = {
                "path": relative,
                "sha256": digest,
                "exists": digest is not None,
                "status": row.get("status"),
                "old_score": row.get("old_score"),
                "old_score_protocol": (
                    "text-paint-sft-pairwise-margin-v1"
                    if label in {"SFT", "RL-step-16"}
                    else "text-paint-sft-pairwise-grounded-v1"
                    if label == "pilot-step-24" else None
                ),
            }
        frozen.append({
            "id": ident,
            "family": case["family"],
            "brief": case["brief"],
            "brief_sha256": sha256_text(case["brief"]),
            "images": images,
        })
    return frozen


def _available(case: dict[str, Any], role: str) -> bool:
    return bool(case.get("images", {}).get(role, {}).get("sha256"))


def _preference_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score > 0.5:
        return "candidate"
    if score < 0.5:
        return "anchor"
    return "tie"


def _old_verdict(case: dict[str, Any], anchor: str, candidate: str) -> dict[str, Any]:
    a_score = case["images"].get(anchor, {}).get("old_score")
    c_score = case["images"].get(candidate, {}).get("old_score")
    # Historical gallery values came from different judge rubrics (and some
    # anchors are weak), so they must not be converted into a direct verdict.
    # The controlled fresh v1 result below is the old-side verdict.
    return {
        "preference_score": None,
        "preferred_role": None,
        "label": "historical_context_only",
        "basis": "different_historical_rubrics_not_comparable",
        "historical_scores": {anchor: a_score, candidate: c_score},
        "protocols": {
            role: case["images"].get(role, {}).get("old_score_protocol")
            for role in (anchor, candidate)
        },
    }


def _pair_id(case_id: str, anchor: str, candidate: str) -> str:
    return f"{case_id}__{ROLE_TOKEN[anchor]}__{ROLE_TOKEN[candidate]}"


def _make_pair(case: dict[str, Any], anchor: str, candidate: str, group: str) -> dict[str, Any] | None:
    if not (_available(case, anchor) and _available(case, candidate)):
        return None
    return {
        "pair_id": _pair_id(case["id"], anchor, candidate),
        "case_id": case["id"],
        "family": case["family"],
        "group": group,
        "brief": case["brief"],
        "brief_sha256": case["brief_sha256"],
        "anchor_role": anchor,
        "candidate_role": candidate,
        "images": {
            anchor: case["images"][anchor],
            candidate: case["images"][candidate],
        },
        "old_scores": {
            role: case["images"][role].get("old_score") for role in (anchor, candidate)
        },
        "old_verdict": _old_verdict(case, anchor, candidate),
    }


def _select_controls(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = set(PANEL_IDS)
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILY_ORDER}
    for case in sorted(cases, key=lambda row: row["id"]):
        if case["id"] in excluded:
            continue
        if _available(case, "RL-step-16") and _available(case, "pilot-step-24"):
            by_family.setdefault(case["family"], []).append(case)
    chosen: list[dict[str, Any]] = []
    # Round-robin gives deterministic 3/3/2 coverage for simple/intermediate/
    # complex when all families have enough valid direct pairs.
    while len(chosen) < 8:
        progress = False
        for family in FAMILY_ORDER:
            if len(chosen) >= 8:
                break
            if by_family.get(family):
                chosen.append(by_family[family].pop(0))
                progress = True
        if not progress:
            break
    return chosen


def build_pairs(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create panel pairs and deterministic stratified direct controls."""

    by_id = {case["id"]: case for case in cases}
    pairs: list[dict[str, Any]] = []
    requested: dict[str, int] = {}
    missing: dict[str, dict[str, list[str]]] = {}

    def add(case: dict[str, Any], anchor: str, candidate: str, group: str) -> None:
        requested[case["id"]] = requested.get(case["id"], 0) + 1
        pair = _make_pair(case, anchor, candidate, group)
        if pair is None:
            missing.setdefault(case["id"], {}).setdefault("pairs", []).append(
                f"{anchor}>{candidate}"
            )
            return
        pairs.append(pair)

    for ident in PANEL_IDS:
        case = by_id.get(ident)
        if case is None:
            raise ValueError(f"panel case missing from gallery: {ident}")
        add(case, "frozen SFT", "RL-step-16", "human_panel")
        add(case, "frozen SFT", "pilot-step-24", "human_panel")
        add(case, "RL-step-16", "pilot-step-24", "human_panel")
        if ident == "t600-516":
            add(case, "frozen SFT", "SFT", "human_panel")

    controls = _select_controls(cases)
    for case in controls:
        add(case, "RL-step-16", "pilot-step-24", "control")
    meta = {
        "panel_case_ids": list(PANEL_IDS),
        "control_case_ids": [case["id"] for case in controls],
        "requested_pair_count_by_case": requested,
        "missing_pairs_by_case": missing,
        "control_family_counts": {
            family: sum(case["family"] == family for case in controls)
            for family in FAMILY_ORDER
        },
    }
    return pairs, meta


def _manifest_digest(manifest: dict[str, Any]) -> str:
    copy = dict(manifest)
    copy.pop("manifest_content_sha256", None)
    return sha256_text(json.dumps(copy, sort_keys=True, separators=(",", ":")))


def prepare_manifest(gallery: Path, output: Path) -> dict[str, Any]:
    cases = parse_gallery(gallery)
    pairs, pair_meta = build_pairs(cases)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol": judge.PROTOCOL,
        "old_protocol": old_judge.PROTOCOL,
        "model": MODEL,
        "prompt_sha256": sha256_text(judge.PROMPT),
        "old_prompt_sha256": sha256_text(old_judge.PROMPT),
        "gallery_html_sha256": sha256_file(gallery),
        "gallery_path": str(gallery.resolve()),
        "cases": cases,
        "pairs": pairs,
        "pair_plan": pair_meta,
        "human_feedback": HUMAN_FEEDBACK,
        "historical_score_note": (
            "Gallery SFT/RL-step-16 scores use text-paint-sft-pairwise-margin-v1; "
            "pilot-step-24 scores use text-paint-sft-pairwise-grounded-v1. "
            "Cross-score comparisons are descriptive historical context, not a "
            "direct quality verdict."
        ),
    }
    manifest["manifest_content_sha256"] = _manifest_digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "input-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read_manifest(output: Path) -> dict[str, Any]:
    path = output / "input-manifest.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported judge-alignment input manifest")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise ValueError("input manifest content hash mismatch")
    if manifest.get("protocol") != judge.PROTOCOL:
        raise ValueError("input manifest protocol mismatch")
    if manifest.get("prompt_sha256") != sha256_text(judge.PROMPT):
        raise ValueError("input manifest prompt hash mismatch; prepare again")
    if manifest.get("old_protocol") != old_judge.PROTOCOL:
        raise ValueError("input manifest old protocol mismatch")
    if manifest.get("old_prompt_sha256") != sha256_text(old_judge.PROMPT):
        raise ValueError("input manifest old prompt hash mismatch; prepare again")
    if manifest.get("model") != MODEL:
        raise ValueError("input manifest model mismatch")
    return manifest


def _pair_identity(manifest: dict[str, Any], pair: dict[str, Any], repeat: int) -> dict[str, Any]:
    return {
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "protocol": manifest["protocol"],
        "old_protocol": manifest["old_protocol"],
        "model": manifest["model"],
        "prompt_sha256": manifest["prompt_sha256"],
        "old_prompt_sha256": manifest["old_prompt_sha256"],
        "pair_id": pair["pair_id"],
        "case_id": pair["case_id"],
        "repeat": repeat,
        "brief_sha256": pair["brief_sha256"],
        "image_sha256": {
            pair["anchor_role"]: pair["images"][pair["anchor_role"]]["sha256"],
            pair["candidate_role"]: pair["images"][pair["candidate_role"]]["sha256"],
        },
    }


def _verify_pair_files(pair: dict[str, Any], gallery: Path) -> None:
    for role in (pair["anchor_role"], pair["candidate_role"]):
        row = pair["images"][role]
        path = (gallery.parent / row["path"]).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        current = sha256_file(path)
        if current != row["sha256"]:
            raise ValueError(f"image hash mismatch for {pair['pair_id']} {role}")


def _materialize_brief(pair: dict[str, Any], output: Path) -> Path:
    briefs = output / "briefs"
    briefs.mkdir(parents=True, exist_ok=True)
    path = briefs / f"{pair['case_id']}.txt"
    if path.is_symlink():
        raise ValueError(f"brief path is symlink: {path}")
    if path.is_file() and sha256_file(path) != pair["brief_sha256"]:
        raise ValueError(f"brief cache mismatch for {pair['case_id']}")
    if not path.is_file():
        path.write_text(pair["brief"], encoding="utf-8")
    return path


def _receipt_path(output: Path, pair_id: str, repeat: int) -> Path:
    _safe_id(pair_id)
    return output / "receipts" / f"{pair_id}--repeat-{repeat}.json"


def _safe_failure(exc: Exception) -> str:
    name = type(exc).__name__
    if not SAFE_NAME.fullmatch(name):
        name = "ProviderError"
    return f"judge_failure_{name}"


async def _one_pair(*, manifest: dict[str, Any], pair: dict[str, Any], repeat: int,
                    gallery: Path, output: Path, key_file: Path,
                    semaphore: asyncio.Semaphore, index: int, total: int) -> dict[str, Any]:
    identity = _pair_identity(manifest, pair, repeat)
    receipt_path = _receipt_path(output, pair["pair_id"], repeat)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Verify frozen bytes even when a cached receipt exists.  A matching old
    # receipt is not enough if a local image was replaced in place.
    _verify_pair_files(pair, gallery)
    brief_path = _materialize_brief(pair, output)
    if receipt_path.is_file():
        if receipt_path.is_symlink():
            raise ValueError(f"receipt path is symlink: {receipt_path}")
        cached = json.loads(receipt_path.read_text(encoding="utf-8"))
        if cached.get("schema") != RECEIPT_SCHEMA or cached.get("identity") != identity:
            raise ValueError(f"stale or incompatible cached receipt: {pair['pair_id']} repeat {repeat}")
        print(f"[{index}/{total}] cached {pair['pair_id']} repeat={repeat}", flush=True)
        return cached

    async with semaphore:
        started = time.monotonic()
        anchor = gallery.parent / pair["images"][pair["anchor_role"]]["path"]
        candidate = gallery.parent / pair["images"][pair["candidate_role"]]["path"]

        async def call_one(module: Any) -> dict[str, Any]:
            try:
                result = await module.judge_pair(
                    brief=brief_path, anchor=anchor, candidate=candidate, key_file=key_file
                )
                if result.get("protocol") != module.PROTOCOL:
                    raise ValueError("judge protocol mismatch")
                return {"status": "scored", "judge": result}
            except Exception as exc:  # Persist only a safe exception class name.
                return {"status": "censored", "reason": _safe_failure(exc)}

        old_result, new_result = await asyncio.gather(
            call_one(old_judge), call_one(judge)
        )
        statuses = {old_result["status"], new_result["status"]}
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "identity": identity,
            "status": "scored" if statuses == {"scored"} else "partial" if "scored" in statuses else "censored",
            "elapsed_seconds_runner": time.monotonic() - started,
            "versions": {"old": old_result, "new": new_result},
        }
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    print(f"[{index}/{total}] {receipt['status']} {pair['pair_id']} repeat={repeat}", flush=True)
    return receipt


async def run_judging(*, gallery: Path, output: Path, key_file: Path,
                      concurrency: int = 6, limit_pairs: int | None = None,
                      repeat: int = 1, extra_repeats: int = 0) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if repeat < 1 or extra_repeats < 0:
        raise ValueError("repeat must be positive and extra_repeats nonnegative")
    manifest = _read_manifest(output)
    if sha256_file(gallery) != manifest["gallery_html_sha256"]:
        raise ValueError("gallery HTML changed since preparation")
    pairs = manifest["pairs"]
    if limit_pairs is not None:
        if limit_pairs < 1:
            raise ValueError("limit-pairs must be positive")
        pairs = pairs[:limit_pairs]
    repeats = repeat + extra_repeats
    jobs = [(pair, index, rep) for index, pair in enumerate(pairs, 1) for rep in range(1, repeats + 1)]
    total = len(jobs)
    sem = asyncio.Semaphore(concurrency)
    receipts = await asyncio.gather(*(
        _one_pair(manifest=manifest, pair=pair, repeat=rep, gallery=gallery,
                  output=output, key_file=key_file, semaphore=sem,
                  index=index, total=total)
        for pair, index, rep in jobs
    ))
    summary = summarize(manifest, pairs, receipts, gallery=gallery, output=output,
                        repeats=repeats)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(manifest, summary, gallery=gallery, output=output)
    return summary


def _label_for_score(score: float | None, anchor: str, candidate: str) -> str | None:
    if score is None:
        return None
    if score > 0.5:
        return candidate
    if score < 0.5:
        return anchor
    return "tie"


def _reported_cost(usage: dict[str, Any]) -> tuple[str, float] | None:
    """Choose one provider cost field; Gateway fields are alternatives."""

    for key in ("cost", "gateway_cost", "market_cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def _flatten_version(version: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    row = {"status": version.get("status"), "reason": version.get("reason")}
    if version.get("status") != "scored":
        return row
    result = version.get("judge") or {}
    score = result.get("score")
    row.update({
        "score": score,
        "raw_pairwise_score": result.get("raw_pairwise_score"),
        "candidate_margin_cap": result.get("candidate_margin_cap"),
        "order_gap": result.get("order_gap"),
        "preferred_role": _label_for_score(score, pair["anchor_role"], pair["candidate_role"]),
        "raw_preferred_role": _label_for_score(result.get("raw_pairwise_score"), pair["anchor_role"], pair["candidate_role"]),
        "capped_preferred_role": _label_for_score(score, pair["anchor_role"], pair["candidate_role"]),
        "judge_protocol": result.get("protocol"),
        "model": result.get("model", MODEL),
        "views": [
            {
                "reason": (view.get("label") or {}).get("reason"),
                "winner": (view.get("label") or {}).get("winner"),
                "strength": (view.get("label") or {}).get("strength"),
                "brief_fidelity_A": (view.get("label") or {}).get("brief_fidelity_A"),
                "brief_fidelity_B": (view.get("label") or {}).get("brief_fidelity_B"),
                "visible_evidence_A": (view.get("label") or {}).get("visible_evidence_A"),
                "visible_evidence_B": (view.get("label") or {}).get("visible_evidence_B"),
                "weaknesses_A": (view.get("label") or {}).get("weaknesses_A"),
                "weaknesses_B": (view.get("label") or {}).get("weaknesses_B"),
                "elapsed_seconds": view.get("elapsed_seconds"),
                "usage": view.get("usage"),
            }
            for view in result.get("views", [])
        ],
    })
    votes = []
    for view_index, view in enumerate(row["views"]):
        winner = view.get("winner")
        if winner == "A":
            votes.append(pair["anchor_role"] if view_index == 0 else pair["candidate_role"])
        elif winner == "B":
            votes.append(pair["candidate_role"] if view_index == 0 else pair["anchor_role"])
        elif winner == "tie":
            votes.append("tie")
        else:
            votes.append(None)
    row["normalized_label_votes"] = votes
    row["two_order_label_consensus"] = (
        len(votes) == 2 and votes[0] is not None and votes[0] == votes[1]
    )
    row["two_order_label_disagreement"] = len(votes) == 2 and votes[0] != votes[1]
    return row


def _flatten_receipt(receipt: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    versions = receipt.get("versions", {})
    old = _flatten_version(versions.get("old", {"status": "censored", "reason": "missing_old_result"}), pair)
    new = _flatten_version(versions.get("new", {"status": "censored", "reason": "missing_new_result"}), pair)
    return {
        "status": receipt.get("status"),
        "elapsed_seconds_runner": receipt.get("elapsed_seconds_runner"),
        "old": old,
        "new": new,
    }


def _case_stats(manifest: dict[str, Any], pairs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for pair, row in zip(pairs, rows):
        by_case.setdefault(pair["case_id"], []).append(row)
    stats: dict[str, Any] = {}
    requested = manifest["pair_plan"]["requested_pair_count_by_case"]
    missing = manifest["pair_plan"]["missing_pairs_by_case"]
    for case in manifest["cases"]:
        ident = case["id"]
        case_rows = by_case.get(ident, [])
        version_stats: dict[str, Any] = {}
        for version in ("old", "new"):
            version_rows = [repeat[version] for row in case_rows for repeat in row["repeats"]]
            scored = [item for item in version_rows if item["status"] == "scored"]
            censored = [item for item in version_rows if item["status"] != "scored"]
            gaps = [float(item["order_gap"]) for item in scored if item.get("order_gap") is not None]
            latencies = [float(view["elapsed_seconds"]) for item in scored for view in item.get("views", []) if view.get("elapsed_seconds") is not None]
            tokens: dict[str, int] = {}
            costs: list[tuple[str, float]] = []
            for item in scored:
                for view in item.get("views", []):
                    usage = view.get("usage") or {}
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if isinstance(usage.get(key), (int, float)):
                            tokens[key] = tokens.get(key, 0) + int(usage[key])
                    observed_cost = _reported_cost(usage)
                    if observed_cost is not None:
                        costs.append(observed_cost)
            version_stats[version] = {
                "scored_receipts": len(scored),
                "censored_receipts": len(censored),
                "order_disagreement_receipts": sum(gap + 1e-9 >= 1 / 3 for gap in gaps),
                "order_disagreement_denominator": len(gaps),
                "api_latency_seconds": {
                    "count": len(latencies),
                    "mean": statistics.fmean(latencies) if latencies else None,
                    "median": statistics.median(latencies) if latencies else None,
                },
                "provider_usage": tokens,
                "provider_reported_cost": sum(value for _, value in costs) if costs else None,
                "provider_reported_cost_observations": len(costs),
            }
        stats[ident] = {
            "family": case["family"],
            "requested_pairs": requested.get(ident, 0),
            "available_pairs": len(case_rows),
            "missing_pairs": missing.get(ident, {}).get("pairs", []),
            "versions": version_stats,
        }
    return stats


def summarize(manifest: dict[str, Any], pairs: list[dict[str, Any]], receipts: list[dict[str, Any]], *, gallery: Path, output: Path, repeats: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {pair["pair_id"]: [] for pair in pairs}
    for receipt in receipts:
        grouped[receipt["identity"]["pair_id"]].append(receipt)
    pair_rows: list[dict[str, Any]] = []
    agreement = {
        version: {
            "explicit_label_denominator": 0,
            "scored_denominator": 0,
            "agreements": 0,
            "disagreements_or_ties": 0,
            "raw_direction_agreements": 0,
            "capped_direction_agreements": 0,
            "two_order_unanimous_denominator": 0,
            "two_order_unanimous_agreements": 0,
            "two_order_unanimous_disagreements_or_ties": 0,
        }
        for version in ("old", "new")
    }
    metrics = {
        version: {"rows": [], "latencies": [], "costs": [], "tokens": {}, "gaps": []}
        for version in ("old", "new")
    }
    for pair in pairs:
        feedback = manifest["human_feedback"].get(pair["case_id"], {})
        repeats_rows = [_flatten_receipt(receipt, pair) for receipt in sorted(grouped[pair["pair_id"]], key=lambda x: x["identity"]["repeat"])]
        for repeat_row in repeats_rows:
            for version in ("old", "new"):
                version_row = repeat_row[version]
                metrics[version]["rows"].append(version_row)
                if version_row["status"] == "scored":
                    if version_row.get("order_gap") is not None:
                        metrics[version]["gaps"].append(float(version_row["order_gap"]))
                    for view in version_row.get("views", []):
                        if view.get("elapsed_seconds") is not None:
                            metrics[version]["latencies"].append(float(view["elapsed_seconds"]))
                        usage = view.get("usage") or {}
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            if isinstance(usage.get(key), (int, float)):
                                metrics[version]["tokens"][key] = metrics[version]["tokens"].get(key, 0) + int(usage[key])
                        observed_cost = _reported_cost(usage)
                        if observed_cost is not None:
                            metrics[version]["costs"].append(observed_cost)
                for preference in feedback.get("explicit_preferences", []):
                    if {preference["preferred_role"], preference["other_role"]} != {pair["anchor_role"], pair["candidate_role"]}:
                        continue
                    agreement[version]["explicit_label_denominator"] += 1
                    if version_row["status"] != "scored":
                        continue
                    agreement[version]["scored_denominator"] += 1
                    if version_row.get("raw_preferred_role") == preference["preferred_role"]:
                        agreement[version]["raw_direction_agreements"] += 1
                    if version_row.get("capped_preferred_role") == preference["preferred_role"]:
                        agreement[version]["capped_direction_agreements"] += 1
                        agreement[version]["agreements"] += 1
                    else:
                        agreement[version]["disagreements_or_ties"] += 1
                    votes = version_row.get("normalized_label_votes", [])
                    if len(votes) == 2 and votes[0] is not None and votes[1] is not None:
                        agreement[version]["two_order_unanimous_denominator"] += 1
                        if votes[0] == votes[1] == preference["preferred_role"]:
                            agreement[version]["two_order_unanimous_agreements"] += 1
                        else:
                            agreement[version]["two_order_unanimous_disagreements_or_ties"] += 1
        pair_rows.append({
            "pair_id": pair["pair_id"],
            "case_id": pair["case_id"],
            "family": pair["family"],
            "group": pair["group"],
            "brief": pair["brief"],
            "anchor_role": pair["anchor_role"],
            "candidate_role": pair["candidate_role"],
            "images": pair["images"],
            "old_scores": pair["old_scores"],
            "old_verdict": pair["old_verdict"],
            "human_feedback": feedback,
            "repeats": repeats_rows,
        })
    status_counts = {status: sum(receipt["status"] == status for receipt in receipts) for status in ("scored", "partial", "censored")}
    version_summary: dict[str, Any] = {}
    for version in ("old", "new"):
        gaps = metrics[version]["gaps"]
        latencies = metrics[version]["latencies"]
        costs = metrics[version]["costs"]
        label_rows = [row for row in metrics[version]["rows"] if row["status"] == "scored" and len(row.get("normalized_label_votes", [])) == 2]
        version_summary[version] = {
            "status_counts": {
                "scored": sum(row["status"] == "scored" for row in metrics[version]["rows"]),
                "censored": sum(row["status"] != "scored" for row in metrics[version]["rows"]),
            },
            "order_disagreement": {
                "threshold": 1 / 3,
                "count": sum(gap + 1e-9 >= 1 / 3 for gap in gaps),
                "denominator": len(gaps),
                "mean_order_gap": statistics.fmean(gaps) if gaps else None,
            },
            "two_order_label_consensus": {
                "consensus_count": sum(bool(row.get("two_order_label_consensus")) for row in label_rows),
                "disagreement_count": sum(bool(row.get("two_order_label_disagreement")) for row in label_rows),
                "denominator": len(label_rows),
            },
            "api_latency_seconds": {
                "count": len(latencies),
                "mean": statistics.fmean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
            },
            "provider_usage": metrics[version]["tokens"],
            "provider_reported_cost": sum(value for _, value in costs) if costs else None,
            "provider_reported_cost_observations": len(costs),
            "provider_reported_cost_fields": {
                key: sum(1 for observed_key, _ in costs if observed_key == key)
                for key in ("cost", "gateway_cost", "market_cost")
                if any(observed_key == key for observed_key, _ in costs)
            },
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol": manifest["protocol"],
        "old_protocol": manifest["old_protocol"],
        "model": MODEL,
        "prompt_sha256": manifest["prompt_sha256"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "gallery_html_sha256": manifest["gallery_html_sha256"],
        "gallery_path": str(gallery.resolve()),
        "pair_count": len(pairs),
        "repeat_count": repeats,
        "receipt_count": len(receipts),
        "status_counts": status_counts,
        "censored_never_scored_as_zero": True,
        "human_agreement": agreement,
        "versions": version_summary,
        "case_stats": _case_stats(manifest, pairs, pair_rows),
        "pairs": pair_rows,
        "historical_score_note": manifest["historical_score_note"],
        "report_note": "Preference scores are oriented pairwise values: 0.5 means tie; they are not absolute quality scores.",
    }


def _img_src(report: Path, gallery: Path, relative: str) -> str:
    path = gallery.parent / relative
    return Path(__import__("os").path.relpath(path, report.parent)).as_posix()


def write_report(manifest: dict[str, Any], summary: dict[str, Any], *, gallery: Path, output: Path) -> Path:
    """Write a compact, readable report with both images and rationales."""

    blocks: list[str] = []
    for pair in summary["pairs"]:
        feedback = pair.get("human_feedback", {})
        feedback_html = f"<aside><strong>Human context:</strong> {html.escape(feedback['note'])}</aside>" if feedback.get("note") else ""
        image_cells = []
        for role in (pair["anchor_role"], pair["candidate_role"]):
            image = pair["images"][role]
            old_score = image.get("old_score")
            old_protocol = image.get("old_score_protocol")
            historical = "missing" if old_score is None else f"{old_score:.3f} ({old_protocol})"
            image_cells.append(
                f"<div class='image'><img src='{html.escape(_img_src(output / 'report.html', gallery, image['path']))}' alt='{html.escape(role)}'>"
                f"<div><b>{html.escape(role)}</b><br>historical gallery score: {html.escape(historical)}</div></div>"
            )
        repeat_blocks = []
        for index, repeat_row in enumerate(pair["repeats"], 1):
            version_blocks = []
            for version, label in (("old", "grounded-v1"), ("new", "grounded-v2")):
                version_row = repeat_row[version]
                if version_row["status"] != "scored":
                    version_blocks.append(f"<div class='receipt bad'>{label}: censored ({html.escape(version_row.get('reason') or 'unknown')})</div>")
                    continue
                reasons = []
                for view in version_row.get("views", []):
                    if view.get("reason"):
                        reasons.append(f"<details><summary>{label} order rationale</summary><p>{html.escape(view['reason'])}</p></details>")
                version_blocks.append(
                    f"<div class='receipt'><b>{label}</b>: capped preference score {version_row.get('score')!r} "
                    f"(preferred {html.escape(str(version_row.get('capped_preferred_role')))}; 0.5 = tie), "
                    f"raw score {version_row.get('raw_pairwise_score')!r} (raw direction {html.escape(str(version_row.get('raw_preferred_role')))}), "
                    f"cap {version_row.get('candidate_margin_cap')!r}, capped direction {html.escape(str(version_row.get('capped_preferred_role')))}, "
                    f"order gap {version_row.get('order_gap')!r}, label votes {html.escape(str(version_row.get('normalized_label_votes')))}, "
                    f"label consensus {version_row.get('two_order_label_consensus')!r}{''.join(reasons)}</div>"
                )
            repeat_blocks.append(f"<div class='receipt'><b>repeat {index}</b> receipt: {html.escape(str(repeat_row.get('status')))}</div>{''.join(version_blocks)}")
        blocks.append(
            f"<section><h2>{html.escape(pair['case_id'])} <small>{html.escape(pair['family'])} · {html.escape(pair['group'])}</small></h2>"
            f"<p class='brief'>{html.escape(pair['brief'])}</p>{feedback_html}"
            f"<div class='images'>{''.join(image_cells)}</div>"
            f"<p><b>Historical gallery values:</b> context only ({html.escape(str(pair['old_verdict'].get('historical_scores')))}); "
            f"fresh old-v1 and new-v2 verdicts are shown per repeat below.</p>"
            f"{''.join(repeat_blocks)}</section>"
        )
    h = summary["human_agreement"]
    h_old = h["old"]
    h_new = h["new"]
    report = """<!doctype html><html><head><meta charset='utf-8'><title>Grounded judge v2 alignment</title>
<style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui,sans-serif;padding:28px}main{max-width:1240px;margin:auto}h1{margin:0 0 8px;color:#f9fafb}h2{margin:0 0 6px;color:#f9fafb}h2 small{font-size:.65em;color:#93c5fd;font-weight:500}.meta{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:16px;margin:16px 0 24px}.note{color:#cbd5e1}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.image{background:#111827;border-radius:10px;padding:10px}.image img{display:block;width:100%;max-width:600px;height:auto;aspect-ratio:1/1;object-fit:cover;border-radius:7px;margin-bottom:8px;cursor:zoom-in}.image img:active{transform:scale(1.25);transform-origin:center;position:relative;z-index:2}section{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:18px;margin:18px 0}.brief{color:#d1d5db;line-height:1.45}aside{border-left:3px solid #fbbf24;padding:8px 12px;margin:10px 0;background:#292014;color:#fde68a}.receipt{border-top:1px solid #374151;padding:10px 0;line-height:1.45}.receipt.bad{color:#fca5a5}.receipt p{color:#cbd5e1;margin:7px 0}.receipt details{margin-top:8px;color:#cbd5e1}.receipt summary{cursor:pointer;color:#bfdbfe}.summary{color:#bfdbfe}@media(max-width:760px){body{padding:14px}.images{grid-template-columns:1fr}.image img{max-width:600px}}</style></head><body><main>"""
    old_votes = summary["versions"]["old"]["two_order_label_consensus"]
    new_votes = summary["versions"]["new"]["two_order_label_consensus"]
    report += f"<h1>Grounded judge v1 → v2 alignment</h1><div class='meta'><div class='summary'>Old {html.escape(manifest['old_protocol'])} vs new {html.escape(manifest['protocol'])} · model {html.escape(MODEL)} · {summary['pair_count']} pairs · {summary['receipt_count']} receipts</div><p class='note'>Preference scores are oriented pairwise values: 0.5 means tie, not absolute quality. Historical gallery scores use different old protocols and are shown as context only. Human agreement counts only explicit preference labels; qualitative notes do not force a winner.</p><p>Fresh old-v1 explicit-label agreement: {h_old['agreements']}/{h_old['scored_denominator']} scored observations; new-v2: {h_new['agreements']}/{h_new['scored_denominator']}. Raw-direction agreements are {h_old['raw_direction_agreements']} and {h_new['raw_direction_agreements']}; capped-direction agreements are {h_old['capped_direction_agreements']} and {h_new['capped_direction_agreements']}.</p><p>Two-order label consensus: old {old_votes['consensus_count']} unanimous, {old_votes['disagreement_count']} disagreeing of {old_votes['denominator']}; new {new_votes['consensus_count']} unanimous, {new_votes['disagreement_count']} disagreeing of {new_votes['denominator']}. Explicit human-label unanimous agreement is old {h_old['two_order_unanimous_agreements']}/{h_old['two_order_unanimous_denominator']} and new {h_new['two_order_unanimous_agreements']}/{h_new['two_order_unanimous_denominator']}. Censored provider failures are omitted from scores.</p></div>"""
    report += "".join(blocks) + "</main></body></html>\n"
    path = output / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--extra-repeats", type=int, default=0)
    args = parser.parse_args(argv)
    if args.prepare_only:
        manifest = prepare_manifest(args.gallery, args.output)
        print(json.dumps({"manifest": str(args.output / 'input-manifest.json'), "pairs": len(manifest["pairs"]), "controls": manifest["pair_plan"]["control_case_ids"]}, sort_keys=True))
        return 0
    summary = asyncio.run(run_judging(
        gallery=args.gallery,
        output=args.output,
        key_file=args.key_file,
        concurrency=args.concurrency,
        limit_pairs=args.limit_pairs,
        repeat=args.repeat,
        extra_repeats=args.extra_repeats,
    ))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"pairs", "case_stats"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
