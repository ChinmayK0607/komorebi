#!/usr/bin/env python3
"""Make a small, auditable summary of a text-paint RL run.

The reader deliberately treats shipped batches and trainer-completed batches as
different quantities.  It only consumes the public, hash-verified evidence
bundle; no model or renderer is needed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


def _jsonl(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _stream_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for outer in _jsonl(root / "stream"):
        for trace in outer.get("traces", []):
            task = outer.get("task", {}).get("data", {})
            info = trace.get("info", {})
            rows.append({
                "trace_id": trace.get("id"), "outer_id": outer.get("id"),
                "step": outer.get("run", {}).get("work", {}).get("step"),
                "task_id": task.get("name") or task.get("task_id"),
                "family": task.get("family") or info.get("family"),
                "group": outer.get("group"), "trace": trace, "info": info,
            })
    return rows


def _annotations(root: Path, kind: str) -> dict[str, dict[str, Any]]:
    out = {}
    for x in _jsonl(root / "annotations" / kind):
        if x.get("trace_id"):
            out[x["trace_id"]] = x
    return out


def _score(row: dict[str, Any]) -> float | None:
    s = row["info"].get("final_canvas_score")
    if s is None:
        rewards = row["trace"].get("rewards") or {}
        s = (rewards.get("final_canvas_aesthetic") or {}).get("score")
    return float(s) if isinstance(s, (int, float)) else None


def _retained_invalid(row: dict[str, Any]) -> bool:
    info = row["info"]
    turns = info.get("turns") or []
    return bool(info.get("final_canvas_valid")) and bool(turns) and not bool(turns[-1].get("valid"))


def _advantage_audit(row: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    """Audit token advantages without guessing a branch or token alignment."""
    trace = row["trace"]
    nodes = trace.get("nodes") or []
    branches = annotation.get("branches") or []
    total = sum(len(n.get("token_ids") or []) for n in nodes)
    branch = branches[0] if len(branches) == 1 and isinstance(branches[0], dict) else None
    adv = branch.get("advantages") if branch is not None else None
    mask_counts_aligned = all(
        isinstance(n.get("mask"), list)
        and len(n.get("mask") or []) == len(n.get("token_ids") or [])
        for n in nodes
    )
    if branch is None:
        alignment = "unknown"
        aligned: bool | None = None
    elif not isinstance(adv, list) or len(adv) != total or not mask_counts_aligned:
        alignment = "unaligned"
        aligned = False
    else:
        alignment = "aligned"
        aligned = True

    # Token fields stay null unless every positional assumption is verified.
    token_fields: dict[str, Any] = {
        "prompt_nonzero": None,
        "assistant_mask_tokens": None,
        "assistant_nonzero": None,
        "assistant_zero": None,
        "assistant_positive": None,
        "assistant_negative": None,
        "retained_last_assistant_nonzero": None,
        "retained_last_assistant_zero": None,
        "retained_last_assistant_positive": None,
        "retained_last_assistant_negative": None,
    }
    prompt_nonzero = assistant_nonzero = assistant_tokens = assistant_zero = 0
    assistant_positive = assistant_negative = 0
    assistant_segments = []
    if alignment == "aligned":
        offset = 0
        for n in nodes:
            ids = n.get("token_ids") or []
            role = (n.get("message") or {}).get("role")
            mask = n.get("mask") or []
            vals = adv[offset:offset + len(ids)]
            for i, value in enumerate(vals):
                try:
                    nonzero = abs(float(value)) > 1e-12
                except (TypeError, ValueError):
                    nonzero = False
                masked = bool(mask[i])
                if role == "assistant" and masked:
                    assistant_tokens += 1
                    if nonzero:
                        assistant_nonzero += 1
                    else:
                        assistant_zero += 1
                    try:
                        if float(value) > 1e-12:
                            assistant_positive += 1
                        elif float(value) < -1e-12:
                            assistant_negative += 1
                    except (TypeError, ValueError):
                        pass
                elif role != "assistant" and nonzero:
                    # Positional: prompt tokens are counted only at their node.
                    prompt_nonzero += 1
            if role == "assistant" and any(bool(x) for x in mask):
                assistant_segments.append((offset, mask))
            offset += len(ids)
        token_fields.update({
            "prompt_nonzero": prompt_nonzero,
            "assistant_mask_tokens": assistant_tokens,
            "assistant_nonzero": assistant_nonzero,
            "assistant_zero": assistant_zero,
            "assistant_positive": assistant_positive,
            "assistant_negative": assistant_negative,
        })

    last_assistant_nonzero = last_assistant_zero = 0
    last_assistant_positive = last_assistant_negative = 0
    if alignment == "aligned" and _retained_invalid(row) and assistant_segments:
        start, mask = assistant_segments[-1]
        for i, marked in enumerate(mask):
            if not marked:
                continue
            try:
                value = float(adv[start + i])
            except (TypeError, ValueError):
                value = 0.0
            if abs(value) > 1e-12:
                last_assistant_nonzero += 1
            else:
                last_assistant_zero += 1
            if value > 1e-12:
                last_assistant_positive += 1
            elif value < -1e-12:
                last_assistant_negative += 1
        token_fields.update({
            "retained_last_assistant_nonzero": last_assistant_nonzero,
            "retained_last_assistant_zero": last_assistant_zero,
            "retained_last_assistant_positive": last_assistant_positive,
            "retained_last_assistant_negative": last_assistant_negative,
        })

    # Scalar status is independent of token alignment and remains auditable.
    last_positive = last_nonpositive = 0
    if _retained_invalid(row):
        scalar = ((annotation.get("info") or {}).get("advantage"))
        if isinstance(scalar, (int, float)):
            if scalar > 0:
                last_positive += 1
            else:
                last_nonpositive += 1
    return {"aligned": aligned, "alignment": alignment,
            "branch_count": len(branches) if isinstance(branches, list) else 0,
            "mask_counts_aligned": mask_counts_aligned,
            **token_fields,
            "retained_last_invalid_positive_scalar": last_positive,
            "retained_last_invalid_nonpositive_scalar": last_nonpositive}


def _receipt_summary(receipts: Path) -> dict[str, Any]:
    status = collections.Counter(); scores = collections.Counter(); fidelity = collections.Counter()
    conflicts = rechecked = recheck_conflicts = 0
    examples = []
    files = sorted(receipts.glob("*/text-judge.json"))
    for path in files:
        x = json.loads(path.read_text()); status[x.get("status", "missing")] += 1
        s = x.get("score")
        if isinstance(s, (int, float)): scores[str(round(float(s), 6))] += 1
        if x.get("initial_conflict"):
            conflicts += 1
            if len(examples) < 5: examples.append(path.parent.name)
        if x.get("recheck_views") or x.get("recheck_votes"):
            rechecked += 1
            votes = x.get("recheck_votes") or []
            if len(set(votes)) > 1: recheck_conflicts += 1
        for view in x.get("views", []):
            assessments = ((view.get("assessment") or {}).get("assessments") or {})
            for a in assessments.values():
                if a.get("brief_fidelity") is not None: fidelity[a["brief_fidelity"]] += 1
    return {"receipt_count": len(files), "status": dict(status), "score_distribution": dict(scores),
            "initial_conflicts": conflicts, "rechecked": rechecked,
            "recheck_conflicts": recheck_conflicts, "fidelity_labels": dict(fidelity),
            "conflict_examples": examples}


def _hash(path: Path) -> dict[str, Any]:
    h = hashlib.sha256(); size = 0
    with path.open("rb") as fh:
        while b := fh.read(1024 * 1024): h.update(b); size += len(b)
    return {"bytes": size, "sha256": h.hexdigest()}


def _completed_steps(log: Path) -> list[int]:
    if not log.exists(): return []
    # ANSI colour/reset sequences can occur between SUCCESS and Step.
    return [int(x) for x in re.findall(r"SUCCESS[^\n]*?Step\s+(\d+)\s+\|", log.read_text(errors="replace"))]


def _hash_group(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    """Hash a small manifest of files without putting raw evidence in output."""
    entries = []
    for p in sorted(paths):
        if p.is_file(): entries.append((str(p.relative_to(root)), _hash(p)["sha256"]))
    h = hashlib.sha256()
    for name, digest in entries: h.update((name + "\0" + digest + "\n").encode())
    return {"files": len(entries), "manifest_sha256": h.hexdigest()}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def _repeated_prompt_steps(
    effective: list[dict[str, Any]], orch: dict[str, dict[str, Any]]
) -> dict[Any, list[Any]]:
    """Find fixed prompts observed at two or more distinct ship steps."""
    steps: dict[Any, set[Any]] = collections.defaultdict(set)
    for row in effective:
        annotation = orch.get(row.get("trace_id"), {})
        step = (annotation.get("info") or {}).get("ship", {}).get("step")
        steps[row.get("task_id")].add(step)
    return {prompt: sorted(values) for prompt, values in steps.items() if len(values) >= 2}


def _batch_reward_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return both scored-only and invalid-as-zero views of one ship batch."""
    scores = [_score(r) for r in rows]
    scored = [s for s in scores if s is not None]
    valid_scores = [
        s for r, s in zip(rows, scores)
        if bool(r["info"].get("final_canvas_valid")) and s is not None
    ]
    zero_values = [s if s is not None else 0.0 for s in scores]
    groups: dict[str, list[float]] = collections.defaultdict(list)
    group_valid: dict[str, int] = collections.Counter()
    for r, s in zip(rows, scores):
        key = str(r.get("task_id"))
        groups[key].append(s if s is not None else 0.0)
        group_valid[key] += int(bool(r["info"].get("final_canvas_valid")))
    group_stats = {}
    group_means = []
    nonzero_groups = 0
    for key, values in sorted(groups.items()):
        group_mean = sum(values) / len(values)
        group_means.append(group_mean)
        nonzero_count = sum(abs(v) > 1e-12 for v in values)
        nonzero_groups += int(nonzero_count > 0)
        group_stats[key] = {
            "episodes": len(values),
            "valid_count": group_valid[key],
            "reward_mean_invalid_zero": group_mean,
            "reward_variance": _population_variance(values),
            "nonzero_count": nonzero_count,
            "nonzero": bool(nonzero_count),
        }
    valid_count = sum(bool(r["info"].get("final_canvas_valid")) for r in rows)
    return {
        "valid_count": valid_count,
        "invalid_count": len(rows) - valid_count,
        "scored_count": len(scored),
        "valid_scored_count": len(valid_scores),
        # Keep reward_mean for compatibility with the first audit while making
        # the two denominators explicit for review.
        "reward_mean": _mean(scored),
        "valid_only_reward_mean": _mean(valid_scores),
        "invalid_zero_reward_mean": _mean(zero_values),
        "valid_only_mean": _mean(valid_scores),
        "invalid_zero_mean": _mean(zero_values),
        "group_count": len(group_stats),
        "nonzero_reward_groups": nonzero_groups,
        "group_reward_variance": _population_variance(group_means),
        "group_reward_stats": group_stats,
        "score_distribution": dict(collections.Counter(
            str(round(s, 6)) for s in scored
        )),
    }


def _prompt_step_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = _batch_reward_stats(rows)
    return {
        "episodes": len(rows),
        "valid_count": stats["valid_count"],
        "scored_count": stats["scored_count"],
        "valid_scored_count": stats["valid_scored_count"],
        "valid_only_mean": stats["valid_only_reward_mean"],
        "invalid_zero_mean": stats["invalid_zero_reward_mean"],
    }


def _cluster_bootstrap_interval(
    lower: list[float], upper: list[float], replicates: int = 1000,
    seed: int = 20260930,
) -> list[float] | None:
    """Descriptive cluster bootstrap for two lists of fixed-prompt changes."""
    if not lower or not upper:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        left = [lower[rng.randrange(len(lower))] for _ in lower]
        right = [upper[rng.randrange(len(upper))] for _ in upper]
        draws.append(sum(right) / len(right) - sum(left) / len(left))
    draws.sort()
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def analyze(raw_root: Path, run_root: Path | None = None) -> dict[str, Any]:
    rows = _stream_rows(raw_root / "async-output/text-sft512-consistent-20260930/monitors/file/traces")
    orch = _annotations(raw_root / "async-output/text-sft512-consistent-20260930/monitors/file/traces", "orch")
    trainer = _annotations(raw_root / "async-output/text-sft512-consistent-20260930/monitors/file/traces", "trainer")
    effective = [
        r for r in rows
        if (orch.get(r["trace_id"], {}).get("info") or {}).get("effective")
    ]
    by_step = collections.defaultdict(list)
    for r in effective:
        by_step[orch[r["trace_id"]]["info"].get("ship", {}).get("step")].append(r)
    step_summary = {}
    for step, rs in sorted(by_step.items(), key=lambda z: (z[0] is None, z[0])):
        reward_stats = _batch_reward_stats(rs)
        step_summary[str(step)] = {"episodes": len(rs),
            "families": dict(collections.Counter(r["family"] for r in rs)),
            "prompt_ids": dict(collections.Counter(r["task_id"] for r in rs)),
            "status": dict(collections.Counter(r["info"].get("reward_status", "missing") for r in rs)),
            "valid_final_canvas": reward_stats["valid_count"],
            **reward_stats}
    prompt_steps = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in effective:
        step = orch[r["trace_id"]]["info"]["ship"]["step"]
        prompt_steps[r["task_id"]][step].append(r)
    # Eight samples in one batch are not a repeated prompt. Repetition means
    # at least two distinct shipped steps for the same fixed prompt ID.
    repeated = _repeated_prompt_steps(effective, orch)
    matched_prompt_group_means = {}
    fixed_prompt_changes = {}
    window_changes: dict[str, list[float]] = {"11-15": [], "16-20": []}
    for prompt, by_prompt_step in sorted(prompt_steps.items(), key=lambda z: str(z[0])):
        if len(by_prompt_step) < 2:
            continue
        per_step = {
            str(step): _prompt_step_stats(rs)
            for step, rs in sorted(by_prompt_step.items())
        }
        matched_prompt_group_means[str(prompt)] = {
            "steps": sorted(by_prompt_step), "per_step": per_step,
        }
        steps = sorted(by_prompt_step)
        first, last = steps[0], steps[-1]
        first_stats, last_stats = per_step[str(first)], per_step[str(last)]
        invalid_delta = (
            last_stats["invalid_zero_mean"] - first_stats["invalid_zero_mean"]
            if last_stats["invalid_zero_mean"] is not None
            and first_stats["invalid_zero_mean"] is not None else None
        )
        valid_delta = (
            last_stats["valid_only_mean"] - first_stats["valid_only_mean"]
            if last_stats["valid_only_mean"] is not None
            and first_stats["valid_only_mean"] is not None else None
        )
        fixed_prompt_changes[str(prompt)] = {
            "from_step": first, "to_step": last,
            "delta_invalid_zero": invalid_delta,
            "delta_valid_only": valid_delta,
            "per_step": per_step,
        }
        if invalid_delta is not None:
            for label, lo, hi in (("11-15", 11, 15), ("16-20", 16, 20)):
                if lo <= last <= hi:
                    window_changes[label].append(invalid_delta)
    window_means = {
        label: _mean(values) for label, values in window_changes.items()
    }
    delta_windows = (
        window_means["16-20"] - window_means["11-15"]
        if window_means["16-20"] is not None and window_means["11-15"] is not None
        else None
    )
    fixed_prompt_comparison = {
        "window_definition": "fixed-prompt latest-minus-earliest change, grouped by latest ship step",
        "11-15": {"clusters": len(window_changes["11-15"]), "mean_delta": window_means["11-15"]},
        "16-20": {"clusters": len(window_changes["16-20"]), "mean_delta": window_means["16-20"]},
        "delta_16-20_minus_11-15": delta_windows,
        "descriptive_cluster_bootstrap_95pct": _cluster_bootstrap_interval(
            window_changes["11-15"], window_changes["16-20"]
        ),
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 20260930,
        "interpretation": "Descriptive interval for this run; it is not a significance claim or held-out estimate.",
    }
    adv = collections.Counter(aligned=0, unaligned=0,
                              branch_alignment_unknown=0,
                              token_metrics_available=0)
    retained = collections.Counter(); examples = []
    token_keys = ("prompt_nonzero", "assistant_mask_tokens", "assistant_nonzero",
                  "assistant_zero", "assistant_positive", "assistant_negative",
                  "retained_last_assistant_nonzero", "retained_last_assistant_zero",
                  "retained_last_assistant_positive", "retained_last_assistant_negative")
    for r in effective:
        a = _advantage_audit(r, orch[r["trace_id"]])
        if a["alignment"] == "aligned":
            adv["aligned"] += 1
        elif a["alignment"] == "unaligned":
            adv["unaligned"] += 1
        else:
            adv["branch_alignment_unknown"] += 1
        adv["token_metrics_available"] += int(a["alignment"] == "aligned")
        for key in token_keys:
            if a[key] is not None:
                adv[key] += a[key]
        adv["retained_last_invalid_positive_scalar"] += a["retained_last_invalid_positive_scalar"]
        adv["retained_last_invalid_nonpositive_scalar"] += a["retained_last_invalid_nonpositive_scalar"]
        if _retained_invalid(r):
            retained["count"] += 1
            retained["by_step_" + str(orch[r["trace_id"]]["info"]["ship"]["step"])] += 1
            if a["retained_last_invalid_positive_scalar"]:
                retained["positive_scalar_episodes"] += 1
                retained["positive_scalar_aligned_episodes"] += int(a["aligned"] is True)
                if a["retained_last_assistant_positive"] is not None:
                    retained["positive_scalar_aligned_positive_tokens"] += a["retained_last_assistant_positive"]
            if len(examples) < 8: examples.append(r["trace_id"])
    all_valid = sum(bool(r["info"].get("final_canvas_valid")) for r in rows)
    invalid_last_valid = sum(_retained_invalid(r) for r in rows)
    log_root = raw_root / "async-output/text-sft512-consistent-20260930/logs/attempt_1"
    completed = _completed_steps(log_root / "trainer.log")
    shipped = sorted(int(k) for k in step_summary if k != "None")
    source_paths = [raw_root / "reward-trend.jsonl", raw_root / "async-output/text-sft512-consistent-20260930/monitors/file/metrics.jsonl"]
    source_hashes = {str(p.relative_to(raw_root)): _hash(p) for p in source_paths if p.exists()}
    trace_root = raw_root / "async-output/text-sft512-consistent-20260930/monitors/file/traces"
    # Explicit names keep the provenance contract stable for downstream review.
    source_hashes["stream"] = _hash_group((trace_root / "stream").glob("*.jsonl"), raw_root)
    source_hashes["orch"] = _hash_group((trace_root / "annotations/orch").glob("*.jsonl"), raw_root)
    source_hashes["trainer"] = _hash_group((trace_root / "annotations/trainer").glob("*.jsonl"), raw_root)
    source_hashes["receipts"] = _hash_group((raw_root / "rl-rollouts").glob("*/text-judge.json"), raw_root)
    score_values = [_score(r) for r in rows]; score_values = [s for s in score_values if s is not None]
    return {"schema": "reward-learning-analysis-v1", "episodes": len(rows), "effective_episodes": len(effective),
        "valid_final_canvas": all_valid, "valid_final_canvas_rate": all_valid / len(rows) if rows else None,
        "no_canvas": len(rows) - all_valid, "no_canvas_rate": (len(rows) - all_valid) / len(rows) if rows else None,
        "scored_final_canvas": len(score_values),
        "scored_reward_distribution": dict(collections.Counter(str(round(s, 6)) for s in score_values)),
        "valid_final_with_invalid_last_turn": invalid_last_valid,
        "effective_valid_final_with_invalid_last_turn": retained.get("count", 0),
        "effective_status": dict(collections.Counter(r["info"].get("reward_status", "missing") for r in effective)),
        "trainer_annotation_coverage": {"records": len(trainer),
            "effective_with_record": sum(r["trace_id"] in trainer for r in effective)},
        "shipped_steps": shipped, "trainer_completed_steps": completed,
        "step_summary": step_summary, "repeated_prompt_steps": repeated,
        "matched_prompt_group_means": matched_prompt_group_means,
        "fixed_prompt_changes": fixed_prompt_changes,
        "fixed_prompt_comparison": fixed_prompt_comparison,
        "advantage_audit": dict(adv), "retained_last_invalid": dict(retained), "retained_examples": examples,
        "receipts": _receipt_summary(raw_root / "rl-rollouts"), "source_hashes": source_hashes,
        "notes": ["Reward means are admitted effective-batch means; step 20 was shipped but trainer completion is determined from SUCCESS lines.",
                  "A valid retained canvas after an invalid final revision is counted as evidence about terminal-credit lineage, not as a scoring error.",
                  "Prompt-token nonzero advantages are counted only where the annotation vector aligns to flattened node tokens; unaligned vectors are reported separately."]}


def _html(report: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape("—" if value is None else str(value))

    def pct(value: Any) -> str:
        return "—" if value is None else f"{100 * value:.1f}%"

    def dec(value: Any) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    shipped = set(report.get("shipped_steps", []))
    completed = set(report.get("trainer_completed_steps", []))
    not_trained = sorted(shipped - completed)
    status = report.get("effective_status", {})
    rows = []
    curve_points = []
    for key, batch in report.get("step_summary", {}).items():
        step = esc(key)
        rows.append(
            "<tr>"
            f"<td>{step}</td><td>{batch.get('episodes', 0)}</td>"
            f"<td>{dec(batch.get('reward_mean'))}</td>"
            f"<td>{dec(batch.get('valid_only_reward_mean'))}</td>"
            f"<td>{pct((batch.get('invalid_count') or 0) / batch.get('episodes', 1))}</td>"
            f"<td>{esc(batch.get('nonzero_reward_groups'))}/{esc(batch.get('group_count'))}</td>"
            "</tr>"
        )
        try:
            curve_points.append((float(key), batch.get("reward_mean"), batch.get("valid_only_reward_mean")))
        except (TypeError, ValueError):
            pass
    curve_points.sort()
    width, height, pad = 720, 190, 28
    # These scores are relative to frozen SFT. Keep the quality axis fixed so
    # a changing per-run range cannot make a small change look large.
    ymin, ymax = 0.0, 1.0
    def point(step: float, value: Any) -> str:
        x = pad + (step - curve_points[0][0]) / max(curve_points[-1][0] - curve_points[0][0], 1) * (width - 2 * pad)
        clipped = min(max(float(value), ymin), ymax)
        y = height - pad - (clipped - ymin) / (ymax - ymin) * (height - 2 * pad)
        return f"{x:.1f},{y:.1f}"
    all_points = " ".join(point(step, value) for step, value, _ in curve_points if isinstance(value, (int, float)))
    valid_points = " ".join(point(step, value) for step, _, value in curve_points if isinstance(value, (int, float)))
    y0, yhalf, y1 = height - pad, height - pad - (height - 2 * pad) / 2, pad
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Batch reward curves relative to frozen SFT; 0.5 is a tie">'
        f'<rect width="100%" height="100%" fill="#fafafa"/>'
        f'<line x1="{pad}" y1="{y0:.1f}" x2="{width-pad}" y2="{y0:.1f}" stroke="#bbb"/>'
        f'<line x1="{pad}" y1="{yhalf:.1f}" x2="{width-pad}" y2="{yhalf:.1f}" stroke="#999" stroke-dasharray="4 3"/>'
        f'<line x1="{pad}" y1="{y1:.1f}" x2="{width-pad}" y2="{y1:.1f}" stroke="#bbb"/>'
        f'<text x="3" y="{y1 + 4:.1f}">1.0</text><text x="3" y="{yhalf + 4:.1f}">0.5</text><text x="3" y="{y0 + 4:.1f}">0.0</text>'
        f'<polyline points="{all_points}" fill="none" stroke="#3568a8" stroke-width="2"/>'
        f'<polyline points="{valid_points}" fill="none" stroke="#c45b39" stroke-width="2"/>'
        f'<text x="{pad}" y="16" fill="#3568a8">batch reward mean</text>'
        f'<text x="{width-185}" y="16" fill="#c45b39">valid-only mean</text></svg>'
    )
    cards = (
        f'<div class="card"><b>{esc(report.get("episodes"))}</b>episodes</div>'
        f'<div class="card"><b>{esc(report.get("effective_episodes"))}</b>effective</div>'
        f'<div class="card"><b>{esc(status.get("scored", 0))}/{esc(report.get("effective_episodes"))}</b>effective scored</div>'
        f'<div class="card"><b>{esc(report.get("no_canvas"))} ({pct(report.get("no_canvas_rate"))})</b>no canvas</div>'
        f'<div class="card"><b>{esc(len(shipped))} / {esc(len(completed))}</b>shipped / trainer complete</div>'
    )
    return (
        '<!doctype html><meta charset=utf-8><title>Reward learning audit</title>'
        '<style>body{font:15px system-ui,sans-serif;max-width:1100px;margin:2em auto;color:#222}h1{margin-bottom:.25em}.cards{display:flex;gap:.7em;flex-wrap:wrap}.card{background:#f2f4f7;padding:.7em 1em;border-radius:7px;min-width:8em}.card b{display:block;font-size:1.35em}table{border-collapse:collapse;width:100%;margin:1em 0}td,th{border:1px solid #ccc;padding:.35em;text-align:right}td:first-child,th:first-child{text-align:left}small{color:#555}.finding{background:#fff7df;border-left:4px solid #d7a500;padding:.6em 1em}svg{width:100%;max-height:210px;border:1px solid #ddd}a{color:#145a9c}</style>'
        '<h1>Reward learning audit</h1>'
        '<p class="finding"><b>Scope:</b> this is a run audit, not held-out evidence. A shipped step is not necessarily a trainer-completed step.</p>'
        '<p><b>Scores are relative to frozen SFT; 0.5 is a tie.</b> Batch reward includes invalid episodes as zero; it is not itself a valid-painting quality score.</p>'
        f'<div class="cards">{cards}</div>'
        f'<p>Shipped steps without trainer completion: <b>{esc(not_trained)}</b>.</p>'
        f'<h2>Batch reward and valid painting quality</h2>{svg}'
        '<table><thead><tr><th>step</th><th>episodes</th><th>batch reward mean (invalid zero)</th><th>valid-only score mean</th><th>no canvas</th><th>nonzero groups</th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table><p><small>Batch reward uses the admitted effective batch. Valid-only means use scored valid final canvases; invalid-zero means are retained in JSON and are not presented as painting quality. Group counts are fixed-prompt groups inside each ship batch.</small></p>'
        + f'<p><small>Fixed-prompt change means: steps 11–15 {dec(report.get("fixed_prompt_comparison", {}).get("11-15", {}).get("mean_delta"))}; steps 16–20 {dec(report.get("fixed_prompt_comparison", {}).get("16-20", {}).get("mean_delta"))}; descriptive cluster bootstrap delta (16–20 minus 11–15) {esc(report.get("fixed_prompt_comparison", {}).get("descriptive_cluster_bootstrap_95pct"))}.</small></p>'
        f'<h2>Terminal credit and judges</h2><p>{esc(report.get("effective_valid_final_with_invalid_last_turn"))} effective episodes retain a valid canvas after an invalid last turn. The scalar/token audit is available in the JSON; token claims are omitted when branch or mask alignment is unknown.</p>'
        f'<p>Judge receipts: {esc(report.get("receipts", {}).get("receipt_count"))} total, {esc(report.get("receipts", {}).get("initial_conflicts"))} initial conflicts, {esc(report.get("receipts", {}).get("rechecked"))} rechecked.</p>'
        '<p><a href="reward-learning-analysis.json">Full JSON report</a></p>'
        '<p><small>Fixed-prompt comparisons are within-run descriptive changes and are not held-out evidence. The JSON includes a deterministic 1,000-replicate cluster bootstrap interval labeled descriptive.</small></p>'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--html-output", type=Path)
    args = ap.parse_args()
    report = analyze(args.raw_root, args.run_root)
    # Keep generated reports beside the collected evidence. The run root is a
    # tracked documentation surface and should not receive generated output.
    audit_root = args.raw_root.parent / "reward-audit"
    output = args.output or audit_root / "reward-learning-analysis.json"
    html_output = args.html_output or audit_root / "reward-learning-analysis.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    html_output.write_text(_html(report))


if __name__ == "__main__": main()
