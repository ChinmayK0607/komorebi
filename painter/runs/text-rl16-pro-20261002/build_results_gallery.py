#!/usr/bin/env python3
"""Build the offline visual review for the Pro step-12 text-painting run.

This builder only reads preserved evidence.  It verifies the frozen control
image and prompt hashes from ``cases.json`` before copying them, verifies the
PNG hashes recorded in every painting receipt, and writes a self-contained
ignored review tree.  It never renders, calls a model, contacts a service, or
turns an invalid/unfinished turn into a valid result.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: Any) -> str:
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe case id: {value!r}")
    return value


def _copy_verified(source: Path, destination: Path, expected: str) -> str:
    """Copy one ordinary file only after checking its recorded SHA-256."""
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"missing or symlinked evidence file: {source}")
    actual = sha256(source)
    if actual != expected:
        raise ValueError(
            f"hash mismatch for {source}: expected {expected}, got {actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return actual


def _copy_present(source: Path, destination: Path) -> bool:
    if not source.is_file() or source.is_symlink():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return value


def _episode_dir(rollout_root: Path, case_id: str) -> Path | None:
    case_root = rollout_root / case_id
    if not case_root.is_dir() or case_root.is_symlink():
        return None
    episodes = sorted(p for p in case_root.iterdir() if p.is_dir() and not p.is_symlink())
    if len(episodes) > 1:
        raise ValueError(f"multiple episode directories for {case_id}: {episodes}")
    return episodes[0] if episodes else None


def _turn_status(
    receipt: dict[str, Any] | None,
    *,
    program: bool,
    png: bool,
    png_hash_ok: bool,
) -> str:
    if receipt is None:
        if program:
            return "program only · missing receipt"
        if png:
            return "PNG without receipt · unverified"
        return "no saved receipt"
    if receipt.get("valid") is True and png_hash_ok:
        return "valid saved canvas"
    if receipt.get("valid") is True:
        return "valid receipt · missing or unverified PNG"
    elapsed = receipt.get("elapsed_seconds")
    if receipt.get("finished") is None and isinstance(elapsed, (int, float)) and elapsed >= 179.5:
        return "hard render timeout · invalid"
    if png_hash_ok:
        return "invalid receipt · partial PNG"
    if png:
        return "invalid receipt · PNG hash not verified"
    return "invalid receipt · no PNG"


def _turn_label(status: str) -> str:
    return status.replace(" · ", " — ")


def _copy_turn_artifacts(
    turn_dir: Path | None,
    *,
    case_id: str,
    turn_number: int,
    assets_root: Path,
) -> dict[str, Any]:
    """Copy raw turn evidence and classify the saved canvas conservatively."""
    if turn_dir is None or not turn_dir.is_dir():
        return {
            "turn": turn_number,
            "status": "no turn folder saved",
            "receipt": None,
            "program": None,
            "png": None,
            "png_sha256": None,
            "png_hash_ok": False,
            "valid_saved": False,
            "invalid_png": False,
            "receipt_data": None,
        }

    program_source = turn_dir / "program.js"
    receipt_source = turn_dir / "painting.json"
    png_source = turn_dir / "painting.png"
    receipt = _load_json(receipt_source)
    png_actual = sha256(png_source) if png_source.is_file() and not png_source.is_symlink() else None
    expected_png = receipt.get("png_sha256") if receipt else None
    png_hash_ok = bool(png_actual and expected_png and png_actual == expected_png)
    valid_saved = bool(receipt and receipt.get("valid") is True and png_hash_ok)
    invalid_png = bool(receipt and receipt.get("valid") is not True and png_hash_ok)

    base = assets_root / "rl" / case_id / f"turn-{turn_number}"
    program_dest = base / "program.js" if _copy_present(program_source, base / "program.js") else None
    receipt_dest = base / "painting.json" if _copy_present(receipt_source, base / "painting.json") else None
    png_dest: Path | None = None
    if png_source.is_file() and not png_source.is_symlink():
        # Every collected PNG has a receipt hash.  If a future run has an
        # unbound PNG, preserve the evidence under an explicit unverified name.
        if valid_saved:
            png_name = "painting.png"
        elif invalid_png:
            png_name = "painting-invalid.png"
        else:
            png_name = "painting-unverified.png"
        png_dest = base / png_name
        _copy_present(png_source, png_dest)

    status = _turn_status(
        receipt,
        program=program_dest is not None,
        png=png_dest is not None,
        png_hash_ok=png_hash_ok,
    )
    return {
        "turn": turn_number,
        "status": status,
        "receipt": str(receipt_dest.relative_to(assets_root.parent).as_posix()) if receipt_dest else None,
        "program": str(program_dest.relative_to(assets_root.parent).as_posix()) if program_dest else None,
        "png": str(png_dest.relative_to(assets_root.parent).as_posix()) if png_dest else None,
        "png_sha256": png_actual,
        "png_hash_ok": png_hash_ok,
        "valid_saved": valid_saved,
        "invalid_png": invalid_png,
        "receipt_data": receipt,
    }


def _href(value: str | None) -> str:
    return html.escape(value or "", quote=True)


def _text(value: Any, fallback: str = "—") -> str:
    return html.escape(fallback if value is None else str(value))


def _image(path: str | None, alt: str, *, class_name: str = "") -> str:
    if not path:
        return '<div class="missing"><b>No saved valid result</b><span>This case has no hash-verified valid RL canvas.</span></div>'
    classes = f' class="{html.escape(class_name)}"' if class_name else ""
    return (
        f'<a class="zoomable" href="{_href(path)}" data-full="{_href(path)}" '
        f'aria-label="Open full size: {html.escape(alt)}">'
        f'<img{classes} src="{_href(path)}" alt="{html.escape(alt)}" loading="lazy"></a>'
    )


def _turn_card(turn: dict[str, Any], *, last_valid: int | None, last_attempt: int | None) -> str:
    n = turn["turn"]
    status = turn["status"]
    if status == "valid saved canvas":
        kind = "valid"
    elif "invalid" in status or "partial" in status or "unverified" in status:
        kind = "invalid"
    elif "program only" in status or "missing" in status:
        kind = "missing"
    else:
        kind = "missing"
    links: list[str] = []
    if turn.get("receipt"):
        links.append(f'<a href="{_href(turn["receipt"])}">full receipt JSON</a>')
    if turn.get("program"):
        links.append(f'<a href="{_href(turn["program"])}">program.js</a>')
    if turn.get("png"):
        links.append(f'<a href="{_href(turn["png"])}">PNG evidence</a>')
    links_html = " · ".join(links) if links else '<span class="muted">no saved files</span>'
    body = ""
    if turn.get("png"):
        caption = "valid saved canvas" if turn.get("valid_saved") else "invalid partial PNG · not used as valid"
        body = (
            f'<a class="zoomable thumb-link" href="{_href(turn["png"])}" data-full="{_href(turn["png"])}">'
            f'<img class="thumb" src="{_href(turn["png"])}" alt="Turn {n}: {html.escape(caption)}" loading="lazy"></a>'
            f'<p class="thumb-caption">{html.escape(caption)}</p>'
        )
    else:
        body = '<div class="turn-empty">No PNG saved</div>'
    marker = ""
    if last_valid == n:
        marker += '<span class="turn-marker good">last valid saved</span>'
    if last_attempt == n:
        marker += '<span class="turn-marker latest">last attempted</span>'
    return (
        f'<article class="turn {kind}"><h4>Turn {n} {marker}</h4>'
        f'<p class="turn-status">{html.escape(_turn_label(status))}</p>{body}'
        f'<p class="turn-links">{links_html}</p></article>'
    )


def _case_card(row: dict[str, Any]) -> str:
    case_id = row["id"]
    control_alt = f"{case_id} frozen SFT step 512 control"
    last_valid = row["last_valid_turn"]
    last_attempt = row["last_attempt_turn"]
    if last_valid is not None:
        rl_caption = f"RL: Pro judge, 12 new updates · last valid saved canvas · turn {last_valid}"
        rl_note = "This retained canvas is the latest turn with a valid receipt and matching PNG hash."
        rl_image = _image(row["last_valid_png"], f"{row['id']} RL last valid saved canvas turn {last_valid}")
    else:
        rl_caption = "RL: Pro judge, 12 new updates · no saved valid result"
        rl_note = "No valid saved RL canvas was found for this case; partial or invalid PNGs remain in the turn strip below."
        rl_image = _image(None, f"{row['id']} RL no saved valid result")
    if last_attempt is None:
        attempt = "No attempted turn recorded"
    else:
        latest = next(t for t in row["turns"] if t["turn"] == last_attempt)
        attempt = f"Last attempted turn {last_attempt} · {latest['status']}"
    if last_valid is not None and last_attempt is not None and last_attempt > last_valid:
        attempt += f" · main image remains turn {last_valid}; later attempts are not substituted"
    return (
        f'<article class="case" id="case-{_href(case_id)}">'
        f'<header class="case-head"><div><p class="eyebrow">{_text(row.get("category"), "case").upper()}</p>'
        f'<h2>{_text(case_id)} <span>{_text(row.get("family"))}</span></h2></div>'
        f'<p class="attempt"><b>{html.escape(attempt)}</b></p></header>'
        f'<p class="prompt"><b>Prompt</b><br>{_text(row.get("prompt_text"), "Prompt unavailable")}</p>'
        '<div class="pair">'
        '<figure class="panel control"><figcaption><strong>Frozen SFT step 512</strong><span>fixed left control</span></figcaption>'
        f'{_image(row["control"], control_alt)}'
        f'<p class="panel-note">SHA-256 verified against the case manifest.</p></figure>'
        '<figure class="panel rl"><figcaption><strong>'
        f'{html.escape(rl_caption)}</strong><span>fixed right continuation</span></figcaption>{rl_image}'
        f'<p class="panel-note">{html.escape(rl_note)}</p></figure>'
        '</div>'
        f'<p class="attempt-detail">{html.escape(attempt)}</p>'
        f'<section class="turns"><h3>Turn evidence</h3><div class="turn-grid">'
        f'{"".join(_turn_card(t, last_valid=last_valid, last_attempt=last_attempt) for t in row["turns"])}'
        '</div></section></article>'
    )


def _page(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    nav = " ".join(
        f'<a href="#case-{_href(row["id"])}">{_text(row["id"])}</a>' for row in rows
    )
    stat_cards = "".join(
        f'<div class="stat"><b>{html.escape(str(summary[key]))}</b><span>{html.escape(label)}</span></div>'
        for key, label in (
            ("cases", "cases"),
            ("receipts", "saved receipts"),
            ("receipt_cases", "prompts with receipts"),
            ("valid_pngs", "valid PNG receipts"),
            ("valid_cases", "with any valid canvas"),
            ("no_valid_cases", "with no saved valid result"),
            ("invalid_runtime_receipts", "invalid runtime receipts"),
            ("hard_timeouts", "hard render timeouts"),
        )
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pro step-12 painting results · fixed visual review</title>
<style>
:root{{--ink:#18212b;--muted:#627182;--paper:#f4f6f8;--card:#fff;--line:#dce3ea;--blue:#135ca8;--good:#176b4c;--warn:#a45113;--bad:#a32222}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.48 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px 24px 70px}} h1{{font-size:clamp(2rem,4vw,3.2rem);line-height:1.04;margin:.2rem 0 .8rem;letter-spacing:-.035em}}
h2,h3,h4,p{{margin-top:0}} h2{{font-size:1.25rem;margin-bottom:.1rem}} h2 span{{font-size:.75rem;font-weight:500;color:var(--muted);margin-left:.35rem}}
h3{{font-size:1rem;margin:1.1rem 0 .6rem}} h4{{font-size:.95rem;margin:0 0 .2rem}} a{{color:var(--blue)}}
.lede{{font-size:1.04rem;max-width:1050px;color:#2e3f50}} .scope{{background:#fff8e7;border:1px solid #eed28c;border-left:5px solid #d99b00;padding:1rem 1.1rem;border-radius:8px;max-width:1250px}}
.scope b{{color:#5f4100}} .stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0 16px}} .stat{{background:var(--card);border:1px solid var(--line);padding:12px 14px;border-radius:9px}}
.stat b{{display:block;font-size:1.55rem;line-height:1.05}} .stat span{{color:var(--muted);font-size:.87rem}} .nav{{background:var(--card);border:1px solid var(--line);padding:10px 13px;border-radius:8px;line-height:2;max-height:125px;overflow:auto}} .nav a{{white-space:nowrap;margin-right:.7rem}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px;margin:22px 0;box-shadow:0 3px 16px #2132440b;scroll-margin-top:18px}} .case-head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:12px}}
.eyebrow{{color:var(--blue);font-size:.73rem;font-weight:750;letter-spacing:.08em;margin:0 0 .15rem}} .attempt{{text-align:right;font-size:.87rem;color:var(--warn);max-width:48%;margin:0}} .prompt{{background:#f7f9fb;border-left:4px solid #8fa7bf;padding:10px 13px;white-space:pre-wrap;margin-bottom:15px}} .prompt b{{font-size:.75rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}}
.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .panel{{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fbfcfd}} .panel figcaption{{display:flex;justify-content:space-between;gap:10px;padding:11px 12px;background:#eef3f7;align-items:baseline}} .panel figcaption strong{{font-size:1rem}} .panel figcaption span{{font-size:.78rem;color:var(--muted);white-space:nowrap}} .panel .zoomable{{display:block;background:#e9edf1}} .panel img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#fff;cursor:zoom-in}} .panel-note{{color:var(--muted);font-size:.83rem;padding:8px 12px;margin:0;min-height:2.6em}}
.missing{{aspect-ratio:1;display:grid;place-content:center;justify-items:center;text-align:center;padding:25px;background:#f3f0ec;color:#7a5c44;gap:5px}} .missing b{{font-size:1.05rem}} .missing span{{font-size:.85rem;max-width:260px}} .rl .missing{{background:#fff4e8;color:#8b4e22}} .attempt-detail{{background:#fff5ea;border-left:4px solid #d97822;padding:8px 11px;color:#7f3f15;margin:12px 0 0}}
.turns{{margin-top:15px}} .turn-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}} .turn{{border:1px solid var(--line);border-top:4px solid #9aa9b7;border-radius:8px;padding:10px;background:#fff}} .turn.valid{{border-top-color:var(--good)}} .turn.invalid{{border-top-color:var(--bad);background:#fffafa}} .turn.missing{{border-top-color:#c28620;background:#fffcf4;aspect-ratio:auto;display:block;text-align:left;color:var(--ink);padding:10px}} .turn h4{{display:flex;flex-wrap:wrap;gap:5px;align-items:center}} .turn-status{{font-size:.82rem;color:var(--muted);min-height:2.35em;margin-bottom:7px}} .thumb-link,.turn .zoomable{{display:block;background:#eef1f4}} .thumb{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#fff;cursor:zoom-in}} .thumb-caption{{color:var(--bad);font-size:.75rem;margin:5px 0 0}} .turn.valid .thumb-caption{{color:var(--good)}} .turn-empty{{display:grid;place-items:center;min-height:120px;background:#f1f3f5;color:var(--muted);font-size:.82rem}} .turn-links{{font-size:.78rem;margin:8px 0 0;line-height:1.7}} .muted{{color:var(--muted)}} .turn-marker{{font-size:.68rem;padding:2px 5px;border-radius:4px;font-weight:650}} .turn-marker.good{{background:#dff4ea;color:var(--good)}} .turn-marker.latest{{background:#ffe3bd;color:#8b4e22}}
.foot{{color:var(--muted);font-size:.84rem;margin-top:30px}} .lightbox{{position:fixed;inset:0;background:#07111dcc;display:none;place-items:center;padding:4vh 4vw;z-index:10}} .lightbox.open{{display:grid}} .lightbox img{{max-width:92vw;max-height:88vh;object-fit:contain;background:#fff;box-shadow:0 12px 45px #0008}} .lightbox button{{position:fixed;top:15px;right:18px;border:0;border-radius:50%;width:40px;height:40px;font-size:1.4rem;cursor:pointer;background:#fff;color:#111}} .lightbox p{{position:fixed;bottom:7px;color:#fff;text-align:center;margin:0;max-width:90vw}}
@media(max-width:900px){{.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}} .turn-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}} .case-head{{display:block}} .attempt{{text-align:left;max-width:none;margin-top:7px}}}}
@media(max-width:650px){{main{{padding:20px 12px 45px}} .pair,.turn-grid{{grid-template-columns:1fr}} .stats{{grid-template-columns:repeat(2,minmax(0,1fr))}} .panel figcaption{{display:block}} .panel figcaption span{{display:block;margin-top:3px}}}}
</style></head><body><main>
<p class="eyebrow">VISUAL EVIDENCE · 2026-10-02</p><h1>Pro step-12 painting results</h1>
<p class="lede">A fixed, side-by-side gallery for all 47 prompts. The left image is always the frozen SFT step 512 control. The right image is always the RL Pro continuation after 12 new updates, initialized from the preferred earlier RL step 16 adapter. The right panel keeps the last valid saved canvas for that case; later invalid, partial, or receipt-missing attempts stay visible in the turn evidence and are never substituted into the main panel.</p>
<p class="scope"><b>Run boundary:</b> the harness reported 45/47 completed episodes before its global timeout. This page does not treat episode completion as painting validity. It records 99 saved receipts across 46 prompts, 76 PNGs whose hashes match valid receipts, and 41 prompts with at least one valid saved canvas. There are 15 invalid runtime receipts and 8 hard render timeouts. No reward score or quality verdict is shown here.</p>
<div class="stats">{stat_cards}</div><nav class="nav" aria-label="Cases">{nav}</nav>
{"".join(_case_card(row) for row in rows)}
<p class="foot">Controls and prompts were copied only after SHA-256 verification against the supplied case manifest. Every turn links to the copied program, receipt JSON, and available PNG evidence. Invalid partial PNGs are labelled and remain excluded from the retained valid canvas.</p>
</main><div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="Enlarged painting"><button type="button" id="close" aria-label="Close">×</button><img id="lightbox-image" alt=""><p id="lightbox-caption"></p></div>
<script>
const box=document.getElementById('lightbox'),big=document.getElementById('lightbox-image'),cap=document.getElementById('lightbox-caption');
document.querySelectorAll('.zoomable').forEach((link)=>link.addEventListener('click',(event)=>{{event.preventDefault();big.src=link.dataset.full||link.href;big.alt=link.getAttribute('aria-label')||'';cap.textContent=big.alt;box.classList.add('open')}}));
function closeBox(){{box.classList.remove('open');big.removeAttribute('src')}}
document.getElementById('close').addEventListener('click',closeBox); box.addEventListener('click',(event)=>{{if(event.target===box)closeBox()}}); document.addEventListener('keydown',(event)=>{{if(event.key==='Escape')closeBox()}});
</script></body></html>'''


def build_gallery(
    manifest_path: Path,
    rollout_root: Path,
    anchors_root: Path,
    prompts_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    rollout_root = rollout_root.expanduser().resolve()
    anchors_root = anchors_root.expanduser().resolve()
    prompts_root = prompts_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("manifest must contain a cases list")
    output_root.mkdir(parents=True, exist_ok=True)
    assets_root = output_root / "assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)
    assets_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    receipt_count = 0
    receipt_cases = 0
    valid_pngs = 0
    valid_cases = 0
    invalid_runtime_receipts = 0
    hard_timeouts = 0
    invalid_receipts = 0
    program_only_turns = 0
    missing_turns = 0

    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("manifest case is not an object")
        case_id = _safe_id(raw.get("id"))
        control = anchors_root / f"{case_id}.png"
        prompt = prompts_root / f"{case_id}.txt"
        control_dest = assets_root / "controls" / f"{case_id}.png"
        prompt_dest = assets_root / "prompts" / f"{case_id}.txt"
        _copy_verified(control, control_dest, raw["opponent_sha256"])
        _copy_verified(prompt, prompt_dest, raw["reference_sha256"])
        prompt_text = prompt.read_text(encoding="utf-8").strip()
        episode = _episode_dir(rollout_root, case_id)
        turns: list[dict[str, Any]] = []
        for turn_number in range(1, int(raw.get("max_turns", 3)) + 1):
            turn_dir = episode / f"turn-{turn_number}" if episode else None
            row_turn = _copy_turn_artifacts(
                turn_dir,
                case_id=case_id,
                turn_number=turn_number,
                assets_root=assets_root,
            )
            turns.append(row_turn)
            receipt = row_turn.get("receipt_data")
            if receipt is not None:
                receipt_count += 1
                if receipt.get("valid") is True and row_turn.get("valid_saved"):
                    valid_pngs += 1
                if receipt.get("valid") is not True:
                    invalid_receipts += 1
                    if receipt.get("finished") is False:
                        invalid_runtime_receipts += 1
                    elapsed = receipt.get("elapsed_seconds")
                    if receipt.get("finished") is None and isinstance(elapsed, (int, float)) and elapsed >= 179.5:
                        hard_timeouts += 1
            elif row_turn.get("status") == "program only · missing receipt":
                program_only_turns += 1
            elif row_turn.get("status") == "no turn folder saved":
                missing_turns += 1
        if any(t.get("receipt_data") is not None for t in turns):
            receipt_cases += 1
        valid_turns = [t for t in turns if t.get("valid_saved")]
        last_valid = max(valid_turns, key=lambda t: t["turn"]) if valid_turns else None
        attempted = [
            t for t in turns
            if t.get("status") != "no turn folder saved"
            and any(t.get(key) for key in ("receipt", "program", "png"))
        ]
        last_attempt = max(attempted, key=lambda t: t["turn"]) if attempted else None
        if last_valid:
            valid_cases += 1
        rows.append({
            "id": case_id,
            "category": raw.get("category"),
            "family": raw.get("family"),
            "phase": raw.get("phase"),
            "max_turns": raw.get("max_turns", 3),
            "prompt_text": prompt_text,
            "control": str(control_dest.relative_to(output_root).as_posix()),
            "control_sha256": raw["opponent_sha256"],
            "prompt": str(prompt_dest.relative_to(output_root).as_posix()),
            "prompt_sha256": raw["reference_sha256"],
            "turns": turns,
            "last_valid_turn": last_valid["turn"] if last_valid else None,
            "last_valid_png": last_valid.get("png") if last_valid else None,
            "last_attempt_turn": last_attempt["turn"] if last_attempt else None,
        })

    rows.sort(key=lambda row: row["id"])
    summary = {
        "cases": len(rows),
        "receipts": receipt_count,
        "receipt_cases": receipt_cases,
        "valid_pngs": valid_pngs,
        "valid_cases": valid_cases,
        "no_valid_cases": len(rows) - valid_cases,
        "invalid_receipts": invalid_receipts,
        "invalid_runtime_receipts": invalid_runtime_receipts,
        "hard_timeouts": hard_timeouts,
        "program_only_turns": program_only_turns,
        "missing_turns": missing_turns,
        "harness_completed_episodes": 45,
        "harness_episode_total": 47,
    }
    (output_root / "index.html").write_text(_page(rows, summary), encoding="utf-8")
    (output_root / "gallery.json").write_text(
        json.dumps({"schema": "pro-step12-results-gallery-v1", "summary": summary, "cases": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"output": str(output_root / "index.html"), "summary": summary, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_gallery(
        manifest_path=args.manifest,
        rollout_root=args.rollout_root,
        anchors_root=args.anchors,
        prompts_root=args.prompts,
        output_root=args.output,
    )
    print(json.dumps({"output": report["output"], "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
