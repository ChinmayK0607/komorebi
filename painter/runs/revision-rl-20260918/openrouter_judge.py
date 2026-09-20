"""Blinded visual pair grading through an OpenRouter vision model.

The API key is read only from OPENROUTER_API_KEY and is never written to the
run directory.  Each orientation is graded independently; reward_queue.py
later reconciles the reversed views and rejects inconsistent judgments.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import getpass
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "inclusionai/ling-3.0-flash-vl:free"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def image_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def parse_json(text: str) -> dict:
    if text is None:
        raise ValueError("judge returned no visible content")
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("judge did not return a JSON object")
        text = text[start:end + 1]
    row = json.loads(text)
    required = {"winner", "confidence", "acceptable_A", "acceptable_B", "reason"}
    if not required.issubset(row):
        raise ValueError(f"missing fields: {sorted(required - set(row))}")
    if row["winner"] not in {"A", "B", "tie", "uncertain"}:
        raise ValueError("invalid winner")
    if row["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("invalid confidence")
    for side in ("A", "B"):
        if row[f"acceptable_{side}"] not in {"yes", "no", "uncertain"}:
            raise ValueError("invalid acceptability")
    if not str(row["reason"]).strip():
        raise ValueError("empty reason")
    return {k: row[k] for k in required}


def judge_one(pair: dict, api_key: str, model: str, attempts: int = 5) -> dict:
    prompt = (
        "You are a strict visual-quality comparator for a painting RL task. "
        "The first image is the reference; A and B are two candidate canvases. "
        "Judge only the visible final paintings, never code validity or token length. "
        "A candidate is acceptable only when the main subject, silhouette, placement, "
        "connected parts, proportions, and completion are faithful to the reference. "
        "Prefer the candidate with better overall reference fidelity and attractive "
        "brushwork. Be conservative: a pretty but structurally wrong image is not "
        "acceptable. Return ONLY one JSON object with exactly these fields: "
        "winner (A|B|tie|uncertain), confidence (high|medium|low), "
        "acceptable_A (yes|no|uncertain), acceptable_B (yes|no|uncertain), "
        "reason (short concrete explanation)."
    )
    content = [{"type": "text", "text": prompt}]
    for label in ("reference", "A", "B"):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_url(Path(pair[label]))}})
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        # This VL model may spend hundreds of tokens on internal reasoning
        # before emitting JSON; a small cap returns content=null instead.
        "max_tokens": 8192,
        "reasoning": {"effort": "low", "exclude": True},
        "stream": False,
    }).encode()
    request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PrimeIntellect-ai/prime-rl",
        "X-Title": "brush-painting-rl-visual-judge",
    })
    last = None
    for attempt in range(attempts):
        try:
            global _LAST_REQUEST
            with _RATE_LOCK:
                interval = float(os.environ.get("OPENROUTER_MIN_INTERVAL", "5"))
                wait = interval - (time.monotonic() - _LAST_REQUEST)
                if wait > 0:
                    time.sleep(wait)
                _LAST_REQUEST = time.monotonic()
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            text = payload["choices"][0]["message"]["content"]
            return parse_json(text)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                break
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                time.sleep(max(30, int(exc.headers.get("Retry-After", "0") or 0)))
            else:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"judge failed for {pair['id']}: {last}")


def run_batch(batch: Path, key: str, model: str, workers: int, attempts: int = 5) -> dict:
    review = json.loads((batch / "review-input.json").read_text())
    outputs = sorted(batch.glob("judge-*.json"))
    total = 0
    for index, source in enumerate(outputs, 1):
        target = batch / f"labels-{index:02d}.json"
        pairs = json.loads(source.read_text())
        saved = {}
        if target.exists():
            existing = json.loads(target.read_text())
            if existing.get('judge_model') != model:
                raise ValueError(f'Judge identity mismatch: {target}')
            saved = {row['id']: row for row in existing.get('judgments', [])}
            if not set(saved).issubset({pair['id'] for pair in pairs}):
                raise ValueError(f'Unexpected cached pair: {target}')
            for pair in pairs:
                if pair['id'] in saved:
                    parse_json(json.dumps(saved[pair['id']]))
                    if saved[pair['id']]['reviewed_images'] != [pair[x] for x in ('reference', 'A', 'B')]:
                        raise ValueError(f'Cached image mismatch: {pair["id"]}')
        judgments = [saved.get(pair['id']) for pair in pairs]
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(judge_one, pair, key, model, attempts): i
                       for i, pair in enumerate(pairs) if judgments[i] is None}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    judgments[i] = {"id": pairs[i]["id"], **future.result(),
                                    "reviewed_images": [pairs[i][x] for x in ("reference", "A", "B")]}
                    temporary = target.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'judge_model': model,
                        'judgments': [row for row in judgments if row is not None]}, indent=2) + '\n')
                    temporary.replace(target)
                    print(json.dumps({'completed': pairs[i]['id'], 'saved': sum(row is not None for row in judgments),
                                      'total': len(pairs)}), flush=True)
                except Exception as exc:
                    errors.append(f"{pairs[i]['id']}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
        target.write_text(json.dumps({"judge_model": model, "judgments": judgments}, indent=2) + "\n")
        total += len(judgments)
    return {"batch": batch.name, "pairs": total, "model": model}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--prompt-key", action="store_true", help="Read the API key without terminal echo")
    args = parser.parse_args()
    key = getpass.getpass('OpenRouter API key: ') if args.prompt_key else os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required and is never stored by this script")
    print(json.dumps(run_batch(args.batch, key, args.model, max(1, args.workers), max(1, args.attempts))))


if __name__ == "__main__":
    main()
