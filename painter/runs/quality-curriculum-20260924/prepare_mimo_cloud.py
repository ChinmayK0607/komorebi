#!/usr/bin/env python3
"""Stage hash-verified training references for finite MiMo cloud episodes."""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
BENCH = ROOT / "painter/benchmarks/openrouter-teachers-20260922"
OUT = ROOT / "painter/collected/quality-curriculum-20260924/mimo-cloud"
DATASET = "CK0607/komorebi-painter-teachers"
REF_REVISION = "076a548be0b731c6a33f31749c905ade8b6546de"
HARD_REVISION = "2a379e4f866ff1bb3e185e89a59ca7fc28367f7e"
COCO_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    for attempt in range(6):
        try:
            with urlopen(url, timeout=180) as response:
                return response.read()
        except (OSError, TimeoutError, HTTPException):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def stage(tier: str, model: str, stage_name: str | None = None) -> Path:
    if tier not in {"easy", "hard"}:
        raise ValueError("tier must be easy or hard")
    if model not in {"xiaomi/mimo-v2.6-flash", "xiaomi/mimo-v2.6-pro"}:
        raise ValueError("unexpected teacher model")
    manifest_path = RUN / "reference-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = [r for r in manifest["references"] if (r["tier"] == "manual hard") == (tier == "hard")]
    if len(rows) != (10 if tier == "hard" else 7):
        raise ValueError("reference manifest has unexpected tier count")
    benchmark = json.loads((BENCH / "refs.json").read_text())
    holdout_ids = {r["id"] for r in benchmark["references"]}
    holdout_hashes = {r["sha256"] for r in benchmark["references"]}
    if any(r["id"] in holdout_ids or r["sha256"] in holdout_hashes or r["split"] != "train" for r in rows):
        raise ValueError("training reference overlaps benchmark holdout or is not train split")
    stage_name = stage_name or f"{tier}-{model.rsplit('-', 1)[-1]}"
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", stage_name):
        raise ValueError("stage name must be a safe relative directory name")
    dest = OUT / stage_name
    (dest / "references").mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows:
        ident = row["id"]
        suffix = ".png" if tier == "easy" else ".jpg"
        revision = REF_REVISION if tier == "easy" else HARD_REVISION
        url = (f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/"
               f"curricula/quality-20260924/references/{ident}{suffix}?download=true")
        data = fetch(url)
        if sha(data) != row["sha256"]:
            raise ValueError(f"reference hash mismatch: {ident}")
        target = dest / "references" / f"{ident}{suffix}"
        target.write_bytes(data)
        selected.append({"id": ident, "image": f"references/{target.name}",
                         "sha256": row["sha256"], "split": "train", "category": row["description"],
                         "tier": row["tier"], "source_kind": row["source_kind"]})
    config = {
        "benchmark": f"quality-curriculum-{tier}-{model.rsplit('-', 1)[-1]}-20260924",
        "models": [model],
        "tracks": {"quality": {"max_turns": 6 if tier == "easy" else 12,
                               "max_tokens": "native", "episode_timeout_seconds": None,
                               "reasoning_effort": "highest_supported"}},
        "temperature": 0.7, "concurrency": 2, "render_concurrency": 1,
        "renderer_timeout": 600, "samples_per_image": 1,
        "catalog": "model-catalog.json", "transport_script": "gateway_transport.ts",
        "references": "refs.json", "prompt": "prompt.txt",
        "request_timeout_seconds": 3600, "max_retries": 2,
    }
    (dest / "refs.json").write_text(json.dumps({"version": 1, "count": len(selected), "references": selected}, indent=2, sort_keys=True) + "\n")
    (dest / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(RUN / "quality-prompt.txt", dest / "prompt.txt")
    for name in ("contract.json", "model-catalog.json", "gateway_transport.ts",
                 "gateway_prompt.ts", "package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml", "tsconfig.json"):
        shutil.copyfile(BENCH / name, dest / name)
    source = {"schema": "painter.mimo-curriculum-source.v1", "tier": tier, "model": model,
              "reference_manifest_sha256": sha(manifest_bytes),
              "public_easy_revision": REF_REVISION if tier == "easy" else None,
              "public_hard_revision": HARD_REVISION if tier == "hard" else None,
              "official_coco128_sha256": COCO_SHA256 if tier == "hard" else None,
              "reference_ids": [r["id"] for r in rows], "status": "candidate_generation_only"}
    (dest / "restored-source.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage_root": str(dest), "tier": tier, "model": model,
                      "references": len(selected), "manifest_sha256": source["reference_manifest_sha256"]}), flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("easy", "hard"), required=True)
    parser.add_argument("--model", choices=("xiaomi/mimo-v2.6-flash", "xiaomi/mimo-v2.6-pro"), required=True)
    parser.add_argument("--stage-name", help="unique stage directory for a fresh probe; default keeps historical behavior")
    args = parser.parse_args()
    stage(args.tier, args.model, args.stage_name)


if __name__ == "__main__":
    main()
