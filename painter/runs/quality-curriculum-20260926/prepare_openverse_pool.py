#!/usr/bin/env python3
"""Collect a small, diverse CC0 photo pool via the Openverse image API.

Search receipts and downloaded thumbnails remain outside Git. The thumbnail
hash, creator/source URL, and explicit CC0 license travel with every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "painter/collected/quality-curriculum-20260926/openverse-cc0"
API = "https://api.openverse.org/v1/images/"
QUERIES = (
    "red bus street", "wild horse field", "shorebird coast", "cat window",
    "dog snow", "bicycle street", "sailboat harbor", "steam train landscape",
    "forest waterfall", "mountain lake", "desert road", "coastal cliff",
    "city street rain", "brick building facade", "wooden cabin", "modern interior room",
    "ceramic vase still life", "teapot table", "fruit market", "bread bakery",
    "flowers in vase", "farm tractor", "bridge river", "person silhouette sunset",
    "sports court", "airplane runway", "garden path", "fishing boat harbor",
)
HEADERS = {"User-Agent": "KomorebiPainterResearch/1.0 (openly licensed photo selection)"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def request_bytes(url: str, timeout: int = 60) -> bytes:
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=timeout) as response:
                return response.read(5_000_001)
        except HTTPError as error:
            if error.code != 429 or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def collect(output: Path, train_per_query: int = 5, queries: tuple[str, ...] = QUERIES) -> dict:
    existing = output / "reference-pool.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if (old.get("search_queries") != list(queries)
                or old.get("train_per_query", 5) != train_per_query):
            raise ValueError("source-pool output already belongs to a different query selection")
    cache = output / "search-receipts"
    cache.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    entries = []
    for index, query in enumerate(queries, 1):
        params = urlencode({"q": query, "license": "cc0", "page_size": 20, "page": 1})
        url = API + "?" + params
        cache_file = cache / f"{index:02d}.json"
        if cache_file.exists():
            raw = cache_file.read_bytes()
        else:
            raw = request_bytes(url)
            cache_file.write_bytes(raw)
        data = json.loads(raw)
        accepted = []
        for item in data.get("results", []):
            if (item.get("id") in seen_ids or item.get("license") != "cc0"
                    or item.get("mature") or item.get("category") != "photograph"
                    or not item.get("thumbnail") or not item.get("foreign_landing_url")):
                continue
            title = (item.get("title") or "").strip()
            title_key = (title.casefold(), (item.get("creator") or "").casefold())
            if not title or title_key in seen_titles:
                continue
            seen_ids.add(item["id"])
            seen_titles.add(title_key)
            accepted.append(item)
            if len(accepted) >= train_per_query + 1:
                break
        for position, item in enumerate(accepted):
            entries.append({
                "id": f"openverse-{item['id']}", "split": "train" if position < train_per_query else "heldout",
                "query": query, "title": item["title"], "creator": item.get("creator"),
                "creator_url": item.get("creator_url"), "attribution": item.get("attribution"),
                "source_url": item["foreign_landing_url"], "provider": item.get("provider"),
                "thumbnail_url": item["thumbnail"], "license": item["license"],
                "license_url": item.get("license_url"), "license_version": item.get("license_version"),
                "tags": [tag["name"] for tag in (item.get("tags") or [])[:12]],
                "local_reference": f"references/{item['id']}.jpg", "review_status": "unreviewed_source",
                "search_receipt_sha256": sha(raw),
            })
        print(f"query {index}/{len(queries)} {query}: {len(accepted)} unique CC0 photos", flush=True)
    manifest = {
        "schema": "painter.openverse-cc0-reference-pool.v1", "api": API,
        "search_queries": list(queries), "license_filter": "cc0",
        "train_count": sum(row["split"] == "train" for row in entries),
        "heldout_count": sum(row["split"] == "heldout" for row in entries),
        "entries": entries,
    }
    if queries != QUERIES or train_per_query != 5:
        manifest["train_per_query"] = train_per_query
    (output / "reference-pool.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def download(manifest: dict, output: Path, count: int) -> dict:
    queries = manifest["search_queries"]
    by_query: dict[str, list[dict]] = {query: [] for query in queries}
    for row in manifest["entries"]:
        if row["split"] == "train":
            by_query[row["query"]].append(row)
    interleaved = [row for offset in range(manifest.get("train_per_query", 5)) for query in queries
                   for row in by_query[query][offset:offset + 1]]
    rows = interleaved[:count]
    refs = output / "references"
    refs.mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, row in enumerate(rows, 1):
        target = output / row["local_reference"]
        raw = target.read_bytes() if target.exists() else request_bytes(row["thumbnail_url"])
        if len(raw) > 5_000_000 or not raw.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"Openverse thumbnail is not a bounded JPEG: {row['id']}")
        if not target.exists():
            target.write_bytes(raw)
        receipts.append({"id": row["id"], "path": row["local_reference"],
                         "bytes": len(raw), "sha256": sha(raw), "thumbnail_url": row["thumbnail_url"]})
        print(f"reference {index}/{len(rows)} {row['id']} {len(raw)} bytes", flush=True)
    receipt = {"schema": "painter.openverse-cc0-download.v1", "count": len(receipts), "files": receipts}
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    (output / f"download-receipt-n{len(receipts)}.json").write_text(payload)
    (output / "download-receipt.json").write_text(payload)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--download-count", type=int, default=0)
    parser.add_argument("--queries-json", type=Path,
                        help="JSON array of search queries for a separate source pool")
    parser.add_argument("--train-per-query", type=int, default=5)
    args = parser.parse_args()
    queries = tuple(json.loads(args.queries_json.read_text())) if args.queries_json else QUERIES
    if (not queries or len(set(queries)) != len(queries)
            or any(not isinstance(query, str) or not query.strip() for query in queries)):
        parser.error("queries must be a nonempty array of unique nonempty strings")
    if not 1 <= args.train_per_query <= 10:
        parser.error("train-per-query must be 1..10")
    if not 0 <= args.download_count <= len(queries) * args.train_per_query:
        parser.error("download count exceeds the planned train pool")
    manifest = collect(args.output, args.train_per_query, queries)
    if args.download_count > manifest["train_count"]:
        parser.error("download count exceeds actual train pool")
    if args.download_count:
        download(manifest, args.output, args.download_count)
    print(json.dumps({"train": manifest["train_count"], "heldout": manifest["heldout_count"],
                      "downloaded": args.download_count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
