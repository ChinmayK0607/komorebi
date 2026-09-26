#!/usr/bin/env python3
"""Summarize local, hash-verified teacher publication receipts without raw data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_candidate_pool import POOL, audit
from publish_teacher_batch import validate_publication_metadata


OUT = Path(__file__).resolve().parent / "PUBLIC_SOURCE_INDEX.json"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build() -> dict:
    candidate_audit = audit()
    if candidate_audit["errors"]:
        raise ValueError("candidate source audit failed")
    published = []
    excluded = []
    for batch in sorted(POOL.iterdir()):
        receipt_path = batch / "source-receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        public_path = batch / "source-public.json"
        if not public_path.is_file():
            try:
                validate_publication_metadata(batch / "source.tar.gz")
                reason = "publication_pending"
            except ValueError:
                reason = "missing_image_source_or_rights_metadata"
            excluded.append({"batch": batch.name, "count": receipt["count"],
                             "reason": reason})
            continue
        raw = public_path.read_bytes()
        remote = json.loads(raw)
        if (remote.get("anonymous_hash_verified") is not True
                or remote.get("batch") != batch.name
                or remote.get("count") != receipt["count"]
                or remote.get("archive_sha256") != receipt["archive_sha256"]
                or remote.get("archive_bytes") != receipt["archive_bytes"]):
            raise ValueError(f"public source receipt mismatch: {batch.name}")
        published.append({"batch": batch.name, "count": receipt["count"],
                          "source_commit": remote["source_commit"],
                          "archive_sha256": remote["archive_sha256"],
                          "dataset_commit": remote["dataset_commit"],
                          "url": remote["url"], "receipt_sha256": sha(raw)})
    result = {"schema": "painter.teacher-source-public-index.v1",
              "candidate_status": "unrendered_unreviewed_not_sft_admitted",
              "candidate_audit": {key: candidate_audit[key] for key in
                                  ("distinct", "distinct_total", "alternatives",
                                   "render_conditioned_corrections",
                                   "new_input_coverage_total", "batches")},
              "published_bundle_count": len(published),
              "published_members": sum(row["count"] for row in published),
              "excluded_bundle_count": len(excluded),
              "published": published, "excluded": excluded}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = build()
    print(json.dumps({key: result[key] for key in
                      ("published_bundle_count", "published_members", "excluded_bundle_count",
                       "candidate_audit")}, sort_keys=True))
