"""Install the minimal group-scoring hook in the pinned Prime checkout.

The hook runs after all member episodes reach TrainSink and immediately before
and after Prime's normal GRPO finalization. Refuse unknown upstream versions.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PINNED_SHA256 = "8618ae01796d616f973bee57966474da184fff50e324f6ac4a93b52becce3ca8"
OLD_IMPORT = "import asyncio\nfrom collections import defaultdict"
NEW_IMPORT = "import asyncio\nimport importlib\nimport os\nfrom collections import defaultdict"
OLD_BLOCK = """        if survivors:
            await env.algorithm.finalize_group(group)
        admitted = self._admit(group) if group else False"""
NEW_BLOCK = """        if survivors:
            painter_module = os.environ.get("PAINTER_GROUP_RANK_MODULE")
            painter_hook = (
                importlib.import_module(painter_module).score_group if painter_module else None
            )
            if painter_hook is not None:
                await painter_hook(group, group_id=group_id, phase="before")
            await env.algorithm.finalize_group(group)
            if painter_hook is not None:
                await painter_hook(group, group_id=group_id, phase="after")
        admitted = self._admit(group) if group else False"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install(source: Path) -> dict:
    original = source.read_bytes()
    text = original.decode()
    if NEW_BLOCK in text and NEW_IMPORT in text:
        return {"status": "already_installed", "sha256": digest(original)}
    if digest(original) != PINNED_SHA256:
        raise ValueError("Prime train_sink source hash differs from pinned checkout")
    if text.count(OLD_IMPORT) != 1 or text.count(OLD_BLOCK) != 1:
        raise ValueError("Prime hook location is not unique")
    result = text.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_BLOCK, NEW_BLOCK)
    compile(result, str(source), "exec")
    source.write_text(result)
    return {"status": "installed", "sha256": digest(source.read_bytes())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prime_root", type=Path)
    args = parser.parse_args()
    source = args.prime_root / "src/prime_rl/orchestrator/train_sink.py"
    print(install(source))


if __name__ == "__main__":
    main()
