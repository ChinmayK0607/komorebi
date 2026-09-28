from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("painter_group_rank_test", HERE / "group_rank.py")
assert spec and spec.loader
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)


class Reward:
    def __init__(self, score: float, weight: float = 1.0):
        self.score = score
        self.weight = weight

    def model_copy(self, update: dict):
        return Reward(update["score"], self.weight)


def fake_trace(root: Path, name: str, reference: Path, candidate: Path | None):
    info = {}
    if candidate is not None:
        info["comparison_images"] = {
            "reference": str(reference), "candidate": str(candidate),
            "image_sha256": {"reference": rank.digest(reference),
                              "candidate": rank.digest(candidate)},
        }
    return types.SimpleNamespace(
        id=name, ok=True, has_error=False,
        agent=types.SimpleNamespace(trainable=True),
        task=types.SimpleNamespace(data=types.SimpleNamespace(root=str(root))),
        info=info, rewards={"final_canvas_aesthetic": Reward(0)},
        nodes=[types.SimpleNamespace(mask=[True, False, True], advantages=[])],
    )


class GroupRankTest(unittest.TestCase):
    def test_all_anchor_losses_still_give_rank_signal_and_credit_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("reference.png", "better.png", "worse.png")]
            for index, path in enumerate(paths):
                Image.new("RGB", (8, 8), (index * 80, 10, 20)).save(path)
            a = fake_trace(root, "aaa", paths[0], paths[1])
            b = fake_trace(root, "bbb", paths[0], paths[2])
            c = fake_trace(root, "ccc", paths[0], None)
            group = [types.SimpleNamespace(traces=[trace]) for trace in (a, b, c)]

            def judge(*, labeled, **_kwargs):
                labels = {path.name: label for label, path in labeled}
                names = [labels["better.png"], labels["worse.png"]]
                return {"label": {"best_to_worst": names, "reason": "better subject",
                                  "weaknesses": {x: "rough" for x in names}},
                        "usage": {"total_tokens": 20}}

            fake_gateway = types.SimpleNamespace(load_key=lambda _path: "dummy")
            group_id = "12345678-1234-1234-1234-123456789abc"
            with patch.dict(sys.modules, {"gateway_judge": fake_gateway}), \
                 patch.object(rank, "_one_view", side_effect=judge), \
                 patch.dict(os.environ, {"PAINTER_GATEWAY_ENV_FILE": str(root / "dummy")}):
                asyncio.run(rank.score_group(group, group_id=group_id, phase="before"))
            self.assertEqual(a.rewards["final_canvas_aesthetic"].score, 0.5)
            self.assertEqual(b.rewards["final_canvas_aesthetic"].score, 0.0)
            self.assertEqual(c.rewards["final_canvas_aesthetic"].score, 0.0)
            for trace, advantage in ((a, 0.3), (b, -0.2), (c, -0.2)):
                trace.nodes[0].advantages = [advantage, advantage]
            asyncio.run(rank.score_group(group, group_id=group_id, phase="after"))
            audit = json.loads((root / "group-credit" / group_id / "audit.json").read_text())
            self.assertEqual(audit["phase"], "credit_assigned")
            self.assertEqual(audit["traces"]["aaa"]["sampled_tokens"], 2)
            self.assertEqual(audit["traces"]["aaa"]["advantage_nonzero_tokens"], 2)

    def test_rank_parser_rejects_omission(self):
        with self.assertRaises(ValueError):
            rank._parse('{"best_to_worst":["A"],"reason":"x","weaknesses":{"A":"x"}}', ["A", "B"])


if __name__ == "__main__":
    unittest.main()
