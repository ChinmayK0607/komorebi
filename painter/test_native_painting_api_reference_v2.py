"""Default-off contract tests for the concise p5.brush API reference v2."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

from painter.test_native_painting import _Agents, _DummyEnv, _load_module


ROOT = Path(__file__).resolve().parent
VENDOR_BRUSH = ROOT / "vendor/integrations/watercolour/assets/p5.brush.js"
V1_REFERENCE_SHA256 = "368930640e89707b91f20eca8c64a3e2130bf69af14793eef3c37c94887edf3f"


class NativePaintingApiReferenceV2Tests(unittest.TestCase):
    def test_v2_is_default_off_and_does_not_change_v1_bytes(self):
        module = _load_module()
        self.assertEqual(
            hashlib.sha256(module.P5_BRUSH_API_REFERENCE.encode("utf-8")).hexdigest(),
            V1_REFERENCE_SHA256,
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(module.brush_api_reference_v2_enabled())
            self.assertIs(module._painting_system_prompt("text"), module.TEXT_SYSTEM)
        with patch.dict("os.environ", {module.P5_BRUSH_API_REFERENCE_ENV: "1"}, clear=True):
            self.assertIs(module._painting_system_prompt("text"), module.TEXT_SYSTEM_API)
            self.assertNotIn(module.P5_BRUSH_API_REFERENCE_V2, module.TEXT_SYSTEM_API)
        with patch.dict(
            "os.environ",
            {
                module.P5_BRUSH_API_REFERENCE_ENV: "1",
                module.P5_BRUSH_API_REFERENCE_V2_ENV: "1",
            },
            clear=True,
        ):
            self.assertIn(module.P5_BRUSH_API_REFERENCE_V2, module._painting_system_prompt("text"))

    def test_v2_prompt_has_only_verified_signatures_and_complete_template(self):
        module = _load_module()
        source = VENDOR_BRUSH.read_text()
        self.assertRegex(source, r"t\.circle=function\([^)]*\)")
        self.assertRegex(source, r"t\.rect=function\([^)]*\)")
        self.assertIn("function br(", source)
        self.assertRegex(source, r"t\.spline=function\([^)]*\)")
        self.assertRegex(source, r"t\.stroke=\w+")
        self.assertIn('["cpencil",', source)
        with patch.dict("os.environ", {module.P5_BRUSH_API_REFERENCE_V2_ENV: "1"}, clear=True):
            prompt = module._painting_system_prompt("text")
            self.assertIn(module.P5_BRUSH_API_REFERENCE_V2, prompt)
            for signature in (
                "brush.circle(x, y, radius, roughness?)",
                "brush.rect(x, y, width, height, mode?)",
                "brush.line(x1, y1, x2, y2)",
                "brush.spline([[x, y, pressure], ...], curvature?)",
                "brush.set(name, color, weight)",
                "brush.stroke(color)",
            ):
                self.assertIn(signature, prompt)
            self.assertIn("createCanvas(600,600,WEBGL);", prompt)
            self.assertIn("translate(-width/2,-height/2);", prompt)
            self.assertIn("noLoop();", prompt)
            self.assertIn("Do not define a second `setup()` or `draw()`", prompt)
            self.assertIn("`circle`, `line`, `ellipse`, or `box`", prompt)
            self.assertEqual(
                hashlib.sha256(module.P5_BRUSH_API_REFERENCE_V2.encode("utf-8")).hexdigest(),
                module.P5_BRUSH_API_REFERENCE_V2_SHA256,
            )

    def test_v2_trace_records_version_and_hash_and_enables_repair_guidance(self):
        module = _load_module()
        with self.subTest("specific guidance"):
            with patch.dict("os.environ", {module.P5_BRUSH_API_REFERENCE_V2_ENV: "1"}, clear=True):
                guidance = module._program_failure_feedback(
                    ["TypeError: brush.ellipse is not a function"],
                    include_guidance=False,
                    include_api_guidance=module.brush_api_reference_v2_enabled(),
                )
                self.assertIn("brush.circle", guidance)

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp, patch.dict(
            "os.environ",
            {
                module.P5_BRUSH_API_REFERENCE_ENV: "1",
                module.P5_BRUSH_API_REFERENCE_V2_ENV: "1",
            },
            clear=True,
        ):
            root = Path(temp)
            brief = root / "brief.txt"
            brief.write_text("Paint a blue cup.\n")
            data = module.NativePaintingData(
                idx=0,
                name="api-v2-trace",
                source_group="api-v2-trace",
                root=str(root),
                task_id="api-v2-trace",
                reference=str(brief),
                reference_sha256=module.sha(brief),
                reference_kind="text",
                max_turns=1,
            )
            agents = _Agents(["FINISHED"])
            asyncio.run(module.NativePaintingEnv.__dict__["run"](_DummyEnv(), module.NativePaintingTask(data), agents))
            info = agents.agent.interaction_obj.trace.info
            self.assertEqual(info["brush_api_reference_v2_version"], module.P5_BRUSH_API_REFERENCE_V2_VERSION)
            self.assertEqual(info["brush_api_reference_v2_sha256"], module.P5_BRUSH_API_REFERENCE_V2_SHA256)
            self.assertEqual(info["brush_api_reference_version"], module.P5_BRUSH_API_REFERENCE_V2_VERSION)
            self.assertEqual(info["brush_api_reference_sha256"], module.P5_BRUSH_API_REFERENCE_V2_SHA256)


if __name__ == "__main__":
    unittest.main()
