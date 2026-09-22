"""Pure cache/provider invariants; no model, renderer, or judge calls."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
import json

from painter.frozen_opponent import (
    FrozenOpponentProvider,
    OpponentUnavailable,
    OpponentSpec,
    deterministic_seed,
    OpponentResult,
    _native_eval_canvas,
)
from painter.test_native_painting_rl import _load_rl_module


PNG = b"\x89PNG\r\n\x1a\nmock-canvas"


class FrozenOpponentTests(unittest.TestCase):
    def _spec(self, root: Path, *, alias: str = "sft-step512") -> OpponentSpec:
        reference = root / "reference.jpg"
        reference.write_bytes(b"reference")
        return OpponentSpec(
            source_group="source-1",
            reference=reference,
            reference_sha256="reference-hash",
            model_alias=alias,
            checkpoint_sha256="checkpoint-sha256",
            protocol_version="native-painting-rl-v1",
            render_protocol="native-painting-v1",
            decoding_config="{}",
            seed=deterministic_seed(20260922, "source-1", "reference-hash"),
        )

    def test_seed_is_stable_and_source_bound(self):
        first = deterministic_seed(10, "a", "h")
        self.assertEqual(first, deterministic_seed(10, "a", "h"))
        self.assertNotEqual(first, deterministic_seed(10, "b", "h"))
        self.assertNotEqual(first, deterministic_seed(11, "a", "h"))

    def test_cache_key_binds_checkpoint_and_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self._spec(root)
            changed = OpponentSpec(
                source_group=spec.source_group,
                reference=spec.reference,
                reference_sha256=spec.reference_sha256,
                model_alias=spec.model_alias,
                checkpoint_sha256="different-checkpoint",
                protocol_version=spec.protocol_version,
                render_protocol=spec.render_protocol,
                decoding_config=spec.decoding_config,
                seed=spec.seed,
            )
            self.assertNotEqual(spec.cache_key, changed.cache_key)

    def test_concurrent_requests_single_flight_and_cache(self):
        async def run():
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                spec = self._spec(root)
                produced = root / "generated.png"
                calls = 0

                def generate(_spec):
                    nonlocal calls
                    calls += 1
                    time.sleep(0.01)
                    produced.write_bytes(PNG)
                    return produced

                provider = FrozenOpponentProvider(
                    root / "cache", model_alias=spec.model_alias, generator=generate,
                    checkpoint_sha256=spec.checkpoint_sha256,
                    protocol_version=spec.protocol_version,
                    render_protocol=spec.render_protocol,
                    decoding_config=spec.decoding_config,
                )
                first, second = await asyncio.gather(provider.resolve(spec), provider.resolve(spec))
                self.assertTrue(first.valid)
                self.assertEqual(first.path, second.path)
                self.assertEqual(calls, 1)
                third = await provider.resolve(spec)
                self.assertEqual(third.sha256, first.sha256)
                self.assertEqual(calls, 1)

        asyncio.run(run())

    def test_model_invalid_is_cached_but_outage_is_retryable(self):
        async def run():
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                spec = self._spec(root)
                invalid_calls = 0

                def invalid(_spec):
                    nonlocal invalid_calls
                    invalid_calls += 1
                    return {"status": "invalid_model_output", "reason": "no canvas"}

                provider = FrozenOpponentProvider(
                    root / "invalid", model_alias=spec.model_alias, generator=invalid,
                    checkpoint_sha256=spec.checkpoint_sha256,
                    protocol_version=spec.protocol_version,
                    render_protocol=spec.render_protocol,
                    decoding_config=spec.decoding_config,
                )
                first = await provider.resolve(spec)
                second = await provider.resolve(spec)
                self.assertEqual(first.status, "invalid_model_output")
                self.assertEqual(second.status, "invalid_model_output")
                self.assertEqual(invalid_calls, 1)

                outage_calls = 0

                def outage(_spec):
                    nonlocal outage_calls
                    outage_calls += 1
                    raise OpponentUnavailable("server unavailable")

                outage_provider = FrozenOpponentProvider(
                    root / "outage", model_alias=spec.model_alias, generator=outage,
                    checkpoint_sha256=spec.checkpoint_sha256,
                    protocol_version=spec.protocol_version,
                    render_protocol=spec.render_protocol,
                    decoding_config=spec.decoding_config,
                )
                self.assertEqual((await outage_provider.resolve(spec)).status, "unavailable")
                self.assertEqual((await outage_provider.resolve(spec)).status, "unavailable")
                self.assertEqual(outage_calls, 2)

        asyncio.run(run())

    def test_advertised_missing_or_non_png_path_is_retryable_unavailable(self):
        async def run():
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                spec = self._spec(root)
                calls = 0
                malformed = root / "malformed.png"
                malformed.write_bytes(b"not-a-png")

                def generator(_spec):
                    nonlocal calls
                    calls += 1
                    return root / ("missing.png" if calls == 1 else "malformed.png")

                provider = FrozenOpponentProvider(
                    root / "cache", model_alias=spec.model_alias, generator=generator,
                    checkpoint_sha256=spec.checkpoint_sha256,
                    protocol_version=spec.protocol_version,
                    render_protocol=spec.render_protocol,
                    decoding_config=spec.decoding_config,
                )
                self.assertEqual((await provider.resolve(spec)).status, "unavailable")
                self.assertEqual((await provider.resolve(spec)).status, "unavailable")
                self.assertEqual(calls, 2)
                self.assertFalse(list((root / "cache" / "receipts").glob("*.json")))

        asyncio.run(run())

    def test_native_eval_parser_accepts_only_recorded_valid_final(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            case_id = "frozen-case-valid"
            canvas = attempt / "evaluation-rollouts/frozen" / case_id / "episode" / "turn-6" / "painting.png"
            canvas.parent.mkdir(parents=True)
            canvas.write_bytes(PNG)
            (attempt / "traces.jsonl").write_text(json.dumps({
                "id": "trace-id",
                "task": {"data": {"name": case_id, "task_id": case_id}},
                "info": {
                    "task_id": case_id,
                    "final_canvas": str(canvas),
                    "final_canvas_valid": True,
                    "turns": [{"turn": 6, "valid": True, "canvas": str(canvas)}],
                },
            }) + "\n")
            result = _native_eval_canvas(attempt, case_id, attempt)
            self.assertEqual(result, canvas.resolve())

    def test_native_eval_parser_rejects_service_error_even_on_clean_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            case_id = "frozen-case-service"
            (attempt / "traces.jsonl").write_text(json.dumps({
                "id": "trace-id",
                "ok": True,
                "task": {"data": {"name": case_id}},
                "info": {
                    "task_id": case_id,
                    "final_canvas_valid": False,
                    "service_failure": "renderer HTTP 503",
                },
            }) + "\n")
            with self.assertRaisesRegex(OpponentUnavailable, "HTTP 503"):
                _native_eval_canvas(attempt, case_id, attempt)

    def test_native_eval_parser_reports_invalid_without_using_stale_unrelated_png(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            case_id = "frozen-case-invalid"
            stale = attempt / "evaluation-rollouts/frozen" / case_id / "turn-1" / "painting.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(PNG)
            (attempt / "traces.jsonl").write_text(json.dumps({
                "task": {"data": {"name": case_id}},
                "info": {
                    "task_id": case_id,
                    "final_canvas_valid": False,
                    "turns": [{"turn": 1, "valid": False, "failure_kind": "program"}],
                },
            }) + "\n")
            result = _native_eval_canvas(attempt, case_id, attempt)
            self.assertEqual(result["status"], "invalid_model_output")
            self.assertNotIn("path", result)

    def test_lazy_loader_defers_anchor_and_can_preserve_manifest_order(self):
        module = _load_rl_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, contents in (("ref-a.png", b"ref-a"), ("ref-b.png", b"ref-b")):
                (root / name).write_bytes(contents)
            rows = [
                {"id": "photo-1", "source_group": "photo-1", "family": "photo", "split": "train", "reference": "ref-a.png"},
                {"id": "proc-1", "source_group": "proc-1", "family": "proc", "split": "train", "reference": "ref-b.png"},
            ]
            manifest = root / "scaled.json"
            manifest.write_text(json.dumps({"training_rows": rows}))
            config = module.NativePaintingRLConfig(
                root=str(root),
                manifest="scaled.json",
                lazy_opponent=True,
                opponent_model_alias="sft-step512",
                opponent_checkpoint_sha256="checkpoint-sha256",
                preserve_manifest_order=True,
            )
            tasks = list(module.NativePaintingRLTaskset(config).load())
            self.assertEqual([task.data.task_id for task in tasks], ["photo-1", "proc-1"])
            self.assertTrue(all(task.data.opponent is None for task in tasks))
            self.assertTrue(all(task.data.opponent_cache_key for task in tasks))

    def test_valid_candidate_gets_fair_score_for_invalid_lazy_anchor(self):
        module = _load_rl_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.jpg"
            candidate = root / "candidate.png"
            reference.write_bytes(b"reference")
            candidate.write_bytes(PNG)
            data = module.NativePaintingRLData(
                root=str(root), task_id="task-1", source_group="group", family="shape",
                reference=str(reference), reference_sha256="reference-hash", mode="fresh",
                opponent_model_alias="sft-step512", opponent_checkpoint_sha256="checkpoint-sha256",
                opponent_protocol_version="native-painting-rl-v1", opponent_render_protocol="native-painting-v1",
                opponent_decoding_config="{}", opponent_seed=7,
            )
            task = module.NativePaintingRLTask(data)

            class InvalidProvider:
                async def resolve(self, _spec):
                    return OpponentResult("invalid_model_output", "cache-key", reason="no canvas")

            trace = type("Trace", (), {"id": "trace-1", "info": {"final_canvas": str(candidate), "final_canvas_valid": True}})()
            with patch.object(module, "_frozen_provider", return_value=InvalidProvider()):
                self.assertEqual(asyncio.run(task.final_canvas_aesthetic(trace)), 1.0)
            self.assertEqual(trace.info["reward_status"], "anchor_invalid_model_output")

    def test_advertised_missing_candidate_is_service_failure(self):
        module = _load_rl_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.jpg"
            reference.write_bytes(b"reference")
            data = module.NativePaintingRLData(
                root=str(root), task_id="task-1", source_group="group", family="shape",
                reference=str(reference), reference_sha256="reference-hash", mode="fresh",
                opponent=str(root / "opponent.png"), opponent_sha256="opponent-hash",
            )
            task = module.NativePaintingRLTask(data)
            trace = type("Trace", (), {"id": "trace-1", "info": {"final_canvas": str(root / "missing.png"), "final_canvas_valid": True}})()
            with self.assertRaisesRegex(RuntimeError, "advertised final canvas is missing"):
                asyncio.run(task.final_canvas_aesthetic(trace))


if __name__ == "__main__":
    unittest.main()
