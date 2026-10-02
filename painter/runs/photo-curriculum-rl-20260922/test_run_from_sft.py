"""Offline checks for adapter preload and pinned Prime command interception."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("photo_rl_runner", ROOT / "run_from_sft.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhotoRLRunnerTests(unittest.TestCase):
    def test_group_rank_environment_reaches_prime_orchestrator_child(self):
        parent = {
            "PAINTER_GROUP_RANK_MODULE": "group_rank",
            "PAINTER_GATEWAY_ENV_FILE": "/root/painter/private/gateway.env",
            "PYTHONPATH": "/root/painter/repo/checkout/painter/runs/quality-relative-rl-20260928",
        }
        with patch.dict(MODULE.os.environ, parent):
            child = MODULE._orchestrator_env({"MODEL_DIR": "/tmp/model"})
        self.assertEqual({name: child[name] for name in parent}, parent)
        self.assertEqual(child["MODEL_DIR"], "/tmp/model")

    def test_painting_protocol_controls_reach_orchestrator_child(self):
        names = (
            "PAINTER_P5_BRUSH_API_REFERENCE",
            "PAINTER_INSPECT_CHOICE_FEEDBACK",
            "PAINTER_RENDER_WORKERS",
            "PAINTER_JUDGE_PROTOCOL",
            "PAINTER_JUDGE_FORMAT",
            "PAINTER_JUDGE_MODEL",
        )
        parent = {name: str(index) for index, name in enumerate(names, 1)}
        with patch.dict(MODULE.os.environ, parent):
            child = MODULE._orchestrator_env({})
        self.assertEqual({name: child[name] for name in names}, parent)

    def test_initial_live_and_frozen_aliases_are_distinct(self):
        self.assertEqual(MODULE._initial_aliases("/tmp/base", "sft-step512"), ("/tmp/base", "sft-step512"))
        with self.assertRaises(ValueError):
            MODULE._initial_aliases("/tmp/base", "/tmp/base")

    def test_recovered_policy_loads_frozen_alias_from_distinct_adapter(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            live = root / "recovered"
            frozen = root / "sft-step512"
            for adapter in (live, frozen):
                adapter.mkdir()
                (adapter / "adapter_model.safetensors").write_bytes(adapter.name.encode())
            calls = []

            class Process:
                returncode = None

                def poll(self):
                    return None

            def request(_url, *, payload=None, timeout=10.0):
                calls.append(payload)
                return 200, b""

            with patch.object(MODULE, "_request", side_effect=request):
                MODULE._wait_and_load(
                    Process(), live, base_url="http://127.0.0.1:8000",
                    aliases=("/base", "sft-step512"), frozen_adapter=frozen,
                )
            self.assertEqual(calls[0], None)
            self.assertEqual(calls[1]["lora_name"], "/base")
            self.assertEqual(calls[1]["lora_path"], live.as_posix())
            self.assertEqual(calls[2]["lora_name"], "sft-step512")
            self.assertEqual(calls[2]["lora_path"], frozen.as_posix())

    def test_trainer_command_only_replaces_pinned_prime_child(self):
        command = [
            "torchrun", "--nproc-per-node=1", "-m", "prime_rl.trainer.rl.train", "@", "trainer.toml"
        ]
        result = MODULE.trainer_command(command, Path("train_from_sft.py"))
        self.assertEqual(result[2], "train_from_sft.py")
        self.assertEqual(result[-2:], ["@", "trainer.toml"])
        self.assertEqual(MODULE.trainer_command(["orchestrator", "@", "orch.toml"], Path("x")), ["orchestrator", "@", "orch.toml"])

    def test_policy_verification_preserves_prime_base_name(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "orchestrator.toml"
            path.write_text('[model]\nname = "/tmp/base-revision"\n')
            MODULE._verify_orchestrator_policy(path, "/tmp/base-revision")
            self.assertEqual(tomllib.loads(path.read_text())["model"]["name"], "/tmp/base-revision")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE._verify_orchestrator_policy(path, "wrong")

    def test_policy_verification_accepts_prime_json_child_config(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "orchestrator.json"
            path.write_text(json.dumps({"model": {"name": "/tmp/base-revision"}}))
            MODULE._verify_orchestrator_policy(path, "/tmp/base-revision")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE._verify_orchestrator_policy(path, "wrong")

    def test_cache_paths_are_role_specific(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inference = MODULE._cache_env(root, "inference", {
                "MODEL_DIR": "/tmp/base", "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            })
            trainer = MODULE._cache_env(root, "trainer", {"MODEL_DIR": "/tmp/base"})
            self.assertNotEqual(inference["TORCHINDUCTOR_CACHE_DIR"], trainer["TORCHINDUCTOR_CACHE_DIR"])
            self.assertNotEqual(inference["PAINTER_COMPILE_CACHE_ROOT"], trainer["PAINTER_COMPILE_CACHE_ROOT"])
            self.assertEqual(inference["PAINTER_COMPILE_CACHE_PROCESS_ISOLATION"], "1")
            self.assertEqual(inference["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            self.assertEqual(trainer["PAINTER_COMPILE_CACHE_PROCESS_ISOLATION"], "0")
            self.assertTrue(Path(inference["VLLM_CACHE_ROOT"]).is_dir())
            self.assertTrue(Path(trainer["TRITON_CACHE_DIR"]).is_dir())

    def test_prewarm_uses_first_real_task_and_accepts_valid_result(self):
        from painter.frozen_opponent import OpponentSpec

        with TemporaryDirectory() as temp:
            reference = Path(temp) / "reference.jpg"
            reference.write_bytes(b"reference")
            data = SimpleNamespace(
                task_id="task-1", source_group="group-1", opponent=None,
                reference=reference, reference_sha256="reference-sha",
                opponent_model_alias="sft-step512", opponent_checkpoint_sha256="checkpoint-sha",
                opponent_protocol_version="native-painting-rl-v1", opponent_render_protocol="native-painting-v1",
                opponent_decoding_config="{}", opponent_seed=1234, max_turns=6,
                opponent_max_completion_tokens=8192, opponent_context_length=32768,
            )
            captured = {}

            class Provider:
                async def resolve(self, spec):
                    captured["spec"] = spec
                    return SimpleNamespace(status="valid", cache_key="cache-key")

            result = MODULE._prewarm_frozen_opponent(
                Path(temp) / "native-rl.toml",
                task_loader=lambda: SimpleNamespace(data=data),
                provider_factory=lambda _data: Provider(),
                spec_type=OpponentSpec,
            )
            self.assertEqual(result.status, "valid")
            self.assertEqual(captured["spec"].seed, 1234)
            self.assertEqual(captured["spec"].max_turns, 6)

    def test_prewarm_fails_for_service_unavailable(self):
        data = SimpleNamespace(
            task_id="task-1", source_group="group-1", opponent=None,
            reference=Path("/tmp/reference.jpg"), reference_sha256="reference-sha",
            opponent_model_alias="sft-step512", opponent_checkpoint_sha256="checkpoint-sha",
            opponent_protocol_version="native-painting-rl-v1", opponent_render_protocol="native-painting-v1",
            opponent_decoding_config="{}", opponent_seed=1234, max_turns=6,
            opponent_max_completion_tokens=8192, opponent_context_length=32768,
        )

        class Provider:
            async def resolve(self, _spec):
                return SimpleNamespace(status="unavailable", reason="renderer service failure")

        from painter.frozen_opponent import OpponentSpec
        with self.assertRaisesRegex(RuntimeError, "prewarm unavailable"):
            MODULE._prewarm_frozen_opponent(
                Path("/tmp/native-rl.toml"),
                task_loader=lambda: SimpleNamespace(data=data),
                provider_factory=lambda _data: Provider(),
                spec_type=OpponentSpec,
            )

    def test_prewarm_verifies_fixed_opponent_without_provider(self):
        import hashlib

        with TemporaryDirectory() as temp:
            anchor = Path(temp) / "anchor.png"
            anchor.write_bytes(b"fixed painting")
            expected = hashlib.sha256(anchor.read_bytes()).hexdigest()
            data = SimpleNamespace(opponent=str(anchor), opponent_sha256=expected)
            result = MODULE._prewarm_frozen_opponent(
                Path(temp) / "native-rl.toml",
                task_loader=lambda: SimpleNamespace(data=data),
                provider_factory=lambda _data: self.fail("lazy provider was called"),
            )
            self.assertEqual(result["status"], "fixed_verified")
            self.assertEqual(result["sha256"], expected)
            anchor.write_bytes(b"changed painting")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                MODULE._prewarm_frozen_opponent(
                    Path(temp) / "native-rl.toml",
                    task_loader=lambda: SimpleNamespace(data=data),
                )


if __name__ == "__main__":
    unittest.main()
