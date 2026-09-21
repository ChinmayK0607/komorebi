"""Offline tests for the bounded Luna-low CLI judge handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from painter import luna_cli_judge as cli


PROTOCOL_DIR = cli.DEFAULT_PROTOCOL_DIR


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pair(
    batch: Path,
    pair_id: str,
    index: int = 0,
    *,
    source_suffix: dict[str, str] | None = None,
    magic: dict[str, str] | None = None,
) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for role in cli.DISPLAY_ROLES:
        suffix = (source_suffix or {}).get(role, ".png")
        if not suffix.startswith("."):
            suffix = "." + suffix
        image_kind = (magic or {}).get(role, suffix.lstrip(".").lower())
        if image_kind in {"jpg", "jpeg"}:
            payload = b"\xff\xd8\xff\xe0" + f"{pair_id}:{index}:{role}".encode()
        elif image_kind == "webp":
            payload = b"RIFF0000WEBP" + f"{pair_id}:{index}:{role}".encode()
        else:
            payload = b"\x89PNG\r\n\x1a\n" + f"{pair_id}:{index}:{role}".encode()
        path = batch / f"{pair_id}-{role}{suffix}"
        path.write_bytes(payload)
        images[role] = path
    pair = {
        "id": pair_id,
        **{role: str(images[role]) for role in cli.DISPLAY_ROLES},
        "image_sha256": {role: _sha(images[role]) for role in cli.DISPLAY_ROLES},
        "protocol_version": "aesthetic-pairwise-v1",
    }
    (batch / f"judge-{pair_id}.json").write_text(json.dumps({
        "protocol_version": "aesthetic-pairwise-v1",
        "judge_model": cli.JUDGE_MODEL,
        "reasoning_effort": cli.REASONING_EFFORT,
        "comparisons": [pair],
    }) + "\n")
    return images


def _valid_label(winner: str = "B") -> dict[str, object]:
    return {
        "winner": winner,
        "reason": "B preserves the complete form more clearly.",
        "weaknesses_A": ["The contour is less complete."],
        "weaknesses_B": [],
    }


def _load_queue_module():
    """Load the pinned queue without making its protocol package installable."""

    path = PROTOCOL_DIR / "judge_queue.py"
    spec = importlib.util.spec_from_file_location("test_luna_judge_queue", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.path.insert(0, str(PROTOCOL_DIR))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class _FakeRunner:
    def __init__(
        self,
        label: object | None = None,
        *,
        returncode: int = 0,
        delay: float = 0.0,
        stdout: str = "event",
        stderr: str = "",
    ):
        self.label = _valid_label() if label is None else label
        self.returncode = returncode
        self.delay = delay
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[tuple[list[str], str, Path, Path]] = []
        self.attachment_snapshots: list[list[tuple[str, str]]] = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __call__(self, command: list[str], prompt: str, cwd: Path, last_message: Path):
        with self._lock:
            self.calls.append((command, prompt, cwd, last_message))
            image_args = [
                Path(command[index + 1])
                for index, value in enumerate(command[:-1])
                if value == "--image"
            ]
            self.attachment_snapshots.append([
                (path.name, _sha(path)) for path in image_args
            ])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.label is not None:
                last_message.write_text(json.dumps(self.label))
            return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)
        finally:
            with self._lock:
                self.active -= 1


class LunaCliJudgeTests(unittest.TestCase):
    def test_two_orientation_inputs_use_three_ordered_attachments_and_queue_label_shape(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            output = root / "output"
            batch.mkdir()
            _write_pair(batch, "orientation-a")
            _write_pair(batch, "orientation-b", index=1)
            runner = _FakeRunner(
                stdout='{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}\n',
                stderr="cli warning\n",
            )

            summary = cli.run_batch(
                batch,
                output_dir=output,
                protocol_dir=PROTOCOL_DIR,
                runner=runner,
                scratch_parent=root,
            )

            self.assertEqual(summary["pairs"], 2)
            self.assertEqual(summary["validated"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["receipt_installation"], "not performed")
            schema = cli.output_schema()
            self.assertEqual(
                schema["required"], ["winner", "reason", "weaknesses_A", "weaknesses_B"]
            )
            self.assertFalse(schema["additionalProperties"])
            expected_prompt = cli.ATTACHMENT_PREAMBLE + (PROTOCOL_DIR / "JUDGE_PROMPT.md").read_text()
            self.assertEqual({call[1] for call in runner.calls}, {expected_prompt})
            self.assertLessEqual(runner.max_active, 2)
            self.assertEqual(
                {tuple(name for name, _digest in snapshot) for snapshot in runner.attachment_snapshots},
                {("reference.png", "A.png", "B.png")},
            )
            self.assertEqual(len({call[2] for call in runner.calls}), 2)
            for command, _prompt, _cwd, _last_message in runner.calls:
                image_values = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--image"]
                self.assertEqual(command.count("--image"), 3)
                self.assertEqual([Path(value).name for value in image_values], ["reference.png", "A.png", "B.png"])
                self.assertIn("--model", command)
                self.assertEqual(command[command.index("--model") + 1], cli.JUDGE_MODEL)
                self.assertIn('model_reasoning_effort="low"', command)
                self.assertIn("--output-schema", command)
                self.assertIn("--output-last-message", command)
                self.assertIn("--json", command)
                self.assertEqual(command[-1], "-")
                self.assertIn("--sandbox", command)
                self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
                self.assertIn("--ephemeral", command)
                self.assertIn("--ignore-user-config", command)
                self.assertIn("--skip-git-repo-check", command)

            for label_path in sorted(output.glob("labels-*.json")):
                label = json.loads(label_path.read_text())
                self.assertEqual(set(label), {"protocol_version", "judge_model", "reasoning_effort", "judgments"})
                self.assertEqual(len(label["judgments"]), 1)
                self.assertEqual(
                    set(label["judgments"][0]),
                    {"id", "winner", "reason", "weaknesses_A", "weaknesses_B"},
                )
            queue = _load_queue_module()
            loaded = queue._label_documents(
                output, {"orientation-a", "orientation-b"}
            )
            self.assertEqual(set(loaded), {"orientation-a", "orientation-b"})
            for execution_path in sorted(output.glob("execution-*.json")):
                execution = json.loads(execution_path.read_text())
                self.assertEqual(execution["image_input_mode"], "cli_attachment")
                self.assertEqual(
                    [item["attachment_status"] for item in execution["input_hashes"]["image_inputs"]],
                    ["attached", "attached", "attached"],
                )
                image_inputs = execution["input_hashes"]["image_inputs"]
                self.assertEqual(
                    [Path(item["attachment_path"]).name for item in image_inputs],
                    ["reference.png", "A.png", "B.png"],
                )
                self.assertTrue(image_inputs[0]["path"].endswith("-reference.png"))
                self.assertTrue(image_inputs[1]["path"].endswith("-A.png"))
                self.assertTrue(image_inputs[2]["path"].endswith("-B.png"))
                self.assertTrue(all(item["path"] != item["attachment_path"] for item in image_inputs))
                self.assertEqual(
                    [item["attachment_sha256"] for item in image_inputs],
                    [item["sha256"] for item in image_inputs],
                )
                self.assertNotIn("inspected", execution)
                self.assertNotIn("tool_inspected", execution)
                self.assertEqual(execution["usage"], {"input_tokens": 12, "output_tokens": 3})
                self.assertTrue(execution["usage_available"])
                self.assertEqual(Path(execution["stdout_path"]).read_text(), runner.stdout)
                self.assertEqual(Path(execution["stderr_path"]).read_text(), runner.stderr)
                self.assertEqual(Path(execution["events_path"]).read_text(), runner.stdout)
                self.assertTrue(execution["events_sha256"])
                self.assertTrue(execution["validated_label"])

    def test_changed_hash_is_rejected_without_runner_or_score(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            output = root / "output"
            batch.mkdir()
            images = _write_pair(batch, "changed")
            images["A"].write_bytes(b"changed-after-stage")
            runner = _FakeRunner()

            summary = cli.run_batch(batch, output_dir=output, protocol_dir=PROTOCOL_DIR, runner=runner)

            self.assertEqual(summary["validated"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertFalse(runner.calls)
            self.assertFalse(list(output.glob("labels-*.json")))
            execution = json.loads(next(output.glob("execution-*.json")).read_text())
            self.assertIn("changed image hash", execution["error"])
            self.assertIsNone(execution["validated_label"])

    def test_jpeg_magic_gets_jpg_attachment_even_when_source_suffix_is_png(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            output = root / "output"
            batch.mkdir()
            images = _write_pair(
                batch,
                "jpeg-reference",
                source_suffix={"reference": ".png"},
                magic={"reference": "jpeg"},
            )
            runner = _FakeRunner()

            summary = cli.run_batch(
                batch,
                output_dir=output,
                protocol_dir=PROTOCOL_DIR,
                runner=runner,
                scratch_parent=root,
            )

            self.assertEqual(summary["validated"], 1)
            command = runner.calls[0][0]
            image_values = [
                command[index + 1]
                for index, value in enumerate(command[:-1])
                if value == "--image"
            ]
            self.assertEqual(
                [Path(value).name for value in image_values],
                ["reference.jpg", "A.png", "B.png"],
            )
            prompt = runner.calls[0][1]
            self.assertIn("REFERENCE (reference.jpg), A (A.png), B (B.png)", prompt)
            self.assertEqual(_sha(images["reference"]), runner.attachment_snapshots[0][0][1])
            execution = json.loads(next(output.glob("execution-*.json")).read_text())
            inputs = execution["input_hashes"]["image_inputs"]
            self.assertEqual(inputs[0]["attachment_sha256"], inputs[0]["sha256"])
            self.assertEqual(Path(inputs[0]["attachment_path"]).name, "reference.jpg")

    def test_invalid_model_output_is_retained_as_raw_but_never_installed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            output = root / "output"
            batch.mkdir()
            _write_pair(batch, "invalid")
            runner = _FakeRunner(label={"winner": "A"})

            summary = cli.run_batch(batch, output_dir=output, protocol_dir=PROTOCOL_DIR, runner=runner)

            self.assertEqual(summary["validated"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertFalse(list(output.glob("labels-*.json")))
            execution = json.loads(next(output.glob("execution-*.json")).read_text())
            self.assertEqual(execution["raw_label"], json.dumps({"winner": "A"}))
            self.assertIsNone(execution["validated_label"])
            self.assertIn("missing fields", execution["error"])

    def test_existing_output_aborts_whole_batch_without_overwriting_evidence(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            output = root / "output"
            batch.mkdir()
            _write_pair(batch, "collision")
            _write_pair(batch, "unstarted")
            output.mkdir()
            prior = output / "execution-collision.json"
            prior.write_text("prior evidence\n")
            runner = _FakeRunner()

            with self.assertRaisesRegex(cli.OutputCollision, "refusing to overwrite"):
                cli.run_batch(batch, output_dir=output, protocol_dir=PROTOCOL_DIR, runner=runner)

            self.assertFalse(runner.calls)
            self.assertEqual(prior.read_text(), "prior evidence\n")
            self.assertFalse((output / "execution-unstarted.json").exists())
            self.assertFalse((output / "cli-judge-summary.json").exists())

    def test_unsafe_pair_id_is_rejected_without_traversal_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            output = root / "output"
            batch.mkdir()
            _write_pair(batch, "unsafe")
            document_path = batch / "judge-unsafe.json"
            document = json.loads(document_path.read_text())
            document["comparisons"][0]["id"] = "../escape"
            document_path.write_text(json.dumps(document))
            runner = _FakeRunner()

            summary = cli.run_batch(batch, output_dir=output, protocol_dir=PROTOCOL_DIR, runner=runner)

            self.assertEqual(summary["failed"], 1)
            self.assertFalse(runner.calls)
            self.assertTrue((output / "execution-unsafe.json").is_file())
            self.assertFalse((root / "execution-../escape.json").exists())

    def test_default_runner_sets_a_finite_timeout(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch("painter.luna_cli_judge.subprocess.run", return_value=completed) as run:
                result = cli._default_runner(
                    ["codex", "exec", "-"], "prompt", root, root / "last.json"
                )
            self.assertIs(result, completed)
            self.assertEqual(run.call_args.kwargs["timeout"], cli.DEFAULT_EXEC_TIMEOUT_SECONDS)

    def test_default_runner_contract_is_an_argument_vector(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch"
            batch.mkdir()
            _write_pair(batch, "command")
            pair = cli.validate_pair_document(
                batch / "judge-command.json", batch, cli._load_judge_module(PROTOCOL_DIR)
            )
            command = cli.build_command(
                pair,
                schema_path=root / "schema.json",
                last_message_path=root / "last.json",
                codex_bin="codex-test",
            )
            self.assertIsInstance(command, list)
            self.assertEqual(command[:2], ["codex-test", "exec"])
            self.assertEqual(
                [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--image"],
                [str(pair.images[role]) for role in cli.DISPLAY_ROLES],
            )


if __name__ == "__main__":
    unittest.main()
