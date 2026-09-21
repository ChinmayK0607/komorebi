"""Pure checks for reviewed teacher programs in the repaired curriculum.

The prompt and contract remain the canonical ``painter/contract.py``.  This
module adds the missing placement check without changing the frozen historical
contract or silently rewriting student programs.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re


RUN_ROOT = Path(__file__).resolve().parent
CANONICAL_CONTRACT = RUN_ROOT.parents[1] / "contract.py"


def canonical_contract_sha256() -> str:
    return hashlib.sha256(CANONICAL_CONTRACT.read_bytes()).hexdigest()


def _canonical_module():
    spec = importlib.util.spec_from_file_location("painter_canonical_contract", CANONICAL_CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical contract: {CANONICAL_CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draw_body(program: str) -> str:
    match = re.search(r"\bfunction\s+draw\s*\([^)]*\)\s*\{", program)
    if match is None:
        raise ValueError("Teacher must define draw()")
    start = match.end() - 1
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(program)):
        char = program[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return program[start + 1:index]
    raise ValueError("Teacher draw() body is not balanced")


def validate_teacher_program(program: str) -> None:
    """Run canonical checks and require the single transform inside draw()."""
    _canonical_module().validate_teacher_program(program)
    body = _draw_body(program)
    if not re.search(r"\btranslate\s*\(\s*-300\s*,\s*-300\s*\)", body):
        raise ValueError("Teacher translate(-300,-300) must be inside draw()")


__all__ = ["CANONICAL_CONTRACT", "canonical_contract_sha256", "validate_teacher_program"]
