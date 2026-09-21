"""Regression checks for the immutable fresh-base model identity."""

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("model_profiles.py")
SPEC = importlib.util.spec_from_file_location("photo_curriculum_model_profiles", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qwen38_27b_identity_is_the_reviewed_pin() -> None:
    assert MODULE.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert MODULE.REVISION == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert MODULE.CONFIG_SHA256 == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
    assert MODULE.PRIME_REVISION == "26b3131d2716a4f8210b165df584b83b4bc54f61"
    assert MODULE.profile()["tensors"] == 992
    assert MODULE.profile()["parameters"] == 116727808
