"""Offline tests for the non-gating Prime batch observer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_module():
    spec = importlib.util.spec_from_file_location("aesthetic_train_from_sft", ROOT / "train_from_sft.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Loader:
    def __init__(self, receiver):
        self.receiver = receiver


class _DataLoader:
    @staticmethod
    def get_batch(loader):
        return {"native": loader.receiver.value}


def _train_class():
    return type("Train", (), {"DataLoader": type("DataLoader", (), {"get_batch": _DataLoader.get_batch})})


def test_private_zmq_step_metadata_does_not_gate_native_batch():
    module = _load_module()
    train = _train_class()
    calls = []

    class Receiver:
        _current_step = 7
        value = "batch"

        def _get_micro_batch_path(self):
            raise RuntimeError("optional path metadata unavailable")

    def record(*args, **kwargs):
        calls.append((args, kwargs))

    module.install_batch_observer(train, Path("/tmp/observer-test"), record)
    result = train.DataLoader.get_batch(_Loader(Receiver()))
    assert result == {"native": "batch"}
    assert calls and calls[0][1]["training_step"] == 7
    assert calls[0][1]["max_off_policy_steps"] == 1


def test_missing_step_metadata_still_returns_native_batch_and_skips_observer():
    module = _load_module()
    train = _train_class()
    calls = []

    class Receiver:
        value = "batch-without-step"

    module.install_batch_observer(
        train,
        Path("/tmp/observer-test"),
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = train.DataLoader.get_batch(_Loader(Receiver()))
    assert result == {"native": "batch-without-step"}
    assert calls == []


def test_native_dataloader_error_propagates_through_observer():
    module = _load_module()

    class FailingDataLoader:
        @staticmethod
        def get_batch(_loader):
            raise RuntimeError("native decode failed")

    class Train:
        DataLoader = FailingDataLoader

    module.install_batch_observer(Train, Path("/tmp/observer-test"), lambda *a, **k: None)
    try:
        Train.DataLoader.get_batch(_Loader(object()))
    except RuntimeError as exc:
        assert str(exc) == "native decode failed"
    else:
        raise AssertionError("native dataloader error was swallowed")
