from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.piper import PiperAdapter
from onnxvoice.types import Installation, InstalledArtifact, TensorSpec


class PiperSession:
    input_names = ("input", "input_lengths", "scales", "sid")
    input_specs = (
        TensorSpec("input", "tensor(int32)", (1, "N")),
        TensorSpec("input_lengths", "tensor(int64)", (1,)),
        TensorSpec("scales", "tensor(float16)", (3,)),
        TensorSpec("sid", "tensor(int64)", (1,)),
    )
    output_names = ("audio",)

    def __init__(self, names=None):
        self.input_names = tuple(names or self.input_names)
        self.seen: dict[str, np.ndarray] | None = None

    def run(self, inputs):
        self.seen = inputs
        return [np.array([[0.1, 0.2]], dtype=np.float16)]

    def close(self):
        pass


def _installation(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    config = tmp_path / "voice.json"
    config.write_text(json.dumps({"audio": {"sample_rate": 16000}}), encoding="utf-8")
    return Installation(
        "piper",
        "voice",
        "voice",
        tmp_path,
        (
            InstalledArtifact("model", model.name, model, "0" * 64, model.stat().st_size),
            InstalledArtifact("config", config.name, config, "1" * 64, config.stat().st_size),
        ),
    )


def test_piper_forwards_model_ready_speaker_id_and_graph_dtypes(tmp_path):
    adapter = PiperAdapter(_installation(tmp_path))
    fake = PiperSession()
    adapter._session = fake  # type: ignore[assignment]

    result = adapter.infer([1, 2], speaker_id=1)

    assert adapter.requires_speaker_id is True
    assert result.sample_rate == 16000
    assert result.audio.dtype == np.float32
    assert fake.seen is not None
    assert fake.seen["input"].dtype == np.int32
    assert fake.seen["scales"].dtype == np.float16
    assert fake.seen["sid"].dtype == np.int64
    assert fake.seen["sid"].item() == 1
    assert adapter._config() is adapter._config()


def test_piper_requires_speaker_id_when_sid_exists(tmp_path):
    adapter = PiperAdapter(_installation(tmp_path))
    adapter._session = PiperSession()  # type: ignore[assignment]

    with pytest.raises(RuntimeContractError, match="speaker_id was not provided"):
        adapter.infer([1])


def test_piper_rejects_speaker_id_when_sid_is_absent(tmp_path):
    adapter = PiperAdapter(_installation(tmp_path))
    adapter._session = PiperSession(("input", "input_lengths", "scales"))  # type: ignore[assignment]

    assert adapter.requires_speaker_id is False
    with pytest.raises(RuntimeContractError, match="no input 'sid'"):
        adapter.infer([1], speaker_id=0)


def test_piper_passes_provider_and_session_options(tmp_path, monkeypatch):
    import onnxvoice.systems.piper as piper_module

    captured = {}
    options = object()

    class CapturingSession:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(piper_module, "OnnxSession", CapturingSession)
    adapter = PiperAdapter(
        _installation(tmp_path),
        providers=("cpu",),
        provider_options={"cpu": {"arena_extend_strategy": "kNextPowerOfTwo"}},
        session_options=options,
    )
    _ = adapter.session

    assert captured["providers"] == ("cpu",)
    assert captured["provider_options"] == {"cpu": {"arena_extend_strategy": "kNextPowerOfTwo"}}
    assert captured["session_options"] is options


class NoInferencePiperSession:
    def __init__(self, input_specs):
        self.input_specs = tuple(input_specs)
        self.run_called = False

    def run(self, _inputs):
        self.run_called = True
        raise AssertionError("capacity inspection must not execute inference")


def _capacity_adapter(tmp_path, input_specs):
    adapter = PiperAdapter(_installation(tmp_path))
    fake = NoInferencePiperSession(input_specs)
    adapter._session = fake  # type: ignore[assignment]
    return adapter, fake


@pytest.mark.parametrize("dimension", ["N", "phonemes", "sequence_length"])
def test_piper_sequence_contract_reports_symbolic_axis_as_variable(tmp_path, dimension):
    adapter, fake = _capacity_adapter(
        tmp_path, (TensorSpec("input", "tensor(int32)", (1, dimension)),)
    )

    contract = adapter.phoneme_sequence_contract

    assert contract.kind == "variable"
    assert contract.dimension == dimension
    assert contract.axis == 1
    assert contract.source == "onnx.input.shape"
    assert fake.run_called is False


def test_piper_sequence_contract_reports_integer_axis_as_fixed(tmp_path):
    adapter, fake = _capacity_adapter(tmp_path, (TensorSpec("input", "tensor(int32)", (1, 256)),))

    contract = adapter.phoneme_sequence_contract

    assert contract.kind == "fixed"
    assert contract.dimension == 256
    assert contract.axis == 1
    assert contract.source == "onnx.input.shape"
    assert fake.run_called is False


@pytest.mark.parametrize("dimension", [None, True, 0, -1, ""])
def test_piper_sequence_contract_classifies_unsupported_dimensions_as_unknown(tmp_path, dimension):
    adapter, fake = _capacity_adapter(
        tmp_path, (TensorSpec("input", "tensor(int32)", (1, dimension)),)
    )

    contract = adapter.phoneme_sequence_contract

    assert contract.kind == "unknown"
    assert contract.dimension == dimension
    assert contract.axis == 1
    assert contract.source == "onnx.input.shape"
    assert fake.run_called is False


@pytest.mark.parametrize(
    "input_specs",
    [
        (),
        (TensorSpec("other", "tensor(int32)", (1, 256)),),
        (TensorSpec("input", "tensor(int32)", ()),),
        (TensorSpec("input", "tensor(int32)", (256,)),),
        (TensorSpec("input", "tensor(int32)", cast(Any, None)),),
        (TensorSpec("input", "tensor(int32)", cast(Any, [])),),
    ],
)
def test_piper_sequence_contract_is_unknown_for_missing_or_malformed_input(tmp_path, input_specs):
    adapter, fake = _capacity_adapter(tmp_path, input_specs)

    contract = adapter.phoneme_sequence_contract

    assert contract.kind == "unknown"
    assert contract.dimension is None
    assert contract.axis == 1
    assert contract.source == "onnx.input.shape"
    assert fake.run_called is False


def test_piper_max_phonemes_does_not_treat_fixed_dimension_as_a_limit(tmp_path):
    adapter, fake = _capacity_adapter(tmp_path, (TensorSpec("input", "tensor(int32)", (1, 256)),))

    assert adapter.max_phonemes is None
    assert fake.run_called is False
