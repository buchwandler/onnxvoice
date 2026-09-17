from __future__ import annotations

from types import SimpleNamespace

import pytest

from onnxvoice import runtime
from onnxvoice.errors import RuntimeContractError


class FakeOrt:
    def __init__(self):
        self.created = None

    def get_available_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def InferenceSession(self, model, **kwargs):
        self.created = (model, kwargs)
        return SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="tokens", type="tensor(int32)", shape=[1, "N"])],
            get_outputs=lambda: [SimpleNamespace(name="audio", type="tensor(float)", shape=["N"])],
            run=lambda *_: [],
        )


def test_provider_alias_and_auto_policy():
    assert runtime.resolve_providers("gpu", available=["CUDAExecutionProvider"]) == (
        "CUDAExecutionProvider",
    )
    assert runtime.resolve_providers("auto", available=["CPUExecutionProvider"]) == (
        "CPUExecutionProvider",
    )
    assert runtime.resolve_providers(
        None,
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
        environment={"ONNXVOICE_PROVIDER": "cuda"},
    ) == ("CUDAExecutionProvider",)


def test_explicit_missing_provider_is_not_silently_replaced():
    with pytest.raises(RuntimeContractError, match="unavailable"):
        runtime.resolve_providers("cuda", available=["CPUExecutionProvider"])


def test_session_passes_options_and_exposes_tensor_specs(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    ort = FakeOrt()
    monkeypatch.setattr(runtime, "_ort", lambda: ort)
    session_options = object()

    session = runtime.OnnxSession(
        model,
        providers="cuda",
        provider_options={"cuda": {"device_id": "1"}},
        session_options=session_options,
    )

    assert session.input_specs[0].name == "tokens"
    assert session.input_specs[0].ort_type == "tensor(int32)"
    assert session.output_specs[0].shape == ("N",)
    assert session.resolved_providers == ("CUDAExecutionProvider",)
    assert ort.created[1]["provider_options"] == [{"device_id": "1"}]
    assert ort.created[1]["sess_options"] is session_options
