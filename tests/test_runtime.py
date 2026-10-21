from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from onnxvoice import runtime
from onnxvoice.errors import RuntimeContractError


class FakeSessionOptions:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries.append((key, value))


class FakeOrt:
    def __init__(self) -> None:
        self.created: tuple[str, dict[str, Any]] | None = None
        self.session_options: FakeSessionOptions | None = None

    def get_available_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def SessionOptions(self) -> FakeSessionOptions:
        self.session_options = FakeSessionOptions()
        return self.session_options

    def InferenceSession(self, model, **kwargs):
        self.created = (model, kwargs)
        return SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(name="tokens", type="tensor(int32)", shape=[1, "N"])
            ],
            get_outputs=lambda: [SimpleNamespace(name="audio", type="tensor(float)", shape=["N"])],
            run=lambda *_: [],
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            disable_fallback=lambda: None,
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
    assert session.active_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert session.diagnostics().providers_active == session.active_providers
    assert ort.created is not None
    assert ort.created[1]["provider_options"] == [{"device_id": "1"}]
    assert ort.created[1]["sess_options"] is session_options


def test_strict_provider_policy_disables_implicit_cpu_fallback(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    ort = FakeOrt()
    monkeypatch.setattr(runtime, "_ort", lambda: ort)
    session = runtime.OnnxSession(model, providers="cuda", strict_providers=True)
    _ = session.session
    assert ort.session_options is not None
    assert ort.session_options.entries == [("session.disable_cpu_ep_fallback", "1")]
    assert session.active_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_session_diagnostics_capture_environment_provider_preference(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    ort = FakeOrt()
    monkeypatch.setattr(runtime, "_ort", lambda: ort)
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)
    monkeypatch.setenv("ONNXVOICE_PROVIDER", "cuda")

    session = runtime.OnnxSession(model, providers=None)
    monkeypatch.setenv("ONNXVOICE_PROVIDER", "cpu")
    diagnostic = session.diagnostics()

    assert diagnostic.providers_requested == ("CUDAExecutionProvider",)
    assert diagnostic.provider_source == "ONNXVOICE_PROVIDER"
    assert diagnostic.providers_resolved == ("CUDAExecutionProvider",)
    assert diagnostic.providers_active == ("CUDAExecutionProvider", "CPUExecutionProvider")
