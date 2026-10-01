from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from onnxvoice import OnnxVoice, registered_systems
from onnxvoice.errors import CapabilityError
from onnxvoice.systems import KittenAdapter
from onnxvoice.types import Installation, TensorSpec


class SessionStub:
    input_names = ("input_ids", "style", "speed")
    input_specs = (
        TensorSpec("input_ids", "tensor(int64)", (1, "tokens")),
        TensorSpec("style", "tensor(float)", (1, 4)),
        TensorSpec("speed", "tensor(float)", (1,)),
    )

    def run(self, _inputs: dict[str, Any]) -> list[Any]:
        return []


def test_kitten_adapter_is_registered_and_exported() -> None:
    assert "kitten" in registered_systems()
    assert KittenAdapter.system == "kitten"


def test_open_local_keeps_roles_metadata_and_provider_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.onnx"
    voices = tmp_path / "voices.npz"
    model.write_bytes(b"fake model")
    voices.write_bytes(b"not opened by OnnxVoice")
    created: list[tuple[Path, dict[str, Any]]] = []
    session = SessionStub()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("local Kitten opening must not access catalogs or the asset store")

    def create_session(path: Path, **kwargs: Any) -> SessionStub:
        created.append((path, kwargs))
        return session

    monkeypatch.setattr("onnxvoice.manager.CatalogClient", forbidden)
    monkeypatch.setattr("onnxvoice.manager.AssetStore", forbidden)
    monkeypatch.setattr("onnxvoice.systems.kitten.OnnxSession", create_session)
    provider_options = [{"device_id": "2"}]
    session_options = object()
    metadata = {
        "voice_aliases": {"Bella": "expr-voice-2-f"},
        "speed_priors": {"expr-voice-2-f": 0.8},
        "runtime": {"profile": "ONNX2"},
    }

    runtime = OnnxVoice.open_local(
        system="kitten",
        model=model,
        voices=voices,
        sample_rate=24000,
        metadata=metadata,
        providers=["CUDAExecutionProvider"],
        provider_options=provider_options,
        session_options=session_options,
    )
    try:
        assert isinstance(runtime, KittenAdapter)
        installation = runtime.installation
        assert installation.system == "kitten"
        assert installation.sample_rate == 24000
        assert installation.artifact("model").path == model.resolve()
        assert installation.artifact("voices").path == voices.resolve()
        assert installation.metadata["voice_aliases"] == metadata["voice_aliases"]
        assert installation.metadata["speed_priors"] == metadata["speed_priors"]
        assert installation.metadata["runtime"] == metadata["runtime"]
        assert created == []

        assert runtime.session is session
        assert created == [
            (
                model.resolve(),
                {
                    "providers": ["CUDAExecutionProvider"],
                    "provider_options": provider_options,
                    "session_options": session_options,
                },
            )
        ]
    finally:
        runtime.close()


def test_kitten_runtime_reports_missing_model_artifact(tmp_path: Path) -> None:

    adapter = KittenAdapter(
        Installation(
            system="kitten",
            id="broken",
            kind="external",
            path=tmp_path,
            artifacts=(),
            sample_rate=24000,
        )
    )

    with pytest.raises(CapabilityError, match="missing its model artifact"):
        _ = adapter.session
