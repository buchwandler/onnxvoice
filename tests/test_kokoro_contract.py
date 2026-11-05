from __future__ import annotations

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.kokoro import KokoroAdapter
from onnxvoice.types import Installation, InstalledArtifact, TensorSpec


class ContractSession:
    input_names = ("input_ids", "ref_s", "speed")
    input_specs = (
        TensorSpec("input_ids", "tensor(int32)", (1, 6)),
        TensorSpec("ref_s", "tensor(float16)", (1, 3)),
        TensorSpec("speed", "tensor(float)", (1,)),
    )
    output_names = ("audio", "durations")

    def __init__(self):
        self.seen = None

    def run(self, inputs):
        self.seen = inputs
        return [np.array([[0.1, 0.2]], dtype=np.float16), np.array([1, 2], dtype=np.int32)]

    def close(self):
        pass


def _installation(tmp_path, *, split=False, metadata=None):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    artifacts = [InstalledArtifact("model", model.name, model, "0" * 64, model.stat().st_size)]
    if split:
        second = tmp_path / "decoder.onnx"
        second.write_bytes(b"decoder")
        artifacts.append(
            InstalledArtifact("model", second.name, second, "1" * 64, second.stat().st_size)
        )
    return Installation(
        "kokoro",
        "v1",
        "model",
        tmp_path,
        tuple(artifacts),
        sample_rate=24000,
        metadata=metadata or {},
    )


def test_kokoro_accepts_explicit_style_and_named_timing_outputs(tmp_path):
    adapter = KokoroAdapter(
        _installation(tmp_path, metadata={"runtime": {"timings_output": "durations"}})
    )
    fake = ContractSession()
    adapter._session = fake  # type: ignore[assignment]

    result = adapter.infer([5, 6], style=np.array([1, 2, 3]), speed=1.0, language="en-US")

    np.testing.assert_array_equal(fake.seen["input_ids"], np.array([[0, 5, 6, 0]], dtype=np.int32))
    assert fake.seen["ref_s"].dtype == np.float16
    assert result.audio.dtype == np.float32
    np.testing.assert_array_equal(result.timings, np.array([1, 2], dtype=np.int32))
    assert set(result.outputs) == {"durations"}
    assert "voice" not in result.metadata


def test_kokoro_requires_explicit_style(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path))
    adapter._session = ContractSession()  # type: ignore[assignment]

    with pytest.raises(RuntimeContractError, match="model-ready Kokoro style"):
        adapter.infer([1, 2])


def test_kokoro_enforces_graph_token_limit(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path))
    adapter._session = ContractSession()  # type: ignore[assignment]

    with pytest.raises(RuntimeContractError, match="maximum"):
        adapter.infer([1, 2, 3, 4, 5], style=np.zeros(3))


def test_kokoro_rejects_split_layout(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path, split=True))

    with pytest.raises(CapabilityError, match="Split"):
        _ = adapter.session


def test_installation_timing_output_tolerates_missing_metadata(tmp_path):
    assert _installation(tmp_path).timing_output is None
    assert (
        _installation(tmp_path, metadata={"runtime": {"timings_output": "durations"}}).timing_output
        == "durations"
    )
    assert (
        _installation(tmp_path, metadata={"runtime": {"timings_output": 1}}).timing_output is None
    )


def _language_router_installation(tmp_path, *, runtime=None):
    artifacts = []
    for component in ("model_en", "model_de"):
        path = tmp_path / f"{component}.onnx"
        path.write_bytes(component.encode())
        artifacts.append(
            InstalledArtifact(
                "model",
                path.name,
                path,
                "0" * 64,
                path.stat().st_size,
                component=component,
            )
        )
    metadata = {
        "runtime": {
            "layout": "language-router-onnx-v1",
            "language_models": {"en": "model_en", "de": "model_de"},
            "language_codes": ["en", "de"],
        }
    }
    if runtime is not None:
        metadata["runtime"].update(runtime)
    return Installation(
        "kokoro",
        "akinvox-multispeaker-v1",
        "model",
        tmp_path,
        tuple(artifacts),
        sample_rate=24000,
        metadata=metadata,
    )


def test_language_router_adapter_dispatches_by_language_and_closes_sessions(tmp_path):
    sessions = {}
    calls = []

    def factory(path, component):
        calls.append((path, component))
        sessions[component] = ContractSession()
        return sessions[component]

    adapter = KokoroAdapter(_language_router_installation(tmp_path), session_factory=factory)

    adapter.infer([5, 6], style=np.array([1, 2, 3]), language="en-US")
    adapter.infer([5, 6], style=np.array([1, 2, 3]), language="de-DE")

    assert [component for _, component in calls] == ["en:model_en", "de:model_de"]
    assert adapter.diagnostics().properties["language_models"] == {
        "en": "model_en",
        "de": "model_de",
    }
    with pytest.raises(CapabilityError, match="language-router"):
        _ = adapter.session

    adapter.close()
    adapter.close()


def test_language_router_adapter_requires_supported_language(tmp_path):
    adapter = KokoroAdapter(_language_router_installation(tmp_path))

    with pytest.raises(RuntimeContractError, match="language is required"):
        adapter.infer([1], style=np.ones(3))
    with pytest.raises(RuntimeContractError, match="Unsupported Kokoro language-router"):
        adapter.infer([1], style=np.ones(3), language="fr")


def test_existing_kokoro_layout_accepts_optional_language(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path))
    adapter._session = ContractSession()  # type: ignore[assignment]

    adapter.infer([1], style=np.ones(3), language="en-US")

    with pytest.raises(RuntimeContractError, match="language"):
        adapter.infer([1], style=np.ones(3), language="")
