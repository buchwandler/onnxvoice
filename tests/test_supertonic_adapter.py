from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.supertonic import SupertonicAdapter
from onnxvoice.types import Installation, InstalledArtifact

COMPONENT_INPUTS = {
    "duration_predictor": ("text_ids", "style_dp", "text_mask"),
    "text_encoder": ("text_ids", "style_ttl", "text_mask"),
    "vector_estimator": (
        "noisy_latent",
        "text_emb",
        "style_ttl",
        "text_mask",
        "latent_mask",
        "current_step",
        "total_step",
    ),
    "vocoder": ("latent",),
}


def _installation(
    root: Path,
    *,
    duration: float = 0.2,
    sample_rate: int | None = 44100,
    missing_config: bool = False,
    missing_component: str | None = None,
    config_override: dict[str, Any] | None = None,
    raw_config: str | None = None,
) -> Installation:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "tts.json"
    config_path.write_text(
        raw_config
        if raw_config is not None
        else json.dumps(
            config_override
            or {
                "ae": {"sample_rate": 44100, "base_chunk_size": 512},
                "ttl": {"chunk_compress_factor": 6, "latent_dim": 24},
            }
        ),
        encoding="utf-8",
    )
    artifacts = []
    if not missing_config:
        artifacts.append(InstalledArtifact("config", config_path.name, config_path, "config", 1))
    components = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
    for component in components:
        if component == missing_component:
            continue
        path = root / f"{component}.onnx"
        path.write_bytes(b"model")
        artifacts.append(
            InstalledArtifact("model", path.name, path, component, 5, component=component)
        )
    return Installation(
        system="supertonic",
        id="test-supertonic",
        kind="external",
        path=root,
        artifacts=tuple(artifacts),
        sample_rate=sample_rate,
        metadata={"runtime": {"layout": "supertonic-3-v1"}},
    )


class FakeSession:
    def __init__(
        self,
        component: str,
        *,
        duration: float = 0.2,
        input_names: tuple[str, ...] | None = None,
        ort_style: bool = False,
    ) -> None:
        self.component = component
        self.duration = duration
        self.input_names = input_names if input_names is not None else COMPONENT_INPUTS[component]
        self.calls: list[dict[str, np.ndarray]] = []
        self.close_count = 0
        self.ort_style = ort_style

    def run(self, *args: Any) -> list[np.ndarray]:
        if self.ort_style and len(args) == 1:
            raise TypeError("run requires output_names and inputs")
        if len(args) == 1:
            inputs = args[0]
        elif len(args) == 2 and args[0] is None:
            inputs = args[1]
        else:
            raise AssertionError("unexpected ONNX run arguments")
        saved = {name: np.array(value, copy=True) for name, value in inputs.items()}
        self.calls.append(saved)
        if self.component == "duration_predictor":
            return [np.asarray([[self.duration]], dtype=np.float32)]
        if self.component == "text_encoder":
            return [np.ones((1, 4, 3), dtype=np.float32)]
        if self.component == "vector_estimator":
            return [inputs["noisy_latent"] + np.float32(0.125)]
        return [inputs["latent"][:, 0, :]]

    def diagnostics(self) -> SimpleNamespace:
        return SimpleNamespace(component=self.component)

    def close(self) -> None:
        self.close_count += 1


def _patch_sessions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration: float = 0.2,
    input_names: dict[str, tuple[str, ...]] | None = None,
    ort_style: bool = False,
) -> tuple[dict[str, FakeSession], list[tuple[Path, dict[str, Any]]]]:
    sessions = {
        component: FakeSession(
            component,
            duration=duration,
            input_names=(input_names or {}).get(component),
            ort_style=ort_style,
        )
        for component in COMPONENT_INPUTS
    }
    constructions: list[tuple[Path, dict[str, Any]]] = []

    def factory(path: Path, **kwargs: Any) -> FakeSession:
        constructions.append((path, kwargs))
        return sessions[kwargs["component"]]

    monkeypatch.setattr("onnxvoice.systems.supertonic.OnnxSession", factory)
    return sessions, constructions


def _infer(adapter: SupertonicAdapter, **kwargs: Any):
    arguments = {
        "token_ids": [7, 11, 13],
        "text_mask": np.ones((1, 1, 3), dtype=np.float32),
        "style_ttl": np.ones((1, 4, 5), dtype=np.float64),
        "style_dp": np.ones((1, 2, 3), dtype=np.float64),
    }
    arguments.update(kwargs)
    return adapter.infer(**arguments)


def test_sessions_are_lazy_and_receive_component_provider_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, constructions = _patch_sessions(monkeypatch)
    options = [{"device_id": "3"}]
    session_options = object()
    adapter = SupertonicAdapter(
        _installation(tmp_path),
        providers=["CUDAExecutionProvider"],
        provider_options=options,
        session_options=session_options,
    )

    assert adapter._sessions == {}
    assert constructions == []
    session = adapter._get_session("duration_predictor")

    assert session is sessions["duration_predictor"]
    assert len(constructions) == 1
    path, kwargs = constructions[0]
    assert path == tmp_path / "duration_predictor.onnx"
    assert kwargs == {
        "component": "duration_predictor",
        "providers": ["CUDAExecutionProvider"],
        "provider_options": options,
        "session_options": session_options,
    }
    assert adapter._sessions == {"duration_predictor": session}


def test_diagnostics_materializes_all_four_named_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, constructions = _patch_sessions(monkeypatch)
    adapter = SupertonicAdapter(_installation(tmp_path))

    diagnostic = adapter.diagnostics()

    assert len(constructions) == 4
    assert tuple(session.component for session in diagnostic.sessions) == tuple(sessions)


def test_infer_passes_archived_inputs_dtypes_and_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, _ = _patch_sessions(monkeypatch)
    adapter = SupertonicAdapter(_installation(tmp_path))

    result = _infer(adapter, steps=3, speed=1.25, seed=7)

    assert tuple(sessions["duration_predictor"].calls[0]) == (
        "text_ids",
        "style_dp",
        "text_mask",
    )
    dp_inputs = sessions["duration_predictor"].calls[0]
    assert dp_inputs["text_ids"].shape == (1, 3)
    assert dp_inputs["text_ids"].dtype == np.int64
    assert dp_inputs["style_dp"].dtype == np.float32
    assert dp_inputs["text_mask"].dtype == np.float32
    text_inputs = sessions["text_encoder"].calls[0]
    assert set(text_inputs) == {"text_ids", "style_ttl", "text_mask"}
    assert text_inputs["style_ttl"].dtype == np.float32

    vector_calls = sessions["vector_estimator"].calls
    expected_noise = np.random.RandomState(7).randn(1, 144, 3).astype(np.float32)
    np.testing.assert_array_equal(vector_calls[0]["noisy_latent"], expected_noise)
    assert len(vector_calls) == 3
    assert [call["current_step"].tolist() for call in vector_calls] == [[0.0], [1.0], [2.0]]
    assert all(call["current_step"].dtype == np.float32 for call in vector_calls)
    assert all(call["total_step"].tolist() == [3.0] for call in vector_calls)
    assert all(call["noisy_latent"].shape == (1, 144, 3) for call in vector_calls)
    assert set(vector_calls[0]) == set(COMPONENT_INPUTS["vector_estimator"])
    assert set(sessions["vocoder"].calls[0]) == {"latent"}

    expected_duration = np.asarray([[0.2]], dtype=np.float32) / np.float32(1.25)
    np.testing.assert_allclose(result.timings, expected_duration)
    np.testing.assert_array_equal(result.outputs["duration"], result.timings)
    assert result.audio.ndim == 1
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 44100
    assert result.metadata == {
        "system": "supertonic",
        "layout": "supertonic-3-v1",
        "steps": 3,
        "speed": 1.25,
        "seed": 7,
    }
    assert set(result.outputs) == {"duration"}


def test_seed_is_deterministic_and_does_not_mutate_global_numpy_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, _ = _patch_sessions(monkeypatch)
    adapter = SupertonicAdapter(_installation(tmp_path))
    np.random.seed(12345)
    before = np.random.get_state()

    first = _infer(adapter, steps=2, seed=99)
    after = np.random.get_state()
    second = _infer(adapter, steps=2, seed=99)

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(first.audio, second.audio)
    np.testing.assert_array_equal(
        sessions["vector_estimator"].calls[0]["noisy_latent"],
        sessions["vector_estimator"].calls[2]["noisy_latent"],
    )


def test_run_supports_onnxruntime_run_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, _ = _patch_sessions(monkeypatch, ort_style=True)
    adapter = SupertonicAdapter(_installation(tmp_path))

    _infer(adapter, steps=1, seed=1)

    assert all(sessions[component].calls for component in COMPONENT_INPUTS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": 101},
        {"steps": 1.5},
        {"steps": True},
        {"speed": 0.69},
        {"speed": 2.01},
        {"speed": np.nan},
        {"seed": -1},
        {"seed": 2**32},
        {"unexpected": 1},
    ],
)
def test_invalid_arguments_fail_before_session_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    _, constructions = _patch_sessions(monkeypatch)
    adapter = SupertonicAdapter(_installation(tmp_path))

    with pytest.raises(RuntimeContractError):
        _infer(adapter, **kwargs)

    assert constructions == []


def test_missing_config_and_model_components_are_capability_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sessions(monkeypatch)
    no_config = SupertonicAdapter(_installation(tmp_path / "without-config", missing_config=True))
    with pytest.raises(CapabilityError, match="missing config artifact"):
        _infer(no_config)

    missing_model = SupertonicAdapter(
        _installation(tmp_path / "without-model", missing_component="duration_predictor")
    )
    with pytest.raises(CapabilityError, match="duration_predictor"):
        _infer(missing_model)


def test_malformed_config_and_missing_config_key_are_contract_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sessions(monkeypatch)
    malformed = SupertonicAdapter(_installation(tmp_path / "malformed", raw_config="{"))
    with pytest.raises(RuntimeContractError, match="Could not read Supertonic config"):
        _infer(malformed)

    incomplete_config = {
        "ae": {"sample_rate": 44100},
        "ttl": {"chunk_compress_factor": 6, "latent_dim": 24},
    }
    incomplete = SupertonicAdapter(
        _installation(tmp_path / "incomplete", config_override=incomplete_config)
    )
    with pytest.raises(RuntimeContractError, match="ae.base_chunk_size"):
        _infer(incomplete)


def test_installation_sample_rate_must_match_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sessions(monkeypatch)
    adapter = SupertonicAdapter(_installation(tmp_path, sample_rate=22050))

    with pytest.raises(RuntimeContractError, match="disagrees with Supertonic config"):
        _infer(adapter)


def test_input_names_are_checked_before_running_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, _ = _patch_sessions(
        monkeypatch, input_names={"duration_predictor": ("text_ids", "text_mask")}
    )
    adapter = SupertonicAdapter(_installation(tmp_path))

    with pytest.raises(RuntimeContractError, match="duration_predictor input contract mismatch"):
        _infer(adapter)

    assert sessions["duration_predictor"].calls == []


def test_close_is_idempotent_and_closes_every_created_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, _ = _patch_sessions(monkeypatch)
    adapter = SupertonicAdapter(_installation(tmp_path))
    adapter._get_session("duration_predictor")
    adapter._get_session("text_encoder")

    adapter.close()
    adapter.close()

    assert sessions["duration_predictor"].close_count == 1
    assert sessions["text_encoder"].close_count == 1
    assert all(sessions[name].close_count == 0 for name in ("vector_estimator", "vocoder"))
    assert adapter._sessions == {}


def test_missing_config_setting_reports_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sessions(monkeypatch)
    config = {"ae": {"sample_rate": 44100, "base_chunk_size": 512}, "ttl": {"latent_dim": 24}}
    adapter = SupertonicAdapter(_installation(tmp_path, config_override=config))

    with pytest.raises(RuntimeContractError, match="ttl.chunk_compress_factor"):
        _infer(adapter)
