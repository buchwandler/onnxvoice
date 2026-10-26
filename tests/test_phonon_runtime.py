from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.phonon import PhononAdapter, PhononStreamChunk, PhononVoiceState
from onnxvoice.types import Installation, InstalledArtifact, SessionDiagnostic, TensorSpec

_GRAPH_FILES = {
    "text_conditioner": "onnx/text_conditioner.onnx",
    "flow_lm_main": "onnx/flow_lm_main.onnx",
    "flow_lm_flow": "onnx/flow_lm_flow.onnx",
    "mimi_decoder": "onnx/mimi_decoder.onnx",
}
_RUNTIME = {
    "kind": "phonon-streaming-onnx-v1",
    "graphs": list(_GRAPH_FILES.values()),
    "sample_rate": 8,
    "frame_samples": 4,
    "frame_rate": 2.0,
    "d_model": 4,
    "latent_dim": 3,
    "num_layers": 2,
    "num_heads": 2,
    "head_dim": 2,
    "mimi_state": [{"name": "cache", "shape": [1, 2], "dtype": "float32"}],
}
_GENERATION = {
    "duration_delta_default": 0.25,
    "temperature_default": 0.5,
    "eos_threshold_default": 1.0,
    "frames_after_eos": 2,
    "min_frames_before_eos": 1,
    "max_frames": 5,
}


def _spec(name: str, ort_type: str, *shape: Any) -> TensorSpec:
    return TensorSpec(name, ort_type, tuple(shape))


class _FakeSession:
    def __init__(
        self,
        component: str,
        input_specs: tuple[TensorSpec, ...],
        output_specs: tuple[TensorSpec, ...],
        run: Any,
        owner: _FakeGraphs,
        model_path: Path,
        kwargs: Mapping[str, Any],
    ) -> None:
        self.component = component
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.input_names = tuple(spec.name for spec in input_specs)
        self.output_names = tuple(spec.name for spec in output_specs)
        self._run = run
        self.owner = owner
        self.model_path = model_path
        self.kwargs = dict(kwargs)
        self.calls: list[dict[str, np.ndarray]] = []
        self.closed = False

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        captured = {name: np.array(value, copy=True) for name, value in feeds.items()}
        self.calls.append(captured)
        return self._run(feeds)

    def close(self) -> None:
        self.closed = True

    def diagnostics(self) -> SessionDiagnostic:
        return SessionDiagnostic(
            component=self.component,
            model_path=self.model_path,
            providers_requested=tuple(self.kwargs.get("providers") or ()),
            providers_active=(),
            inputs=self.input_names,
            outputs=self.output_names,
        )


class _FakeGraphs:
    def __init__(
        self,
        *,
        eos_values: tuple[float, ...] = (-1.0, 5.0, 5.0, 5.0, 5.0),
        invalid_audio: bool = False,
        invalid_main_signature: bool = False,
    ) -> None:
        self.eos_values = eos_values
        self.invalid_audio = invalid_audio
        self.invalid_main_signature = invalid_main_signature
        self.sessions: dict[str, _FakeSession] = {}
        self.events: list[tuple[str, str]] = []
        self.main_stage = ""
        self.generation_step = 0
        self.main_calls: list[dict[str, np.ndarray]] = []
        self.flow_calls: list[dict[str, np.ndarray]] = []
        self.mimi_calls: list[dict[str, np.ndarray]] = []

    def create(self, model_path: Path, *, component: str, **kwargs: Any) -> _FakeSession:
        if component == "text_conditioner":
            inputs = (
                _spec("input_ids", "tensor(int64)", 1, "sequence"),
                _spec("duration_delta", "tensor(float)", 1),
            )
            outputs = (
                _spec("inputs_embeds", "tensor(float)", 1, "sequence", 4),
                _spec("bos_embeds", "tensor(float)", 1, 1, 4),
            )

            def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                self.events.append((component, "text"))
                ids = feeds["input_ids"][0].astype(np.float32)
                embeddings = np.repeat(ids[None, :, None], 4, axis=2)
                return {
                    "inputs_embeds": embeddings,
                    "bos_embeds": np.full((1, 1, 4), 9.0, dtype=np.float32),
                }

        elif component == "flow_lm_main":
            kv_inputs = tuple(
                _spec(f"past_key_values.{layer}.{kind}", "tensor(float)", 1, 2, "past", 2)
                for layer in range(2)
                for kind in ("key", "value")
            )
            inputs = (_spec("inputs_embeds", "tensor(float)", 1, "sequence", 4), *kv_inputs)
            kv_outputs = tuple(
                _spec(f"present.{layer}.{kind}", "tensor(float)", 1, 2, "total", 2)
                for layer in range(2)
                for kind in ("key", "value")
            )
            outputs = (
                _spec("hidden", "tensor(float)", 1, "sequence", 4),
                _spec("eos_logit", "tensor(float)", 1, 1),
                *kv_outputs,
            )
            if self.invalid_main_signature:
                outputs = tuple(spec for spec in outputs if spec.name != "eos_logit")

            def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                embeds = feeds["inputs_embeds"]
                past = feeds["past_key_values.0.key"]
                past_length = past.shape[-2]
                if past_length == 0:
                    self.main_stage = "voice"
                    self.generation_step = 0
                elif self.main_stage == "voice":
                    self.main_stage = "text"
                else:
                    self.main_stage = "generation"
                self.events.append((component, self.main_stage))
                self.main_calls.append(
                    {name: np.array(value, copy=True) for name, value in feeds.items()}
                )
                if self.main_stage == "generation":
                    index = min(self.generation_step, len(self.eos_values) - 1)
                    eos = self.eos_values[index]
                    self.generation_step += 1
                else:
                    eos = 0.0
                result = {
                    "hidden": embeds[:, -1:, :].astype(np.float32, copy=True),
                    "eos_logit": np.asarray([[eos]], dtype=np.float32),
                }
                extension = np.full(
                    (1, 2, embeds.shape[1], 2), len(self.main_calls), dtype=np.float32
                )
                for layer in range(2):
                    for kind in ("key", "value"):
                        name = f"past_key_values.{layer}.{kind}"
                        result[f"present.{layer}.{kind}"] = np.concatenate(
                            (feeds[name], extension), axis=2
                        )
                return result

        elif component == "flow_lm_flow":
            inputs = (
                _spec("hidden", "tensor(float)", 1, 1, 4),
                _spec("noise", "tensor(float)", 1, 1, 3),
                _spec("duration_delta", "tensor(float)", 1),
            )
            outputs = (
                _spec("latent", "tensor(float)", 1, 1, 3),
                _spec("next_embeds", "tensor(float)", 1, 1, 4),
            )

            def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                self.events.append((component, "flow"))
                self.flow_calls.append(
                    {name: np.array(value, copy=True) for name, value in feeds.items()}
                )
                delta = np.float32(feeds["noise"].mean())
                return {
                    "latent": feeds["noise"].copy(),
                    "next_embeds": feeds["hidden"] + delta,
                }

        elif component == "mimi_decoder":
            inputs = (
                _spec("latent", "tensor(float)", 1, 1, 3),
                _spec("offset", "tensor(int64)", 1),
                _spec("state.cache", "tensor(float)", 1, 2),
            )
            outputs = (
                _spec("audio", "tensor(float)", 1, 4),
                _spec("next_offset", "tensor(int64)", 1),
                _spec("next_state.cache", "tensor(float)", 1, 2),
            )

            def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                self.events.append((component, "mimi"))
                self.mimi_calls.append(
                    {name: np.array(value, copy=True) for name, value in feeds.items()}
                )
                if self.invalid_audio:
                    audio = np.full((1, 4), np.nan, dtype=np.float32)
                else:
                    audio = np.full((1, 4), float(feeds["offset"][0] + 1), dtype=np.float32)
                return {
                    "audio": audio,
                    "next_offset": feeds["offset"] + 1,
                    "next_state.cache": feeds["state.cache"] + 1,
                }

        else:
            raise AssertionError(f"unexpected graph component {component!r}")

        session = _FakeSession(
            component, tuple(inputs), tuple(outputs), run, self, model_path, kwargs
        )
        self.sessions[component] = session
        return session


def _make_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake: _FakeGraphs | None = None,
    default_voice: str | None = "easton",
    voice_data: np.ndarray | None = None,
    generation_overrides: Mapping[str, Any] | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> tuple[PhononAdapter, _FakeGraphs]:
    root = tmp_path / "installation"
    root.mkdir(parents=True)
    runtime = {**_RUNTIME, **(runtime_overrides or {})}
    generation = {**_GENERATION, **(generation_overrides or {})}
    voices = [{"name": "easton", "display_name": "Easton", "locale": "en-US", "frames": 2}]
    config = {"runtime": runtime, "generation": generation, "voices": voices}
    if default_voice is not None:
        config["default_voice"] = default_voice
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifacts: list[InstalledArtifact] = []

    def add_artifact(role: str, filename: str, path: Path, *, component: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"fake")
        artifacts.append(
            InstalledArtifact(
                role=role,
                filename=filename,
                path=path,
                sha256="0" * 64,
                size=path.stat().st_size,
                component=component,
            )
        )

    add_artifact("config", "config.json", config_path)
    for component, filename in _GRAPH_FILES.items():
        add_artifact("model", filename, root / filename, component=component)
    prompt_path = root / "voices" / "easton.f32"
    prompt = (
        np.arange(8, dtype="<f4") if voice_data is None else np.asarray(voice_data, dtype="<f4")
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt.tofile(prompt_path)
    add_artifact("voice_prompt", "voices/easton.f32", prompt_path, component="easton")

    metadata = {
        "runtime": dict(_RUNTIME),
        "generation": generation,
        "voices": ["easton"],
        "selected_quality": "fp32",
    }
    installation = Installation(
        system="phonon",
        id="phonon-31m-en",
        kind="voice-model",
        path=root,
        artifacts=tuple(artifacts),
        sample_rate=cast(int, runtime["sample_rate"]),
        voices=("easton",),
        default_voice=default_voice,
        metadata=metadata,
    )
    fake_graphs = fake or _FakeGraphs()
    monkeypatch.setattr("onnxvoice.systems.phonon.OnnxSession", fake_graphs.create)
    adapter = PhononAdapter(
        installation,
        providers=("CPUExecutionProvider",),
        provider_options=({"arena_extend_strategy": "kSameAsRequested"},),
        session_options=object(),
    )
    return adapter, fake_graphs


def test_stream_runs_prefix_cache_flow_mimi_and_eos_tail_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, fake = _make_adapter(tmp_path, monkeypatch)
    voice = adapter.prepare_voice()
    assert voice.name == "easton"
    assert voice.embeddings.shape == (1, 2, 4)
    assert voice.embeddings.dtype == np.float32
    assert voice.embeddings.flags.c_contiguous and not voice.embeddings.flags.writeable
    assert fake.sessions == {}

    controls: dict[str, Any] = {
        "duration_delta": 0.75,
        "temperature": 0.4,
        "seed": 42,
        "eos_threshold": 1.0,
        "frames_after_eos": 2,
        "min_frames_before_eos": 1,
        "max_frames": 5,
    }
    chunks = list(adapter.stream([11, 12], **controls))
    assert [chunk.frame_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.eos_detected for chunk in chunks] == [False, True, False, False]
    assert all(chunk.sample_rate == 8 and chunk.audio.shape == (4,) for chunk in chunks)
    assert [chunk.audio[0] for chunk in chunks] == [1.0, 2.0, 3.0, 4.0]
    assert not chunks[0].audio.flags.writeable
    assert list(fake.events[:4]) == [
        ("flow_lm_main", "voice"),
        ("text_conditioner", "text"),
        ("flow_lm_main", "text"),
        ("flow_lm_main", "generation"),
    ]
    assert fake.main_calls[0]["inputs_embeds"].shape == (1, 2, 4)
    np.testing.assert_array_equal(fake.main_calls[0]["inputs_embeds"], voice.embeddings)
    np.testing.assert_array_equal(fake.main_calls[2]["inputs_embeds"], np.full((1, 1, 4), 9.0))
    assert fake.main_calls[0]["past_key_values.0.key"].shape[2] == 0
    assert fake.main_calls[1]["past_key_values.0.key"].shape[2] == 2
    np.testing.assert_array_equal(
        fake.sessions["text_conditioner"].calls[0]["input_ids"], [[11, 12]]
    )
    np.testing.assert_array_equal(
        fake.sessions["text_conditioner"].calls[0]["duration_delta"], [0.75]
    )
    assert all(call["duration_delta"].tolist() == [0.75] for call in fake.flow_calls)
    assert [call["offset"].tolist() for call in fake.mimi_calls] == [[0], [1], [2], [3]]
    assert [call["state.cache"][0, 0] for call in fake.mimi_calls] == [0.0, 1.0, 2.0, 3.0]
    assert set(fake.sessions) == set(_GRAPH_FILES)
    assert all(
        session.kwargs["providers"] == ("CPUExecutionProvider",)
        for session in fake.sessions.values()
    )

    second = list(adapter.stream([11, 12], **controls))
    np.testing.assert_array_equal(
        np.stack([call["noise"] for call in fake.flow_calls[:4]]),
        np.stack([call["noise"] for call in fake.flow_calls[4:8]]),
    )
    voice_prefixes = [
        call for call in fake.main_calls if np.array_equal(call["inputs_embeds"], voice.embeddings)
    ]
    assert len(voice_prefixes) == 2
    assert all(call["past_key_values.0.key"].shape[2] == 0 for call in voice_prefixes)
    assert fake.mimi_calls[4]["offset"].tolist() == [0]
    assert fake.mimi_calls[4]["state.cache"].tolist() == [[0.0, 0.0]]
    assert [chunk.frame_index for chunk in second] == [0, 1, 2, 3]

    silent = list(
        adapter.stream(
            [11],
            duration_delta=0.75,
            temperature=0.0,
            seed=42,
            eos_threshold=100.0,
            max_frames=1,
        )
    )
    np.testing.assert_array_equal(
        fake.flow_calls[-1]["noise"], np.zeros((1, 1, 3), dtype=np.float32)
    )
    assert len(silent) == 1


def test_infer_collects_stream_audio_and_reports_frame_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = _make_adapter(tmp_path, monkeypatch, fake=_FakeGraphs(eos_values=(-3.0,) * 8))
    result = adapter.infer(
        [1, 2], duration_delta=0.25, temperature=0.0, eos_threshold=10.0, max_frames=2
    )
    assert result.audio.dtype == np.float32 and result.audio.shape == (8,)
    np.testing.assert_array_equal(result.audio, np.concatenate((np.ones(4), np.full(4, 2))))
    assert result.sample_rate == 8
    assert result.metadata["frames"] == 2
    assert result.metadata["terminated_by"] == "max_frames"
    assert result.metadata["terminated_by_frame_cap"] is True
    assert result.metadata["voice"] == "easton"
    assert result.metadata["quality"] == "fp32"


def test_infer_preserves_empty_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _make_adapter(tmp_path, monkeypatch)
    adapter._load_contract()

    def empty_stream(*args: Any, **kwargs: Any) -> Any:
        assert args[0] == [1]
        assert kwargs["duration_delta"] == 0.25
        return iter(())

    monkeypatch.setattr(adapter, "stream", empty_stream)
    result = adapter.infer([1], duration_delta=0.25, max_frames=2)
    assert result.audio.shape == (0,)
    assert result.audio.dtype == np.float32
    assert result.metadata["frames"] == 0
    assert result.metadata["terminated_by"] == "max_frames"


def test_voice_defaults_and_required_controls_are_metadata_driven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, fake = _make_adapter(
        tmp_path,
        monkeypatch,
        default_voice=None,
        generation_overrides={"duration_delta_default": None},
    )
    with pytest.raises(RuntimeContractError, match="select a voice explicitly"):
        adapter.prepare_voice()
    voice = adapter.prepare_voice("easton")
    assert voice.metadata["locale"] == "en-US"
    with pytest.raises(RuntimeContractError, match="no bundle default"):
        list(adapter.stream([1], voice=voice, max_frames=1))
    assert fake.sessions == {}
    assert adapter.prepare_voice("easton") is voice


def test_voice_prompt_and_voice_state_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_size, _ = _make_adapter(
        tmp_path / "bad-size", monkeypatch, voice_data=np.arange(5, dtype=np.float32)
    )
    with pytest.raises(RuntimeContractError, match="not divisible"):
        bad_size.prepare_voice()

    wrong_frames, _ = _make_adapter(
        tmp_path / "wrong-frames", monkeypatch, voice_data=np.arange(12, dtype=np.float32)
    )
    with pytest.raises(RuntimeContractError, match="expected 2"):
        wrong_frames.prepare_voice()

    nonfinite, _ = _make_adapter(
        tmp_path / "nonfinite", monkeypatch, voice_data=np.array([0, 1, 2, 3, 4, 5, np.nan, 7])
    )
    with pytest.raises(RuntimeContractError, match="NaN or infinity"):
        nonfinite.prepare_voice()

    with pytest.raises(RuntimeContractError, match="shape"):
        PhononVoiceState("easton", np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(RuntimeContractError, match="finite"):
        PhononVoiceState("easton", np.full((1, 2, 4), np.inf, dtype=np.float32))


def test_missing_voice_default_unknown_voice_and_invalid_controls_fail_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, fake = _make_adapter(tmp_path, monkeypatch, default_voice=None)
    with pytest.raises(RuntimeContractError, match="no default voice"):
        list(adapter.stream([1], duration_delta=0.25, max_frames=1))
    with pytest.raises(RuntimeContractError, match="Unknown Phonon voice"):
        adapter.prepare_voice("missing")
    with pytest.raises(RuntimeContractError, match="non-negative integer"):
        list(adapter.stream([1], voice="easton", duration_delta=0.25, seed=-1))
    with pytest.raises(RuntimeContractError, match="non-empty"):
        list(adapter.stream([], voice="easton", duration_delta=0.25))
    assert fake.sessions == {}


def test_runtime_and_graph_signature_mismatches_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mismatched, fake = _make_adapter(
        tmp_path / "runtime", monkeypatch, runtime_overrides={"d_model": 8}
    )
    with pytest.raises(RuntimeContractError, match="catalog/config runtime mismatch"):
        list(mismatched.stream([1], duration_delta=0.25))
    assert fake.sessions == {}

    bad_graph, invalid = _make_adapter(
        tmp_path / "signature", monkeypatch, fake=_FakeGraphs(invalid_main_signature=True)
    )
    with pytest.raises(RuntimeContractError, match="signature mismatch"):
        list(bad_graph.stream([1], duration_delta=0.25))
    assert invalid.sessions["flow_lm_main"].closed


def test_invalid_mimi_audio_and_chunk_values_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = _make_adapter(tmp_path, monkeypatch, fake=_FakeGraphs(invalid_audio=True))
    with pytest.raises(RuntimeContractError, match="NaN or infinity"):
        list(adapter.stream([1], duration_delta=0.25, max_frames=1))

    with pytest.raises(RuntimeContractError, match="finite mono float32"):
        PhononStreamChunk(np.array([np.nan]), 8, 0, False)
    with pytest.raises(RuntimeContractError, match="sample_rate"):
        PhononStreamChunk(np.zeros(1), 0, 0, False)


def test_adapter_close_is_idempotent_and_closes_all_lazy_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, fake = _make_adapter(tmp_path, monkeypatch)
    list(adapter.stream([1], duration_delta=0.25, max_frames=1))
    assert set(fake.sessions) == set(_GRAPH_FILES)
    diagnostics = adapter.diagnostics()
    assert diagnostics.layout == "phonon-four-graph-streaming-v1"
    assert {session.component for session in diagnostics.sessions} == set(_GRAPH_FILES)
    assert diagnostics.properties["created_graphs"] == sorted(_GRAPH_FILES)
    adapter.close()
    adapter.close()
    assert all(session.closed for session in fake.sessions.values())
    assert adapter.diagnostics().sessions == ()
