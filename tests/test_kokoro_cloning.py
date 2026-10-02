from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.manager import OnnxVoice
from onnxvoice.systems.kokoro import KokoroAdapter
from onnxvoice.systems.kokoro_cloning import (
    CLONING_MODEL_COMPONENTS,
    KokoroCloningRuntime,
    KokoroReferenceState,
    kokoro_model_fingerprint,
    reference_mels,
)
from onnxvoice.types import Installation, InstalledArtifact, SessionDiagnostic


def _installation(tmp_path, *, layout="cloning-onnx-v1", digest_offset=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index, component in enumerate(CLONING_MODEL_COMPONENTS):
        path = tmp_path / f"{component}.onnx"
        path.write_bytes(f"model-{component}".encode())
        artifacts.append(
            InstalledArtifact(
                "model",
                path.name,
                path,
                f"{index + digest_offset + 1:064x}",
                path.stat().st_size,
                component=component,
            )
        )

    source_path = tmp_path / "source_params.npz"
    np.savez(
        source_path,
        weight=np.ones((1, 9), dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
        window=np.ones(20, dtype=np.float32),
    )
    artifacts.append(
        InstalledArtifact(
            "metadata",
            source_path.name,
            source_path,
            f"{len(CLONING_MODEL_COMPONENTS) + digest_offset + 1:064x}",
            source_path.stat().st_size,
            component="source_params",
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"max_tokens": 510}), encoding="utf-8")
    artifacts.append(
        InstalledArtifact(
            "config", config_path.name, config_path, "f" * 64, config_path.stat().st_size
        )
    )
    return Installation(
        "kokoro",
        "clone",
        "model",
        tmp_path,
        tuple(artifacts),
        sample_rate=24000,
        metadata={"runtime": {"layout": layout}},
    )


def _state(*, fingerprint="a" * 64):
    return KokoroReferenceState(
        style=np.zeros((1, 256), dtype=np.float64),
        memory=np.ones((1, 3, 192), dtype=np.float64),
        memory_mask=np.array([[1, 0, 1]], dtype=np.int8),
        model_fingerprint=fingerprint,
        metadata={"reference_seconds": 3.0},
    )


class EnrollmentSession:
    def __init__(self, component, order):
        self.component = component
        self.order = order
        self.inputs = None
        self.closed = False
        self.input_names = {
            "reference_encoders": ("mel",),
            "reference_wavlm": ("input_values",),
            "reference_mapper": (
                "mel",
                "mel_lengths",
                "reference_ids",
                "reference_lengths",
                "wavlm",
                "raw_sdec",
                "raw_spred",
            ),
        }[component]
        self.output_names = {
            "reference_encoders": ("raw_sdec", "raw_spred"),
            "reference_wavlm": ("wavlm",),
            "reference_mapper": ("style", "reference_memory", "reference_mask"),
        }[component]

    def run(self, inputs):
        self.order.append(self.component)
        self.inputs = inputs
        if self.component == "reference_encoders":
            return [np.full((1, 128), 0.25, np.float32), np.full((1, 128), 0.5, np.float32)]
        if self.component == "reference_wavlm":
            return [np.ones((1, 512), np.float32)]
        return [
            np.arange(256, dtype=np.float32)[None],
            np.ones((1, 3, 192), np.float32),
            np.array([[False, False, True]]),
        ]

    def close(self):
        self.closed = True


def _audio_pair():
    time24 = np.arange(72000, dtype=np.float32) / 24000.0
    time16 = np.arange(48000, dtype=np.float32) / 16000.0
    audio24 = 0.1 * np.sin(2 * np.pi * 220 * time24)
    audio16 = 0.1 * np.sin(2 * np.pi * 220 * time16)
    return audio24.astype(np.float32), audio16.astype(np.float32)


class SynthesisSession:
    def __init__(self, component, run_order, durations=None):
        self.component = component
        self.run_order = run_order
        self.durations = np.asarray(durations if durations is not None else [1, 2, 1, 1])
        self.inputs = None
        self.closed = False
        self.input_names = {
            "prosody": ("input_ids", "style_dur", "reference_memory", "reference_mask"),
            "curves": ("en", "style_dur"),
            "decoder": ("asr", "f0_curve", "n_curve", "style_acou", "har"),
        }[component]
        self.output_names = {
            "prosody": ("pred_dur", "d", "t_en"),
            "curves": ("f0_curve", "n_curve"),
            "decoder": ("audio",),
        }[component]

    def run(self, inputs):
        self.run_order.append(self.component)
        self.inputs = inputs
        if self.component == "prosody":
            count = len(self.durations)
            return [
                self.durations,
                np.arange(count * 512, dtype=np.float32).reshape(1, count, 512),
                np.arange(512 * count, dtype=np.float32).reshape(1, 512, count),
            ]
        if self.component == "curves":
            frames = inputs["en"].shape[-1]
            return [np.zeros((1, frames), np.float32), np.ones((1, frames), np.float32)]
        return [np.array([[0.1, 0.2, 0.3]], dtype=np.float32)]

    def diagnostics(self):
        return SessionDiagnostic(
            component=self.component,
            model_path=Path(f"{self.component}.onnx"),
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=self.input_names,
            outputs=self.output_names,
        )

    def close(self):
        self.closed = True


def test_reference_state_owns_canonical_read_only_arrays():
    style = np.zeros((1, 256), dtype=np.float64)
    memory = np.ones((1, 3, 192), dtype=np.float64)
    mask = np.array([[1, 0, 1]], dtype=np.int8)
    state = KokoroReferenceState(style, memory, mask, "a" * 64)

    assert state.style.dtype == np.float32
    assert state.memory.dtype == np.float32
    assert state.memory_mask.dtype == np.bool_
    assert state.style.flags.c_contiguous and not state.style.flags.writeable
    assert state.memory.flags.c_contiguous and not state.memory.flags.writeable
    assert state.memory_mask.flags.c_contiguous and not state.memory_mask.flags.writeable
    assert not np.shares_memory(state.style, style)
    assert not np.shares_memory(state.memory, memory)
    assert not np.shares_memory(state.memory_mask, mask)


def test_reference_state_validates_shapes_finiteness_and_fingerprint():
    with pytest.raises(RuntimeContractError, match="style must have shape"):
        KokoroReferenceState(np.zeros(256), np.zeros((1, 1, 192)), np.ones((1, 1)), "f")
    with pytest.raises(RuntimeContractError, match="memory must have shape"):
        KokoroReferenceState(np.zeros((1, 256)), np.zeros((1, 0, 192)), np.ones((1, 0)), "f")
    with pytest.raises(RuntimeContractError, match="mask must have shape"):
        KokoroReferenceState(np.zeros((1, 256)), np.zeros((1, 1, 192)), np.ones((1, 2)), "f")
    with pytest.raises(RuntimeContractError, match="finite"):
        KokoroReferenceState(np.full((1, 256), np.nan), np.zeros((1, 1, 192)), np.ones((1, 1)), "f")
    with pytest.raises(RuntimeContractError, match="non-empty model fingerprint"):
        KokoroReferenceState(np.zeros((1, 256)), np.zeros((1, 1, 192)), np.ones((1, 1)), " ")


def test_model_fingerprint_uses_artifact_hashes_not_local_paths(tmp_path):
    first = kokoro_model_fingerprint(_installation(tmp_path / "one"))
    second = kokoro_model_fingerprint(_installation(tmp_path / "two"))
    changed = kokoro_model_fingerprint(_installation(tmp_path / "three", digest_offset=10))

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_adapter_routes_cloning_layout_without_exposing_single_session(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path))

    assert adapter._uses_cloning_runtime()
    with pytest.raises(CapabilityError, match="do not expose a single session"):
        _ = adapter.session
    with pytest.raises(RuntimeContractError, match="reference state must be provided"):
        adapter.infer([1, 2])
    with pytest.raises(CapabilityError, match="Static style input"):
        adapter.infer([1, 2], style=np.zeros(256))


def test_reference_state_arguments_are_clone_only(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path, layout="single-onnx-v1"))
    state = _state()

    with pytest.raises(CapabilityError, match="requires the cloning-onnx-v1 layout"):
        adapter.infer([1, 2], reference=state)
    with pytest.raises(CapabilityError, match="Reference enrollment requires"):
        adapter.prepare_reference([1], audio_24k=np.zeros(72000), audio_16k=np.zeros(48000))


def test_reference_mels_match_upstream_torchaudio_fixture():
    """Fixture uses AkinVox's published reference_mel.py preprocessing."""
    fixture_path = Path(__file__).parent / "fixtures" / "kokoro_cloning_reference_mel.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        mapper_mel, encoder_mel = reference_mels(fixture["wave"])
        assert mapper_mel.shape[-1] == fixture["mapper_frames"]
        assert encoder_mel.shape[-1] == fixture["encoder_frames"]
        np.testing.assert_allclose(mapper_mel, fixture["mapper_mel"], rtol=1.0e-5, atol=2.0e-4)
        np.testing.assert_allclose(
            encoder_mel, fixture["padded_encoder_mel"], rtol=1.0e-5, atol=2.0e-4
        )


def test_reference_enrollment_runs_only_enrollment_components(tmp_path):
    order = []
    sessions = {}

    def factory(_path, component):
        session = EnrollmentSession(component, order)
        sessions[component] = session
        return session

    runtime = KokoroCloningRuntime(_installation(tmp_path), session_factory=factory)
    audio24, audio16 = _audio_pair()
    state = runtime.prepare_reference([12, 24], audio_24k=audio24, audio_16k=audio16)

    assert order == ["reference_encoders", "reference_wavlm", "reference_mapper"]
    assert set(sessions) == set(order)
    assert sessions["reference_encoders"].inputs["mel"].shape == (1, 1, 80, 274)
    assert sessions["reference_wavlm"].inputs["input_values"].shape == (1, 48000)
    np.testing.assert_array_equal(sessions["reference_wavlm"].inputs["input_values"][0], audio16)
    mapper_inputs = sessions["reference_mapper"].inputs
    assert mapper_inputs["mel"].shape == (1, 80, 241)
    np.testing.assert_array_equal(mapper_inputs["mel_lengths"], np.array([241], np.int64))
    np.testing.assert_array_equal(mapper_inputs["reference_ids"], np.array([[0, 12, 24, 0]]))
    np.testing.assert_array_equal(mapper_inputs["reference_lengths"], np.array([4], np.int64))
    np.testing.assert_array_equal(mapper_inputs["raw_sdec"], np.full((1, 128), 0.25, np.float32))
    np.testing.assert_array_equal(mapper_inputs["raw_spred"], np.full((1, 128), 0.5, np.float32))
    np.testing.assert_allclose(np.linalg.norm(mapper_inputs["wavlm"], axis=-1), [1.0])
    np.testing.assert_array_equal(state.style, np.arange(256, dtype=np.float32)[None])
    assert state.memory.shape == (1, 3, 192)
    assert state.memory_mask.dtype == np.bool_
    assert state.model_fingerprint == runtime.model_fingerprint
    assert state.metadata["format"] == "akinvox-cloning-reference-v1"
    assert not any("audio" in key for key in state.metadata)
    assert not hasattr(state, "audio_24k")


def test_reference_validation_precedes_session_creation(tmp_path):
    sessions = []

    def factory(_path, component):
        sessions.append(component)
        return EnrollmentSession(component, [])

    runtime = KokoroCloningRuntime(_installation(tmp_path), session_factory=factory)
    audio24, audio16 = _audio_pair()
    invalid_inputs = [
        ([], audio24, audio16),
        ([1] * 509, audio24, audio16),
        ([1], audio24[: 2 * 24000], audio16[: 2 * 16000]),
        ([1], audio24, audio16[:-1000]),
        ([1], np.zeros_like(audio24), audio16),
        ([1], np.ones_like(audio24), audio16),
        ([1], np.where(np.arange(len(audio24)) == 0, np.nan, audio24), audio16),
        ([1], audio24.astype(np.int16), audio16),
        ([1], audio24, np.zeros_like(audio16)),
        ([1], audio24, np.ones_like(audio16)),
        ([1], audio24, np.where(np.arange(len(audio16)) == 0, np.nan, audio16)),
    ]

    for tokens, wave24, wave16 in invalid_inputs:
        with pytest.raises(RuntimeContractError):
            runtime.prepare_reference(tokens, audio_24k=wave24, audio_16k=wave16)
        assert not sessions
        assert not runtime._sessions


def _run_synthesis(tmp_path, *, durations=None):
    run_order = []
    sessions = {}

    def factory(_path, component):
        session = SynthesisSession(component, run_order, durations)
        sessions[component] = session
        return session

    adapter = KokoroAdapter(_installation(tmp_path), session_factory=factory)
    state = KokoroReferenceState(
        style=np.arange(256, dtype=np.float32)[None],
        memory=np.ones((1, 3, 192), dtype=np.float32),
        memory_mask=np.array([[False, False, True]]),
        model_fingerprint=kokoro_model_fingerprint(adapter.installation),
    )
    return adapter, state, sessions, run_order


def test_cloning_synthesis_expands_duration_and_routes_style_slices(tmp_path):
    adapter, state, sessions, run_order = _run_synthesis(tmp_path)

    result = adapter.infer([7, 8], reference=state, seed=17)

    assert run_order == ["prosody", "curves", "decoder"]
    assert set(sessions) == {"prosody", "curves", "decoder"}
    prosody_inputs = sessions["prosody"].inputs
    np.testing.assert_array_equal(prosody_inputs["input_ids"], [[0, 7, 8, 0]])
    np.testing.assert_array_equal(prosody_inputs["style_dur"], np.arange(128, 256)[None])
    np.testing.assert_array_equal(prosody_inputs["reference_memory"], state.memory)
    np.testing.assert_array_equal(prosody_inputs["reference_mask"], state.memory_mask)
    curve_inputs = sessions["curves"].inputs
    np.testing.assert_array_equal(curve_inputs["style_dur"], np.arange(128, 256)[None])
    assert curve_inputs["en"].shape == (1, 512, 5)
    expansion_index = np.array([0, 1, 1, 2, 3])
    duration_embedding = np.arange(4 * 512, dtype=np.float32).reshape(1, 4, 512)
    np.testing.assert_array_equal(
        curve_inputs["en"], duration_embedding.transpose(0, 2, 1)[:, :, expansion_index]
    )
    decoder_inputs = sessions["decoder"].inputs
    np.testing.assert_array_equal(decoder_inputs["style_acou"], np.arange(128)[None])
    assert decoder_inputs["asr"].shape == (1, 512, 5)
    np.testing.assert_array_equal(result.timings, [1, 2, 1, 1])
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert result.metadata["layout"] == "cloning-onnx-v1"
    assert result.metadata["seed"] == 17
    assert result.metadata["speed"] == 1.0
    assert len(result.metadata["reference_fingerprint"]) == 64


def test_precomputed_state_skips_enrollment_sessions_and_tracks_lifecycle(tmp_path):
    adapter, state, sessions, run_order = _run_synthesis(tmp_path)

    first = adapter.infer([7, 8], reference=state, seed=17)
    first_har = sessions["decoder"].inputs["har"].copy()
    second = adapter.infer([7, 8], reference=state, seed=17)
    second_har = sessions["decoder"].inputs["har"].copy()
    adapter.infer([7, 8], reference=state, seed=18)
    third_har = sessions["decoder"].inputs["har"].copy()

    assert first.metadata["reference_fingerprint"] == second.metadata["reference_fingerprint"]
    np.testing.assert_array_equal(first.audio, second.audio)
    np.testing.assert_array_equal(first_har, second_har)
    assert not np.array_equal(first_har, third_har)
    assert not any(component.startswith("reference_") for component in sessions)
    diagnostics = adapter.diagnostics()
    assert {session.component for session in diagnostics.sessions} == {
        "prosody",
        "curves",
        "decoder",
    }
    assert all(
        session.providers_active == ("CPUExecutionProvider",) for session in diagnostics.sessions
    )
    assert run_order == ["prosody", "curves", "decoder"] * 3

    adapter.close()
    adapter.close()
    assert all(session.closed for session in sessions.values())


@pytest.mark.parametrize(
    "durations, message",
    [
        ([1, 0, 1, 1], "at least one frame"),
        ([1, np.nan, 1, 1], "finite"),
        ([1, 1.5, 1, 1], "integers"),
        ([1001, 1001, 1001, 1001], "exceeds 4000 frames"),
    ],
)
def test_cloning_duration_safety_limits(tmp_path, durations, message):
    adapter, state, sessions, _ = _run_synthesis(tmp_path, durations=durations)

    with pytest.raises(RuntimeContractError, match=message):
        adapter.infer([7, 8], reference=state)
    assert set(sessions) == {"prosody"}


@pytest.mark.parametrize(
    "speed, fingerprint, error, message",
    [
        (1.25, None, CapabilityError, "does not support speed"),
        (1.0, "b" * 64, RuntimeContractError, "different model build"),
    ],
)
def test_cloning_rejects_unsupported_speed_and_model_state(
    tmp_path, speed, fingerprint, error, message
):
    adapter, state, sessions, _ = _run_synthesis(tmp_path)
    if fingerprint is not None:
        state = KokoroReferenceState(state.style, state.memory, state.memory_mask, fingerprint)

    with pytest.raises(error, match=message):
        adapter.infer([7, 8], reference=state, speed=speed)
    assert not sessions


def test_local_component_mapping_runs_fake_enrollment_and_synthesis(tmp_path):
    components = (
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
    )
    artifacts = {}
    for component in components:
        path = tmp_path / f"{component}.onnx"
        path.write_bytes(b"model")
        artifacts[f"model:{component}"] = path
    source_path = tmp_path / "source_params.npz"
    np.savez(
        source_path,
        weight=np.ones((1, 9), np.float32),
        bias=np.zeros(1, np.float32),
        window=np.ones(20, np.float32),
    )
    artifacts["metadata:source_params"] = source_path
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"max_tokens": 510}), encoding="utf-8")
    artifacts["config"] = config_path
    adapter = OnnxVoice.open_local(
        system="kokoro",
        artifacts=artifacts,
        runtime={"layout": "cloning-onnx-v1", "voice_mode": "reference"},
        sample_rate=24000,
    )
    order = []
    sessions = {}

    def factory(_path, component):
        if component in {"reference_encoders", "reference_wavlm", "reference_mapper"}:
            session = EnrollmentSession(component, order)
        else:
            session = SynthesisSession(component, order)
        sessions[component] = session
        return session

    adapter._split_session_factory = factory
    audio24, audio16 = _audio_pair()
    state = adapter.prepare_reference([5, 6], audio_24k=audio24, audio_16k=audio16)
    result = adapter.infer([7, 8], reference=state, seed=9)

    assert order == [
        "reference_encoders",
        "reference_wavlm",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
    ]
    assert adapter.installation.voices == ()
    assert adapter.installation.default_voice is None
    assert result.metadata["layout"] == "cloning-onnx-v1"
    adapter.close()
    assert all(session.closed for session in sessions.values())
