from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems import kokoro_inno as inno_module
from onnxvoice.systems.kokoro import KokoroAdapter
from onnxvoice.systems.kokoro_inno import (
    InnoVoiceTuner,
    KokoroVoicePack,
    blend_weights,
    ceiling,
    dense_blend_weights,
    head_stats,
    inno_model_fingerprint,
    rate,
    resample_audio,
    speaker_fbank,
    stats,
    tilt,
)
from onnxvoice.types import Installation, InstalledArtifact, SessionDiagnostic


def test_inno_fbank_matches_pinned_upstream_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "inno_fbank.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        actual = speaker_fbank(fixture["waveform"])
        expected = fixture["expected_fbank"]
        assert fixture["upstream_version"].item() == "0.2.0"
        assert fixture["upstream_commit"].item() == "892ef184bc932aa3ff9d72c1509d5b81ff6941e6"
        assert fixture["generator"].item().endswith("_ResNet34.fbank")

    assert actual.shape == (1, 300, 80)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=1.0e-4)


def test_inno_fbank_uses_snip_edges_frame_count_and_rejects_invalid_audio():
    waveform = np.zeros(400 + 160 * 3, dtype=np.float32)
    assert speaker_fbank(waveform).shape == (1, 4, 80)

    with pytest.raises(RuntimeContractError, match="one-dimensional"):
        speaker_fbank(np.zeros((2, 400), dtype=np.float32))
    with pytest.raises(RuntimeContractError, match="floating-point"):
        speaker_fbank(np.zeros(400, dtype=np.int16))
    with pytest.raises(RuntimeContractError, match="finite"):
        speaker_fbank(np.full(400, np.nan, dtype=np.float32))
    with pytest.raises(RuntimeContractError, match="400 samples"):
        speaker_fbank(np.zeros(399, dtype=np.float32))


@pytest.mark.parametrize(
    ("source_rate", "target_rate"), [(48000, 16000), (22050, 24000), (16000, 24000)]
)
def test_inno_resampling_matches_upstream_polyphase_contract(source_rate, target_rate):
    signal = pytest.importorskip("scipy.signal")
    waveform = np.random.default_rng(1024).normal(0, 0.1, source_rate // 2).astype(np.float32)
    divisor = np.gcd(source_rate, target_rate)

    actual = resample_audio(waveform, source_rate, target_rate)
    expected = signal.resample_poly(
        waveform, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)

    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


def test_inno_resampling_identity_owns_an_output_copy():
    waveform = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    actual = resample_audio(waveform, 16000, 16000)

    np.testing.assert_array_equal(actual, waveform)
    assert not np.shares_memory(actual, waveform)


@pytest.mark.parametrize(("source_rate", "target_rate"), [(0, 16000), (16000, 0)])
def test_inno_resampling_rejects_nonpositive_rates(source_rate, target_rate):
    with pytest.raises(RuntimeContractError, match="positive integer"):
        resample_audio(np.ones(16, dtype=np.float32), source_rate, target_rate)


def _inno_prosody_fixture():
    fixture_dir = Path(__file__).parent / "fixtures"
    metadata = json.loads((fixture_dir / "inno_prosody.json").read_text(encoding="utf-8"))
    with np.load(fixture_dir / "inno_fbank.npz", allow_pickle=False) as fixture:
        waveform = fixture["prosody_waveform"]
        sample_rate = int(fixture["sample_rate"])
    digest = hashlib.sha256(waveform.tobytes()).hexdigest()
    assert metadata["source"]["version"] == "0.2.0"
    assert metadata["source"]["commit"] == "892ef184bc932aa3ff9d72c1509d5b81ff6941e6"
    assert metadata["source"]["waveform_sha256"] == digest
    return waveform, sample_rate, metadata


def test_inno_prosody_matches_pinned_upstream_fixture():
    pytest.importorskip("scipy")
    waveform, sample_rate, fixture = _inno_prosody_fixture()
    expected = fixture["prosody"]

    assert ceiling(waveform, sample_rate) == expected["ceiling_default"]
    assert ceiling(waveform, sample_rate, fmax=190.0) == expected["ceiling_override"]
    np.testing.assert_allclose(stats(waveform, sample_rate), expected["stats_default"], atol=1.0e-6)
    np.testing.assert_allclose(
        stats(waveform, sample_rate, fmax=190.0), expected["stats_override"], atol=1.0e-6
    )
    assert rate(waveform, sample_rate) == pytest.approx(expected["rate"], abs=1.0e-7)
    assert tilt(waveform, sample_rate) == pytest.approx(expected["tilt"], abs=1.0e-5)


def test_inno_praat_head_stats_match_multichunk_upstream_fixture():
    pytest.importorskip("parselmouth")
    waveform, sample_rate, fixture = _inno_prosody_fixture()
    expected = fixture["prosody"]

    np.testing.assert_allclose(
        head_stats(waveform, sample_rate), expected["head_stats_default"], atol=1.0e-10
    )
    np.testing.assert_allclose(
        head_stats(waveform, sample_rate, fmax=190.0),
        expected["head_stats_override"],
        atol=1.0e-10,
    )


def test_inno_blend_weights_match_upstream_nnls_cases():
    pytest.importorskip("scipy.optimize")
    fixture = _inno_prosody_fixture()[2]["blend"]
    tuner = SimpleNamespace(
        names=fixture["names"],
        stats=np.asarray(fixture["stats"], dtype=fixture["stats_dtype"]),
        grades=np.asarray(fixture["grades"], dtype=fixture["grades_dtype"]),
        gate=fixture["gate"],
        scale=np.asarray(fixture["scale"], dtype=fixture["scale_dtype"]),
        grade_pen=fixture["grade_pen"],
    )

    for case in fixture["cases"]:
        actual = blend_weights(tuner, np.asarray(case["target"], dtype=np.float64))
        expected = case["weights"]
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(
            list(actual.values()), list(expected.values()), rtol=1.0e-10, atol=1.0e-12
        )
        dense = dense_blend_weights(fixture["names"], actual)
        assert dense.shape == (1, len(fixture["names"]))
        np.testing.assert_allclose(
            dense,
            [[expected.get(name, 0.0) for name in fixture["names"]]],
            rtol=1.0e-7,
            atol=1.0e-8,
        )


def test_inno_silence_has_no_pitch_or_syllable_rate():
    silence = np.zeros(3 * 16000, dtype=np.float32)
    assert ceiling(silence, 16000) == 400.0
    assert stats(silence, 16000) == (0.0, 0.0, 0.0)
    assert rate(silence, 16000) == 0.0


def _fake_inno_installation(tmp_path: Path, *, version: str = "0.2.0") -> Installation:
    base_path = tmp_path / "base.onnx"
    graph_path = tmp_path / "inno_voicepack.onnx"
    metadata_path = tmp_path / "inno_tuner.npz"
    base_path.write_bytes(b"kokoro base graph")
    graph_path.write_bytes(b"fake Inno graph")
    np.savez(
        metadata_path,
        blend_names=np.asarray(["af_one", "af_two", "am_one"]),
        blend_stats=np.zeros((3, 3), dtype=np.float32),
        blend_grades=np.ones(3, dtype=np.float32),
        blend_scale=np.ones(3, dtype=np.float32),
        blend_gate=np.asarray(4.0),
        grade_pen=np.asarray(1.0),
        tilt_mean=np.asarray(-7.8),
        tilt_sd=np.asarray(1.8),
        version=np.asarray(version),
        upstream_commit=np.asarray("892ef184bc932aa3ff9d72c1509d5b81ff6941e6"),
        checkpoint_sha256=np.asarray("a" * 64),
    )

    def installed(role: str, path: Path, component: str | None = None) -> InstalledArtifact:
        content = path.read_bytes()
        return InstalledArtifact(
            role=role,
            filename=path.name,
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
            component=component,
        )

    return Installation(
        system="kokoro",
        id="test-kokoro",
        kind="model",
        path=tmp_path,
        artifacts=(
            installed("model", base_path),
            installed("model", graph_path, "inno_voicepack"),
            installed("metadata", metadata_path, "inno_tuner"),
        ),
        metadata={
            "runtime": {
                "layout": "single-onnx-v1",
                "voice_enrollers": [
                    {
                        "id": "inno-v0.2",
                        "kind": "kokoro-voicepack-tuner",
                        "input": "reference-audio",
                        "transcript_required": False,
                        "min_seconds": 3.0,
                        "recommended_seconds": 5.0,
                        "max_seconds": 30.0,
                        "output": {
                            "format": "kokoro-voicepack-v1",
                            "shape": [510, 1, 256],
                            "dtype": "float32",
                        },
                        "model_component": "inno_voicepack",
                        "metadata_component": "inno_tuner",
                    }
                ],
            }
        },
    )


def test_inno_tuner_validates_inputs_runs_lazily_and_closes(tmp_path):
    installation = _fake_inno_installation(tmp_path)
    created = []

    class FakeSession:
        input_names = ("fbank", "tilt", "head_stats", "blend_weights", "head_enabled")
        output_names = ("voicepack",)

        def __init__(self):
            self.inputs = None
            self.closed = False

        def run(self, inputs):
            self.inputs = inputs
            return [np.zeros((510, 1, 256), dtype=np.float32)]

        def close(self):
            self.closed = True

    def session_factory(path, component):
        created.append((path, component))
        session = FakeSession()
        created.append(session)
        return session

    tuner = InnoVoiceTuner(installation, session_factory=session_factory)
    assert created == []
    assert inno_model_fingerprint(installation) == tuner.model_fingerprint
    with pytest.raises(RuntimeContractError, match="at least one frame"):
        tuner.infer_graph(np.zeros((1, 0, 80)), 0.0, np.zeros((1, 2)), np.ones((1, 3)))
    assert created == []

    output = tuner.infer_graph(
        np.zeros((1, 8, 80)),
        -0.25,
        np.asarray([[6.0, 1.2]], dtype=np.float32),
        np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32),
        head=False,
    )
    assert output.shape == (510, 1, 256)
    assert output.dtype == np.float32
    assert created[0] == (
        installation.artifact_path("model", component="inno_voicepack"),
        "inno_voicepack",
    )
    session = created[1]
    assert session.inputs["head_enabled"].tolist() == [0.0]
    assert all(value.flags.c_contiguous for value in session.inputs.values())
    tuner.close()
    assert session.closed


def test_inno_tuner_rejects_incompatible_metadata_version(tmp_path):
    installation = _fake_inno_installation(tmp_path, version="0.1.0")
    with pytest.raises(RuntimeContractError, match="Expected Inno tuner metadata v0.2.0"):
        InnoVoiceTuner(installation)


def test_inno_fingerprint_requires_advertised_capability(tmp_path):
    installation = _fake_inno_installation(tmp_path)
    installation.metadata["runtime"]["voice_enrollers"] = []
    with pytest.raises(CapabilityError, match="does not advertise"):
        inno_model_fingerprint(installation)


def test_kokoro_voice_pack_owns_canonical_data_and_selects_style_rows():
    source = np.ones((510, 1, 256), dtype=np.float64)
    pack = KokoroVoicePack(
        data=source,
        enroller="inno-v0.2",
        model_fingerprint="a" * 64,
        metadata={"upstream_version": "0.2.0", "blend_weights": {"af_one": 1.0}},
    )

    assert pack.data.dtype == np.float32
    assert pack.data.flags.c_contiguous and not pack.data.flags.writeable
    assert not np.shares_memory(pack.data, source)
    assert len(pack.fingerprint) == 64
    assert len(pack.enroller_fingerprint) == 64
    with pytest.raises(TypeError):
        pack.metadata["blend_weights"]["af_one"] = 0.0
    np.testing.assert_array_equal(pack.style_for(12), np.ones((1, 256), dtype=np.float32))
    assert pack.style_for(1000).shape == (1, 256)
    with pytest.raises(RuntimeContractError, match="non-negative integer"):
        pack.style_for(-1)
    with pytest.raises(RuntimeContractError, match="shape"):
        KokoroVoicePack(np.zeros((510, 256)), "inno-v0.2", "a" * 64)
    with pytest.raises(RuntimeContractError, match="finite"):
        KokoroVoicePack(np.full((510, 1, 256), np.nan), "inno-v0.2", "a" * 64)
    with pytest.raises(RuntimeContractError, match="fingerprint"):
        KokoroVoicePack(
            np.zeros((510, 1, 256)),
            "inno-v0.2",
            "a" * 64,
            fingerprint="b" * 64,
        )


class _RecordingInnoSession:
    input_names = ("fbank", "tilt", "head_stats", "blend_weights", "head_enabled")
    output_names = ("voicepack",)

    def __init__(self, path: Path):
        self.path = path
        self.inputs = None
        self.closed = False

    def run(self, inputs):
        self.inputs = inputs
        return [np.zeros((510, 1, 256), dtype=np.float32)]

    def diagnostics(self):
        return SessionDiagnostic(
            component="inno_voicepack",
            model_path=self.path,
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=self.input_names,
            outputs=self.output_names,
        )

    def close(self):
        self.closed = True


class _RecordingKokoroSession:
    input_names = ("input_ids", "ref_s", "speed")
    output_names = ("audio",)

    def __init__(self, path: Path):
        self.path = path
        self.inputs = None
        self.closed = False

    def run(self, inputs):
        self.inputs = inputs
        return [np.zeros((1, 32), dtype=np.float32)]

    def diagnostics(self):
        return SessionDiagnostic(
            component=None,
            model_path=self.path,
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=self.input_names,
            outputs=self.output_names,
        )

    def close(self):
        self.closed = True


def test_adapter_enrollment_downmixes_truncates_and_reuses_style_without_changing_infer(
    tmp_path, monkeypatch
):
    installation = _fake_inno_installation(tmp_path)
    graph_sessions = []

    def session_factory(path, component):
        assert component == "inno_voicepack"
        session = _RecordingInnoSession(path)
        graph_sessions.append(session)
        return session

    observed = {"resamples": [], "stats_fmax": None, "target": None}

    def fake_resample(audio, source_rate, target_rate):
        observed["resamples"].append((audio.size, source_rate, target_rate))
        return np.array(audio, dtype=np.float32, copy=True)

    def fake_stats(audio, sample_rate, *, fmax):
        observed["stats_fmax"] = fmax
        return 5.0, 1.25, 0.75

    def fake_blend(tuner, target):
        observed["target"] = target.copy()
        return {"af_one": 1.0}

    monkeypatch.setattr(inno_module, "resample_audio", fake_resample)
    monkeypatch.setattr(inno_module, "speaker_fbank", lambda audio: np.ones((1, 8, 80), np.float32))
    monkeypatch.setattr(
        inno_module, "ceiling", lambda *args, **kwargs: pytest.fail("fmax override ignored")
    )
    monkeypatch.setattr(inno_module, "stats", fake_stats)
    monkeypatch.setattr(inno_module, "rate", lambda audio, sample_rate: 4.0)
    monkeypatch.setattr(inno_module, "tilt", lambda audio, sample_rate: -8.0)
    monkeypatch.setattr(
        inno_module, "head_stats", lambda *args, **kwargs: pytest.fail("head disabled")
    )
    monkeypatch.setattr(inno_module, "blend_weights", fake_blend)

    adapter = KokoroAdapter(installation, session_factory=session_factory)
    sample_rate = 16000
    time = np.arange(31 * sample_rate, dtype=np.float32) / sample_rate
    stereo = np.column_stack(
        [0.1 * np.sin(2 * np.pi * 200 * time), 0.08 * np.sin(2 * np.pi * 180 * time)]
    ).astype(np.float32)
    with pytest.raises(CapabilityError, match="Unsupported Kokoro voice enroller"):
        adapter.enroll_voice(stereo, sample_rate=sample_rate, enroller="not-available")
    assert graph_sessions == []
    pack = adapter.enroll_voice(
        stereo,
        sample_rate=sample_rate,
        options={"fmax": 190.0, "head": False},
    )

    reused = adapter.enroll_voice(
        stereo,
        sample_rate=sample_rate,
        options={"fmax": 190.0, "head": False},
    )
    assert reused.fingerprint == pack.fingerprint
    assert len(graph_sessions) == 1
    assert isinstance(pack, KokoroVoicePack)
    assert pack.enroller == "inno-v0.2"
    assert pack.data.shape == (510, 1, 256)
    assert pack.metadata["duration_seconds"] == 30.0
    assert "audio" not in pack.metadata
    assert (
        observed["resamples"]
        == [(30 * sample_rate, sample_rate, 16000), (30 * sample_rate, sample_rate, 24000)] * 2
    )
    assert observed["stats_fmax"] == 190.0
    np.testing.assert_array_equal(observed["target"], [5.0, 1.25, 4.0])
    assert graph_sessions[0].inputs["head_enabled"].tolist() == [0.0]
    assert adapter._session is None

    monkeypatch.setattr(
        "onnxvoice.systems.kokoro.OnnxSession",
        lambda path, **kwargs: _RecordingKokoroSession(path),
    )
    result = adapter.infer([4, 5, 6], style=pack.style_for(3))
    base_session = adapter._session
    assert isinstance(base_session, _RecordingKokoroSession)
    assert base_session.path == installation.artifact_path("model")
    assert result.audio.shape == (32,)
    assert base_session.inputs["ref_s"].shape == (1, 256)
    np.testing.assert_array_equal(base_session.inputs["ref_s"], pack.style_for(3))
    diagnostics = adapter.diagnostics()
    assert {session.component for session in diagnostics.sessions} == {None, "inno_voicepack"}

    adapter.close()
    adapter.close()
    assert graph_sessions[0].closed and base_session.closed
    assert pack.style_for(3).shape == (1, 256)


def test_adapter_requires_advertised_inno_capability(tmp_path):
    installation = _fake_inno_installation(tmp_path)
    installation.metadata["runtime"]["voice_enrollers"] = []
    adapter = KokoroAdapter(
        installation, session_factory=lambda *args: pytest.fail("no graph session expected")
    )
    with pytest.raises(CapabilityError, match="does not advertise"):
        adapter.enroll_voice(np.ones(5 * 16000, dtype=np.float32), sample_rate=16000)
    assert adapter._inno_runtime is None


def test_inno_enrollment_validates_audio_and_warns_for_short_references(tmp_path):
    installation = _fake_inno_installation(tmp_path)
    created = []
    tuner = InnoVoiceTuner(installation, session_factory=lambda *args: created.append(args))

    with pytest.raises(RuntimeContractError, match="positive integer"):
        tuner.enroll(np.ones(4 * 16000, np.float32), sample_rate=0)
    with pytest.raises(RuntimeContractError, match="floating-point"):
        tuner.enroll(np.ones(4 * 16000, np.int16), sample_rate=16000)
    with pytest.raises(RuntimeContractError, match="downmixable"):
        tuner.enroll(np.ones((12, 12), np.float32), sample_rate=16000)
    with pytest.raises(RuntimeContractError, match="finite"):
        tuner.enroll(np.full(4 * 16000, np.nan, np.float32), sample_rate=16000)
    with pytest.raises(RuntimeContractError, match="at least 3 seconds"):
        tuner.enroll(np.ones(2 * 16000, np.float32) * 0.1, sample_rate=16000)
    with (
        pytest.warns(UserWarning, match="at least 5 seconds"),
        pytest.raises(RuntimeContractError, match="non-silent RMS"),
    ):
        tuner.enroll(np.zeros(4 * 16000, np.float32), sample_rate=16000)
    with (
        pytest.warns(UserWarning, match="at least 5 seconds"),
        pytest.raises(RuntimeContractError, match="grossly clipped"),
    ):
        tuner.enroll(np.ones(4 * 16000, np.float32), sample_rate=16000)
    with pytest.raises(RuntimeContractError, match="Unsupported Inno enrollment options"):
        tuner.enroll(
            np.full(5 * 16000, 0.1, np.float32), sample_rate=16000, options={"unknown": True}
        )
    with pytest.raises(RuntimeContractError, match="fmax"):
        tuner.enroll(np.full(5 * 16000, 0.1, np.float32), sample_rate=16000, options={"fmax": 0})
    with pytest.raises(RuntimeContractError, match="head.*boolean"):
        tuner.enroll(
            np.full(5 * 16000, 0.1, np.float32), sample_rate=16000, options={"head": "yes"}
        )
    assert created == []
