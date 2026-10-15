from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.qwen3 import Qwen3Adapter
from onnxvoice.systems.qwen3_audio import (
    QWEN3_BASE_MAX_REFERENCE_SAMPLES,
    Qwen3VoiceState,
    compute_log_mel,
    mel_filterbank,
    validate_reference_audio,
)
from onnxvoice.systems.qwen3_contract import CODE_GROUPS, Qwen3Config, Qwen3EmbeddingStore


def _config() -> Qwen3Config:
    return Qwen3Config.from_mapping(
        {
            "tts_model_type": "base",
            "model_id": "fixture/base",
            "sample_rate": 24000,
            "im_start_token_id": 0,
            "im_end_token_id": 1,
            "tts_pad_token_id": 2,
            "tts_bos_token_id": 3,
            "tts_eos_token_id": 4,
            "codec_eos_token_id": 5,
            "codec_think_id": 6,
            "codec_nothink_id": 7,
            "codec_think_bos_id": 8,
            "codec_think_eos_id": 9,
            "codec_pad_id": 10,
            "codec_bos_id": 11,
            "codec_language_id": {"english": 12},
            "talker_hidden_size": 4,
            "talker_num_layers": 2,
            "talker_num_attention_heads": 2,
            "talker_num_kv_heads": 1,
            "talker_head_dim": 4,
            "talker_vocab_size": 1100,
            "talker_text_hidden_size": 3,
            "talker_num_code_groups": CODE_GROUPS,
            "cp_hidden_size": 4,
            "cp_num_layers": 2,
            "cp_num_attention_heads": 2,
            "cp_num_kv_heads": 1,
            "cp_head_dim": 4,
            "cp_vocab_size": 8,
            "cp_num_code_groups": CODE_GROUPS,
        }
    )


def _embeddings(config: Qwen3Config) -> Qwen3EmbeddingStore:
    arrays: dict[str, np.ndarray] = {
        "text_embedding": np.zeros((64, config.talker_text_hidden_size), np.float32),
        "text_projection_fc1_weight": np.zeros((5, config.talker_text_hidden_size), np.float32),
        "text_projection_fc1_bias": np.zeros((5,), np.float32),
        "text_projection_fc2_weight": np.zeros((config.talker_hidden_size, 5), np.float32),
        "text_projection_fc2_bias": np.zeros((config.talker_hidden_size,), np.float32),
        "talker_codec_embedding": np.repeat(
            np.arange(config.talker_vocab_size, dtype=np.float32)[:, None],
            config.talker_hidden_size,
            axis=1,
        ),
    }
    for index in range(CODE_GROUPS - 1):
        arrays[f"cp_codec_embedding_{index}"] = np.repeat(
            np.arange(config.cp_vocab_size, dtype=np.float32)[:, None],
            config.talker_hidden_size,
            axis=1,
        )
    return Qwen3EmbeddingStore(MappingProxyType(arrays))


def _reference_ids() -> tuple[int, ...]:
    return (0, 24, 25, 26, 1, 22)


def _adapter() -> Any:
    config = _config()
    adapter = cast(Any, Qwen3Adapter.__new__(Qwen3Adapter))
    adapter.model_type = "base"
    adapter.quality = "int4"
    adapter.config = config
    adapter.embeddings = _embeddings(config)
    adapter.installation = SimpleNamespace(id="base-0.6b")
    return adapter


def test_numpy_log_mel_matches_committed_legacy_reference_array() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "qwen3_log_mel_golden.json").read_text(
            encoding="utf-8"
        )
    )
    samples = (
        np.random.default_rng(fixture["seed"])
        .normal(0.0, 0.1, fixture["sample_count"])
        .astype(np.float32)
    )
    mel = compute_log_mel(samples)
    assert mel.shape == (1, 1, 128)
    assert mel.dtype == np.float32
    np.testing.assert_allclose(
        mel[0, 0], np.asarray(fixture["expected"], np.float32), rtol=0, atol=1e-6
    )


def test_numpy_log_mel_uses_periodic_hann_center_false_and_slaney_bank() -> None:
    bank = mel_filterbank()
    assert bank.shape == (128, 513)
    assert bank.dtype == np.float32
    assert not bank.flags.writeable
    silence = compute_log_mel(np.zeros((1536,), np.float32))
    assert silence.shape == (1, 3, 128)
    np.testing.assert_array_equal(silence, np.full((1, 3, 128), np.log(1e-5), np.float32))
    with pytest.raises(RuntimeContractError, match="at least 1024"):
        compute_log_mel(np.zeros((1023,), np.float32))


def test_reference_audio_contract_requires_mono_finite_24khz_and_at_most_ten_seconds() -> None:
    samples = np.linspace(-1, 1, 2048, dtype=np.float64)
    validated = validate_reference_audio(samples, sample_rate=24_000)
    assert validated.dtype == np.float32
    assert validated.flags.c_contiguous
    assert not np.shares_memory(validated, samples)
    assert validated.shape == samples.shape

    with pytest.raises(RuntimeContractError, match="24000 Hz"):
        validate_reference_audio(samples, sample_rate=16_000)
    with pytest.raises(RuntimeContractError, match="mono"):
        validate_reference_audio(np.zeros((2048, 1), np.float32), sample_rate=24_000)
    with pytest.raises(RuntimeContractError, match="float-compatible"):
        validate_reference_audio(np.asarray(["audio"], dtype=object), sample_rate=24_000)
    with pytest.raises(RuntimeContractError, match="finite"):
        validate_reference_audio(np.asarray([0.0, np.nan], np.float32), sample_rate=24_000)
    with pytest.raises(RuntimeContractError, match="empty"):
        validate_reference_audio(np.asarray([], np.float32), sample_rate=24_000)
    with pytest.raises(RuntimeContractError, match="at least 1024"):
        validate_reference_audio(np.zeros((100,), np.float32), sample_rate=24_000)
    with pytest.raises(RuntimeContractError, match="no longer than 10 seconds"):
        validate_reference_audio(
            np.zeros((QWEN3_BASE_MAX_REFERENCE_SAMPLES + 1,), np.float32), sample_rate=24_000
        )


def test_voice_state_copies_arrays_into_immutable_storage() -> None:
    speaker = np.ones((1, 4), np.float32)
    codes = np.zeros((1, CODE_GROUPS, 2), np.int64)
    state = Qwen3VoiceState(
        speaker_embedding=speaker,
        reference_codes=codes,
        reference_token_ids=_reference_ids(),
        sample_rate=24_000,
        reference_samples=1921,
        model_id="fixture/base",
    )
    speaker[0, 0] = 99
    codes[0, 0, 0] = 99
    assert state.speaker_embedding[0, 0] == 1
    assert state.reference_codes[0, 0, 0] == 0
    assert not state.speaker_embedding.flags.writeable
    assert not state.reference_codes.flags.writeable
    with pytest.raises(ValueError):
        state.speaker_embedding.setflags(write=True)
    with pytest.raises(ValueError):
        state.reference_codes.setflags(write=True)


class _SpeakerEncoder:
    def __init__(self) -> None:
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs.append(inputs)
        return [np.asarray([[0.25, 0.5, 0.75, 1.0]], np.float32)]


class _TokenizerEncoder:
    def __init__(self) -> None:
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs.append(inputs)
        return [np.zeros((1, CODE_GROUPS, 125), np.int64)]


def test_prepare_voice_runs_both_base_encoders_pads_and_trims_codec_frames() -> None:
    adapter = _adapter()
    speaker_encoder = _SpeakerEncoder()
    tokenizer_encoder = _TokenizerEncoder()
    sessions = {"speaker_encoder": speaker_encoder, "tokenizer_encoder": tokenizer_encoder}
    adapter._ensure_sessions = lambda: None
    adapter._get_session = lambda name: sessions[name]
    samples = np.linspace(-0.5, 0.5, 1921, dtype=np.float32)
    voice = adapter.prepare_voice(samples, sample_rate=24_000, reference_token_ids=_reference_ids())

    mel = speaker_encoder.inputs[0]["mels"]
    assert mel.shape == (1, 4, 128)
    waveform = tokenizer_encoder.inputs[0]["waveform"]
    assert waveform.shape == (1, 240_000)
    np.testing.assert_array_equal(waveform[0, : samples.size], samples)
    assert np.all(waveform[0, samples.size :] == 0)
    assert voice.speaker_embedding.shape == (1, 4)
    assert voice.reference_codes.shape == (1, CODE_GROUPS, 2)
    assert voice.reference_token_ids == _reference_ids()
    assert voice.reference_samples == 1921
    assert voice.sample_rate == 24_000
    assert voice.model_id == "fixture/base"


def test_prepare_voice_rejects_wrong_model_before_creating_sessions() -> None:
    adapter = _adapter()
    adapter.model_type = "voice_design"
    adapter._ensure_sessions = lambda: pytest.fail("sessions should not be initialized")
    with pytest.raises(CapabilityError, match="requires a base"):
        adapter.prepare_voice(
            np.ones((2048,), np.float32), sample_rate=24_000, reference_token_ids=_reference_ids()
        )
