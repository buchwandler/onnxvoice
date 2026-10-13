from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError
from onnxvoice.systems.sopro_frontend import (
    SoproTokenizer,
    prepare_reference_audio,
    reference_features,
    resample_sopro,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sopro"


def _golden() -> dict[str, Any]:
    return json.loads((FIXTURES / "reference_golden.json").read_text(encoding="utf-8"))


def _steady_tone(sample_rate: int, samples: int) -> np.ndarray:
    indices = np.arange(samples, dtype=np.float64)
    waveform = 0.08 * np.sin(2 * math.pi * 220 * indices / sample_rate)
    waveform += 0.025 * np.sin(2 * math.pi * 791 * indices / sample_rate)
    return waveform.astype(np.float32)


def _room_tone_input(sample_rate: int, samples: int, active_from: int) -> np.ndarray:
    indices = np.arange(samples, dtype=np.float64)
    waveform = np.zeros(samples, dtype=np.float32)
    waveform[active_from:] = (
        0.12 * np.sin(2 * math.pi * 220 * indices[active_from:] / sample_rate)
    ).astype(np.float32)
    return waveform


def _assert_samples(
    actual: np.ndarray, expected: dict[str, float], *, tolerance: float = 5.0e-6
) -> None:
    indices = np.asarray([int(index) for index in expected], dtype=np.int64)
    values = np.asarray([expected[str(index)] for index in indices], dtype=np.float64)
    np.testing.assert_allclose(actual[indices], values, rtol=1.0e-5, atol=tolerance)


def test_official_reference_audio_dsp_and_feature_golden() -> None:
    golden = _golden()
    steady = golden["tones"]["steady"]
    source = _steady_tone(steady["sample_rate"], steady["samples"])
    prepared, level_db = prepare_reference_audio(source, steady["sample_rate"], seconds=10.0)
    assert prepared.size == steady["prepared_samples"]
    assert level_db == pytest.approx(steady["level_db"], abs=1.0e-6)
    _assert_samples(prepared, steady["prepared"])

    features = reference_features(
        prepared,
        golden["frontend_config"],
        token_samples=1024,
    )
    expected = steady["features"]
    assert features.token_count == expected["token_count"]
    assert list(features.semantic_mel.shape) == expected["semantic_shape"]
    assert list(features.speaker_mel.shape) == expected["speaker_shape"]
    assert list(features.acoustic_mel.shape) == expected["acoustic_shape"]
    for name, array in (
        ("semantic", features.semantic_mel),
        ("speaker", features.speaker_mel),
        ("acoustic", features.acoustic_mel),
    ):
        _assert_samples(array.reshape(-1), expected[name])
    for name, array in (
        ("interp_left", features.interp_left),
        ("interp_right", features.interp_right),
        ("interp_weight", features.interp_weight),
    ):
        expected_values = expected[name]
        indices = np.asarray([int(index) for index in expected_values], dtype=np.int64)
        values = np.asarray([expected_values[str(index)] for index in indices])
        np.testing.assert_allclose(array[indices], values, rtol=0.0, atol=1.0e-7)


def test_official_source_rate_resampling_golden() -> None:
    golden = _golden()
    expected = golden["tones"]["resampled_16k"]
    source = _steady_tone(expected["sample_rate"], expected["samples"])
    waveform_24k = resample_sopro(source, expected["sample_rate"], 24000)
    assert waveform_24k.size == expected["resampled_samples"]
    _assert_samples(waveform_24k, expected["resampled"], tolerance=1.0e-7)

    prepared, level_db = prepare_reference_audio(waveform_24k, 24000, seconds=10.0)
    assert prepared.size == expected["prepared_samples"]
    assert level_db == pytest.approx(expected["level_db"], abs=1.0e-6)
    _assert_samples(prepared, expected["prepared"])
    features = reference_features(prepared, golden["frontend_config"], token_samples=1024)
    expected_features = expected["features"]
    assert features.token_count == expected_features["token_count"]
    assert list(features.semantic_mel.shape) == expected_features["semantic_shape"]
    assert list(features.speaker_mel.shape) == expected_features["speaker_shape"]
    assert list(features.acoustic_mel.shape) == expected_features["acoustic_shape"]
    for name, array in (
        ("semantic", features.semantic_mel),
        ("speaker", features.speaker_mel),
        ("acoustic", features.acoustic_mel),
    ):
        _assert_samples(array.reshape(-1), expected_features[name], tolerance=3.0e-5)


def test_official_reference_room_tone_and_fade_golden() -> None:
    golden = _golden()["tones"]["room_tone"]
    source = _room_tone_input(
        golden["sample_rate"], golden["samples"], golden["active_from_sample"]
    )
    prepared, level_db = prepare_reference_audio(source, golden["sample_rate"], seconds=10.0)
    assert prepared.size == golden["prepared_samples"]
    assert level_db == pytest.approx(golden["level_db"], abs=1.0e-6)
    _assert_samples(prepared, golden["prepared"], tolerance=1.0e-8)


def test_official_sentencepiece_text_ids_match_multilingual_runtime() -> None:
    golden = _golden()
    tokenizer_bytes = gzip.decompress((FIXTURES / "tokenizer-v2-turbo.json.gz").read_bytes())
    assert hashlib.sha256(tokenizer_bytes).hexdigest() == golden["source"]["tokenizer_sha256"]
    tokenizer = SoproTokenizer(json.loads(tokenizer_bytes))
    for item in golden["tokenization"]:
        result = tokenizer.encode(item["text"], item["language"])
        effective_language = None if item["language"] == "auto" else item["language"]
        assert result.normalized_text == item["normalized"]
        assert list(result.token_ids) == item["token_ids"]
        assert result.language == effective_language


def test_unsupported_tokenizer_language_is_rejected() -> None:
    tokenizer_bytes = gzip.decompress((FIXTURES / "tokenizer-v2-turbo.json.gz").read_bytes())
    tokenizer = SoproTokenizer(json.loads(tokenizer_bytes))
    with pytest.raises(CapabilityError, match="Unsupported Sopro language"):
        tokenizer.encode("hello", "es")
