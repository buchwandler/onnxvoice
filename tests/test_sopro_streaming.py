from __future__ import annotations

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.sopro import _sopro_istft
from onnxvoice.systems.sopro_streaming import (
    SoproStreamChunk,
    _chunk_attention_bias,
    _select_stream_graph_mode,
    _SoproNoise,
    _stable_prompt_frames,
    _stream_target_frames,
    _StreamingISTFT,
)


def test_chunk_attention_bias_masks_future_chunks() -> None:
    bias = _chunk_attention_bias(5, 2)
    assert bias.shape == (1, 1, 5, 5)
    assert bias[0, 0, 0].tolist() == [0.0, 0.0, -1.0e4, -1.0e4, -1.0e4]
    assert bias[0, 0, 2].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0e4]
    assert bias[0, 0, 4].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeContractError):
        _chunk_attention_bias(0, 2)
    with pytest.raises(RuntimeContractError):
        _chunk_attention_bias(2, 0)


def test_stream_graph_selection_prefers_split_then_falls_back_to_monolithic() -> None:
    common = ("acoustic_stream_prefill_2", "acoustic_stream_step_2")
    split = common + (
        "acoustic_stream_prefill_ode_2_0",
        "acoustic_stream_prefill_ode_2_1",
        "acoustic_stream_ode_2_0",
        "acoustic_stream_ode_2_1",
    )
    mode, selected = _select_stream_graph_mode(split, 2)
    assert mode == "split" and len(selected) == 6
    assert _select_stream_graph_mode(common, 2) == ("monolithic", common)
    with pytest.raises(CapabilityError):
        _select_stream_graph_mode(("acoustic_stream_prefill_2",), 2)


def test_stream_prompt_alignment_and_acoustic_target_advancement() -> None:
    assert _stable_prompt_frames(100, 30, 4, 4, 8) == 96
    assert _stable_prompt_frames(20, 2, 4, 4, 8) == 0
    assert _stream_target_frames(100, 10, 4, 4, 8, final=False) == 120
    assert _stream_target_frames(100, 10, 4, 4, 8, final=True) == 140
    with pytest.raises(RuntimeContractError):
        _stable_prompt_frames(10, 4, -1, 4, 8)


def test_stream_noise_extension_preserves_prior_frames_and_layout() -> None:
    noise = _SoproNoise(123)
    noise.ensure(3)
    first = noise.slice(0, 3)
    assert first.shape == (1, 100, 3)
    noise.ensure(7)
    np.testing.assert_array_equal(noise.slice(0, 3), first)
    reference = _SoproNoise(123)
    reference.ensure(7)
    np.testing.assert_array_equal(noise.slice(0, 7), reference.slice(0, 7))
    with pytest.raises(RuntimeContractError):
        noise.slice(6, 8)


def test_stream_chunk_rejects_invalid_audio_contract() -> None:
    chunk = SoproStreamChunk(
        np.asarray([0.0, 1.0], dtype=np.float32),
        sample_rate=24000,
        is_final=True,
        metadata={"streaming": True},
    )
    assert chunk.audio.dtype == np.float32
    assert chunk.audio.ndim == 1 and chunk.audio.flags.c_contiguous
    with pytest.raises(RuntimeContractError):
        SoproStreamChunk(
            np.arange(4, dtype=np.float32)[::2],
            sample_rate=24000,
            is_final=True,
            metadata={},
        )


def test_streaming_istft_defers_samples_that_might_be_trimmed_on_flush() -> None:
    rng = np.random.default_rng(8)
    n_fft, hop, frames = 16, 12, 5
    features = np.empty((frames, n_fft + 2), dtype=np.float32)
    features[:, : n_fft // 2 + 1] = rng.uniform(-2.0, 1.0, (frames, n_fft // 2 + 1))
    features[:, n_fft // 2 + 1 :] = rng.uniform(-np.pi, np.pi, (frames, n_fft // 2 + 1))
    expected = _sopro_istft(features, frames, n_fft=n_fft, hop=hop)
    stream = _StreamingISTFT(n_fft, hop)
    chunks = [stream.push(features[index : index + 1], 1, flush=False) for index in range(frames)]
    chunks.append(stream.push(np.empty((0, n_fft + 2), dtype=np.float32), 0, flush=True))
    actual = np.concatenate(chunks)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_streaming_istft_matches_one_shot_with_multiple_pushes() -> None:
    rng = np.random.default_rng(2026)
    n_fft, hop, frames = 16, 4, 11
    features = np.empty((frames, n_fft + 2), dtype=np.float32)
    features[:, : n_fft // 2 + 1] = rng.uniform(-3.0, 1.0, (frames, n_fft // 2 + 1))
    features[:, n_fft // 2 + 1 :] = rng.uniform(-np.pi, np.pi, (frames, n_fft // 2 + 1))
    expected = _sopro_istft(features, frames, n_fft=n_fft, hop=hop)

    stream = _StreamingISTFT(n_fft, hop)
    chunks = [stream.push(features[:3], 3, flush=False)]
    chunks.append(stream.push(features[3:7], 4, flush=False))
    chunks.append(stream.push(features[7:], 4, flush=False))
    chunks.append(stream.push(np.empty((0, n_fft + 2), dtype=np.float32), 0, flush=True))
    actual = np.concatenate(chunks)
    assert actual.dtype == np.float32
    assert actual.ndim == 1 and actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    assert actual.size == (frames - 1) * hop
    with pytest.raises(RuntimeContractError, match="already flushed"):
        stream.push(np.empty((0, n_fft + 2), dtype=np.float32), 0, flush=True)
