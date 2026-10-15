from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..errors import RuntimeContractError
from .qwen3_contract import CODE_GROUPS, LAYOUT, SAMPLE_RATE

QWEN3_BASE_MAX_REFERENCE_SAMPLES = 240_000
QWEN3_BASE_MAX_REFERENCE_SECONDS = 10.0
QWEN3_CODEC_SAMPLES_PER_FRAME = 1_920
QWEN3_MEL_N_FFT = 1_024
QWEN3_MEL_HOP_LENGTH = 256
QWEN3_MEL_N_BINS = 128


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float64)
    mel = values * (3.0 / 200.0)
    min_log_hz = 1_000.0
    min_log_mel = min_log_hz * (3.0 / 200.0)
    log_step = math.log(6.4) / 27.0
    logarithmic = values >= min_log_hz
    mel[logarithmic] = min_log_mel + np.log(values[logarithmic] / min_log_hz) / log_step
    return mel


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    values = np.asarray(mels, dtype=np.float64)
    frequencies = values * (200.0 / 3.0)
    min_log_hz = 1_000.0
    min_log_mel = min_log_hz * (3.0 / 200.0)
    log_step = math.log(6.4) / 27.0
    logarithmic = values >= min_log_mel
    frequencies[logarithmic] = min_log_hz * np.exp(log_step * (values[logarithmic] - min_log_mel))
    return frequencies


def mel_filterbank(
    *,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = QWEN3_MEL_N_FFT,
    n_mels: int = QWEN3_MEL_N_BINS,
    fmin: float = 0.0,
    fmax: float = SAMPLE_RATE / 2,
) -> np.ndarray:
    """Construct librosa-compatible Slaney mel triangles and area normalization."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0:
        raise RuntimeContractError("Qwen3 mel sample_rate must be a positive integer")
    if isinstance(n_fft, bool) or not isinstance(n_fft, Integral) or n_fft <= 0 or n_fft % 2:
        raise RuntimeContractError("Qwen3 mel n_fft must be a positive even integer")
    if isinstance(n_mels, bool) or not isinstance(n_mels, Integral) or n_mels <= 0:
        raise RuntimeContractError("Qwen3 mel n_mels must be a positive integer")
    if (
        not math.isfinite(float(fmin))
        or not math.isfinite(float(fmax))
        or fmin < 0
        or fmax <= fmin
        or fmax > sample_rate / 2
    ):
        raise RuntimeContractError(
            "Qwen3 mel frequency bounds must satisfy 0 <= fmin < fmax <= Nyquist"
        )

    fft_frequencies = np.linspace(0.0, sample_rate / 2, 1 + n_fft // 2, dtype=np.float64)
    min_mel, max_mel = _hz_to_mel(np.asarray([fmin, fmax], dtype=np.float64))
    mel_frequencies = _mel_to_hz(np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64))
    differences = np.diff(mel_frequencies)
    ramps = mel_frequencies[:, None] - fft_frequencies[None, :]
    weights = np.zeros((n_mels, fft_frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        lower = -ramps[index] / differences[index]
        upper = ramps[index + 2] / differences[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_frequencies[2 : n_mels + 2] - mel_frequencies[:n_mels]))[:, None]
    result = weights.astype(np.float32)
    result.setflags(write=False)
    return result


def compute_log_mel(samples: np.ndarray) -> np.ndarray:
    """Compute WaveKat's uncentered periodic-Hann power/log-mel tensor [1, frames, 128]."""
    try:
        source = np.asarray(samples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError("Qwen3 reference audio must be float-compatible") from exc
    if source.dtype.kind not in "iuf":
        raise RuntimeContractError("Qwen3 reference audio must be float-compatible numeric samples")
    try:
        audio = np.asarray(source, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError("Qwen3 reference audio must be float-compatible") from exc
    if audio.ndim != 1 or audio.size < QWEN3_MEL_N_FFT:
        raise RuntimeContractError("Qwen3 mel input must be mono with at least 1024 samples")
    if not np.isfinite(audio).all():
        raise RuntimeContractError("Qwen3 mel input must contain only finite samples")
    frame_count = 1 + (audio.size - QWEN3_MEL_N_FFT) // QWEN3_MEL_HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(audio, QWEN3_MEL_N_FFT)[
        ::QWEN3_MEL_HOP_LENGTH
    ][:frame_count].copy()
    sample_indices = np.arange(QWEN3_MEL_N_FFT, dtype=np.float64)
    window = (0.5 - 0.5 * np.cos((2.0 * np.pi * sample_indices) / QWEN3_MEL_N_FFT)).astype(
        np.float32
    )
    spectrum = np.fft.rfft(frames * window[None, :], n=QWEN3_MEL_N_FFT, axis=1)
    power = np.asarray(np.abs(spectrum) ** 2, dtype=np.float32)
    mel = power @ mel_filterbank().T
    logged = np.log(np.maximum(mel, np.float32(1e-5))).astype(np.float32)
    result = logged[None, :, :]
    if result.shape[1:] != (frame_count, QWEN3_MEL_N_BINS) or not np.isfinite(result).all():
        raise RuntimeContractError("Qwen3 NumPy log-mel transform returned invalid values")
    return np.ascontiguousarray(result)


def validate_reference_audio(audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """Validate and copy mono float-compatible 24-kHz Base conditioning audio."""
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, Integral)
        or sample_rate != SAMPLE_RATE
    ):
        raise RuntimeContractError(f"Qwen3 reference audio must be prepared at {SAMPLE_RATE} Hz")
    try:
        values = np.asarray(audio)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(
            "Qwen3 reference audio must be float-compatible numeric samples"
        ) from exc
    if values.ndim != 1:
        raise RuntimeContractError("Qwen3 reference audio must be a mono one-dimensional waveform")
    if values.dtype.kind not in "iuf" or values.dtype.kind == "b":
        raise RuntimeContractError("Qwen3 reference audio must be float-compatible numeric samples")
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError("Qwen3 reference audio must be float-compatible") from exc
    if result.size == 0:
        raise RuntimeContractError("Qwen3 reference audio must not be empty")
    if not np.isfinite(result).all():
        raise RuntimeContractError("Qwen3 reference audio must contain only finite samples")
    if result.size < QWEN3_MEL_N_FFT:
        raise RuntimeContractError("Qwen3 reference audio must contain at least 1024 samples")
    if result.size > QWEN3_BASE_MAX_REFERENCE_SAMPLES:
        raise RuntimeContractError("Qwen3 Base reference audio must be no longer than 10 seconds")
    return np.ascontiguousarray(result)


def validate_reference_token_ids(
    token_ids: Sequence[int],
    *,
    text_embedding_rows: int,
    im_start_token_id: int,
    im_end_token_id: int,
) -> tuple[int, ...]:
    """Validate a frontend-tokenized, assistant-wrapped Base transcript."""
    if isinstance(token_ids, (str, bytes)):
        raise RuntimeContractError(
            "Qwen3 reference_token_ids must be a one-dimensional integer sequence"
        )
    try:
        values = np.asarray(list(token_ids))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(
            "Qwen3 reference_token_ids must be a one-dimensional integer sequence"
        ) from exc
    if values.ndim != 1 or values.size <= 5 or not np.issubdtype(values.dtype, np.integer):
        raise RuntimeContractError(
            "Qwen3 reference_token_ids must contain wrapped non-empty transcript text"
        )
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= text_embedding_rows):
        raise RuntimeContractError(
            "Qwen3 reference_token_ids contain IDs outside the text embedding"
        )
    if values[0] != im_start_token_id or values[-2] != im_end_token_id:
        raise RuntimeContractError(
            "Qwen3 reference_token_ids must use the supported wrapped chat format"
        )
    return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True, eq=False)
class Qwen3VoiceState:
    """Immutable Base voice conditioning prepared from a single 24-kHz reference."""

    speaker_embedding: np.ndarray
    reference_codes: np.ndarray
    reference_token_ids: tuple[int, ...]
    sample_rate: int
    reference_samples: int
    model_id: str
    layout: str = LAYOUT

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise RuntimeContractError(f"Qwen3 voice state sample_rate must be {SAMPLE_RATE}")
        if (
            isinstance(self.reference_samples, bool)
            or not isinstance(self.reference_samples, Integral)
            or not 1 <= self.reference_samples <= QWEN3_BASE_MAX_REFERENCE_SAMPLES
        ):
            raise RuntimeContractError(
                "Qwen3 voice state reference_samples must be between 1 and 240000"
            )
        if not isinstance(self.model_id, str) or not self.model_id:
            raise RuntimeContractError("Qwen3 voice state model_id must be a non-empty string")
        if self.layout != LAYOUT:
            raise RuntimeContractError(f"Qwen3 voice state layout must be {LAYOUT!r}")
        token_ids = tuple(self.reference_token_ids)
        if not token_ids or any(
            isinstance(token_id, bool) or not isinstance(token_id, Integral) or token_id < 0
            for token_id in token_ids
        ):
            raise RuntimeContractError(
                "Qwen3 voice state reference_token_ids must be non-negative integers"
            )
        speaker = np.asarray(self.speaker_embedding, dtype=np.float32)
        if speaker.ndim != 2 or speaker.shape[0] != 1 or speaker.shape[1] == 0:
            raise RuntimeContractError("Qwen3 speaker_embedding must have shape [1, hidden]")
        if not np.isfinite(speaker).all():
            raise RuntimeContractError("Qwen3 speaker_embedding must be finite")
        codes = np.asarray(self.reference_codes)
        if codes.ndim != 3 or codes.shape[:2] != (1, CODE_GROUPS) or codes.shape[2] == 0:
            raise RuntimeContractError(
                "Qwen3 reference_codes must have non-empty [1, 16, frames] shape"
            )
        if (
            not np.issubdtype(codes.dtype, np.integer)
            or np.any(codes < 0)
            or np.any(codes > np.iinfo(np.int64).max)
        ):
            raise RuntimeContractError(
                "Qwen3 reference_codes must contain non-negative integer IDs"
            )
        speaker_copy = np.frombuffer(
            np.ascontiguousarray(speaker).tobytes(), dtype=np.float32
        ).reshape(speaker.shape)
        codes_copy = np.frombuffer(
            np.ascontiguousarray(codes, dtype=np.int64).tobytes(), dtype=np.int64
        ).reshape(codes.shape)
        object.__setattr__(self, "speaker_embedding", speaker_copy)
        object.__setattr__(self, "reference_codes", codes_copy)
        object.__setattr__(
            self, "reference_token_ids", tuple(int(token_id) for token_id in token_ids)
        )
        object.__setattr__(self, "reference_samples", int(self.reference_samples))
