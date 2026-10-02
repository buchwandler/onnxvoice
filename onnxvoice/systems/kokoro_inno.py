from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import CapabilityError, OptionalDependencyError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import Installation, RuntimeDiagnostic

INNO_UPSTREAM_VERSION = "0.2.0"
INNO_UPSTREAM_COMMIT = "892ef184bc932aa3ff9d72c1509d5b81ff6941e6"
INNO_ENCODER_SAMPLE_RATE = 16000
INNO_TILT_SAMPLE_RATE = 24000
INNO_FBANK_FRAME_LENGTH = 400
INNO_FBANK_FRAME_HOP = 160
INNO_FBANK_FFT_SIZE = 512
INNO_FBANK_MEL_BINS = 80


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class KokoroVoicePack:
    """Immutable, stock-compatible Kokoro style table with enrollment provenance."""

    data: np.ndarray
    enroller: str
    model_fingerprint: str
    fingerprint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    enroller_fingerprint: str = ""

    def __post_init__(self) -> None:
        try:
            data = np.array(self.data, dtype=np.float32, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError("Kokoro voice-pack data must be numeric") from exc
        if data.shape != (510, 1, 256):
            raise RuntimeContractError(
                f"Kokoro voice-pack data must have shape [510, 1, 256], got {data.shape}"
            )
        if not np.isfinite(data).all():
            raise RuntimeContractError("Kokoro voice-pack data must be finite")
        if not isinstance(self.enroller, str) or not self.enroller:
            raise RuntimeContractError("Kokoro voice-pack enroller provenance must be non-empty")
        if (
            not isinstance(self.model_fingerprint, str)
            or len(self.model_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.model_fingerprint)
        ):
            raise RuntimeContractError(
                "Kokoro voice-pack model fingerprint must be a lowercase SHA-256 digest"
            )
        enroller_fingerprint = (
            self.enroller_fingerprint
            or hashlib.sha256(f"{self.enroller}:{self.model_fingerprint}".encode()).hexdigest()
        )
        if (
            not isinstance(enroller_fingerprint, str)
            or len(enroller_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in enroller_fingerprint)
        ):
            raise RuntimeContractError(
                "Kokoro voice-pack enroller fingerprint must be a lowercase SHA-256 digest"
            )
        if not isinstance(self.metadata, Mapping):
            raise RuntimeContractError("Kokoro voice-pack metadata must be a mapping")
        try:
            metadata = dict(self.metadata)
            encoded = json.dumps(
                {
                    "enroller": self.enroller,
                    "model_fingerprint": self.model_fingerprint,
                    "enroller_fingerprint": enroller_fingerprint,
                    "metadata": metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(
                "Kokoro voice-pack metadata must be finite JSON data"
            ) from exc
        digest = hashlib.sha256(data.tobytes(order="C") + encoded).hexdigest()
        frozen_metadata = _freeze_json(json.loads(encoded)["metadata"])
        if self.fingerprint and self.fingerprint != digest:
            raise RuntimeContractError("Kokoro voice-pack fingerprint does not match its data")
        data.setflags(write=False)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", frozen_metadata)
        object.__setattr__(self, "enroller_fingerprint", enroller_fingerprint)
        object.__setattr__(self, "fingerprint", digest)

    def style_for(self, token_count: int) -> np.ndarray:
        """Select the stock style row used by the existing Kokoro inference API."""
        if isinstance(token_count, bool) or not isinstance(token_count, (int, np.integer)):
            raise RuntimeContractError("Kokoro style token count must be a non-negative integer")
        if token_count < 0:
            raise RuntimeContractError("Kokoro style token count must be a non-negative integer")
        return np.array(self.data[min(int(token_count), 509)], copy=True).reshape(1, 256)


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if values.ndim != 1:
        raise RuntimeContractError("Inno preprocessing expects a mono one-dimensional waveform")
    if not np.issubdtype(values.dtype, np.floating):
        raise RuntimeContractError("Inno preprocessing expects floating-point audio")
    waveform = np.asarray(values, dtype=np.float32)
    if not waveform.size or not np.isfinite(waveform).all():
        raise RuntimeContractError("Inno audio must be non-empty and finite")
    return np.ascontiguousarray(waveform)


@lru_cache(maxsize=1)
def _mel_filterbank() -> np.ndarray:
    """Return Inno v0.2.0's [80, 257] Kaldi-style mel filters."""
    mel_min = 1127.0 * np.log1p(20.0 / 700.0)
    mel_max = 1127.0 * np.log1p(8000.0 / 700.0)
    spacing = (mel_max - mel_min) / 81
    left = mel_min + np.arange(INNO_FBANK_MEL_BINS, dtype=np.float64)[:, None] * spacing
    frequencies = np.arange(INNO_FBANK_FFT_SIZE // 2, dtype=np.float64) * (
        INNO_ENCODER_SAMPLE_RATE / INNO_FBANK_FFT_SIZE
    )
    mel_frequencies = 1127.0 * np.log1p(frequencies / 700.0)
    banks = np.minimum(
        (mel_frequencies - left) / spacing,
        (left + 2 * spacing - mel_frequencies) / spacing,
    )
    banks = np.maximum(banks, 0.0).astype(np.float32)
    filters = np.pad(banks, ((0, 0), (0, 1)))
    filters.setflags(write=False)
    return filters


def speaker_fbank(audio_16k: np.ndarray) -> np.ndarray:
    """Compute Inno v0.2.0's Kaldi-style fbank as float32 ``[1, F, 80]``."""
    waveform = _mono_float32(audio_16k)
    frame_length = INNO_FBANK_FRAME_LENGTH
    if waveform.size < frame_length:
        raise RuntimeContractError("Inno fbank needs at least 400 samples at 16 kHz")

    scaled = waveform * np.float32(32768.0)
    frames = np.lib.stride_tricks.sliding_window_view(scaled, frame_length)[
        ::INNO_FBANK_FRAME_HOP
    ].copy()
    frames -= frames.mean(axis=-1, keepdims=True, dtype=np.float32)
    previous = np.concatenate((frames[:, :1], frames[:, :-1]), axis=-1)
    frames -= np.float32(0.97) * previous

    window = np.hamming(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=INNO_FBANK_FFT_SIZE, axis=-1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    energy = power @ _mel_filterbank().T
    features = np.log(np.maximum(energy, np.finfo(np.float32).eps))
    features -= features.mean(axis=0, keepdims=True, dtype=np.float32)
    return np.ascontiguousarray(features[None], dtype=np.float32)


def resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono float waveform with the polyphase filter used by Inno v0.2.0."""
    waveform = _mono_float32(audio)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RuntimeContractError("Inno source sample_rate must be a positive integer")
    if not isinstance(target_rate, int) or target_rate <= 0:
        raise RuntimeContractError("Inno target sample_rate must be a positive integer")
    if sample_rate == target_rate:
        return waveform.copy()
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise OptionalDependencyError(
            "Inno audio resampling requires SciPy. Install onnxvoice[inno]."
        ) from exc

    divisor = math.gcd(sample_rate, target_rate)
    return np.ascontiguousarray(
        resample_poly(waveform, target_rate // divisor, sample_rate // divisor), dtype=np.float32
    )


F0_MIN = 60.0
F0_MAX = 400.0
PITCH_HOP_SAMPLES = 320
PITCH_FRAME_SAMPLES = 2 * PITCH_HOP_SAMPLES
PITCH_MEDIAN_FRAMES = 30
PITCH_LAG_TOLERANCE = 0.9
SEMITONE_REFERENCE_HZ = 100.0
CEPSTRUM_FRAME_SAMPLES = 4096
CEPSTRUM_MIN_PEAK = 0.01
CEPSTRUM_CEILING_FACTOR = 1.6
CEPSTRUM_TOLERANCE = 0.25
TILT_WINDOW_SAMPLES = 1024
TILT_HOP_SAMPLES = 256
MIN_BLEND_PACKS = 3


def _frame_rms(waveform: np.ndarray, hop: int) -> np.ndarray:
    frame_count = waveform.size // hop
    frames = waveform[: frame_count * hop].reshape(frame_count, hop)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1, dtype=np.float32))


def _fundamental(rows: np.ndarray, offset: int, tolerance: float) -> np.ndarray:
    best = np.argmax(rows, axis=1)
    peak = np.take_along_axis(rows, best[:, None], axis=1)[:, 0]
    index = best.copy()
    neighbor_offsets = np.array([-1, 0, 1])
    for divisor in (2, 3, 4):
        candidate = np.rint((best + offset) / divisor).astype(np.int64) - offset
        neighbors = np.clip(candidate[:, None] + neighbor_offsets, 0, rows.shape[1] - 1)
        values = np.take_along_axis(rows, neighbors, axis=1)
        argmax = np.argmax(values, axis=1)
        candidate_values = np.take_along_axis(values, argmax[:, None], axis=1)[:, 0]
        candidate_indices = np.take_along_axis(neighbors, argmax[:, None], axis=1)[:, 0]
        use_candidate = (candidate >= 0) & (peak > 0) & (candidate_values >= tolerance * peak)
        index = np.where(use_candidate, candidate_indices, index)
    return index


def _track_f0(
    waveform: np.ndarray, lower_hz: float, upper_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    first_lag = int(INNO_ENCODER_SAMPLE_RATE / upper_hz)
    last_lag = int(INNO_ENCODER_SAMPLE_RATE / lower_hz)
    lags = np.arange(first_lag, last_lag + 1, dtype=np.int64)
    frame_length = PITCH_FRAME_SAMPLES + int(lags[-1])
    padded = np.pad(waveform, (0, frame_length))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::PITCH_HOP_SAMPLES]
    x = frames[:, :PITCH_FRAME_SAMPLES]
    x_energy = np.sum(np.square(x, dtype=np.float32), axis=1, dtype=np.float32)
    correlations = np.empty((frames.shape[0], lags.size), dtype=np.float32)
    for column, lag in enumerate(lags):
        y = frames[:, lag : lag + PITCH_FRAME_SAMPLES]
        y_energy = np.sum(np.square(y, dtype=np.float32), axis=1, dtype=np.float32)
        numerator = np.sum(x * y, axis=1, dtype=np.float32)
        denominator = np.sqrt(x_energy * y_energy, dtype=np.float32)
        correlations[:, column] = numerator / np.maximum(denominator, np.float32(1.0e-8))

    best_lag = (first_lag + _fundamental(correlations, first_lag, PITCH_LAG_TOLERANCE))[
        : waveform.size // PITCH_HOP_SAMPLES
    ].astype(np.float32)
    left = PITCH_MEDIAN_FRAMES // 2
    right = PITCH_MEDIAN_FRAMES - 1 - left
    padded_lags = np.pad(best_lag, (left, right), mode="edge")
    lag_windows = np.lib.stride_tricks.sliding_window_view(padded_lags, PITCH_MEDIAN_FRAMES)
    best_lag = np.partition(lag_windows, PITCH_MEDIAN_FRAMES // 2 - 1, axis=1)[
        :, PITCH_MEDIAN_FRAMES // 2 - 1
    ]
    f0 = np.float32(INNO_ENCODER_SAMPLE_RATE) / best_lag
    return f0, _frame_rms(waveform, PITCH_HOP_SAMPLES)


def _cepstral_f0(waveform: np.ndarray) -> tuple[float, float]:
    frame_count = waveform.size // CEPSTRUM_FRAME_SAMPLES
    frames = waveform[: frame_count * CEPSTRUM_FRAME_SAMPLES].reshape(
        frame_count, CEPSTRUM_FRAME_SAMPLES
    )
    if frame_count:
        energy = np.mean(np.square(frames, dtype=np.float32), axis=1, dtype=np.float32)
        frames = frames[energy > np.quantile(energy, 0.7)]
    if not frames.shape[0]:
        return 0.0, 0.0

    sample_indices = np.arange(CEPSTRUM_FRAME_SAMPLES, dtype=np.float32)
    window = np.float32(0.5) - np.float32(0.5) * np.cos(
        np.float32(2.0 * np.pi) * sample_indices / np.float32(CEPSTRUM_FRAME_SAMPLES)
    )
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    log_magnitude = np.log(magnitude + np.float32(1.0e-8))
    log_magnitude -= log_magnitude.mean(axis=1, keepdims=True, dtype=np.float32)
    cepstrum = np.fft.irfft(log_magnitude, n=CEPSTRUM_FRAME_SAMPLES, axis=1).astype(np.float32)
    cepstrum = cepstrum.mean(axis=0, dtype=np.float32)
    quefrency = np.arange(cepstrum.size, dtype=np.float32) / np.float32(INNO_ENCODER_SAMPLE_RATE)
    band_mask = (quefrency > 1.0 / 300.0) & (quefrency < 1.0 / F0_MIN)
    band = cepstrum[band_mask]
    start = int(np.flatnonzero(band_mask)[0])
    index = int(_fundamental(band[None], start, CEPSTRUM_TOLERANCE)[0])
    return INNO_ENCODER_SAMPLE_RATE / (start + index), float(band[index])


def ceiling(audio: np.ndarray, sample_rate: int, fmax: float | None = None) -> float:
    """Return the upstream cepstral pitch ceiling, or the explicit override."""
    if fmax is not None:
        return float(fmax)
    waveform = resample_audio(audio, sample_rate, INNO_ENCODER_SAMPLE_RATE)
    fundamental, peak = _cepstral_f0(waveform)
    return (
        min(F0_MAX, CEPSTRUM_CEILING_FACTOR * fundamental) if peak > CEPSTRUM_MIN_PEAK else F0_MAX
    )


def stats(
    audio: np.ndarray, sample_rate: int, fmax: float | None = None
) -> tuple[float, float, float]:
    """Return upstream log-F0 mean/std in semitones and voiced fraction."""
    waveform = _mono_float32(audio)
    effective_fmax = ceiling(waveform, sample_rate, fmax)
    encoder_waveform = resample_audio(waveform, sample_rate, INNO_ENCODER_SAMPLE_RATE)
    lower_hz = F0_MIN
    for _ in range(2):
        f0, energy = _track_f0(encoder_waveform, lower_hz, effective_fmax)
        voiced = (energy > np.float32(0.1) * energy.max()) & (f0 > lower_hz) & (f0 < effective_fmax)
        voiced_st = np.float32(12.0) * np.log2(f0[voiced] / np.float32(SEMITONE_REFERENCE_HZ))
        if not voiced_st.size:
            return 0.0, 0.0, 0.0
        upper_quartile = np.quantile(voiced_st, 0.75)
        lower_hz = min(
            0.65 * SEMITONE_REFERENCE_HZ * 2 ** (upper_quartile / 12.0),
            180.0,
        )

    return (
        float(voiced_st.mean(dtype=np.float32)),
        float(voiced_st.std(dtype=np.float32, ddof=1)),
        float(voiced.mean(dtype=np.float32)),
    )


def rate(audio: np.ndarray, sample_rate: int, limit: int = 60) -> float:
    """Return Inno's smoothed energy-peak syllable-rate proxy."""
    waveform = _mono_float32(audio)
    hop = sample_rate // 100
    energy = _frame_rms(waveform[: limit * sample_rate], hop)
    if not energy.size:
        return 0.0
    padded = np.pad(energy, (1, 1), mode="constant")
    energy = (padded[:-2] + padded[1:-1] + padded[2:]) / np.float32(3.0)
    speech = energy > np.float32(0.05) * energy.max()
    peaks = np.flatnonzero(
        (energy[1:-1] > energy[:-2]) & (energy[1:-1] >= energy[2:]) & speech[1:-1]
    )
    if not peaks.size:
        return 0.0
    syllables = 1 + np.count_nonzero(np.diff(peaks) >= 8)
    speech_seconds = float(speech.mean(dtype=np.float32)) * energy.size / 100.0
    return float(syllables / (speech_seconds + 1.0e-9))


def tilt(audio: np.ndarray, sample_rate: int) -> float:
    """Return the Inno long-term spectral slope in dB per octave."""
    waveform = resample_audio(audio, sample_rate, INNO_TILT_SAMPLE_RATE)
    fft_size = TILT_WINDOW_SAMPLES
    if waveform.size <= fft_size // 2:
        raise RuntimeContractError("Inno spectral tilt needs more than 512 samples at 24 kHz")
    padded = np.pad(waveform, (fft_size // 2, fft_size // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, fft_size)[::TILT_HOP_SAMPLES]
    index = np.arange(fft_size, dtype=np.float32)
    window = np.float32(0.5) - np.float32(0.5) * np.cos(
        np.float32(2.0 * np.pi) * index / np.float32(fft_size)
    )
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32).mean(axis=0, dtype=np.float32)
    frequencies = np.linspace(0.0, INNO_TILT_SAMPLE_RATE / 2, power.size, dtype=np.float32)
    band = (frequencies > 300.0) & (frequencies < 4000.0)
    x = np.log2(frequencies[band])
    y = np.float32(10.0) * np.log10(power[band] + np.float32(1.0e-12))
    centered_x = x - x.mean(dtype=np.float32)
    centered_y = y - y.mean(dtype=np.float32)
    return float(
        np.sum(centered_x * centered_y, dtype=np.float32)
        / np.sum(centered_x * centered_x, dtype=np.float32)
    )


def head_stats(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = F0_MIN,
    fmax: float = F0_MAX,
) -> tuple[float, float]:
    """Return upstream Praat-based, six-second chunked prosody-head statistics."""
    try:
        import parselmouth
    except ImportError as exc:
        raise OptionalDependencyError(
            "Inno prosody-head analysis requires praat-parselmouth. Install onnxvoice[inno]."
        ) from exc

    waveform = _mono_float32(audio).astype(np.float64)
    chunk_samples = int(6.0 * sample_rate)
    hop_seconds = 0.01
    chunk_results: list[tuple[float, float, int]] = []
    for offset in range(0, waveform.size, chunk_samples):
        chunk = waveform[offset : offset + chunk_samples]
        if chunk.size < 1.5 * sample_rate:
            continue
        pitch = parselmouth.Sound(chunk, sampling_frequency=sample_rate).to_pitch_ac(
            time_step=hop_seconds, pitch_floor=fmin, pitch_ceiling=fmax
        )
        frequencies = pitch.selected_array["frequency"]
        rms_hop = int(sample_rate * hop_seconds)
        frame_count = chunk.size // rms_hop
        rms_frames = chunk[: frame_count * rms_hop].reshape(frame_count, rms_hop)
        energy = np.sqrt(np.mean(np.square(rms_frames), axis=1))
        energy_indices = np.clip((pitch.xs() / hop_seconds).astype(int), 0, energy.size - 1)
        frame_energy = energy[energy_indices]
        voiced = frequencies[(frequencies > 0) & (frame_energy > 0.1 * frame_energy.max())]
        if voiced.size:
            semitones = 12.0 * np.log2(voiced / SEMITONE_REFERENCE_HZ)
            chunk_results.append((float(semitones.mean()), float(semitones.std()), int(chunk.size)))
    if not chunk_results:
        return 0.0, 0.0
    total_samples = sum(result[2] for result in chunk_results)
    return (
        float(sum(result[0] * result[2] for result in chunk_results) / total_samples),
        float(sum(result[1] * result[2] for result in chunk_results) / total_samples),
    )


def blend_weights(tuner: Any, z: np.ndarray) -> dict[str, float]:
    """Match upstream pitch-gated SciPy NNLS stock-pack blending."""
    stats_table = np.asarray(tuner.stats)
    target = np.asarray(z)
    names = np.asarray(tuner.names)
    grades = np.asarray(tuner.grades)
    scales = np.asarray(tuner.scale)
    pitch_distance = np.abs(stats_table[:, 0] - target[0])
    selected = pitch_distance <= tuner.gate
    if selected.sum() < MIN_BLEND_PACKS:
        selected = pitch_distance <= np.sort(pitch_distance)[MIN_BLEND_PACKS - 1]

    penalties = tuner.grade_pen * np.maximum(0.0, 2.0 - grades[selected])
    matrix = np.vstack(
        [
            stats_table[selected].T / scales[:, None],
            10.0 * np.ones((1, selected.sum())),
            np.diag(penalties),
        ]
    )
    vector = np.concatenate([target / scales, [10.0], np.zeros(selected.sum(), dtype=np.float64)])
    try:
        from scipy.optimize import nnls
    except ImportError as exc:
        raise OptionalDependencyError(
            "Inno voice blending requires SciPy. Install onnxvoice[inno]."
        ) from exc

    weights, _ = nnls(matrix, vector)
    weights /= weights.sum()
    return {
        str(name): float(weight)
        for name, weight in zip(names[selected], weights, strict=True)
        if weight > 1.0e-3
    }


def dense_blend_weights(
    names: tuple[str, ...] | list[str], weights: dict[str, float]
) -> np.ndarray:
    """Expand upstream's sparse blend map to the ONNX graph's ``[1, V]`` input."""
    return np.asarray([[weights.get(name, 0.0) for name in names]], dtype=np.float32)


INNO_ENROLLER_ID = "inno-v0.2"
INNO_GRAPH_COMPONENT = "inno_voicepack"
INNO_METADATA_COMPONENT = "inno_tuner"
INNO_GRAPH_INPUTS = frozenset({"fbank", "tilt", "head_stats", "blend_weights", "head_enabled"})


INNO_SUPPORTED_LAYOUTS = frozenset({"single", "single-onnx-v1", "split", "multi", "split-onnx-v1"})


def _inno_capability(installation: Installation) -> Mapping[str, Any]:
    runtime = installation.metadata.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CapabilityError("Kokoro installation does not advertise Inno voice enrollment")
    capabilities = runtime.get("voice_enrollers")
    if not isinstance(capabilities, (list, tuple)):
        raise CapabilityError("Kokoro installation does not advertise Inno voice enrollment")
    capability = next(
        (
            value
            for value in capabilities
            if isinstance(value, Mapping) and value.get("id") == INNO_ENROLLER_ID
        ),
        None,
    )
    if capability is None:
        raise CapabilityError(f"Kokoro installation does not advertise {INNO_ENROLLER_ID}")

    expected_values = {
        "kind": "kokoro-voicepack-tuner",
        "input": "reference-audio",
        "model_component": INNO_GRAPH_COMPONENT,
        "metadata_component": INNO_METADATA_COMPONENT,
    }
    for key, expected in expected_values.items():
        if capability.get(key) != expected:
            raise CapabilityError(f"Inno voice enroller metadata has invalid {key!r}")
    if capability.get("transcript_required") is not False:
        raise CapabilityError("Inno voice enroller must declare transcript_required=false")
    for key, expected_seconds in (
        ("min_seconds", 3.0),
        ("recommended_seconds", 5.0),
        ("max_seconds", 30.0),
    ):
        value = capability.get(key)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or float(value) != expected_seconds
        ):
            raise CapabilityError(f"Inno voice enroller metadata has invalid {key!r}")
    output = capability.get("output")
    shape = output.get("shape") if isinstance(output, Mapping) else None
    if (
        not isinstance(output, Mapping)
        or output.get("format") != "kokoro-voicepack-v1"
        or not isinstance(shape, (list, tuple))
        or tuple(shape) != (510, 1, 256)
        or output.get("dtype") != "float32"
    ):
        raise CapabilityError("Inno voice enroller metadata has an invalid output contract")
    return capability


def validate_inno_installation(installation: Installation) -> None:
    """Validate declared Inno components without creating an ONNX Runtime session."""
    _inno_capability(installation)
    runtime = installation.metadata.get("runtime")
    layout = runtime.get("layout", "single") if isinstance(runtime, Mapping) else "single"
    if not isinstance(layout, str) or layout not in INNO_SUPPORTED_LAYOUTS:
        raise CapabilityError(f"Inno voice enrollment does not support Kokoro layout {layout!r}")
    models = [
        artifact
        for artifact in installation.artifacts
        if artifact.role == "model" and artifact.component != INNO_GRAPH_COMPONENT
    ]
    if not models:
        raise CapabilityError("Inno enrollment requires a base Kokoro model artifact")
    if layout in {"single", "single-onnx-v1"}:
        if not any(artifact.component is None for artifact in models):
            raise CapabilityError("Inno enrollment requires the single Kokoro model component")
    elif not {"prosody", "curves", "decoder"} <= {artifact.component for artifact in models}:
        raise CapabilityError("Inno enrollment requires the split Kokoro components")
    try:
        installation.require_artifact("model", component=INNO_GRAPH_COMPONENT)
        installation.require_artifact("metadata", component=INNO_METADATA_COMPONENT)
    except KeyError as exc:
        raise CapabilityError(f"Inno enrollment is missing a required artifact: {exc}") from exc


def inno_model_fingerprint(installation: Installation) -> str:
    """Fingerprint base Kokoro and Inno artifacts without including local paths."""
    validate_inno_installation(installation)
    capability = _inno_capability(installation)
    base_artifacts = [
        artifact
        for artifact in installation.artifacts
        if artifact.role == "model" and artifact.component != INNO_GRAPH_COMPONENT
    ]
    inno_artifacts = [
        artifact
        for artifact in installation.artifacts
        if (artifact.role, artifact.component)
        in {
            ("model", INNO_GRAPH_COMPONENT),
            ("metadata", INNO_METADATA_COMPONENT),
        }
    ]
    if not base_artifacts:
        raise CapabilityError("Inno enrollment requires a base Kokoro model artifact")
    if len(inno_artifacts) != 2:
        raise CapabilityError(
            "Inno enrollment requires its ONNX graph and tuner metadata artifacts"
        )

    artifacts = [*base_artifacts, *inno_artifacts]
    records = []
    for artifact in artifacts:
        digest = artifact.sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeContractError(
                f"Inno fingerprint artifact {artifact.component or artifact.role!r} "
                "must have a lowercase SHA-256 digest"
            )
        records.append(
            {
                "role": artifact.role,
                "component": artifact.component,
                "quality": artifact.quality,
                "sha256": digest,
            }
        )
    records.sort(
        key=lambda record: (record["role"], record["component"] or "", record["quality"] or "")
    )
    fingerprint_data = {
        "system": installation.system,
        "model": installation.id,
        "distribution": installation.selected_distribution,
        "quality": installation.selected_quality,
        "enroller": capability["id"],
        "upstream_version": INNO_UPSTREAM_VERSION,
        "upstream_commit": INNO_UPSTREAM_COMMIT,
        "artifacts": records,
    }
    encoded = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_scalar(metadata: Mapping[str, Any], key: str) -> Any:
    if key not in metadata:
        raise RuntimeContractError(f"Inno tuner metadata is missing {key!r}")
    value = np.asarray(metadata[key])
    if value.size != 1:
        raise RuntimeContractError(f"Inno tuner metadata {key!r} must be scalar")
    return value.reshape(()).item()


def _read_inno_metadata(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                metadata = {key: np.array(archive[key], copy=True) for key in archive.files}
        else:
            metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"Could not read Inno tuner metadata {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeContractError("Inno tuner metadata must contain a mapping")
    return metadata


class InnoVoiceTuner:
    """Lazy ONNX voice-pack graph and validated host-side Inno metadata."""

    def __init__(
        self,
        installation: Installation,
        *,
        session_factory: Callable[[Path, str], Any] | None = None,
        providers: Any | None = None,
        provider_options: Any | None = None,
        session_options: Any | None = None,
    ) -> None:
        self.installation = installation
        validate_inno_installation(installation)
        self.capability = _inno_capability(installation)
        try:
            model_artifact = installation.require_artifact("model", component=INNO_GRAPH_COMPONENT)
            metadata_artifact = installation.require_artifact(
                "metadata", component=INNO_METADATA_COMPONENT
            )
        except KeyError as exc:
            raise CapabilityError(f"Inno enrollment is missing a required artifact: {exc}") from exc
        self.model_path = model_artifact.path
        self.graph_sha256 = model_artifact.sha256
        self.metadata_path = metadata_artifact.path
        self.metadata_sha256 = metadata_artifact.sha256
        self.model_fingerprint = inno_model_fingerprint(installation)
        self.session_factory = session_factory
        self.providers = providers
        self.provider_options = provider_options
        self.session_options = session_options
        self._session: Any | None = None
        self._load_metadata()
        enroller_provenance = json.dumps(
            {
                "id": INNO_ENROLLER_ID,
                "version": self.version,
                "commit": self.upstream_commit,
                "checkpoint_sha256": self.checkpoint_sha256,
                "graph_sha256": self.graph_sha256,
                "metadata_sha256": self.metadata_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.enroller_fingerprint = hashlib.sha256(enroller_provenance).hexdigest()

    def _load_metadata(self) -> None:
        metadata = _read_inno_metadata(self.metadata_path)
        names = np.asarray(metadata.get("blend_names", ())).reshape(-1)
        stats_table = np.asarray(metadata.get("blend_stats", ()), dtype=np.float32)
        grades = np.asarray(metadata.get("blend_grades", ()), dtype=np.float32)
        scales = np.asarray(metadata.get("blend_scale", ()), dtype=np.float64)
        if names.size < MIN_BLEND_PACKS or stats_table.shape != (names.size, 3):
            raise RuntimeContractError("Inno blend names and statistics have incompatible shapes")
        if grades.shape != (names.size,) or scales.shape != (3,):
            raise RuntimeContractError("Inno blend grades or scales have incompatible shapes")
        if not np.isfinite(stats_table).all() or not np.isfinite(grades).all():
            raise RuntimeContractError("Inno blend statistics and grades must be finite")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise RuntimeContractError("Inno blend scales must be positive and finite")
        name_values = tuple(str(name) for name in names.tolist())
        if any(not name for name in name_values) or len(set(name_values)) != len(name_values):
            raise RuntimeContractError("Inno blend names must be non-empty and unique")

        self.version = str(_metadata_scalar(metadata, "version"))
        if self.version != INNO_UPSTREAM_VERSION:
            raise RuntimeContractError(
                f"Expected Inno tuner metadata v{INNO_UPSTREAM_VERSION}, found v{self.version}"
            )
        self.upstream_commit = str(_metadata_scalar(metadata, "upstream_commit"))
        if self.upstream_commit != INNO_UPSTREAM_COMMIT:
            raise RuntimeContractError("Inno tuner metadata has an unpinned upstream commit")
        self.checkpoint_sha256 = str(_metadata_scalar(metadata, "checkpoint_sha256"))
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_sha256
        ):
            raise RuntimeContractError(
                "Inno checkpoint provenance must be a lowercase SHA-256 digest"
            )
        self.names = name_values
        self.stats = np.ascontiguousarray(stats_table)
        self.grades = np.ascontiguousarray(grades)
        self.scale = np.ascontiguousarray(scales)
        self.gate = float(_metadata_scalar(metadata, "blend_gate"))
        self.grade_pen = float(_metadata_scalar(metadata, "grade_pen"))
        self.tilt_mean = float(_metadata_scalar(metadata, "tilt_mean"))
        self.tilt_sd = float(_metadata_scalar(metadata, "tilt_sd"))
        scalar_values = np.asarray(
            [self.gate, self.grade_pen, self.tilt_mean, self.tilt_sd], dtype=np.float64
        )
        if not np.isfinite(scalar_values).all() or self.gate < 0 or self.grade_pen < 0:
            raise RuntimeContractError("Inno tuner scalar metadata must be finite and non-negative")
        if self.tilt_sd <= 0:
            raise RuntimeContractError("Inno tilt standard deviation must be positive")

    def _verify_graph_provenance(self, session: Any) -> None:
        runtime_session = getattr(session, "session", None)
        get_modelmeta = getattr(runtime_session, "get_modelmeta", None)
        if get_modelmeta is None:
            return
        custom_metadata = get_modelmeta().custom_metadata_map
        expected = {
            "inno_upstream_version": self.version,
            "inno_upstream_commit": self.upstream_commit,
            "inno_checkpoint_sha256": self.checkpoint_sha256,
            "inno_metadata_sha256": self.metadata_sha256,
            "inno_blend_names": ",".join(self.names),
        }
        mismatches = [key for key, value in expected.items() if custom_metadata.get(key) != value]
        if mismatches:
            raise RuntimeContractError(
                "Inno ONNX graph provenance does not match its tuner metadata: "
                + ", ".join(mismatches)
            )

    def _get_session(self) -> Any:
        if self._session is None:
            if self.session_factory is not None:
                session = self.session_factory(self.model_path, INNO_GRAPH_COMPONENT)
            else:
                session = OnnxSession(
                    self.model_path,
                    component=INNO_GRAPH_COMPONENT,
                    providers=self.providers,
                    provider_options=self.provider_options,
                    session_options=self.session_options,
                )
            self._verify_graph_provenance(session)
            self._session = session
        return self._session

    @staticmethod
    def _input_array(value: np.ndarray, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(f"Inno graph input {name!r} must be numeric") from exc
        if array.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape, strict=True)
        ):
            raise RuntimeContractError(
                f"Inno graph input {name!r} has incompatible shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise RuntimeContractError(f"Inno graph input {name!r} must be finite")
        return np.ascontiguousarray(array)

    def infer_graph(
        self,
        fbank: np.ndarray,
        tilt_value: np.ndarray | float,
        head_stats_value: np.ndarray,
        weights: np.ndarray,
        *,
        head: bool = True,
    ) -> np.ndarray:
        """Run the learned voice-pack graph after validating all host-side tensors."""
        fbank_array = self._input_array(fbank, "fbank", (1, None, 80))
        if fbank_array.shape[1] < 1:
            raise RuntimeContractError("Inno fbank must contain at least one frame")
        tilt_array = self._input_array(np.asarray(tilt_value).reshape(-1), "tilt", (1,))
        head_array = self._input_array(head_stats_value, "head_stats", (1, 2))
        weight_array = self._input_array(weights, "blend_weights", (1, len(self.names)))
        enabled_array = np.asarray([1.0 if head else 0.0], dtype=np.float32)
        inputs = {
            "fbank": fbank_array,
            "tilt": tilt_array,
            "head_stats": head_array,
            "blend_weights": weight_array,
            "head_enabled": enabled_array,
        }

        session = self._get_session()
        input_names = set(getattr(session, "input_names", ()))
        missing = INNO_GRAPH_INPUTS - input_names
        unexpected = input_names - INNO_GRAPH_INPUTS
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing inputs: " + ", ".join(sorted(missing)))
            if unexpected:
                details.append("unexpected inputs: " + ", ".join(sorted(unexpected)))
            raise RuntimeContractError("Inno voicepack graph has " + "; ".join(details))
        outputs = session.run(inputs)
        output_names = tuple(getattr(session, "output_names", ()))
        if output_names and "voicepack" not in output_names:
            raise RuntimeContractError("Inno voicepack graph output name must include 'voicepack'")
        if isinstance(outputs, Mapping):
            output = outputs.get("voicepack")
        elif len(outputs) == 1 and (not output_names or output_names == ("voicepack",)):
            output = outputs[0]
        elif len(outputs) == len(output_names) and "voicepack" in output_names:
            output = outputs[output_names.index("voicepack")]
        else:
            output = None
        if output is None:
            raise RuntimeContractError("Inno voicepack graph returned no voicepack output")
        output_array = np.asarray(output, dtype=np.float32)
        if output_array.shape != (510, 1, 256):
            raise RuntimeContractError(
                f"Inno voicepack graph output must have shape [510, 1, 256], got {output_array.shape}"
            )
        if not np.isfinite(output_array).all():
            raise RuntimeContractError("Inno voicepack graph output must be finite")
        return np.ascontiguousarray(output_array)

    def enroll(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        options: Mapping[str, Any] | None = None,
    ) -> KokoroVoicePack:
        """Measure a reference and return a reusable stock Kokoro voice pack."""
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, (int, np.integer))
            or sample_rate <= 0
        ):
            raise RuntimeContractError("Inno reference sample_rate must be a positive integer")
        if options is None:
            enrollment_options: Mapping[str, Any] = {}
        elif isinstance(options, Mapping):
            enrollment_options = options
        else:
            raise RuntimeContractError("Inno enrollment options must be a mapping")
        unknown_options = set(enrollment_options) - {"fmax", "head"}
        if unknown_options:
            raise RuntimeContractError(
                "Unsupported Inno enrollment options: "
                + ", ".join(sorted(map(str, unknown_options)))
            )
        head = enrollment_options.get("head", True)
        if not isinstance(head, (bool, np.bool_)):
            raise RuntimeContractError("Inno enrollment option 'head' must be boolean")

        source = np.asarray(audio)
        if not np.issubdtype(source.dtype, np.floating):
            raise RuntimeContractError("Inno reference audio must be floating-point")
        if source.ndim == 1:
            mono = np.asarray(source, dtype=np.float32)
        elif source.ndim == 2 and source.shape[0] > 8 and source.shape[1] <= 8:
            mono = np.asarray(source, dtype=np.float32).mean(axis=1, dtype=np.float32)
        elif source.ndim == 2 and source.shape[0] <= 8 and source.shape[1] > 8:
            mono = np.asarray(source, dtype=np.float32).mean(axis=0, dtype=np.float32)
        else:
            raise RuntimeContractError(
                "Inno reference audio must be mono or a downmixable 2D waveform"
            )
        if mono.size == 0 or not np.isfinite(mono).all():
            raise RuntimeContractError("Inno reference audio must be non-empty and finite")

        analysis_audio = np.ascontiguousarray(mono[: 30 * int(sample_rate)], dtype=np.float32)
        if analysis_audio.size < 3 * int(sample_rate):
            raise RuntimeContractError("Inno reference audio must contain at least 3 seconds")
        duration_seconds = analysis_audio.size / int(sample_rate)
        if duration_seconds < 5.0:
            warnings.warn(
                "Inno recommends at least 5 seconds of reference audio",
                UserWarning,
                stacklevel=2,
            )
        rms = float(np.sqrt(np.mean(np.square(analysis_audio, dtype=np.float64))))
        if rms <= 1.0e-5:
            raise RuntimeContractError("Inno reference audio must have non-silent RMS")
        clipped_fraction = float(np.mean(np.abs(analysis_audio) >= 0.999))
        if clipped_fraction > 0.05:
            raise RuntimeContractError("Inno reference audio is grossly clipped")

        fmax = enrollment_options.get("fmax")
        if fmax is None:
            selected_fmax = ceiling(analysis_audio, int(sample_rate))
        else:
            if isinstance(fmax, (bool, np.bool_)):
                raise RuntimeContractError(
                    "Inno enrollment option 'fmax' must be positive and finite"
                )
            try:
                selected_fmax = float(fmax)
            except (TypeError, ValueError) as exc:
                raise RuntimeContractError(
                    "Inno enrollment option 'fmax' must be positive and finite"
                ) from exc
            if not math.isfinite(selected_fmax) or selected_fmax <= 0:
                raise RuntimeContractError(
                    "Inno enrollment option 'fmax' must be positive and finite"
                )

        audio_16k = resample_audio(analysis_audio, int(sample_rate), INNO_ENCODER_SAMPLE_RATE)
        audio_24k = resample_audio(analysis_audio, int(sample_rate), INNO_TILT_SAMPLE_RATE)
        f0_mean, f0_sd, voiced_fraction = stats(
            analysis_audio, int(sample_rate), fmax=selected_fmax
        )
        syllables_per_second = rate(analysis_audio, int(sample_rate))
        target = np.asarray([f0_mean, f0_sd, syllables_per_second], dtype=np.float64)
        sparse_weights = blend_weights(self, target)
        dense_weights = dense_blend_weights(self.names, sparse_weights)
        spectral_tilt = tilt(audio_24k, INNO_TILT_SAMPLE_RATE)
        normalized_tilt = (spectral_tilt - self.tilt_mean) / self.tilt_sd
        head_values = (
            np.asarray(
                [head_stats(analysis_audio, int(sample_rate), fmax=selected_fmax)],
                dtype=np.float32,
            )
            if head
            else np.zeros((1, 2), dtype=np.float32)
        )
        data = self.infer_graph(
            speaker_fbank(audio_16k),
            normalized_tilt,
            head_values,
            dense_weights,
            head=bool(head),
        )
        metadata = {
            "sample_rate": int(sample_rate),
            "encoder_sample_rate": INNO_ENCODER_SAMPLE_RATE,
            "tilt_sample_rate": INNO_TILT_SAMPLE_RATE,
            "prosody_sample_rate": int(sample_rate),
            "duration_seconds": float(duration_seconds),
            "truncated": bool(mono.size > analysis_audio.size),
            "f0_mean": float(f0_mean),
            "f0_sd": float(f0_sd),
            "voiced_fraction": float(voiced_fraction),
            "syllables_per_second": float(syllables_per_second),
            "spectral_tilt": float(spectral_tilt),
            "fmax": float(selected_fmax),
            "head_stats": [float(value) for value in head_values.reshape(-1)] if head else None,
            "blend_weights": sparse_weights,
            "upstream_version": self.version,
            "upstream_commit": self.upstream_commit,
        }
        return KokoroVoicePack(
            data=data,
            enroller=INNO_ENROLLER_ID,
            model_fingerprint=self.model_fingerprint,
            enroller_fingerprint=self.enroller_fingerprint,
            metadata=metadata,
        )

    def diagnostics(self) -> RuntimeDiagnostic:
        sessions = () if self._session is None else (self._session.diagnostics(),)
        runtime = self.installation.metadata.get("runtime")
        layout = (
            runtime.get("layout", "single-onnx-v1")
            if isinstance(runtime, Mapping)
            else "single-onnx-v1"
        )
        return RuntimeDiagnostic(
            system="kokoro",
            ref=self.installation.ref,
            layout=str(layout),
            sessions=sessions,
        )

    def close(self) -> None:
        session = self._session
        self._session = None
        close = getattr(session, "close", None)
        if close is not None:
            close()
