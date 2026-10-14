from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError


def _chunk_attention_bias(length: int, chunk_frames: int) -> np.ndarray:
    if type(length) is not int or length < 1:
        raise RuntimeContractError("Sopro attention length must be a positive integer")
    if type(chunk_frames) is not int or chunk_frames < 1:
        raise RuntimeContractError("Sopro attention chunk size must be a positive integer")
    bias = np.zeros((1, 1, length, length), dtype=np.float32)
    for query in range(length):
        end = min(length, ((query // chunk_frames) + 1) * chunk_frames)
        bias[0, 0, query, end:] = -1.0e4
    return bias


def _select_stream_graph_mode(graphs: Sequence[str], steps: int) -> tuple[str, tuple[str, ...]]:
    if type(steps) is not int or steps <= 0:
        raise RuntimeContractError("Sopro streaming steps must be a positive integer")
    available = set(graphs)
    prefill = f"acoustic_stream_prefill_{steps}"
    step = f"acoustic_stream_step_{steps}"
    if prefill not in available or step not in available:
        raise CapabilityError(
            "Sopro bundle does not contain the required streaming prefill/step graphs"
        )
    split = [prefill]
    split.extend(f"acoustic_stream_prefill_ode_{steps}_{index}" for index in range(steps))
    split.append(step)
    split.extend(f"acoustic_stream_ode_{steps}_{index}" for index in range(steps))
    if all(name in available for name in split):
        return "split", tuple(split)
    return "monolithic", (prefill, step)


def _stable_prompt_frames(
    reference_mel_frames: int,
    reference_semantic_tokens: int,
    pre_lookahead_frames: int,
    hop_ratio: int,
    chunk_frames: int,
) -> int:
    values = (
        reference_mel_frames,
        reference_semantic_tokens,
        pre_lookahead_frames,
        hop_ratio,
        chunk_frames,
    )
    if any(type(value) is not int for value in values):
        raise RuntimeContractError("Sopro prompt frame dimensions must be integers")
    if reference_mel_frames < 0 or reference_semantic_tokens < 0 or pre_lookahead_frames < 0:
        raise RuntimeContractError("Sopro prompt frame dimensions must be non-negative")
    if hop_ratio < 1 or chunk_frames < 1:
        raise RuntimeContractError("Sopro prompt hop/chunk dimensions must be positive")
    usable = min(
        reference_mel_frames,
        max(0, reference_semantic_tokens - pre_lookahead_frames) * hop_ratio,
    )
    return (usable // chunk_frames) * chunk_frames


def _stream_target_frames(
    reference_mel_frames: int,
    generated_tokens: int,
    hop_ratio: int,
    pre_lookahead_frames: int,
    chunk_frames: int,
    *,
    final: bool,
) -> int:
    values = (reference_mel_frames, generated_tokens, hop_ratio, pre_lookahead_frames, chunk_frames)
    if any(type(value) is not int for value in values) or type(final) is not bool:
        raise RuntimeContractError("Sopro stream frame dimensions must be integers")
    if reference_mel_frames < 0 or generated_tokens < 0 or pre_lookahead_frames < 0:
        raise RuntimeContractError("Sopro stream frame dimensions must be non-negative")
    if hop_ratio < 1 or chunk_frames < 1:
        raise RuntimeContractError("Sopro stream hop/chunk dimensions must be positive")
    total = reference_mel_frames + generated_tokens * hop_ratio
    if final:
        return total
    stable = max(0, total - pre_lookahead_frames * hop_ratio)
    return (stable // chunk_frames) * chunk_frames


@dataclass(frozen=True, slots=True)
class SoproStreamChunk:
    audio: np.ndarray
    sample_rate: int
    is_final: bool
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.flags.c_contiguous or not np.isfinite(audio).all():
            raise RuntimeContractError("Sopro stream audio must be finite contiguous mono float32")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise RuntimeContractError("Sopro stream sample rate must be a positive integer")
        if type(self.is_final) is not bool or not isinstance(self.metadata, Mapping):
            raise RuntimeContractError("Sopro stream chunk metadata/finality is invalid")
        object.__setattr__(self, "audio", audio)


class _SoproNoise:
    """Extendable channel-major Gaussian noise sequence for one stream segment."""

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise RuntimeContractError("Sopro stream noise seed must be a uint32 integer")
        self._state = (seed ^ 0x71D3A4C9) & 0xFFFFFFFF or 1
        self._spare: float | None = None
        self._frames = 0
        self._values = np.empty((100, 0), dtype=np.float32)

    def _uniform(self) -> float:
        value = self._state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self._state = value & 0xFFFFFFFF
        return (self._state + 0.5) / 4294967296.0

    def _normal(self) -> float:
        if self._spare is not None:
            value = self._spare
            self._spare = None
            return value
        radius = math.sqrt(-2.0 * math.log(max(1.0e-12, self._uniform())))
        angle = 2.0 * math.pi * self._uniform()
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)

    def ensure(self, frame_count: int) -> None:
        if type(frame_count) is not int or frame_count < self._frames:
            raise RuntimeContractError("Sopro stream noise frame count must not decrease")
        added = frame_count - self._frames
        if added == 0:
            return
        suffix = np.fromiter(
            (self._normal() for _ in range(added * 100)),
            dtype=np.float32,
            count=added * 100,
        ).reshape(added, 100)
        self._values = np.ascontiguousarray(
            np.concatenate((self._values, suffix.T), axis=1), dtype=np.float32
        )
        self._frames = frame_count

    def slice(self, start: int, end: int) -> np.ndarray:
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > self._frames
        ):
            raise RuntimeContractError("Sopro stream noise slice is outside the generated range")
        return np.ascontiguousarray(self._values[None, :, start:end], dtype=np.float32)


class _StreamingISTFT:
    """Bounded overlap-add ISTFT whose concatenated output matches offline Sopro ISTFT."""

    def __init__(self, n_fft: int, hop: int) -> None:
        if type(n_fft) is not int or n_fft <= 0 or n_fft % 2 or n_fft & (n_fft - 1):
            raise RuntimeContractError(
                "Sopro streaming ISTFT FFT size must be an even power of two"
            )
        if type(hop) is not int or hop <= 0 or hop > n_fft:
            raise RuntimeContractError("Sopro streaming ISTFT hop must be in [1, n_fft]")
        self.n_fft = n_fft
        self.hop = hop
        self._pad = n_fft // 2
        self._frame_count = 0
        self._buffer_start = 0
        self._next_output = self._pad
        self._overlap = np.zeros(0, dtype=np.float64)
        self._envelope = np.zeros(0, dtype=np.float64)
        self._window = 0.5 - 0.5 * np.cos(
            2.0 * math.pi * np.arange(n_fft, dtype=np.float64) / n_fft
        )
        self._flushed = False

    def _ensure_capacity(self, end: int) -> None:
        required = end - self._buffer_start
        if required > self._overlap.size:
            growth = required - self._overlap.size
            self._overlap = np.pad(self._overlap, (0, growth))
            self._envelope = np.pad(self._envelope, (0, growth))

    def push(
        self,
        features: np.ndarray,
        frames: int,
        *,
        flush: bool,
    ) -> np.ndarray:
        if self._flushed:
            raise RuntimeContractError("Sopro streaming ISTFT was already flushed")
        if type(frames) is not int or frames < 0 or type(flush) is not bool:
            raise RuntimeContractError("Sopro streaming ISTFT frame/flush arguments are invalid")
        spectrum = np.asarray(features)
        if (
            not np.issubdtype(spectrum.dtype, np.floating)
            or spectrum.size != frames * (self.n_fft + 2)
            or not np.isfinite(spectrum).all()
        ):
            raise RuntimeContractError(
                "Sopro streaming ISTFT features have invalid shape or values"
            )
        spectrum = spectrum.reshape(frames, self.n_fft + 2)
        bins = self.n_fft // 2 + 1
        log_maximum = math.log(100.0)
        for frame_index in range(frames):
            row = spectrum[frame_index]
            magnitude = np.exp(np.minimum(log_maximum, row[:bins].astype(np.float64)))
            phase = row[bins:].astype(np.float64)
            half = magnitude * np.exp(1j * phase)
            full = np.empty(self.n_fft, dtype=np.complex128)
            full[:bins] = half
            full[bins:] = np.conjugate(half[1:-1][::-1])
            waveform = np.fft.ifft(full).real
            global_frame = self._frame_count
            start = global_frame * self.hop
            end = start + self.n_fft
            self._ensure_capacity(end)
            offset = start - self._buffer_start
            self._overlap[offset : offset + self.n_fft] += waveform * self._window
            self._envelope[offset : offset + self.n_fft] += self._window * self._window
            self._frame_count += 1

        if self._frame_count == 0:
            return np.empty(0, dtype=np.float32)
        if flush:
            safe_end = (self._frame_count - 1) * self.hop + self._pad
            self._flushed = True
        else:
            safe_end = (self._frame_count - 1) * self.hop
        output_start = max(self._next_output, self._buffer_start)
        output_end = max(output_start, min(safe_end, self._buffer_start + self._overlap.size))
        left = output_start - self._buffer_start
        right = output_end - self._buffer_start
        if right <= left:
            output = np.empty(0, dtype=np.float32)
        else:
            values = self._overlap[left:right] / np.maximum(1.0e-8, self._envelope[left:right])
            output = np.ascontiguousarray(values, dtype=np.float32)
        self._next_output = output_end

        trim = min(max(0, safe_end - self._buffer_start), self._overlap.size)
        if trim:
            self._overlap = self._overlap[trim:].copy()
            self._envelope = self._envelope[trim:].copy()
            self._buffer_start += trim
        return output


@dataclass(slots=True)
class _SoproVocoderState:
    embed_state: np.ndarray
    conv_state: np.ndarray
    pending: np.ndarray


@dataclass(slots=True)
class _SoproAcousticState:
    cached_frames: int
    x_context: np.ndarray
    past_k: np.ndarray
    past_v: np.ndarray


@dataclass(frozen=True, slots=True)
class _SoproPromptState:
    cached_frames: int
    x_context: np.ndarray
    past_k: np.ndarray
    past_v: np.ndarray
    split_ode: bool
    mu: np.ndarray
    mu_frames: int


@dataclass(frozen=True, slots=True)
class _SoproVocoderPromptState:
    warm_frames: int
    istft_features: np.ndarray
    embed_state: np.ndarray
    conv_state: np.ndarray
    pending: np.ndarray
