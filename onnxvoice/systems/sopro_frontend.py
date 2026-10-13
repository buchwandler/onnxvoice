from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import CapabilityError, ManifestError, RuntimeContractError

OFFICIAL_RUNTIME_COMMIT = "7bfcf9a0539d274f6593959a21da948f506ceee1"
_TOKENIZER_FORMAT = "sentencepiece-unigram-v1"
_LANGUAGE_TAGS = {
    "en": "<|lang_en|>",
    "pt": "<|lang_pt|>",
    "fr": "<|lang_fr|>",
    "de": "<|lang_de|>",
}
_TERMINALS = frozenset(".!?-,;:")
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, slots=True)
class SoproTextInput:
    """Normalized text and exact model token IDs expected by Sopro graphs."""

    normalized_text: str
    token_ids: tuple[int, ...]
    language: str | None


@dataclass(frozen=True, slots=True)
class SoproVoiceState:
    """Reusable reference-audio state emitted by the Sopro reference graph."""

    cond_vec: np.ndarray
    semantic_tokens: np.ndarray
    mel: np.ndarray
    sample_rate: int
    level_db: float

    def __post_init__(self) -> None:
        condition = _readonly_array(self.cond_vec, np.float32, ndim=1, label="condition vector")
        raw_tokens = np.asarray(self.semantic_tokens)
        if not np.issubdtype(raw_tokens.dtype, np.integer):
            raise RuntimeContractError("Sopro semantic tokens must use an integer dtype")
        tokens = _readonly_array(raw_tokens, np.int64, ndim=1, label="semantic tokens")
        mel = _readonly_array(self.mel, np.float32, ndim=2, label="reference mel")
        if condition.size == 0 or not np.isfinite(condition).all():
            raise RuntimeContractError("Sopro condition vector must be non-empty and finite")
        if tokens.size == 0 or np.any(tokens < 0):
            raise RuntimeContractError(
                "Sopro semantic tokens must be non-empty non-negative integers"
            )
        if mel.shape[0] != 100 or mel.shape[1] == 0 or not np.isfinite(mel).all():
            raise RuntimeContractError(
                "Sopro reference mel must be finite with shape [100, frames]"
            )
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise RuntimeContractError("Sopro voice sample rate must be a positive integer")
        if not math.isfinite(float(self.level_db)):
            raise RuntimeContractError("Sopro reference level must be finite")
        object.__setattr__(self, "cond_vec", condition)
        object.__setattr__(self, "semantic_tokens", tokens)
        object.__setattr__(self, "mel", mel)
        object.__setattr__(self, "level_db", float(self.level_db))

    @property
    def mel_frames(self) -> int:
        return int(self.mel.shape[1])


def _readonly_array(value: Any, dtype: Any, *, ndim: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise RuntimeContractError(f"Sopro {label} must have rank {ndim}, got {array.ndim}")
    array = np.ascontiguousarray(array).copy()
    array.flags.writeable = False
    return array


def _effective_language(language: Any) -> tuple[str | None, str | None]:
    code = "" if language is None else str(language).strip().lower()
    if not code or code == "auto":
        return None, None
    try:
        return code, _LANGUAGE_TAGS[code]
    except KeyError as exc:
        raise CapabilityError(
            f"Unsupported Sopro language {language!r}; choices: auto, en, pt, fr, de"
        ) from exc


def normalize_sopro_text(value: Any, language: Any = None) -> tuple[str, str | None]:
    """Match the official browser tokenizer's text and optional-language handling."""
    effective_language, tag = _effective_language(language)
    text = "" if value is None else str(value)
    text = text.strip()
    if not text and tag:
        return tag, effective_language
    if not text:
        text = "You need to add some text for me to talk."
    first = text[0]
    if first.islower() and first.upper() != first:
        text = first.upper() + text[1:]
    text = text.replace("…", "...").replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text[-1] not in _TERMINALS:
        text += "."
    if tag:
        text = f"{tag} {text}"
    return text, effective_language


@dataclass(slots=True)
class _TrieNode:
    next: dict[str, _TrieNode]
    token_id: int = -1
    score: float = -math.inf

    @classmethod
    def create(cls) -> _TrieNode:
        return cls(next={})


class SoproTokenizer:
    """Pure-Python port of the pinned SentencePiece unigram browser tokenizer."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        if spec.get("format") != _TOKENIZER_FORMAT:
            raise ManifestError(f"Unsupported Sopro tokenizer format: {spec.get('format')!r}")
        pieces = spec.get("pieces")
        scores = spec.get("scores")
        types = spec.get("types")
        if (
            not isinstance(pieces, Sequence)
            or isinstance(pieces, (str, bytes))
            or not pieces
            or not isinstance(scores, Sequence)
            or isinstance(scores, (str, bytes))
            or not isinstance(types, Sequence)
            or isinstance(types, (str, bytes))
            or len(pieces) != len(scores)
            or len(pieces) != len(types)
        ):
            raise ManifestError(
                "Sopro tokenizer pieces, scores, and types must be equally sized arrays"
            )
        self.bos_id = self._token_id(spec.get("bosId"), "bosId", len(pieces))
        self.eos_id = self._token_id(spec.get("eosId"), "eosId", len(pieces))
        self.unk_id = self._token_id(spec.get("unkId"), "unkId", len(pieces))
        max_length = spec.get("maxLength")
        if type(max_length) is not int or max_length <= 0:
            raise ManifestError("Sopro tokenizer maxLength must be a positive integer")
        self.max_length = max_length
        self.root = _TrieNode.create()
        self.byte_ids: dict[int, int] = {}
        for token_id, (piece, score, piece_type) in enumerate(
            zip(pieces, scores, types, strict=True)
        ):
            if not isinstance(piece, str) or not isinstance(piece_type, str):
                raise ManifestError("Sopro tokenizer pieces and types must contain strings")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ManifestError("Sopro tokenizer scores must be finite numbers")
            if piece_type == "byte":
                match = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
                if match is None:
                    raise ManifestError(f"Malformed Sopro byte piece {piece!r}")
                byte = int(match.group(1), 16)
                if byte in self.byte_ids:
                    raise ManifestError(f"Duplicate Sopro byte token for 0x{byte:02X}")
                self.byte_ids[byte] = token_id
            elif piece_type == "normal":
                node = self.root
                for char in piece:
                    node = node.next.setdefault(char, _TrieNode.create())
                node.token_id = token_id
                node.score = float(score)
            elif piece_type not in {"control", "unused", "unknown"}:
                raise ManifestError(f"Unsupported Sopro tokenizer piece type {piece_type!r}")

    @staticmethod
    def _token_id(value: Any, name: str, vocab_size: int) -> int:
        if type(value) is not int or not 0 <= value < vocab_size:
            raise ManifestError(f"Sopro tokenizer {name} must be a valid vocabulary index")
        return value

    def encode(self, value: Any, language: Any = None) -> SoproTextInput:
        normalized, effective_language = normalize_sopro_text(value, language)
        normalized = unicodedata.normalize("NFKC", normalized)
        chars = list("▁" + re.sub(r"\s+", "▁", normalized))
        count = len(chars)
        best = [-math.inf] * (count + 1)
        back: list[tuple[int, tuple[int, ...]] | None] = [None] * (count + 1)
        best[0] = 0.0
        for start in range(count):
            if not math.isfinite(best[start]):
                continue
            node: _TrieNode | None = self.root
            matched = False
            for end in range(start, count):
                if node is None:
                    break
                node = node.next.get(chars[end])
                if node is None:
                    break
                if node.token_id >= 0:
                    matched = True
                    score = best[start] + node.score
                    if score > best[end + 1]:
                        best[end + 1] = score
                        back[end + 1] = (start, (node.token_id,))
            if not matched or not math.isfinite(best[start + 1]):
                byte_tokens = tuple(
                    self.byte_ids.get(byte, self.unk_id) for byte in chars[start].encode("utf-8")
                )
                score = best[start] - 100.0 - len(byte_tokens)
                if score > best[start + 1]:
                    best[start + 1] = score
                    back[start + 1] = (start, byte_tokens)
        tokens: list[int] = []
        cursor = count
        while cursor > 0:
            item = back[cursor]
            if item is None:
                tokens.insert(0, self.unk_id)
                cursor -= 1
            else:
                start, token_ids = item
                tokens[0:0] = token_ids
                cursor = start
        output = tuple(([self.bos_id, *tokens, self.eos_id])[: self.max_length])
        return SoproTextInput(
            normalized_text=normalized, token_ids=output, language=effective_language
        )


@dataclass(frozen=True, slots=True)
class ReferenceFeatures:
    semantic_mel: np.ndarray
    speaker_mel: np.ndarray
    acoustic_mel: np.ndarray
    interp_left: np.ndarray
    interp_right: np.ndarray
    interp_weight: np.ndarray
    token_count: int


def _positive_int(section: Mapping[str, Any], key: str, label: str) -> int:
    value = section.get(key)
    if type(value) is not int or value <= 0:
        raise ManifestError(f"Sopro reference config {label} must be a positive integer")
    return value


def _quantile(values: np.ndarray, quantile: float, empty: float = 1.0e-6) -> float:
    if values.size == 0:
        return empty
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = quantile * (ordered.size - 1)
    low = int(math.floor(position))
    weight = position - low
    if low + 1 < ordered.size:
        return float(ordered[low] * (1.0 - weight) + ordered[low + 1] * weight)
    return float(ordered[low])


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0e-6
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[(ordered.size - 1) // 2])


def resample_sopro(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Port the official radius-16 windowed-sinc resampler."""
    source = np.ascontiguousarray(waveform, dtype=np.float32)
    if source_rate == target_rate:
        return source.copy()
    length = max(1, math.floor(source.size * target_rate / source_rate + 0.5))
    output = np.zeros(length, dtype=np.float32)
    radius = 16
    ratio = source_rate / target_rate
    taps = np.arange(-radius + 1, radius + 1, dtype=np.int64)
    for start in range(0, length, 8192):
        stop = min(length, start + 8192)
        indices = np.arange(start, stop, dtype=np.float64)
        centers = (indices + 0.5) * ratio - 0.5
        bases = np.floor(centers).astype(np.int64)
        source_indices = bases[:, None] + taps[None, :]
        in_bounds = (source_indices >= 0) & (source_indices < source.size)
        safe_indices = np.clip(source_indices, 0, max(0, source.size - 1))
        distance = centers[:, None] - source_indices
        sinc = np.sinc(distance)
        window = 0.5 + 0.5 * np.cos(math.pi * distance / radius)
        weights = np.where(in_bounds, sinc * window, 0.0)
        samples = source[safe_indices].astype(np.float64)
        totals = np.sum(samples * weights, axis=1)
        weight_sums = np.sum(weights, axis=1)
        output[start:stop] = np.divide(
            totals,
            weight_sums,
            out=np.zeros_like(totals),
            where=weight_sums != 0.0,
        ).astype(np.float32)
    return output


def _frame_rms(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    window = math.floor(sample_rate * 0.025)
    hop = math.floor(sample_rate * 0.010)
    values = []
    for start in range(0, max(0, waveform.size - window + 1), hop):
        frame = waveform[start : start + window].astype(np.float64)
        values.append(max(1.0e-6, math.sqrt(float(np.dot(frame, frame)) / window)))
    return np.asarray(values, dtype=np.float64), window, hop


def _pause_runs(quiet: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < quiet.size:
        if not quiet[index]:
            index += 1
            continue
        end = index
        while end < quiet.size and quiet[end]:
            end += 1
        if end - index >= min_run:
            runs.append((index, end))
        index = end
    return runs


class _RoomToneRandom:
    def __init__(self) -> None:
        self.state = 1
        self.spare: float | None = None

    def value(self) -> float:
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return (self.state + 0.5) / 4294967296.0

    def normal(self) -> float:
        if self.spare is not None:
            value = self.spare
            self.spare = None
            return value
        radius = math.sqrt(-2.0 * math.log(max(1.0e-12, self.value())))
        angle = 2.0 * math.pi * self.value()
        self.spare = radius * math.sin(angle)
        return radius * math.cos(angle)


def _finish_with_room_tone(waveform: np.ndarray, sample_rate: int, floor: float) -> np.ndarray:
    fade = math.floor(0.02 * sample_rate)
    room = np.zeros(waveform.size + math.floor(0.25 * sample_rate), dtype=np.float32)
    room[: waveform.size] = waveform
    for index in range(fade):
        room[waveform.size - fade + index] *= 1.0 - index / (fade - 1)
    random = _RoomToneRandom()
    for index in range(waveform.size, room.size):
        room[index] = random.normal() * floor
    return room


def crop_on_pause(waveform: np.ndarray, seconds: float, sample_rate: int) -> np.ndarray:
    """Match the browser's reference crop and deterministic room-tone policy."""
    window = math.floor(sample_rate * 0.025)
    hop = math.floor(sample_rate * 0.010)
    total = waveform.size
    if total < 4 * window:
        return waveform
    rms, _, _ = _frame_rms(waveform, sample_rate)
    floor = _quantile(rms, 0.1)
    quiet = rms < floor * 4.0
    min_run = max(1, math.floor(0.10 / 0.010 + 0.5))
    runs = _pause_runs(quiet, min_run)
    keep = math.floor(0.15 / 0.010 + 0.5)
    target = math.floor(seconds * sample_rate + 0.5)

    def cut_at(run: tuple[int, int]) -> np.ndarray:
        start, end = run
        return waveform[: min(total, (start + min(end - start, keep)) * hop + window)]

    if total <= target:
        if runs and runs[-1][1] * hop + window >= total - hop:
            return waveform
        inside = [run for run in runs if run[0] * hop >= total * 0.75]
        return (
            cut_at(inside[-1]) if inside else _finish_with_room_tone(waveform, sample_rate, floor)
        )
    forward = [run for run in runs if target <= run[0] * hop <= target + 5.0 * sample_rate]
    if forward:
        return cut_at(forward[0])
    backward = [
        run for run in runs if run[0] * hop < target and run[1] * hop >= target - 5.0 * sample_rate
    ]
    if backward:
        return cut_at(backward[-1])
    return _finish_with_room_tone(waveform[:target], sample_rate, floor)


def _speech_level_db(waveform: np.ndarray, sample_rate: int) -> tuple[float, float]:
    window = math.floor(sample_rate * 0.025)
    hop = math.floor(sample_rate * 0.010)
    if waveform.size < window:
        rms = math.sqrt(
            float(np.dot(waveform.astype(np.float64), waveform.astype(np.float64)))
            / max(1, waveform.size)
        )
        return 20.0 * math.log10(max(rms, 1.0e-6)), 0.0
    values, _, _ = _frame_rms(waveform, sample_rate)
    threshold = _quantile(values, 0.2) * 1.5
    active = values[values > threshold]
    kept = active if active.size else values
    return 20.0 * math.log10(_median(kept)), kept.size * hop / sample_rate


def prepare_reference_audio(
    waveform: np.ndarray, sample_rate: int, *, seconds: float
) -> tuple[np.ndarray, float]:
    """Resample, crop, and normalize one mono reference using official web DSP."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeContractError("Sopro reference duration must be finite and positive")
    reference = crop_on_pause(waveform, seconds, sample_rate)
    level_db, _ = _speech_level_db(reference, sample_rate)
    gain_db = min(max(-19.8 - level_db, 0.0), 30.0)
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    if peak > 0.0:
        gain_db = min(gain_db, max(0.0, 20.0 * math.log10(0.95 / peak)))
    gain = 10.0 ** (gain_db / 20.0)
    return np.ascontiguousarray(reference * gain, dtype=np.float32), level_db + gain_db


def _hz_to_mel(hz: np.ndarray, scale: str) -> np.ndarray:
    if scale == "htk":
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    linear = 3.0 * hz / 200.0
    logarithmic = 15.0 + np.log(np.maximum(hz, 1000.0) / 1000.0) / (math.log(6.4) / 27.0)
    return np.where(hz < 1000.0, linear, logarithmic)


def _mel_to_hz(mel: np.ndarray, scale: str) -> np.ndarray:
    if scale == "htk":
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    return np.where(
        mel < 15.0, 200.0 * mel / 3.0, 1000.0 * np.exp((math.log(6.4) / 27.0) * (mel - 15.0))
    )


def _mel_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    scale: str,
    norm: str | None,
) -> np.ndarray:
    bins = n_fft // 2 + 1
    limits = _hz_to_mel(np.asarray([f_min, f_max], dtype=np.float64), scale)
    points = _mel_to_hz(np.linspace(limits[0], limits[1], n_mels + 2, dtype=np.float64), scale)
    frequencies = sample_rate * np.arange(bins, dtype=np.float64) / n_fft
    bank = np.empty((n_mels, bins), dtype=np.float32)
    for mel_index in range(n_mels):
        left, center, right = points[mel_index : mel_index + 3]
        area = 2.0 / (right - left) if norm == "slaney" else 1.0
        triangle = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left), (right - frequencies) / (right - center)
            ),
        )
        bank[mel_index] = (area * triangle).astype(np.float32)
    return bank


def _reflect_indices(indices: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros(indices.shape, dtype=np.int64)
    period = 2 * length - 2
    folded = np.mod(indices, period)
    return np.where(folded < length, folded, period - folded).astype(np.int64)


def _mel_spectrogram(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    power: int,
    scale: str,
    norm: str | None,
    frames: int | None = None,
) -> np.ndarray:
    pad = n_fft // 2
    frame_count = frames if frames is not None else math.floor(waveform.size / hop_length) + 1
    window = np.zeros(n_fft, dtype=np.float64)
    window_start = math.floor((n_fft - win_length) / 2)
    window_indices = np.arange(win_length, dtype=np.float64)
    window[window_start : window_start + win_length] = 0.5 - 0.5 * np.cos(
        _TWO_PI * window_indices / win_length
    )
    mel_filter = _mel_bank(sample_rate, n_fft, n_mels, f_min, f_max, scale, norm).astype(np.float64)
    output = np.empty((n_mels, frame_count), dtype=np.float32)
    frame_offsets = np.arange(n_fft, dtype=np.int64)
    for frame_index in range(frame_count):
        base = frame_index * hop_length - pad
        indices = _reflect_indices(base + frame_offsets, waveform.size)
        frame = waveform[indices].astype(np.float64) * window
        spectrum = np.fft.rfft(frame)
        magnitude_squared = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        spectrum_values = np.sqrt(magnitude_squared) if power == 1 else magnitude_squared
        output[:, frame_index] = mel_filter @ spectrum_values
    return output


def reference_features(
    waveform_24k: np.ndarray,
    config: Mapping[str, Any],
    *,
    token_samples: int,
) -> ReferenceFeatures:
    """Reproduce the pinned web runtime's three reference mel inputs and alignment."""
    semantic_cfg = config.get("semantic_encoder")
    speaker_cfg = config.get("speaker_encoder")
    model_cfg = config.get("model")
    vocoder_cfg = config.get("vocoder")
    if (
        not isinstance(semantic_cfg, Mapping)
        or not isinstance(speaker_cfg, Mapping)
        or not isinstance(model_cfg, Mapping)
        or not isinstance(vocoder_cfg, Mapping)
    ):
        raise ManifestError("Sopro manifest config is missing reference encoder sections")
    semantic_rate = _positive_int(semantic_cfg, "sample_rate", "semantic_encoder.sample_rate")
    semantic_hop = _positive_int(semantic_cfg, "hop_length", "semantic_encoder.hop_length")
    semantic_fft = _positive_int(semantic_cfg, "n_fft", "semantic_encoder.n_fft")
    semantic_mels = _positive_int(semantic_cfg, "n_mels", "semantic_encoder.n_mels")
    speaker_rate = _positive_int(speaker_cfg, "sample_rate", "speaker_encoder.sample_rate")
    speaker_fft = _positive_int(speaker_cfg, "n_fft", "speaker_encoder.n_fft")
    speaker_win = _positive_int(speaker_cfg, "win_length", "speaker_encoder.win_length")
    speaker_hop = _positive_int(speaker_cfg, "hop_length", "speaker_encoder.hop_length")
    speaker_mels = _positive_int(speaker_cfg, "n_mels", "speaker_encoder.n_mels")
    acoustic_hop = _positive_int(
        model_cfg, "acoustic_mel_hop_length", "model.acoustic_mel_hop_length"
    )
    acoustic_fft = _positive_int(vocoder_cfg, "n_fft", "vocoder.n_fft")
    acoustic_mels = _positive_int(model_cfg, "acoustic_mel_n_mels", "model.acoustic_mel_n_mels")
    if type(config.get("sample_rate")) is not int or config["sample_rate"] != 24000:
        raise CapabilityError(
            "Sopro reference preprocessing currently requires a 24-kHz model waveform"
        )
    if semantic_rate != 16000 or speaker_rate != 16000:
        raise CapabilityError("Sopro reference encoders require a shared 16-kHz waveform")
    if type(token_samples) is not int or token_samples <= 0:
        raise ManifestError("Sopro manifest tokenSamples must be positive")

    waveform_16k = resample_sopro(waveform_24k, 24000, semantic_rate)
    semantic_frames = math.ceil(waveform_16k.size / semantic_hop)
    semantic_audio = np.pad(waveform_16k, (0, semantic_fft))
    semantic = _mel_spectrogram(
        semantic_audio,
        sample_rate=semantic_rate,
        n_fft=semantic_fft,
        win_length=semantic_fft,
        hop_length=semantic_hop,
        n_mels=semantic_mels,
        f_min=0.0,
        f_max=semantic_rate / 2.0,
        power=2,
        scale="slaney",
        norm="slaney",
        frames=semantic_frames + 2,
    )
    semantic_log = np.log10(np.maximum(1.0e-10, semantic)).astype(np.float32)
    maximum = float(np.max(semantic_log))
    semantic = ((np.maximum(semantic_log, maximum - 8.0) + 4.0) / 4.0).astype(np.float32)

    token_count = math.ceil(waveform_24k.size / token_samples)
    n50 = math.ceil(semantic_frames / 2)
    interp_left = np.empty(token_count, dtype=np.int64)
    interp_right = np.empty(token_count, dtype=np.int64)
    interp_weight = np.empty(token_count, dtype=np.float32)
    for index in range(token_count):
        source = max(0.0, min(float(n50 - 1), (index + 0.5) * n50 / token_count - 0.5))
        left = math.floor(source)
        interp_left[index] = left
        interp_right[index] = min(n50 - 1, left + 1)
        interp_weight[index] = source - left

    speaker = _mel_spectrogram(
        waveform_16k,
        sample_rate=speaker_rate,
        n_fft=speaker_fft,
        win_length=speaker_win,
        hop_length=speaker_hop,
        n_mels=speaker_mels,
        f_min=float(speaker_cfg.get("f_min", 0.0)),
        f_max=float(speaker_cfg.get("f_max", speaker_rate / 2.0)),
        power=2,
        scale="slaney",
        norm="slaney",
    )
    log_floor = float(speaker_cfg.get("mel_log_floor", 1.0e-5))
    speaker = np.log(np.maximum(log_floor, speaker)).astype(np.float32)
    for frame_index in range(speaker.shape[1]):
        column = speaker[:, frame_index]
        mean = float(np.sum(column, dtype=np.float64)) / speaker_mels
        variance = (
            float(np.sum((column.astype(np.float64) - mean) ** 2, dtype=np.float64)) / speaker_mels
        )
        speaker[:, frame_index] = ((column - mean) / math.sqrt(variance + 1.0e-5)).astype(
            np.float32
        )

    acoustic_rate = 24000
    acoustic = _mel_spectrogram(
        waveform_24k,
        sample_rate=acoustic_rate,
        n_fft=acoustic_fft,
        win_length=acoustic_fft,
        hop_length=acoustic_hop,
        n_mels=acoustic_mels,
        f_min=0.0,
        f_max=acoustic_rate / 2.0,
        power=1,
        scale="htk",
        norm=None,
    )
    acoustic = np.log(np.maximum(1.0e-7, acoustic)).astype(np.float32)
    return ReferenceFeatures(
        semantic_mel=np.ascontiguousarray(semantic),
        speaker_mel=np.ascontiguousarray(speaker),
        acoustic_mel=np.ascontiguousarray(acoustic),
        interp_left=np.ascontiguousarray(interp_left),
        interp_right=np.ascontiguousarray(interp_right),
        interp_weight=np.ascontiguousarray(interp_weight),
        token_count=token_count,
    )
