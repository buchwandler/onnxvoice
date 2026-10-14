from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from ..errors import (
    CapabilityError,
    IntegrityError,
    ManifestError,
    RuntimeContractError,
    UnsafePathError,
)
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, RuntimeDiagnostic, validate_safe_component
from .base import SystemAdapter
from .sopro_frontend import (
    SoproTextInput,
    SoproTokenizer,
    SoproVoiceState,
    prepare_reference_audio,
    reference_features,
    resample_sopro,
)
from .sopro_generation import SoproGenerationDefaults
from .sopro_streaming import (
    SoproStreamChunk,
    _chunk_attention_bias,
    _select_stream_graph_mode,
    _SoproAcousticState,
    _SoproNoise,
    _SoproPromptState,
    _SoproVocoderPromptState,
    _SoproVocoderState,
    _stable_prompt_frames,
    _stream_target_frames,
    _StreamingISTFT,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class _SemanticOptions:
    temperature: float
    top_p: float
    top_k: int
    max_seconds: float
    min_seconds: float
    seed: int
    max_steps: int
    min_steps: int


@dataclass(frozen=True, slots=True)
class _SoproRuntimeConfig:
    sample_rate: int
    token_samples: int
    hop_ratio: int
    fixed_steps: int
    position_context: int | None
    stream_chunk_frames: int
    acoustic_pre_lookahead_frames: int | None
    acoustic_dim_head: int | None
    vocoder_n_fft: int | None
    vocoder_hop_length: int | None
    vocoder_dim: int | None
    vocoder_layers: int | None
    vocoder_lookahead: int | None


@dataclass(frozen=True, slots=True)
class _SemanticResult:
    tokens: np.ndarray
    sampled_steps: int
    stopped_on_eos: bool
    max_steps: int
    min_steps: int
    prompt_token_limit: int


class _SoproSemanticDecoder:
    """Incremental semantic token decoder over one prepared ONNX prefix/KV state."""

    def __init__(
        self,
        step: Callable[[], tuple[int | None, bool]],
        *,
        max_steps: int,
        min_steps: int,
        prompt_token_limit: int,
    ) -> None:
        self._step = step
        self._max_steps = max_steps
        self._min_steps = min_steps
        self._prompt_token_limit = prompt_token_limit
        self._generated: list[int] = []
        self._sampled_steps = 0
        self._stopped_on_eos = False
        self._done = False
        self._closed = False

    def next_token(self) -> int | None:
        if self._closed:
            raise RuntimeContractError("Sopro semantic decoder is closed")
        if self._done:
            return None
        token, stopped_on_eos = self._step()
        self._sampled_steps += 1
        if stopped_on_eos:
            self._stopped_on_eos = True
            self._done = True
            return None
        if token is None:
            self._done = True
            return None
        self._generated.append(token)
        if len(self._generated) >= self._max_steps:
            self._done = True
        return token

    @property
    def generated(self) -> np.ndarray:
        tokens = np.asarray(self._generated, dtype=np.uint32)
        tokens.flags.writeable = False
        return tokens

    @property
    def stopped_on_eos(self) -> bool:
        return self._stopped_on_eos

    @property
    def result(self) -> _SemanticResult:
        return _SemanticResult(
            tokens=self.generated,
            sampled_steps=self._sampled_steps,
            stopped_on_eos=self._stopped_on_eos,
            max_steps=self._max_steps,
            min_steps=self._min_steps,
            prompt_token_limit=self._prompt_token_limit,
        )

    def iter_chunks(self, chunk_tokens: int) -> Iterator[np.ndarray]:
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise RuntimeContractError("Sopro semantic chunk size must be a positive integer")
        chunk: list[int] = []
        while (token := self.next_token()) is not None:
            chunk.append(token)
            if len(chunk) == chunk_tokens:
                values = np.asarray(chunk, dtype=np.uint32)
                values.flags.writeable = False
                yield values
                chunk = []
        if chunk:
            values = np.asarray(chunk, dtype=np.uint32)
            values.flags.writeable = False
            yield values

    def close(self) -> None:
        self._closed = True
        self._step = lambda: (None, False)
        self._generated.clear()


class _SoproRandom:
    def __init__(self, seed: int) -> None:
        self.state = (seed & 0xFFFFFFFF) or 1
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


def _rotary(length: int, dimension: int, *, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.empty((length, dimension), dtype=np.float32)
    sine = np.empty((length, dimension), dtype=np.float32)
    for position in range(length):
        for index in range(dimension):
            frequency = 1.0 / (10000.0 ** ((index % (dimension // 2)) * 2.0 / dimension))
            angle = (position + offset) * frequency
            cosine[position, index] = math.cos(angle)
            sine[position, index] = math.sin(angle)
    return cosine, sine


def _causal_attention_bias(length: int) -> np.ndarray:
    bias = np.zeros((1, 1, length, length), dtype=np.float32)
    for query in range(length):
        bias[0, 0, query, query + 1 :] = -1.0e4
    return bias


def _fft_pow2(real: np.ndarray, imag: np.ndarray, *, inverse: bool = False) -> None:
    size = real.size
    if size < 2 or size & (size - 1):
        raise RuntimeContractError("Sopro inverse FFT requires a power-of-two length")
    index = 0
    for current in range(1, size):
        bit = size >> 1
        while index & bit:
            index ^= bit
            bit >>= 1
        index ^= bit
        if current < index:
            real[current], real[index] = real[index], real[current]
            imag[current], imag[index] = imag[index], imag[current]
    length = 2
    while length <= size:
        angle = (2.0 * math.pi if inverse else -2.0 * math.pi) / length
        step_real = math.cos(angle)
        step_imag = math.sin(angle)
        for offset in range(0, size, length):
            weight_real = 1.0
            weight_imag = 0.0
            for column in range(length // 2):
                even = offset + column
                odd = even + length // 2
                temp_real = real[odd] * weight_real - imag[odd] * weight_imag
                temp_imag = real[odd] * weight_imag + imag[odd] * weight_real
                real[odd] = real[even] - temp_real
                imag[odd] = imag[even] - temp_imag
                real[even] += temp_real
                imag[even] += temp_imag
                next_real = weight_real * step_real - weight_imag * step_imag
                weight_imag = weight_real * step_imag + weight_imag * step_real
                weight_real = next_real
        length <<= 1
    if inverse:
        real /= size
        imag /= size


def _sopro_istft(features: np.ndarray, frames: int, *, n_fft: int, hop: int) -> np.ndarray:
    if type(frames) is not int or frames < 0 or type(n_fft) is not int or n_fft <= 0 or n_fft % 2:
        raise RuntimeContractError("Sopro vocoder frame/FFT dimensions are invalid")
    if type(hop) is not int or hop <= 0:
        raise RuntimeContractError("Sopro vocoder hop length must be a positive integer")
    spectrum = np.asarray(features)
    if not np.issubdtype(spectrum.dtype, np.floating) or spectrum.size != frames * (n_fft + 2):
        raise RuntimeContractError("Sopro vocoder features have an invalid dtype or shape")
    if not np.isfinite(spectrum).all():
        raise RuntimeContractError("Sopro vocoder features contain non-finite values")
    spectrum = spectrum.reshape(-1).astype(np.float64, copy=False)
    size = max(0, (frames - 1) * hop + n_fft)
    overlap = np.zeros(size, dtype=np.float64)
    envelope = np.zeros(size, dtype=np.float64)
    bins = n_fft // 2 + 1
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n_fft, dtype=np.float64) / n_fft)
    log_maximum = math.log(100.0)
    for frame in range(frames):
        offset = frame * (n_fft + 2)
        real = np.zeros(n_fft, dtype=np.float64)
        imag = np.zeros(n_fft, dtype=np.float64)
        for frequency in range(bins):
            magnitude = math.exp(min(log_maximum, float(spectrum[offset + frequency])))
            phase = float(spectrum[offset + bins + frequency])
            real[frequency] = magnitude * math.cos(phase)
            imag[frequency] = magnitude * math.sin(phase)
        for frequency in range(1, bins - 1):
            real[n_fft - frequency] = real[frequency]
            imag[n_fft - frequency] = -imag[frequency]
        _fft_pow2(real, imag, inverse=True)
        start = frame * hop
        overlap[start : start + n_fft] += real * window
        envelope[start : start + n_fft] += window * window
    pad = n_fft // 2
    output_length = max(0, size - 2 * pad)
    output = np.empty(output_length, dtype=np.float32)
    for index in range(output_length):
        source = index + pad
        output[index] = overlap[source] / max(1.0e-8, envelope[source])
    return np.ascontiguousarray(output)


def _sample_semantic_token(
    logits: np.ndarray,
    options: _SemanticOptions,
    random: _SoproRandom,
    *,
    vocab_size: int,
    allow_eos: bool,
) -> int:
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.size != vocab_size + 2 or not np.isfinite(scores).all():
        raise RuntimeContractError(
            "Sopro semantic logits have an invalid shape or non-finite values"
        )
    bos_id, eos_id = vocab_size, vocab_size + 1
    if options.temperature == 0.0:
        best_id = -1
        best_score = -math.inf
        for token_id, score in enumerate(scores):
            if token_id == bos_id or (not allow_eos and token_id == eos_id):
                continue
            if score > best_score:
                best_id = token_id
                best_score = float(score)
        if best_id < 0:
            raise RuntimeContractError("Sopro semantic logits contain no sampleable token")
        return best_id

    limit = min(options.top_k, scores.size) if options.top_k > 0 else scores.size
    indices: list[int] = []
    candidates: list[float] = []
    min_score = math.inf
    min_at = 0
    for token_id, raw_score in enumerate(scores):
        if token_id == bos_id or (not allow_eos and token_id == eos_id):
            continue
        score = float(raw_score)
        if len(indices) < limit:
            indices.append(token_id)
            candidates.append(score)
            if score < min_score:
                min_score = score
                min_at = len(indices) - 1
            if len(indices) == limit and limit < scores.size:
                min_score = math.inf
                for candidate_index, candidate_score in enumerate(candidates):
                    if candidate_score < min_score:
                        min_score = candidate_score
                        min_at = candidate_index
        elif score > min_score:
            indices[min_at] = token_id
            candidates[min_at] = score
            min_score = math.inf
            for candidate_index, candidate_score in enumerate(candidates):
                if candidate_score < min_score:
                    min_score = candidate_score
                    min_at = candidate_index

    order = sorted(range(len(indices)), key=lambda index: -candidates[index])
    maximum = candidates[order[0]] / options.temperature
    probabilities = [math.exp(candidates[index] / options.temperature - maximum) for index in order]
    total = 0.0
    for probability in probabilities:
        total += probability
    probabilities = [probability / total for probability in probabilities]
    if options.top_p < 1.0:
        cumulative = 0.0
        keep = 0
        while True:
            cumulative += probabilities[keep]
            keep += 1
            if keep >= len(probabilities) or cumulative >= options.top_p:
                break
        order = order[:keep]
        probabilities = probabilities[:keep]
        total = 0.0
        for probability in probabilities:
            total += probability
        probabilities = [probability / total for probability in probabilities]
    draw = random.value()
    for index, probability in zip(order, probabilities, strict=True):
        draw -= probability
        if draw <= 0.0:
            return indices[index]
    return indices[order[-1]]


def _resolve_bundle_path(root: Path, profile: str | None, value: Any, *, label: str) -> Path:
    """Resolve an upstream manifest path without allowing bundle-root escape."""
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or ":" in value
        or "?" in value
        or "#" in value
        or value.startswith("/")
    ):
        raise UnsafePathError(f"Unsafe Sopro {label} path: {value!r}")
    if profile is not None:
        try:
            validate_safe_component(profile, field_name="Sopro profile")
        except ValueError as exc:
            raise UnsafePathError(str(exc)) from exc
        relative = posixpath.normpath(posixpath.join(profile, value))
    else:
        relative = posixpath.normpath(value)
    if relative in {".", ".."} or relative.startswith("../") or relative.startswith("/"):
        raise UnsafePathError(f"Sopro {label} path escapes the bundle: {value!r}")
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafePathError(f"Unsafe Sopro {label} path: {value!r}")
    bundle_root = root.resolve()
    resolved = root.joinpath(*parts).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise UnsafePathError(f"Sopro {label} path escapes the bundle: {value!r}") from exc
    if not resolved.is_file():
        raise ManifestError(f"Sopro {label} file does not exist: {resolved}")
    return resolved


def _verify_declared_file(path: Path, declaration: Mapping[str, Any], *, label: str) -> None:
    size = declaration.get("bytes")
    if size is not None:
        if type(size) is not int or size <= 0:
            raise ManifestError(f"Sopro {label} bytes must be a positive integer")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise IntegrityError(
                f"Sopro {label} size mismatch for {path}: expected {size}, got {actual_size}"
            )
    digest = declaration.get("sha256")
    if digest is not None:
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ManifestError(f"Sopro {label} sha256 must be a lowercase 64-character digest")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        if actual != digest:
            raise IntegrityError(f"Sopro {label} SHA-256 mismatch for {path}")


class SoproAdapter(SystemAdapter):
    """Built-in adapter for manifest-backed Sopro ONNX bundles.

    Graph sessions are initialized only when requested. This first-layer adapter
    exposes graph inspection and execution while higher-level generation contracts
    are implemented and parity-gated.
    """

    system = "sopro"

    def __init__(
        self,
        installation: Installation,
        *,
        providers: str | Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ) -> None:
        super().__init__(
            installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        self._sessions: dict[str, OnnxSession] = {}
        self._stream_cache_capacity = 1
        self._prompt_state_cache: OrderedDict[tuple[Any, ...], _SoproPromptState] = OrderedDict()
        self._vocoder_prompt_cache: OrderedDict[tuple[Any, ...], _SoproVocoderPromptState] = (
            OrderedDict()
        )
        self._generation_token = 0
        self._runtime_shape_config: _SoproRuntimeConfig | None = None
        self.profile = self._selected_profile(installation)
        manifest_artifacts = [
            artifact for artifact in installation.artifacts if artifact.role == "manifest"
        ]
        if len(manifest_artifacts) != 1:
            raise ManifestError("Sopro installation must contain exactly one manifest artifact")
        self.manifest_path = manifest_artifacts[0].path.resolve()
        if self.manifest_path.name != "manifest.json":
            raise ManifestError("Sopro manifest artifact must be named manifest.json")
        self.bundle_root = self.manifest_path.parent.resolve()
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(
                f"Could not read Sopro manifest {self.manifest_path}: {exc}"
            ) from exc
        if (
            not isinstance(manifest, Mapping)
            or type(manifest.get("format")) is not int
            or manifest["format"] != 1
        ):
            raise ManifestError("Unsupported Sopro manifest; expected format=1")
        profiles = manifest.get("profiles")
        if not isinstance(profiles, Mapping) or not profiles:
            raise ManifestError("Sopro manifest profiles must be a non-empty object")
        if self.profile not in profiles:
            raise CapabilityError(
                f"Unknown Sopro profile {self.profile!r}; choices: {sorted(map(str, profiles))}"
            )
        profile_data = profiles[self.profile]
        if not isinstance(profile_data, Mapping):
            raise ManifestError(f"Sopro profile {self.profile!r} must be an object")
        self.manifest = manifest
        self.profile_metadata = dict(profile_data)
        self._tokenizer: SoproTokenizer | None = None
        revision = manifest.get("revision")
        declared_revision = installation.metadata.get("manifest_revision")
        if declared_revision is not None and revision != declared_revision:
            raise ManifestError(
                "Sopro manifest revision does not match the selected catalog revision"
            )
        sample_rate = manifest.get("sampleRate")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ManifestError("Sopro manifest sampleRate must be a positive integer")
        if installation.sample_rate is not None and installation.sample_rate != sample_rate:
            raise ManifestError("Sopro manifest sample rate does not match the installation")
        self.sample_rate = sample_rate
        tokenizer_artifacts = [
            artifact for artifact in installation.artifacts if artifact.role == "tokenizer"
        ]
        if len(tokenizer_artifacts) != 1:
            raise ManifestError("Sopro installation must contain exactly one tokenizer artifact")
        tokenizer_declaration = manifest.get("tokenizer")
        self.tokenizer_path = _resolve_bundle_path(
            self.bundle_root, None, tokenizer_declaration, label="tokenizer"
        )
        if tokenizer_artifacts[0].path.resolve() != self.tokenizer_path:
            raise ManifestError(
                "Sopro tokenizer artifact does not match the manifest tokenizer path"
            )

        raw_graphs = profile_data.get("graphs")
        raw_shards = profile_data.get("shards")
        if not isinstance(raw_graphs, Mapping) or not raw_graphs:
            raise ManifestError(f"Sopro profile {self.profile!r} has no graph definitions")
        if not isinstance(raw_shards, Mapping) or not raw_shards:
            raise ManifestError(f"Sopro profile {self.profile!r} has no external-data shards")
        self._shard_paths: dict[str, Path] = {}
        self._shard_aliases: dict[str, Path] = {}
        for raw_name, declaration in raw_shards.items():
            name = self._component_name(raw_name, "shard")
            if not isinstance(declaration, Mapping):
                raise ManifestError(f"Sopro shard {name!r} must be an object")
            shard_path = _resolve_bundle_path(
                self.bundle_root,
                self.profile,
                declaration.get("url") or declaration.get("file"),
                label=f"shard {name}",
            )
            _verify_declared_file(shard_path, declaration, label=f"shard {name}")
            self._shard_paths[name] = shard_path
            for alias in (name, declaration.get("file"), declaration.get("url")):
                if not isinstance(alias, str) or not alias:
                    continue
                for candidate in {alias, PurePosixPath(alias).name}:
                    existing = self._shard_aliases.get(candidate)
                    if existing is not None and existing != shard_path:
                        raise ManifestError(f"Sopro shard reference {candidate!r} is ambiguous")
                    self._shard_aliases[candidate] = shard_path
        self._graph_paths: dict[str, Path] = {}
        self._graph_shards: dict[str, Path] = {}
        for raw_name, declaration in raw_graphs.items():
            name = self._component_name(raw_name, "graph")
            if not isinstance(declaration, Mapping):
                raise ManifestError(f"Sopro graph {name!r} must be an object")
            graph_path = _resolve_bundle_path(
                self.bundle_root,
                self.profile,
                declaration.get("url") or declaration.get("file"),
                label=f"graph {name}",
            )
            _verify_declared_file(graph_path, declaration, label=f"graph {name}")
            shard_name = declaration.get("shard")
            if not isinstance(shard_name, str) or not shard_name:
                raise ManifestError(f"Sopro graph {name!r} has no external-data shard")
            self._component_name(shard_name, "graph shard")
            resolved_shard_path = self._shard_aliases.get(shard_name)
            if resolved_shard_path is None:
                raise ManifestError(
                    f"Sopro graph {name!r} references undeclared shard {shard_name!r}"
                )
            self._graph_paths[name] = graph_path
            self._graph_shards[name] = resolved_shard_path

        self._validate_supplied_artifacts(installation)

    @staticmethod
    def _selected_profile(installation: Installation) -> str:
        runtime = installation.metadata.get("runtime") or {}
        profile = runtime.get("profile") if isinstance(runtime, Mapping) else None
        profile = (
            profile
            or installation.metadata.get("selected_quality")
            or installation.metadata.get("onnx_profile")
        )
        if not isinstance(profile, str) or not profile:
            raise CapabilityError("Sopro profile must be selected explicitly when opening a bundle")
        try:
            validate_safe_component(profile, field_name="Sopro profile")
        except ValueError as exc:
            raise CapabilityError(str(exc)) from exc
        return profile

    @staticmethod
    def _component_name(value: Any, label: str) -> str:
        try:
            return validate_safe_component(value, field_name=f"Sopro {label}")
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc

    def _validate_supplied_artifacts(self, installation: Installation) -> None:
        graph_artifacts = [
            artifact for artifact in installation.artifacts if artifact.role == "graph"
        ]
        shard_artifacts = [
            artifact for artifact in installation.artifacts if artifact.role == "shard"
        ]
        if any(
            artifact.quality not in {None, self.profile}
            for artifact in graph_artifacts + shard_artifacts
        ):
            raise ManifestError("Sopro installation contains artifacts from an unselected profile")
        selected_graphs = {artifact.component for artifact in graph_artifacts}
        selected_shards = {artifact.component for artifact in shard_artifacts}
        if selected_graphs != set(self._graph_paths):
            raise ManifestError("Sopro graph artifacts do not match the selected manifest profile")
        if selected_shards != set(self._shard_paths):
            raise ManifestError("Sopro shard artifacts do not match the selected manifest profile")
        for role, paths in (("graph", self._graph_paths), ("shard", self._shard_paths)):
            for component, expected_path in paths.items():
                supplied = [
                    artifact
                    for artifact in installation.artifacts
                    if artifact.role == role and artifact.component == component
                ]
                if len(supplied) != 1:
                    raise ManifestError(f"Sopro requires exactly one {role} artifact {component!r}")
                if supplied[0].path.resolve() != expected_path:
                    raise ManifestError(
                        f"Sopro {role} artifact {component!r} does not match the manifest path"
                    )

    def _generation_integer(
        self, config: Mapping[str, Any], name: str, default: int | None = None
    ) -> int:
        value = config.get(name, default)
        if type(value) is not int or value < 0:
            raise ManifestError(f"Sopro generation {name} must be a non-negative integer")
        return value

    @property
    def generation_defaults(self) -> SoproGenerationDefaults:
        config = self._frontend_config()
        generation = config.get("generation")
        if not isinstance(generation, Mapping):
            raise ManifestError("Sopro manifest is missing generation settings")

        def finite(name: str, *, positive: bool = False) -> float:
            value = generation.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ManifestError(f"Sopro generation {name} must be a finite number")
            result = float(value)
            if not math.isfinite(result) or (positive and result <= 0):
                raise ManifestError(f"Sopro generation {name} must be finite and valid")
            return result

        temperature = finite("temperature")
        top_p = finite("top_p")
        max_seconds = finite("max_seconds", positive=True)
        min_seconds = finite("min_seconds")
        if temperature < 0 or not 0 <= top_p <= 1 or min_seconds < 0 or min_seconds > max_seconds:
            raise ManifestError("Sopro generation defaults are outside their supported ranges")
        return SoproGenerationDefaults(
            temperature=temperature,
            top_p=top_p,
            top_k=self._generation_integer(generation, "top_k"),
            steps=self.fixed_steps,
            max_seconds=max_seconds,
            min_seconds=min_seconds,
            ref_seconds=finite("ref_seconds", positive=True),
            max_segment_chars=self._generation_integer(generation, "max_segment_chars", 500),
            style_tokens=self._generation_integer(generation, "style_tokens"),
            prompt_tokens=self._generation_integer(generation, "prompt_tokens"),
            stream_chunk_frames=self._generation_integer(generation, "stream_chunk_frames", 8),
        )

    @property
    def _runtime_config(self) -> _SoproRuntimeConfig:
        cached = self._runtime_shape_config
        if cached is not None:
            return cached
        position_context = self.manifest.get("positionContext")
        if position_context is not None and (
            type(position_context) is not int or position_context <= 0
        ):
            raise ManifestError("Sopro positionContext must be a positive integer")
        frontend = self._frontend_config()
        model = frontend.get("model")
        vocoder = frontend.get("vocoder")
        if not isinstance(model, Mapping) or not isinstance(vocoder, Mapping):
            raise ManifestError("Sopro manifest is missing acoustic/vocoder shape settings")

        def optional_int(values: Mapping[str, Any], name: str, *, minimum: int = 1) -> int | None:
            value = values.get(name)
            if value is None:
                return None
            if type(value) is not int or value < minimum:
                raise ManifestError(f"Sopro model shape {name} is invalid")
            return value

        vocoder_layers = optional_int(vocoder, "num_layers")
        lookaheads = vocoder.get("block_lookaheads")
        if lookaheads is None:
            vocoder_lookahead = None
        elif (
            not isinstance(lookaheads, Sequence)
            or isinstance(lookaheads, (str, bytes))
            or not lookaheads
            or any(type(value) is not int or value < 0 for value in lookaheads)
            or len(set(lookaheads)) != 1
            or (vocoder_layers is not None and len(lookaheads) != vocoder_layers)
        ):
            raise ManifestError("Sopro vocoder lookahead dimensions are invalid")
        else:
            vocoder_lookahead = int(lookaheads[0])
        cached = _SoproRuntimeConfig(
            sample_rate=self.sample_rate,
            token_samples=self.token_samples,
            hop_ratio=self.hop_ratio,
            fixed_steps=self.fixed_steps,
            position_context=position_context,
            stream_chunk_frames=self.generation_defaults.stream_chunk_frames,
            acoustic_pre_lookahead_frames=optional_int(
                model, "acoustic_pre_lookahead_frames", minimum=0
            ),
            acoustic_dim_head=optional_int(model, "acoustic_dit_dim_head"),
            vocoder_n_fft=optional_int(vocoder, "n_fft"),
            vocoder_hop_length=optional_int(vocoder, "hop_length"),
            vocoder_dim=optional_int(vocoder, "dim"),
            vocoder_layers=vocoder_layers,
            vocoder_lookahead=vocoder_lookahead,
        )
        self._runtime_shape_config = cached
        return cached

    @property
    def fixed_steps(self) -> int:
        config = self._frontend_config().get("generation")
        if not isinstance(config, Mapping):
            raise ManifestError("Sopro manifest is missing generation settings")
        value = self.manifest.get("steps", config.get("steps"))
        if type(value) is not int or value <= 0:
            raise ManifestError("Sopro manifest must declare a positive fixed acoustic step count")
        return value

    @property
    def max_segment_chars(self) -> int:
        return self.generation_defaults.max_segment_chars

    @property
    def token_samples(self) -> int:
        value = self.manifest.get("tokenSamples")
        if type(value) is not int or value <= 0:
            raise ManifestError("Sopro manifest tokenSamples must be a positive integer")
        return value

    @property
    def hop_ratio(self) -> int:
        value = self.manifest.get("hopRatio")
        if type(value) is not int or value <= 0:
            raise ManifestError("Sopro manifest hopRatio must be a positive integer")
        return value

    @property
    def graphs(self) -> tuple[str, ...]:
        """Graph names declared by the selected upstream profile."""
        return tuple(sorted(self._graph_paths))

    @property
    def shards(self) -> tuple[str, ...]:
        """External-data shard names declared by the selected upstream profile."""
        return tuple(sorted(self._shard_paths))

    @property
    def graph_shards(self) -> Mapping[str, Path]:
        """Read-only mapping from graph role to its external-data shard path."""
        return dict(self._graph_shards)

    def session(self, graph: str) -> OnnxSession:
        """Return a provider-configured, lazily created graph session."""
        try:
            path = self._graph_paths[graph]
        except KeyError as exc:
            raise CapabilityError(f"Unknown Sopro graph {graph!r}") from exc
        if graph not in self._sessions:
            self._sessions[graph] = OnnxSession(
                path,
                component=graph,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._sessions[graph]

    def graph_specs(self, graph: str) -> dict[str, Any]:
        """Return the selected graph's input, output, and active-provider inventory."""
        session = self.session(graph)
        return {
            "inputs": session.input_specs,
            "outputs": session.output_specs,
            "providers": session.resolved_providers,
            "external_data": self._graph_shards[graph],
        }

    def run_graph(self, graph: str, inputs: Mapping[str, Any]) -> list[Any]:
        """Execute one graph with caller-supplied tensors matching its ONNX contract."""
        return self.session(graph).run(dict(inputs))

    def _frontend_config(self) -> Mapping[str, Any]:
        config = self.manifest.get("config")
        if not isinstance(config, Mapping):
            raise ManifestError("Sopro manifest is missing its model config")
        return config

    def _get_tokenizer(self) -> SoproTokenizer:
        if self._tokenizer is None:
            try:
                spec = json.loads(self.tokenizer_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ManifestError(
                    f"Could not read Sopro tokenizer {self.tokenizer_path}: {exc}"
                ) from exc
            if not isinstance(spec, Mapping):
                raise ManifestError("Sopro tokenizer must be a JSON object")
            self._tokenizer = SoproTokenizer(spec)
        return self._tokenizer

    def prepare_text(self, text: str, language: str | None = None) -> SoproTextInput:
        """Normalize and tokenize one unsplit text segment for Sopro's semantic graph."""
        if not isinstance(text, str):
            raise RuntimeContractError("Sopro text input must be a string")
        return self._get_tokenizer().encode(text, language)

    @staticmethod
    def _ort_numpy_dtype(ort_type: str) -> np.dtype[Any]:
        dtypes: dict[str, Any] = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_,
        }
        try:
            return np.dtype(dtypes[ort_type])
        except KeyError as exc:
            raise RuntimeContractError(
                f"Unsupported Sopro reference tensor type {ort_type!r}"
            ) from exc

    def _semantic_generation_options(
        self, options: _SemanticOptions | Mapping[str, Any] | None = None
    ) -> _SemanticOptions:
        if isinstance(options, _SemanticOptions):
            return options
        if options is None:
            options = {}
        if not isinstance(options, Mapping):
            raise RuntimeContractError("Sopro generation options must be a mapping")
        allowed = {
            "temperature",
            "top_p",
            "top_k",
            "max_seconds",
            "min_seconds",
            "seed",
            "steps",
        }
        unknown = set(options) - allowed
        if any(not isinstance(key, str) for key in options):
            raise RuntimeContractError("Sopro generation option names must be strings")
        if unknown:
            raise RuntimeContractError(f"Unsupported Sopro generation option(s): {sorted(unknown)}")
        config = self._frontend_config()
        generation = config.get("generation")
        model = config.get("model")
        if not isinstance(generation, Mapping) or not isinstance(model, Mapping):
            raise ManifestError("Sopro manifest config is missing generation/model settings")

        def number(name: str, default_name: str) -> float:
            value = options.get(name)
            if value is None:
                value = generation.get(default_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeContractError(f"Sopro {name} must be a finite number")
            result = float(value)
            if not math.isfinite(result):
                raise RuntimeContractError(f"Sopro {name} must be a finite number")
            return result

        temperature = number("temperature", "temperature")
        top_p = number("top_p", "top_p")
        max_seconds = number("max_seconds", "max_seconds")
        min_seconds = number("min_seconds", "min_seconds")
        if temperature < 0.0:
            raise RuntimeContractError("Sopro temperature must be non-negative")
        if not 0.0 <= top_p <= 1.0:
            raise RuntimeContractError("Sopro top_p must be in [0, 1]")
        if max_seconds <= 0.0 or min_seconds < 0.0 or min_seconds > max_seconds:
            raise RuntimeContractError(
                "Sopro duration controls require 0 <= min_seconds <= max_seconds"
            )
        declared_max = generation.get("max_seconds")
        if (
            isinstance(declared_max, bool)
            or not isinstance(declared_max, (int, float))
            or not math.isfinite(float(declared_max))
            or max_seconds > float(declared_max)
        ):
            raise RuntimeContractError(
                "Sopro max_seconds exceeds the manifest-declared generation limit"
            )

        top_k = options.get("top_k")
        if top_k is None:
            top_k = generation.get("top_k")
        if type(top_k) is not int or top_k < 0:
            raise RuntimeContractError("Sopro top_k must be a non-negative integer")
        seed = options.get("seed", 1)
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise RuntimeContractError("Sopro seed must be an integer in [0, 2**32 - 1]")

        fixed_steps = self.manifest.get("steps", generation.get("steps"))
        requested_steps = options.get("steps")
        if requested_steps is None:
            requested_steps = fixed_steps
        if type(fixed_steps) is not int or fixed_steps <= 0:
            raise ManifestError("Sopro manifest must declare a positive fixed acoustic step count")
        if type(requested_steps) is not int or requested_steps != fixed_steps:
            raise CapabilityError(f"Sopro acoustic steps are fixed at {fixed_steps}")
        token_samples = self.manifest.get("tokenSamples")
        if type(token_samples) is not int or token_samples <= 0:
            raise ManifestError("Sopro manifest tokenSamples must be a positive integer")
        max_steps = math.ceil(max_seconds * self.sample_rate / token_samples)
        min_steps = max(1, math.ceil(min_seconds * self.sample_rate / token_samples))
        if min_steps > max_steps:
            raise RuntimeContractError(
                "Sopro minimum duration exceeds the semantic-token generation limit"
            )
        return _SemanticOptions(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_seconds=max_seconds,
            min_seconds=min_seconds,
            seed=seed,
            max_steps=max_steps,
            min_steps=min_steps,
        )

    def _run_contract_graph(
        self,
        graph: str,
        inputs: Mapping[str, np.ndarray],
        expected_outputs: Sequence[str],
    ) -> dict[str, np.ndarray]:
        session = self.session(graph)
        input_specs = tuple(getattr(session, "input_specs", ()))
        expected_inputs = set(inputs)
        if input_specs:
            names = [spec.name for spec in input_specs]
            if len(names) != len(set(names)) or set(names) != expected_inputs:
                raise RuntimeContractError(
                    f"Sopro {graph} graph input names do not match its contract"
                )
            dtypes = {spec.name: self._ort_numpy_dtype(spec.ort_type) for spec in input_specs}
        else:
            dtypes = {name: np.asarray(value).dtype for name, value in inputs.items()}
        graph_inputs = {
            name: np.ascontiguousarray(value, dtype=dtypes[name]) for name, value in inputs.items()
        }
        values = session.run(graph_inputs)
        output_names = tuple(getattr(session, "output_names", ()))
        if output_names:
            if len(output_names) != len(values):
                raise RuntimeContractError(
                    f"Sopro {graph} graph output inventory does not match its results"
                )
            outputs = dict(zip(output_names, values, strict=True))
        elif len(values) == len(expected_outputs):
            outputs = dict(zip(expected_outputs, values, strict=True))
        else:
            raise RuntimeContractError(
                f"Sopro {graph} graph returned an unexpected number of outputs"
            )
        if set(outputs) != set(expected_outputs):
            raise RuntimeContractError(
                f"Sopro {graph} graph output names do not match its contract"
            )
        return {name: np.asarray(outputs[name]) for name in expected_outputs}

    def _generate_semantic_tokens(
        self,
        text_input: SoproTextInput,
        voice_state: SoproVoiceState,
        *,
        options: _SemanticOptions | Mapping[str, Any] | None = None,
        prompt_token_override: Sequence[int] | np.ndarray | None = None,
    ) -> _SemanticResult:
        """Drain the incremental decoder for the compatibility offline API."""
        decoder = self._create_semantic_decoder(
            text_input,
            voice_state,
            options=options,
            prompt_token_override=prompt_token_override,
        )
        try:
            for _chunk in decoder.iter_chunks(32):
                pass
            return decoder.result
        finally:
            decoder.close()

    def _create_semantic_decoder(
        self,
        text_input: SoproTextInput,
        voice_state: SoproVoiceState,
        *,
        options: _SemanticOptions | Mapping[str, Any] | None = None,
        prompt_token_override: Sequence[int] | np.ndarray | None = None,
    ) -> _SoproSemanticDecoder:
        if not isinstance(text_input, SoproTextInput):
            raise RuntimeContractError("Sopro semantic generation requires a SoproTextInput")
        if not isinstance(voice_state, SoproVoiceState):
            raise RuntimeContractError("Sopro semantic generation requires a SoproVoiceState")
        config = self._frontend_config()
        model = config.get("model")
        generation = config.get("generation")
        if not isinstance(model, Mapping) or not isinstance(generation, Mapping):
            raise ManifestError("Sopro manifest config is missing semantic generation settings")
        settings = self._semantic_generation_options(options)
        if voice_state.sample_rate != self.sample_rate:
            raise RuntimeContractError(
                "Sopro voice state sample rate does not match this installation"
            )
        language = text_input.language
        if language is not None and language not in {"en", "pt", "fr", "de"}:
            raise CapabilityError(f"Unsupported Sopro language {language!r}")

        vocab_size = model.get("semantic_vocab_size")
        text_vocab_size = model.get("text_vocab_size")
        max_text_len = model.get("max_text_len")
        model_dim = model.get("ar_model_dim")
        ar_blocks = model.get("ar_blocks")
        ar_heads = model.get("ar_heads")
        kv_heads = model.get("ar_kv_heads")
        style_prefix = model.get("style_prefix_tokens")
        required_ints = (
            vocab_size,
            text_vocab_size,
            max_text_len,
            model_dim,
            ar_blocks,
            ar_heads,
            kv_heads,
            style_prefix,
        )
        if any(type(value) is not int or value <= 0 for value in required_ints):
            raise ManifestError("Sopro semantic model dimensions must be positive integers")
        vocab_size = cast(int, vocab_size)
        text_vocab_size = cast(int, text_vocab_size)
        max_text_len = cast(int, max_text_len)
        model_dim = cast(int, model_dim)
        ar_blocks = cast(int, ar_blocks)
        ar_heads = cast(int, ar_heads)
        kv_heads = cast(int, kv_heads)
        style_prefix = cast(int, style_prefix)
        if model_dim % ar_heads or model_dim % 2 or (model_dim // ar_heads) % 2:
            raise ManifestError("Sopro autoregressive model dimensions are inconsistent")
        head_dim = model_dim // ar_heads
        token_ids = np.asarray(text_input.token_ids)
        if (
            token_ids.ndim != 1
            or token_ids.size == 0
            or token_ids.size > max_text_len
            or not np.issubdtype(token_ids.dtype, np.integer)
            or np.any(token_ids < 0)
            or np.any(token_ids >= text_vocab_size)
        ):
            raise RuntimeContractError(
                "Sopro model-ready text IDs have an invalid rank, length, or value"
            )
        if (
            voice_state.semantic_tokens.size == 0
            or np.any(voice_state.semantic_tokens >= vocab_size)
            or voice_state.cond_vec.size != model.get("cond_hidden_dim")
        ):
            raise RuntimeContractError(
                "Sopro reference voice state does not match semantic model dimensions"
            )

        style_length = generation.get("style_tokens")
        prompt_length = generation.get("prompt_tokens")
        if (
            type(style_length) is not int
            or style_length < 0
            or type(prompt_length) is not int
            or prompt_length < 0
        ):
            raise ManifestError(
                "Sopro style_tokens and prompt_tokens must be non-negative integers"
            )
        style_tokens = np.ascontiguousarray(
            voice_state.semantic_tokens[:style_length][None, :], dtype=np.int64
        )
        if prompt_token_override is None:
            prompt_values = voice_state.semantic_tokens[:prompt_length]
        else:
            if isinstance(prompt_token_override, (str, bytes)) or not isinstance(
                prompt_token_override, (Sequence, np.ndarray)
            ):
                raise RuntimeContractError(
                    "Sopro prompt_tokens override must be a one-dimensional integer sequence"
                )
            try:
                prompt_values = np.asarray(prompt_token_override)
            except (TypeError, ValueError) as exc:
                raise RuntimeContractError(
                    "Sopro prompt_tokens override must be a one-dimensional integer sequence"
                ) from exc
            if (
                prompt_values.ndim != 1
                or prompt_values.size > prompt_length
                or (prompt_values.size > 0 and not np.issubdtype(prompt_values.dtype, np.integer))
            ):
                raise RuntimeContractError(
                    "Sopro prompt_tokens override has an invalid rank, length, or dtype"
                )
            if prompt_values.size and (
                np.any(prompt_values < 0) or np.any(prompt_values >= vocab_size)
            ):
                raise RuntimeContractError("Sopro prompt_tokens override contains an invalid token")
        prompt_input = np.ascontiguousarray(prompt_values[None, :], dtype=np.int64)
        model_text_ids = np.ascontiguousarray(token_ids[None, :], dtype=np.int64)
        prefix_length = style_prefix + token_ids.size + prompt_input.shape[1] + 1
        hidden_length = prefix_length - 1

        prefix = self._run_contract_graph(
            "semantic_prefix",
            {
                "text_ids": model_text_ids,
                "style_tokens": style_tokens,
                "prompt_tokens": prompt_input,
            },
            ("hidden", "bos_hidden"),
        )
        hidden = prefix["hidden"]
        bos_hidden = prefix["bos_hidden"]
        if (
            hidden.shape != (1, hidden_length, model_dim)
            or not np.issubdtype(hidden.dtype, np.floating)
            or not np.isfinite(hidden).all()
            or bos_hidden.shape != (1, 1, model_dim)
            or not np.issubdtype(bos_hidden.dtype, np.floating)
            or not np.isfinite(bos_hidden).all()
        ):
            raise RuntimeContractError(
                "Sopro semantic prefix outputs have invalid shape, dtype, or values"
            )

        cosine, sine = _rotary(prefix_length, head_dim)
        core = self._run_contract_graph(
            "semantic_core",
            {
                "hidden": hidden,
                "token": np.asarray([[vocab_size]], dtype=np.int64),
                "past_k": np.empty((ar_blocks, 1, kv_heads, 0, head_dim), dtype=np.float32),
                "past_v": np.empty((ar_blocks, 1, kv_heads, 0, head_dim), dtype=np.float32),
                "cos": cosine,
                "sin": sine,
                "attention_bias": _causal_attention_bias(prefix_length),
            },
            ("logits", "present_k", "present_v"),
        )

        def validate_core_outputs(
            outputs: Mapping[str, np.ndarray], expected_length: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            logits = np.asarray(outputs["logits"])
            expected_cache = (ar_blocks, 1, kv_heads, expected_length, head_dim)
            if (
                logits.shape != (1, vocab_size + 2)
                or not np.issubdtype(logits.dtype, np.floating)
                or not np.isfinite(logits).all()
            ):
                raise RuntimeContractError(
                    "Sopro semantic logits have invalid shape, dtype, or values"
                )
            caches = []
            for name in ("present_k", "present_v"):
                cache = np.asarray(outputs[name])
                if (
                    cache.shape != expected_cache
                    or not np.issubdtype(cache.dtype, np.floating)
                    or not np.isfinite(cache).all()
                ):
                    raise RuntimeContractError(
                        f"Sopro semantic {name} cache has an invalid shape or value"
                    )
                caches.append(np.ascontiguousarray(cache, dtype=np.float32))
            return np.ascontiguousarray(logits.reshape(-1), dtype=np.float32), caches[0], caches[1]

        logits, past_k, past_v = validate_core_outputs(core, prefix_length)
        random = _SoproRandom(settings.seed)
        stopped = False
        sampled_steps = 0
        position = prefix_length
        empty_hidden = np.empty((1, 0, model_dim), dtype=np.float32)
        eos_id = vocab_size + 1
        previous_token: int | None = None

        def next_step() -> tuple[int | None, bool]:
            nonlocal logits, past_k, past_v, position, previous_token, sampled_steps, stopped
            if stopped:
                return None, False
            if previous_token is not None:
                step_cosine, step_sine = _rotary(1, head_dim, offset=position)
                present_length = position + 1
                step_outputs = self._run_contract_graph(
                    "semantic_core",
                    {
                        "hidden": empty_hidden,
                        "token": np.asarray([[previous_token]], dtype=np.int64),
                        "past_k": past_k,
                        "past_v": past_v,
                        "cos": step_cosine,
                        "sin": step_sine,
                        "attention_bias": np.zeros((1, 1, 1, present_length), dtype=np.float32),
                    },
                    ("logits", "present_k", "present_v"),
                )
                logits, past_k, past_v = validate_core_outputs(step_outputs, present_length)
                position += 1
            sampled_steps += 1
            token = _sample_semantic_token(
                logits,
                settings,
                random,
                vocab_size=vocab_size,
                allow_eos=sampled_steps >= settings.min_steps,
            )
            if sampled_steps >= settings.min_steps and token == eos_id:
                stopped = True
                return None, True
            previous_token = min(vocab_size - 1, max(0, token))
            return previous_token, False

        return _SoproSemanticDecoder(
            next_step,
            max_steps=settings.max_steps,
            min_steps=settings.min_steps,
            prompt_token_limit=prompt_length,
        )

    def prepare_voice(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        seconds: float | None = None,
    ) -> SoproVoiceState:
        """Prepare reusable reference state through Sopro's DSP and reference graph."""
        started = time.perf_counter()
        if type(sample_rate) is not int or sample_rate <= 0:
            raise RuntimeContractError("Sopro reference sample rate must be a positive integer")
        audio = np.asarray(waveform)
        if audio.ndim != 1:
            raise RuntimeContractError("Sopro reference audio must be mono (one-dimensional)")
        if audio.size == 0 or not np.issubdtype(audio.dtype, np.floating):
            raise RuntimeContractError(
                "Sopro reference audio must be a non-empty floating-point waveform"
            )
        if not np.isfinite(audio).all():
            raise RuntimeContractError("Sopro reference audio must contain only finite samples")

        config = self._frontend_config()
        if self.sample_rate != 24000:
            raise CapabilityError("Sopro reference preparation currently requires a 24-kHz profile")
        generation = config.get("generation")
        if not isinstance(generation, Mapping):
            raise ManifestError("Sopro manifest config is missing generation settings")
        duration = generation.get("ref_seconds", 10.0) if seconds is None else seconds
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise RuntimeContractError("Sopro reference duration must be finite and positive")
        token_samples = self.manifest.get("tokenSamples")
        if type(token_samples) is not int or token_samples <= 0:
            raise ManifestError("Sopro manifest tokenSamples must be a positive integer")

        audio_24k = resample_sopro(audio.astype(np.float32), sample_rate, self.sample_rate)
        normalized, level_db = prepare_reference_audio(
            audio_24k, self.sample_rate, seconds=float(duration)
        )
        features = reference_features(normalized, config, token_samples=token_samples)
        model_config = config.get("model")
        if not isinstance(model_config, Mapping):
            raise ManifestError("Sopro manifest config is missing model settings")
        mel_mean = np.asarray(model_config.get("acoustic_mel_mean"), dtype=np.float64)
        mel_std = np.asarray(model_config.get("acoustic_mel_std"), dtype=np.float64)
        if (
            mel_mean.shape != (100,)
            or mel_std.shape != (100,)
            or not np.isfinite(mel_mean).all()
            or not np.isfinite(mel_std).all()
            or np.any(mel_std <= 0)
        ):
            raise ManifestError("Sopro acoustic mel mean/std must be finite 100-element vectors")
        normalized_mel = (
            (features.acoustic_mel.astype(np.float64) - mel_mean[:, None]) / mel_std[:, None]
        ).astype(np.float32)

        raw_inputs: dict[str, np.ndarray] = {
            "semantic_mel": features.semantic_mel[np.newaxis, :, :],
            "interp_left": features.interp_left,
            "interp_right": features.interp_right,
            "interp_weight": features.interp_weight,
            "speaker_mel": features.speaker_mel[np.newaxis, :, :],
        }
        expected_inputs = set(raw_inputs)
        session = self.session("reference")
        input_specs = tuple(getattr(session, "input_specs", ()))
        if input_specs:
            spec_names = [spec.name for spec in input_specs]
            if len(spec_names) != len(set(spec_names)) or set(spec_names) != expected_inputs:
                raise RuntimeContractError(
                    "Sopro reference graph input names do not match the official reference contract"
                )
            input_dtypes = {spec.name: self._ort_numpy_dtype(spec.ort_type) for spec in input_specs}
        else:
            input_dtypes = {
                name: np.dtype(np.int64 if name in {"interp_left", "interp_right"} else np.float32)
                for name in raw_inputs
            }
        graph_inputs = {
            name: np.ascontiguousarray(value, dtype=input_dtypes[name])
            for name, value in raw_inputs.items()
        }
        outputs = session.run(graph_inputs)
        output_names = tuple(getattr(session, "output_names", ()))
        if output_names:
            if len(output_names) != len(outputs):
                raise RuntimeContractError(
                    "Sopro reference graph output inventory does not match its results"
                )
            named_outputs = dict(zip(output_names, outputs, strict=True))
        elif len(outputs) == 2:
            named_outputs = dict(zip(("cond_vec", "semantic_tokens"), outputs, strict=True))
        else:
            raise RuntimeContractError(
                "Sopro reference graph must return cond_vec and semantic_tokens"
            )
        if set(named_outputs) != {"cond_vec", "semantic_tokens"}:
            raise RuntimeContractError(
                "Sopro reference graph output names do not match the official contract"
            )

        condition = np.asarray(named_outputs["cond_vec"])
        if not np.issubdtype(condition.dtype, np.floating):
            raise RuntimeContractError("Sopro reference condition output must be floating point")
        condition = condition.reshape(-1)
        cond_dim = model_config.get("cond_hidden_dim")
        if type(cond_dim) is not int or cond_dim <= 0 or condition.size != cond_dim:
            raise RuntimeContractError("Sopro reference condition output has an invalid dimension")
        if not np.isfinite(condition).all():
            raise RuntimeContractError(
                "Sopro reference condition output contains non-finite values"
            )

        semantic_tokens = np.asarray(named_outputs["semantic_tokens"])
        semantic_tokens = semantic_tokens.reshape(-1)
        if not np.issubdtype(semantic_tokens.dtype, np.integer) and (
            not np.issubdtype(semantic_tokens.dtype, np.floating)
            or not np.isfinite(semantic_tokens).all()
            or not np.equal(semantic_tokens, np.floor(semantic_tokens)).all()
        ):
            raise RuntimeContractError("Sopro semantic-token output must contain integers")
        semantic_tokens = semantic_tokens.astype(np.int64)
        vocab_size = model_config.get("semantic_vocab_size")
        if (
            semantic_tokens.size != features.token_count
            or type(vocab_size) is not int
            or np.any(semantic_tokens < 0)
            or np.any(semantic_tokens >= vocab_size)
        ):
            raise RuntimeContractError(
                "Sopro reference semantic-token output has an invalid shape or value"
            )
        cache_hasher = hashlib.sha256()
        cache_hasher.update(str(self.manifest.get("revision")).encode("utf-8"))
        cache_hasher.update(b"\0")
        cache_hasher.update(self.profile.encode("utf-8"))
        cache_hasher.update(b"\0")
        cache_hasher.update(float(duration).hex().encode("ascii"))
        cache_hasher.update(b"\0")
        cache_hasher.update(np.ascontiguousarray(normalized, dtype=np.float32).tobytes())
        cache_key = cache_hasher.hexdigest()
        return SoproVoiceState(
            cond_vec=condition,
            semantic_tokens=semantic_tokens,
            mel=normalized_mel,
            sample_rate=self.sample_rate,
            level_db=level_db,
            cache_key=cache_key,
            reference_prepare_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _decode_vocoder(self, mel: np.ndarray) -> np.ndarray:
        config = self._frontend_config().get("vocoder")
        if not isinstance(config, Mapping):
            raise ManifestError("Sopro manifest is missing vocoder configuration")
        sample_rate = config.get("sample_rate")
        n_fft = config.get("n_fft")
        hop = config.get("hop_length")
        n_mels = config.get("n_mels")
        dimension = config.get("dim")
        layers = config.get("num_layers")
        lookaheads = config.get("block_lookaheads")
        if any(
            type(value) is not int or value <= 0
            for value in (sample_rate, n_fft, hop, n_mels, dimension, layers)
        ):
            raise ManifestError("Sopro vocoder dimensions must be positive integers")
        sample_rate = cast(int, sample_rate)
        n_fft = cast(int, n_fft)
        hop = cast(int, hop)
        n_mels = cast(int, n_mels)
        dimension = cast(int, dimension)
        layers = cast(int, layers)
        if sample_rate != self.sample_rate or n_mels != 100:
            raise CapabilityError(
                "Sopro vocoder profile is not compatible with 24-kHz, 100-mel inference"
            )
        if (
            not isinstance(lookaheads, Sequence)
            or isinstance(lookaheads, (str, bytes))
            or len(lookaheads) != layers
            or any(type(value) is not int or value < 0 for value in lookaheads)
            or len(set(lookaheads)) != 1
        ):
            raise CapabilityError("Sopro offline vocoder requires uniform per-layer lookahead")
        lookaheads = [int(value) for value in lookaheads]
        mel = np.asarray(mel)
        if mel.ndim != 2:
            raise RuntimeContractError("Sopro vocoder mel input must be a rank-two matrix")
        frame_count = mel.shape[-1]
        if mel.shape != (n_mels, frame_count) or frame_count <= 0 or not np.isfinite(mel).all():
            raise RuntimeContractError("Sopro vocoder mel input has an invalid shape or values")
        name = "vocoder"
        lookahead = lookaheads[0]
        initial_state: dict[str, np.ndarray] = {
            "embed_state": np.zeros((1, n_mels, 6), dtype=np.float32),
            "conv_state": np.zeros((layers, 1, dimension, 6), dtype=np.float32),
            "pending": np.zeros((layers, 1, dimension, lookahead), dtype=np.float32),
        }
        expected_outputs = ("istft_features", "embed_state_out", "conv_state_out", "pending_out")

        def features_and_frames(outputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, int]:
            features = np.asarray(outputs["istft_features"])
            if (
                features.ndim != 3
                or features.shape[0] != 1
                or features.shape[2] != n_fft + 2
                or not np.issubdtype(features.dtype, np.floating)
                or not np.isfinite(features).all()
            ):
                raise RuntimeContractError(
                    "Sopro vocoder ISTFT features have an invalid shape or values"
                )
            return np.ascontiguousarray(features.reshape(-1), dtype=np.float32), features.shape[1]

        def validate_states(outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
            expected = {
                "embed_state_out": (1, n_mels, 6),
                "conv_state_out": (layers, 1, dimension, 6),
                "pending_out": (layers, 1, dimension, lookahead),
            }
            states: dict[str, np.ndarray] = {}
            for output_name, shape in expected.items():
                state = np.asarray(outputs[output_name])
                if (
                    state.shape != shape
                    or not np.issubdtype(state.dtype, np.floating)
                    or not np.isfinite(state).all()
                ):
                    raise RuntimeContractError(
                        f"Sopro vocoder {output_name} has an invalid shape or values"
                    )
                states[output_name] = np.ascontiguousarray(state, dtype=np.float32)
            return states

        started = self._run_contract_graph(
            name,
            {
                "mel": np.ascontiguousarray(mel[None, :, :], dtype=np.float32),
                "is_start": np.asarray(True, dtype=np.bool_),
                "is_flush": np.asarray(False, dtype=np.bool_),
                **initial_state,
            },
            expected_outputs,
        )
        first_features, first_frames = features_and_frames(started)
        state = validate_states(started)
        tail = self._run_contract_graph(
            name,
            {
                "mel": np.empty((1, n_mels, 0), dtype=np.float32),
                "is_start": np.asarray(False, dtype=np.bool_),
                "is_flush": np.asarray(True, dtype=np.bool_),
                "embed_state": state["embed_state_out"],
                "conv_state": state["conv_state_out"],
                "pending": state["pending_out"],
            },
            expected_outputs,
        )
        tail_features, tail_frames = features_and_frames(tail)
        validate_states(tail)
        features = np.concatenate((first_features, tail_features))
        return _sopro_istft(features, first_frames + tail_frames, n_fft=n_fft, hop=hop)

    def _stream_condition(
        self, reference_tokens: np.ndarray, generated_tokens: np.ndarray, reference_frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        hop_ratio = self._runtime_config.hop_ratio
        tokens = np.concatenate((reference_tokens, generated_tokens.astype(np.int64, copy=False)))
        frames = reference_frames + generated_tokens.size * hop_ratio
        if tokens.size == 0 or frames <= 0:
            raise RuntimeContractError(
                "Sopro streaming condition requires non-empty tokens and frames"
            )
        frame_to_token = np.fromiter(
            (min(tokens.size - 1, (frame * tokens.size) // frames) for frame in range(frames)),
            dtype=np.int64,
            count=frames,
        )
        condition = self._run_contract_graph(
            "acoustic_condition",
            {
                "semantic_tokens": np.ascontiguousarray(tokens[None, :], dtype=np.int64),
                "frame_to_token": frame_to_token,
            },
            ("mu",),
        )["mu"]
        if condition.shape != (1, 100, frames) or not np.isfinite(condition).all():
            raise RuntimeContractError(
                "Sopro streaming condition output has an invalid shape or values"
            )
        return np.ascontiguousarray(condition, dtype=np.float32), tokens

    def _stream_prefill(
        self,
        *,
        steps: int,
        split: bool,
        frames: int,
        x_init: np.ndarray,
        mu: np.ndarray,
        cond_vec: np.ndarray,
        cond_mel: np.ndarray,
        cond_mask: np.ndarray,
        cosine: np.ndarray,
        sine: np.ndarray,
        chunk_bias: np.ndarray,
    ) -> _SoproAcousticState:
        shared = {
            "x_init": x_init,
            "mu": mu,
            "cond_vec": cond_vec,
            "cond_mel": cond_mel,
            "cond_mask": cond_mask,
            "cos": cosine,
            "sin": sine,
            "chunk_bias": chunk_bias,
        }
        output = self._run_contract_graph(
            f"acoustic_stream_prefill_{steps}",
            shared,
            ("mel", "x_context", "present_k", "present_v"),
        )
        if split:
            cache_context: list[np.ndarray] = []
            cache_k: list[np.ndarray] = []
            cache_v: list[np.ndarray] = []
            x = output["mel"]
            for index in range(steps):
                ode = self._run_contract_graph(
                    f"acoustic_stream_prefill_ode_{steps}_{index}",
                    {"x": x, **shared},
                    ("x_out", "mel", "x_context_out", "present_k", "present_v"),
                )
                x = ode["x_out"]
                cache_context.append(ode["x_context_out"])
                cache_k.append(ode["present_k"])
                cache_v.append(ode["present_v"])
            state = _SoproAcousticState(
                cached_frames=frames,
                x_context=np.ascontiguousarray(np.stack(cache_context), dtype=np.float32),
                past_k=np.ascontiguousarray(np.stack(cache_k)),
                past_v=np.ascontiguousarray(np.stack(cache_v)),
            )
        else:
            state = _SoproAcousticState(
                cached_frames=frames,
                x_context=np.ascontiguousarray(output["x_context"], dtype=np.float32),
                past_k=np.ascontiguousarray(output["present_k"]),
                past_v=np.ascontiguousarray(output["present_v"]),
            )
        self._validate_stream_acoustic_state(state, frames)
        return state

    def _validate_stream_acoustic_state(self, state: _SoproAcousticState, past_frames: int) -> None:
        context = self._runtime_config.position_context
        steps = self._runtime_config.fixed_steps
        if (
            context is None
            or steps <= 0
            or state.cached_frames < 0
            or state.x_context.shape != (steps, 1, 100, context)
            or state.past_k.ndim != 6
            or state.past_v.shape != state.past_k.shape
            or state.past_k.shape[0] != steps
            or state.past_k.shape[-2] < past_frames
            or not np.isfinite(state.x_context).all()
            or not np.isfinite(state.past_k).all()
            or not np.isfinite(state.past_v).all()
        ):
            raise RuntimeContractError(
                "Sopro acoustic streaming cache has invalid dimensions or values"
            )

    def _stream_acoustic_step(
        self,
        *,
        steps: int,
        split: bool,
        state: _SoproAcousticState,
        x_init: np.ndarray,
        mu_window: np.ndarray,
        cond_vec: np.ndarray,
        cond_mel_window: np.ndarray,
        cond_mask_window: np.ndarray,
        cosine: np.ndarray,
        sine: np.ndarray,
    ) -> tuple[np.ndarray, _SoproAcousticState]:
        new_frames = x_init.shape[-1]
        output = self._run_contract_graph(
            f"acoustic_stream_step_{steps}",
            {
                "x_init": x_init,
                "mu_window": mu_window,
                "cond_vec": cond_vec,
                "cond_mel_window": cond_mel_window,
                "cond_mask_window": cond_mask_window,
                "cos": cosine,
                "sin": sine,
                "x_context": state.x_context,
                "past_k": state.past_k,
                "past_v": state.past_v,
            },
            ("mel", "x_context_out", "present_k", "present_v"),
        )
        if output["mel"].shape != (1, 100, new_frames):
            raise RuntimeContractError(
                "Sopro acoustic streaming step returned invalid mel dimensions"
            )
        if not split:
            next_state = _SoproAcousticState(
                cached_frames=state.cached_frames + new_frames,
                x_context=np.ascontiguousarray(output["x_context_out"], dtype=np.float32),
                past_k=np.ascontiguousarray(output["present_k"]),
                past_v=np.ascontiguousarray(output["present_v"]),
            )
            self._validate_stream_acoustic_state(next_state, next_state.cached_frames)
            mel = np.asarray(output["mel"])
            if not np.isfinite(mel).all():
                raise RuntimeContractError(
                    "Sopro acoustic streaming output contains non-finite values"
                )
            return np.ascontiguousarray(mel, dtype=np.float32), next_state
        cache_context: list[np.ndarray] = []
        cache_k: list[np.ndarray] = []
        cache_v: list[np.ndarray] = []
        x = output["mel"]
        for index in range(steps):
            ode = self._run_contract_graph(
                f"acoustic_stream_ode_{steps}_{index}",
                {
                    "x": x,
                    "x_init": x_init,
                    "mu_window": mu_window,
                    "cond_vec": cond_vec,
                    "cond_mel_window": cond_mel_window,
                    "cond_mask_window": cond_mask_window,
                    "cos": cosine,
                    "sin": sine,
                    "x_context": output["x_context_out"][index],
                    "past_k": output["present_k"][index],
                    "past_v": output["present_v"][index],
                },
                ("x_out", "mel", "x_context_out", "present_k", "present_v"),
            )
            x = ode["x_out"]
            if ode["mel"].shape != (1, 100, new_frames):
                raise RuntimeContractError(
                    "Sopro acoustic streaming ODE returned invalid mel dimensions"
                )
            cache_context.append(ode["x_context_out"])
            cache_k.append(ode["present_k"])
            cache_v.append(ode["present_v"])
        next_state = _SoproAcousticState(
            cached_frames=state.cached_frames + new_frames,
            x_context=np.ascontiguousarray(np.stack(cache_context), dtype=np.float32),
            past_k=np.ascontiguousarray(np.stack(cache_k)),
            past_v=np.ascontiguousarray(np.stack(cache_v)),
        )
        self._validate_stream_acoustic_state(next_state, next_state.cached_frames)
        mel = np.asarray(ode["mel"])
        if not np.isfinite(mel).all():
            raise RuntimeContractError("Sopro acoustic streaming output contains non-finite values")
        return np.ascontiguousarray(mel, dtype=np.float32), next_state

    @property
    def stream_cache_capacity(self) -> int:
        return self._stream_cache_capacity

    @stream_cache_capacity.setter
    def stream_cache_capacity(self, value: int) -> None:
        if type(value) is not int or value < 0:
            raise RuntimeContractError("Sopro stream cache capacity must be a non-negative integer")
        self._stream_cache_capacity = value
        if value == 0:
            self._prompt_state_cache.clear()
            self._vocoder_prompt_cache.clear()
        else:
            while len(self._prompt_state_cache) > value:
                self._prompt_state_cache.popitem(last=False)
            while len(self._vocoder_prompt_cache) > value:
                self._vocoder_prompt_cache.popitem(last=False)

    @staticmethod
    def _readonly_cache_array(value: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(value).copy()
        result.flags.writeable = False
        return result

    def _store_stream_cache(
        self, cache: OrderedDict[tuple[Any, ...], Any], key: tuple[Any, ...], value: Any
    ) -> None:
        if self._stream_cache_capacity == 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._stream_cache_capacity:
            cache.popitem(last=False)

    def _voice_stream_cache_key(self, voice_state: SoproVoiceState) -> str:
        if voice_state.cache_key:
            return voice_state.cache_key
        digest = hashlib.sha256()
        for value in (str(self.manifest.get("revision")), self.profile):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for array in (voice_state.cond_vec, voice_state.semantic_tokens, voice_state.mel):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def _get_prompt_state(
        self,
        voice_state: SoproVoiceState,
        reference_tokens: np.ndarray,
        reference_frames: int,
        reference_mel: np.ndarray,
        stable_prompt: int,
        seed: int,
        mode: str,
    ) -> tuple[_SoproPromptState, bool]:
        config = self._runtime_config
        steps = config.fixed_steps
        dim_head = config.acoustic_dim_head
        if dim_head is None:
            raise ManifestError("Sopro acoustic rotary dimension is missing from the manifest")
        key = (
            self._voice_stream_cache_key(voice_state),
            seed,
            steps,
            stable_prompt,
            config.stream_chunk_frames,
        )
        if self._stream_cache_capacity:
            cached = self._prompt_state_cache.get(key)
            if cached is not None:
                self._prompt_state_cache.move_to_end(key)
                return cached, True
        condition, _ = self._stream_condition(
            reference_tokens, np.empty(0, dtype=np.int64), reference_frames
        )
        noise = _SoproNoise(seed)
        noise.ensure(stable_prompt)
        cond_mel = np.zeros((1, 100, stable_prompt), dtype=np.float32)
        cond_mel[0] = reference_mel[:, :stable_prompt]
        state = self._stream_prefill(
            steps=steps,
            split=mode == "split",
            frames=stable_prompt,
            x_init=noise.slice(0, stable_prompt),
            mu=np.ascontiguousarray(condition[:, :, :stable_prompt]),
            cond_vec=np.ascontiguousarray(voice_state.cond_vec[None, :]),
            cond_mel=cond_mel,
            cond_mask=np.ones((1, 1, stable_prompt), dtype=np.float32),
            cosine=_rotary(stable_prompt, dim_head)[0],
            sine=_rotary(stable_prompt, dim_head)[1],
            chunk_bias=_chunk_attention_bias(stable_prompt, config.stream_chunk_frames),
        )
        prompt = _SoproPromptState(
            cached_frames=state.cached_frames,
            x_context=self._readonly_cache_array(state.x_context),
            past_k=self._readonly_cache_array(state.past_k),
            past_v=self._readonly_cache_array(state.past_v),
            split_ode=mode == "split",
            mu=self._readonly_cache_array(condition),
            mu_frames=reference_frames,
        )
        self._store_stream_cache(self._prompt_state_cache, key, prompt)
        return prompt, False

    def _get_vocoder_prompt_state(
        self, voice_state: SoproVoiceState, reference_mel: np.ndarray
    ) -> tuple[_SoproVocoderPromptState, bool]:
        config = self._runtime_config
        if any(
            value is None
            for value in (
                config.vocoder_n_fft,
                config.vocoder_dim,
                config.vocoder_layers,
                config.vocoder_lookahead,
            )
        ):
            raise ManifestError("Sopro vocoder streaming dimensions are missing from the manifest")
        n_fft = cast(int, config.vocoder_n_fft)
        dimension = cast(int, config.vocoder_dim)
        layers = cast(int, config.vocoder_layers)
        lookahead = cast(int, config.vocoder_lookahead)
        warm_frames = min(32, reference_mel.shape[1])
        key = (self._voice_stream_cache_key(voice_state), self.profile, warm_frames)
        if self._stream_cache_capacity:
            cached = self._vocoder_prompt_cache.get(key)
            if cached is not None:
                self._vocoder_prompt_cache.move_to_end(key)
                return cached, True
        model = self._frontend_config().get("model")
        if not isinstance(model, Mapping):
            raise ManifestError("Sopro manifest is missing acoustic mel statistics")
        mean = np.asarray(model.get("acoustic_mel_mean"), dtype=np.float32)
        std = np.asarray(model.get("acoustic_mel_std"), dtype=np.float32)
        initial = _SoproVocoderState(
            embed_state=np.zeros((1, 100, 6), dtype=np.float32),
            conv_state=np.zeros((layers, 1, dimension, 6), dtype=np.float32),
            pending=np.zeros((layers, 1, dimension, lookahead), dtype=np.float32),
        )
        warm_mel = reference_mel[:, -warm_frames:] * std[:, None] + mean[:, None]
        output = self._run_contract_graph(
            "vocoder",
            {
                "mel": np.ascontiguousarray(warm_mel[None, :, :], dtype=np.float32),
                "is_start": np.asarray(True, dtype=np.bool_),
                "is_flush": np.asarray(False, dtype=np.bool_),
                "embed_state": initial.embed_state,
                "conv_state": initial.conv_state,
                "pending": initial.pending,
            },
            ("istft_features", "embed_state_out", "conv_state_out", "pending_out"),
        )
        features = np.asarray(output["istft_features"])
        states = (
            np.asarray(output["embed_state_out"]),
            np.asarray(output["conv_state_out"]),
            np.asarray(output["pending_out"]),
        )
        expected = ((1, 100, 6), (layers, 1, dimension, 6), (layers, 1, dimension, lookahead))
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[2] != n_fft + 2
            or not np.isfinite(features).all()
            or any(
                state.shape != shape
                or not np.issubdtype(state.dtype, np.floating)
                or not np.isfinite(state).all()
                for state, shape in zip(states, expected, strict=True)
            )
        ):
            raise RuntimeContractError("Sopro vocoder prompt warmup returned invalid tensors")
        prompt = _SoproVocoderPromptState(
            warm_frames=warm_frames,
            istft_features=self._readonly_cache_array(features),
            embed_state=self._readonly_cache_array(states[0].astype(np.float32)),
            conv_state=self._readonly_cache_array(states[1].astype(np.float32)),
            pending=self._readonly_cache_array(states[2].astype(np.float32)),
        )
        self._store_stream_cache(self._vocoder_prompt_cache, key, prompt)
        return prompt, False

    def prepare_streaming(
        self,
        voice_state: SoproVoiceState,
        *,
        seed: int = 1,
        steps: int | None = None,
    ) -> None:
        """Synchronously prewarm bounded prompt/vocoder state and streaming sessions."""
        if not isinstance(voice_state, SoproVoiceState):
            raise RuntimeContractError("Sopro streaming preparation requires a prepared voice")
        config = self._runtime_config
        self._semantic_generation_options({"seed": seed})
        if steps is not None and (type(steps) is not int or steps != config.fixed_steps):
            raise CapabilityError(
                "Sopro streaming steps must match the installed fixed-step graphs"
            )
        providers = (
            (self.providers,) if isinstance(self.providers, str) else tuple(self.providers or ())
        )
        if self.profile != "wasm-uint8" or providers not in {
            ("cpu",),
            ("CPUExecutionProvider",),
        }:
            raise CapabilityError(
                "Sopro streaming prewarm is parity-gated for wasm-uint8 on CPUExecutionProvider"
            )
        mode, graphs = _select_stream_graph_mode(self.graphs, config.fixed_steps)
        for graph in (*graphs, "acoustic_condition", "vocoder"):
            self.session(graph)
        if config.position_context is None or config.acoustic_pre_lookahead_frames is None:
            raise ManifestError("Sopro streaming prompt dimensions are missing from the manifest")
        hop_ratio = config.hop_ratio
        aligned_tokens = min(
            voice_state.semantic_tokens.size, voice_state.mel.shape[1] // hop_ratio
        )
        reference_frames = (voice_state.mel.shape[1] // hop_ratio) * hop_ratio
        if reference_frames <= 0 or aligned_tokens <= 0:
            raise RuntimeContractError("Sopro prepared voice has no aligned reference prompt")
        reference_tokens = np.ascontiguousarray(voice_state.semantic_tokens[:aligned_tokens])
        reference_mel = np.ascontiguousarray(voice_state.mel[:, :reference_frames])
        stable_prompt = _stable_prompt_frames(
            reference_frames,
            aligned_tokens,
            config.acoustic_pre_lookahead_frames,
            hop_ratio,
            config.stream_chunk_frames,
        )
        if stable_prompt >= config.position_context:
            self._get_prompt_state(
                voice_state,
                reference_tokens,
                reference_frames,
                reference_mel,
                stable_prompt,
                seed,
                mode,
            )
        self._get_vocoder_prompt_state(voice_state, reference_mel)

    def _stream_one_segment(
        self,
        text_input: SoproTextInput,
        voice_state: SoproVoiceState,
        options: _SemanticOptions,
        prompt_tokens: Sequence[int] | np.ndarray | None,
        *,
        segment_index: int,
    ) -> Generator[SoproStreamChunk, None, np.ndarray | None]:
        stream_started = time.perf_counter()
        config = self._runtime_config
        steps = config.fixed_steps
        if config.position_context is None or config.acoustic_pre_lookahead_frames is None:
            raise ManifestError("Sopro streaming prompt dimensions are missing from the manifest")
        position_context = config.position_context
        model = self._frontend_config().get("model")
        vocoder_config = self._frontend_config().get("vocoder")
        if not isinstance(model, Mapping) or not isinstance(vocoder_config, Mapping):
            raise ManifestError("Sopro streaming model and vocoder configuration are required")
        mean = np.asarray(model.get("acoustic_mel_mean"), dtype=np.float32)
        std = np.asarray(model.get("acoustic_mel_std"), dtype=np.float32)
        dim_head = config.acoustic_dim_head
        n_fft = config.vocoder_n_fft
        vocoder_hop = config.vocoder_hop_length
        dimension = config.vocoder_dim
        layers = config.vocoder_layers
        lookahead = config.vocoder_lookahead
        if (
            mean.shape != (100,)
            or std.shape != (100,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
            or None in (dim_head, n_fft, vocoder_hop, dimension, layers, lookahead)
        ):
            raise ManifestError("Sopro streaming acoustic or vocoder dimensions are invalid")
        dim_head = cast(int, dim_head)
        n_fft = cast(int, n_fft)
        vocoder_hop = cast(int, vocoder_hop)
        dimension = cast(int, dimension)
        layers = cast(int, layers)
        lookahead = cast(int, lookahead)
        hop_ratio = config.hop_ratio
        stable_prompt = _stable_prompt_frames(
            voice_state.mel.shape[1],
            int(voice_state.semantic_tokens.size),
            config.acoustic_pre_lookahead_frames,
            hop_ratio,
            config.stream_chunk_frames,
        )
        if stable_prompt < config.position_context:
            fallback_started = time.perf_counter()
            result = self._synthesize_one_segment(
                text_input,
                voice_state,
                options=options,
                prompt_tokens=prompt_tokens,
            )
            generated = np.asarray(
                result.metadata.get("generated_semantic_tokens", ()), dtype=np.int64
            )
            prompt_limit = result.metadata.get("prompt_token_limit", 0)
            carry = (
                generated[-prompt_limit:].copy()
                if type(prompt_limit) is int and prompt_limit > 0 and generated.size
                else None
            )
            fallback_elapsed_ms = (time.perf_counter() - fallback_started) * 1000.0
            metadata = {
                "system": "sopro",
                "profile": self.profile,
                "revision": self.manifest.get("revision"),
                "streaming": True,
                "streaming_fallback": "short_reference",
                "segment_index": segment_index,
                "chunk_index": 0,
                "seed": options.seed,
                "steps": steps,
                "semantic_tokens_total": int(generated.size),
                "acoustic_cached_frames": int(result.metadata.get("acoustic_frames", 0)),
                "audio_samples": int(result.audio.size),
                "reference_prepare_ms": voice_state.reference_prepare_ms,
                "stream_prompt_prepare_ms": 0.0,
                "first_audio_chunk_ms": fallback_elapsed_ms,
                "total_stream_ms": fallback_elapsed_ms,
                "chunk_count": 1,
                "generated_semantic_tokens": int(generated.size),
                "is_final": True,
            }
            yield SoproStreamChunk(
                audio=np.ascontiguousarray(result.audio, dtype=np.float32),
                sample_rate=result.sample_rate,
                is_final=True,
                metadata=metadata,
            )
            return carry

        mode, _ = _select_stream_graph_mode(self.graphs, steps)
        aligned_tokens = min(
            int(voice_state.semantic_tokens.size), voice_state.mel.shape[1] // hop_ratio
        )
        reference_tokens = np.ascontiguousarray(voice_state.semantic_tokens[:aligned_tokens])
        reference_frames = (voice_state.mel.shape[1] // hop_ratio) * hop_ratio
        reference_mel = np.ascontiguousarray(voice_state.mel[:, :reference_frames])
        if reference_tokens.size == 0 or reference_frames == 0:
            raise RuntimeContractError("Sopro stream reference has no aligned semantic/mel frames")
        decoder = self._create_semantic_decoder(
            text_input,
            voice_state,
            options=options,
            prompt_token_override=prompt_tokens,
        )
        try:
            prompt_prepare_started = time.perf_counter()
            prompt_state, prompt_cache_hit = self._get_prompt_state(
                voice_state,
                reference_tokens,
                reference_frames,
                reference_mel,
                stable_prompt,
                options.seed,
                mode,
            )
            condition = prompt_state.mu
            state = _SoproAcousticState(
                cached_frames=prompt_state.cached_frames,
                x_context=prompt_state.x_context,
                past_k=prompt_state.past_k,
                past_v=prompt_state.past_v,
            )
            x_noise = _SoproNoise(options.seed)
            x_noise.ensure(reference_frames)
            cond_mel = np.zeros((1, 100, reference_frames), dtype=np.float32)
            cond_mel[0] = reference_mel
            vocoder_prompt, vocoder_cache_hit = self._get_vocoder_prompt_state(
                voice_state, reference_mel
            )
            istft = _StreamingISTFT(int(n_fft), int(vocoder_hop))
            warm_features = vocoder_prompt.istft_features
            warm_audio = istft.push(warm_features, warm_features.shape[1], flush=False)
            vocoder_state = _SoproVocoderState(
                embed_state=vocoder_prompt.embed_state,
                conv_state=vocoder_prompt.conv_state,
                pending=vocoder_prompt.pending,
            )
            warm_frames = vocoder_prompt.warm_frames
            skip_samples = max(0, warm_frames * int(vocoder_hop) - warm_audio.size)
            stream_prompt_prepare_ms = (time.perf_counter() - prompt_prepare_started) * 1000.0
            emitted_samples = 0
            chunk_index = 0
            semantic_first_token_ms: float | None = None
            first_audio_chunk_ms: float | None = None
            condition_token_count = 0

            def feed_vocoder(mel_values: np.ndarray, *, start: bool, flush: bool) -> np.ndarray:
                nonlocal vocoder_state, skip_samples
                mel_values = np.asarray(mel_values, dtype=np.float32)
                if (
                    mel_values.ndim != 2
                    or mel_values.shape[0] != 100
                    or not np.isfinite(mel_values).all()
                ):
                    raise RuntimeContractError(
                        "Sopro streaming vocoder mel has invalid shape or values"
                    )
                outputs = self._run_contract_graph(
                    "vocoder",
                    {
                        "mel": np.ascontiguousarray(mel_values[None, :, :]),
                        "is_start": np.asarray(start, dtype=np.bool_),
                        "is_flush": np.asarray(flush, dtype=np.bool_),
                        "embed_state": vocoder_state.embed_state,
                        "conv_state": vocoder_state.conv_state,
                        "pending": vocoder_state.pending,
                    },
                    ("istft_features", "embed_state_out", "conv_state_out", "pending_out"),
                )
                states = (
                    np.asarray(outputs["embed_state_out"]),
                    np.asarray(outputs["conv_state_out"]),
                    np.asarray(outputs["pending_out"]),
                )
                expected = (
                    (1, 100, 6),
                    (int(layers), 1, int(dimension), 6),
                    (int(layers), 1, int(dimension), int(lookahead)),
                )
                if any(
                    value.shape != shape
                    or not np.issubdtype(value.dtype, np.floating)
                    or not np.isfinite(value).all()
                    for value, shape in zip(states, expected, strict=True)
                ):
                    raise RuntimeContractError(
                        "Sopro streaming vocoder returned invalid state tensors"
                    )
                vocoder_state = _SoproVocoderState(
                    *(np.ascontiguousarray(value, dtype=np.float32) for value in states)
                )
                features = np.asarray(outputs["istft_features"])
                if (
                    features.ndim != 3
                    or features.shape[0] != 1
                    or features.shape[2] != int(n_fft) + 2
                    or not np.isfinite(features).all()
                ):
                    raise RuntimeContractError(
                        "Sopro streaming vocoder features have invalid dimensions or values"
                    )
                audio = istft.push(features, features.shape[1], flush=flush)
                if skip_samples:
                    drop = min(skip_samples, audio.size)
                    audio = audio[drop:]
                    skip_samples -= drop
                return np.ascontiguousarray(audio, dtype=np.float32)

            def process_until(
                target_frames: int, generated: np.ndarray
            ) -> Iterator[tuple[np.ndarray, int]]:
                nonlocal state, condition, condition_token_count
                if generated.size != condition_token_count:
                    condition, _ = self._stream_condition(
                        reference_tokens, generated, reference_frames
                    )
                    condition_token_count = int(generated.size)
                stream_cond_mel = np.zeros((1, 100, condition.shape[-1]), dtype=np.float32)
                stream_cond_mask = np.zeros((1, 1, condition.shape[-1]), dtype=np.float32)
                stream_cond_mel[:, :, :reference_frames] = reference_mel[None, :, :]
                stream_cond_mask[:, :, :reference_frames] = 1.0
                while state.cached_frames < target_frames:
                    cached = state.cached_frames
                    count = min(config.stream_chunk_frames, target_frames - cached)
                    end = cached + count
                    start = cached - position_context
                    if start < 0 or end > condition.shape[-1]:
                        raise RuntimeContractError(
                            "Sopro streaming acoustic window exceeds condition frames"
                        )
                    mel_values, next_state = self._stream_acoustic_step(
                        steps=steps,
                        split=mode == "split",
                        state=state,
                        x_init=x_noise.slice(cached, end),
                        mu_window=np.ascontiguousarray(condition[:, :, start:end]),
                        cond_vec=np.ascontiguousarray(voice_state.cond_vec[None, :]),
                        cond_mel_window=np.ascontiguousarray(stream_cond_mel[:, :, start:end]),
                        cond_mask_window=np.ascontiguousarray(stream_cond_mask[:, :, start:end]),
                        cosine=_rotary(count, int(dim_head), offset=cached)[0],
                        sine=_rotary(count, int(dim_head), offset=cached)[1],
                    )
                    state = next_state
                    denormalized = mel_values[0] * std[:, None] + mean[:, None]
                    audio = feed_vocoder(denormalized, start=False, flush=False)
                    yield audio, state.cached_frames

            for _tokens in decoder.iter_chunks(1):
                generated = decoder.generated
                if semantic_first_token_ms is None:
                    semantic_first_token_ms = (time.perf_counter() - stream_started) * 1000.0
                x_noise.ensure(reference_frames + generated.size * hop_ratio)
                target = _stream_target_frames(
                    reference_frames,
                    int(generated.size),
                    hop_ratio,
                    config.acoustic_pre_lookahead_frames,
                    config.stream_chunk_frames,
                    final=False,
                )
                for audio, cached_frames in (
                    process_until(target, generated) if target > state.cached_frames else ()
                ):
                    target_audio_samples = int(generated.size) * config.token_samples
                    remaining = max(0, target_audio_samples - emitted_samples)
                    chunk_audio = np.ascontiguousarray(audio[:remaining], dtype=np.float32)
                    if chunk_audio.size:
                        if first_audio_chunk_ms is None:
                            first_audio_chunk_ms = (time.perf_counter() - stream_started) * 1000.0
                        metadata = {
                            "system": "sopro",
                            "profile": self.profile,
                            "revision": self.manifest.get("revision"),
                            "streaming": True,
                            "segment_index": segment_index,
                            "chunk_index": chunk_index,
                            "seed": options.seed,
                            "steps": steps,
                            "semantic_tokens_total": int(generated.size),
                            "acoustic_cached_frames": cached_frames,
                            "audio_samples": int(chunk_audio.size),
                            "is_final": False,
                        }
                        emitted_samples += chunk_audio.size
                        chunk_index += 1
                        yield SoproStreamChunk(chunk_audio, self.sample_rate, False, metadata)

            semantic = decoder.result
            generated = semantic.tokens
            if generated.size == 0:
                return None
            x_noise.ensure(reference_frames + generated.size * hop_ratio)
            final_target = _stream_target_frames(
                reference_frames,
                int(generated.size),
                hop_ratio,
                config.acoustic_pre_lookahead_frames,
                config.stream_chunk_frames,
                final=True,
            )
            for audio, cached_frames in (
                process_until(final_target, generated) if final_target > state.cached_frames else ()
            ):
                target_audio_samples = int(generated.size) * config.token_samples
                remaining = max(0, target_audio_samples - emitted_samples)
                chunk_audio = np.ascontiguousarray(audio[:remaining], dtype=np.float32)
                if chunk_audio.size:
                    if first_audio_chunk_ms is None:
                        first_audio_chunk_ms = (time.perf_counter() - stream_started) * 1000.0
                    metadata = {
                        "system": "sopro",
                        "profile": self.profile,
                        "revision": self.manifest.get("revision"),
                        "streaming": True,
                        "segment_index": segment_index,
                        "chunk_index": chunk_index,
                        "seed": options.seed,
                        "steps": steps,
                        "semantic_tokens_total": int(generated.size),
                        "acoustic_cached_frames": cached_frames,
                        "audio_samples": int(chunk_audio.size),
                        "is_final": False,
                    }
                    emitted_samples += chunk_audio.size
                    chunk_index += 1
                    yield SoproStreamChunk(chunk_audio, self.sample_rate, False, metadata)
            tail = feed_vocoder(np.empty((100, 0), dtype=np.float32), start=False, flush=True)
            target_audio_samples = int(generated.size) * config.token_samples
            final_audio = np.ascontiguousarray(
                tail[: max(0, target_audio_samples - emitted_samples)], dtype=np.float32
            )
            emitted_samples += final_audio.size
            if emitted_samples != target_audio_samples:
                raise RuntimeContractError(
                    f"Sopro streaming audio length {emitted_samples} does not match target {target_audio_samples}"
                )
            prompt_limit = semantic.prompt_token_limit
            carry = generated[-prompt_limit:].copy() if prompt_limit > 0 else None
            total_stream_ms = (time.perf_counter() - stream_started) * 1000.0
            if first_audio_chunk_ms is None:
                first_audio_chunk_ms = total_stream_ms
            metadata = {
                "system": "sopro",
                "profile": self.profile,
                "revision": self.manifest.get("revision"),
                "streaming": True,
                "segment_index": segment_index,
                "chunk_index": chunk_index,
                "seed": options.seed,
                "steps": steps,
                "semantic_tokens_total": int(generated.size),
                "acoustic_cached_frames": state.cached_frames,
                "audio_samples": int(final_audio.size),
                "reference_prepare_ms": voice_state.reference_prepare_ms,
                "stream_prompt_prepare_ms": stream_prompt_prepare_ms,
                "semantic_first_token_ms": semantic_first_token_ms,
                "first_audio_chunk_ms": first_audio_chunk_ms,
                "total_stream_ms": total_stream_ms,
                "chunk_count": chunk_index + 1,
                "generated_semantic_tokens": int(generated.size),
                "prompt_cache_hit": prompt_cache_hit,
                "vocoder_cache_hit": vocoder_cache_hit,
                "is_final": True,
            }
            yield SoproStreamChunk(final_audio, self.sample_rate, True, metadata)
            return carry
        finally:
            decoder.close()

    def _synthesize_one_segment(
        self,
        text_input: SoproTextInput,
        voice_state: SoproVoiceState,
        *,
        options: _SemanticOptions | Mapping[str, Any] | None = None,
        prompt_tokens: Sequence[int] | np.ndarray | None = None,
    ) -> InferenceResult:
        settings = self._semantic_generation_options(options)
        decoder = self._create_semantic_decoder(
            text_input,
            voice_state,
            options=options,
            prompt_token_override=prompt_tokens,
        )
        try:
            for _chunk in decoder.iter_chunks(32):
                pass
            semantic = decoder.result
        finally:
            decoder.close()
        runtime_config = self._runtime_config
        token_samples = runtime_config.token_samples
        hop_ratio = runtime_config.hop_ratio
        steps = runtime_config.fixed_steps
        model = self._frontend_config().get("model")
        if not isinstance(model, Mapping):
            raise ManifestError("Sopro manifest is missing acoustic model configuration")
        if semantic.tokens.size == 0:
            return InferenceResult(
                audio=np.empty(0, dtype=np.float32),
                sample_rate=self.sample_rate,
                metadata={
                    "system": "sopro",
                    "profile": self.profile,
                    "revision": self.manifest.get("revision"),
                    "language": text_input.language,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "top_k": settings.top_k,
                    "max_seconds": settings.max_seconds,
                    "min_seconds": settings.min_seconds,
                    "seed": settings.seed,
                    "steps": steps,
                    "semantic_steps": semantic.sampled_steps,
                    "semantic_tokens": 0,
                    "generated_semantic_tokens": tuple(int(token) for token in semantic.tokens),
                    "prompt_token_limit": semantic.prompt_token_limit,
                    "stopped_on_eos": semantic.stopped_on_eos,
                    "sample_rate": self.sample_rate,
                    "audio_samples": 0,
                },
            )

        reference_frames = voice_state.mel.shape[1]
        if (
            voice_state.mel.shape != (100, reference_frames)
            or reference_frames <= 0
            or not np.isfinite(voice_state.mel).all()
        ):
            raise RuntimeContractError("Sopro reference mel state has an invalid shape or values")
        all_tokens = np.concatenate((voice_state.semantic_tokens, semantic.tokens.astype(np.int64)))
        frames = reference_frames + semantic.tokens.size * hop_ratio
        frame_to_token = np.fromiter(
            (
                min(all_tokens.size - 1, (frame * all_tokens.size) // frames)
                for frame in range(frames)
            ),
            dtype=np.int64,
            count=frames,
        )
        condition = self._run_contract_graph(
            "acoustic_condition",
            {
                "semantic_tokens": np.ascontiguousarray(all_tokens[None, :], dtype=np.int64),
                "frame_to_token": frame_to_token,
            },
            ("mu",),
        )["mu"]
        if (
            condition.shape != (1, 100, frames)
            or not np.issubdtype(condition.dtype, np.floating)
            or not np.isfinite(condition).all()
        ):
            raise RuntimeContractError(
                "Sopro acoustic conditioning output has an invalid shape or values"
            )

        mean = np.asarray(model.get("acoustic_mel_mean"), dtype=np.float64)
        std = np.asarray(model.get("acoustic_mel_std"), dtype=np.float64)
        acoustic_dim = model.get("acoustic_dit_dim_head")
        if (
            mean.shape != (100,)
            or std.shape != (100,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0.0)
            or type(acoustic_dim) is not int
            or acoustic_dim <= 0
            or acoustic_dim % 2
        ):
            raise ManifestError("Sopro acoustic mel statistics and rotary dimensions are invalid")
        noise = _SoproRandom(settings.seed ^ 0xA53C9E1D)
        x_init = np.fromiter(
            (noise.normal() for _ in range(100 * frames)), dtype=np.float32, count=100 * frames
        ).reshape(1, 100, frames)
        cond_mel = np.zeros((1, 100, frames), dtype=np.float32)
        cond_mel[0, :, :reference_frames] = voice_state.mel
        cond_mask = np.zeros((1, 1, frames), dtype=np.float32)
        cond_mask[0, 0, :reference_frames] = 1.0
        cosine, sine = _rotary(frames, acoustic_dim)
        graph = f"acoustic_offline_{steps}"
        acoustic = self._run_contract_graph(
            graph,
            {
                "x_init": x_init,
                "mu": np.ascontiguousarray(condition, dtype=np.float32),
                "cond_vec": np.ascontiguousarray(voice_state.cond_vec[None, :], dtype=np.float32),
                "cond_mel": cond_mel,
                "cond_mask": cond_mask,
                "cos": cosine,
                "sin": sine,
            },
            ("mel",),
        )["mel"]
        if (
            acoustic.shape != (1, 100, frames)
            or not np.issubdtype(acoustic.dtype, np.floating)
            or not np.isfinite(acoustic).all()
        ):
            raise RuntimeContractError(
                "Sopro offline acoustic output has an invalid shape or values"
            )

        context = min(32, reference_frames)
        decode_frames = context + semantic.tokens.size * hop_ratio
        decode_start = reference_frames - context
        mel = np.ascontiguousarray(acoustic[0, :, decode_start:], dtype=np.float32)
        if mel.shape != (100, decode_frames):
            raise RuntimeContractError(
                "Sopro offline acoustic output cannot be cropped to the decoder context"
            )
        for channel in range(100):
            mel[channel] = mel[channel] * std[channel] + mean[channel]
        waveform = self._decode_vocoder(mel)
        start = context * int(self._frontend_config()["vocoder"]["hop_length"])
        target = semantic.tokens.size * token_samples
        audio = np.ascontiguousarray(
            waveform[start : min(waveform.size, start + target)], dtype=np.float32
        )
        if not np.isfinite(audio).all():
            raise RuntimeContractError("Sopro generated audio contains non-finite samples")
        return InferenceResult(
            audio=audio,
            sample_rate=self.sample_rate,
            metadata={
                "system": "sopro",
                "profile": self.profile,
                "revision": self.manifest.get("revision"),
                "language": text_input.language,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "max_seconds": settings.max_seconds,
                "min_seconds": settings.min_seconds,
                "seed": settings.seed,
                "steps": steps,
                "semantic_steps": semantic.sampled_steps,
                "semantic_tokens": int(semantic.tokens.size),
                "generated_semantic_tokens": tuple(int(token) for token in semantic.tokens),
                "prompt_token_limit": semantic.prompt_token_limit,
                "stopped_on_eos": semantic.stopped_on_eos,
                "reference_tokens": int(voice_state.semantic_tokens.size),
                "acoustic_frames": frames,
                "sample_rate": self.sample_rate,
                "audio_samples": int(audio.size),
            },
        )

    def begin_run(
        self,
        voice_state: SoproVoiceState,
        *,
        language: str | None = None,
        **options: Any,
    ) -> SoproRun:
        """Create a segment run with validated, reusable generation settings."""
        if not isinstance(voice_state, SoproVoiceState):
            raise RuntimeContractError("Sopro run requires a prepared SoproVoiceState")
        requested = self.providers
        provider_names = (requested,) if isinstance(requested, str) else tuple(requested or ())
        if self.profile != "wasm-uint8" or provider_names not in {
            ("cpu",),
            ("CPUExecutionProvider",),
        }:
            raise CapabilityError(
                "Sopro runs are parity-gated for wasm-uint8 on CPUExecutionProvider"
            )
        if language is not None and language not in {"auto", "en", "pt", "fr", "de"}:
            raise CapabilityError(f"Unsupported Sopro language {language!r}")
        normalized_language = None if language in {None, "auto"} else language
        resolved = self._semantic_generation_options(options)
        return SoproRun(self, voice_state, normalized_language, resolved)

    def close(self) -> None:
        """Invalidate active runs and close all owned ONNX Runtime sessions."""
        self._prompt_state_cache.clear()
        self._vocoder_prompt_cache.clear()
        self._generation_token += 1
        super().close()

    def infer(self, token_ids: Sequence[int], **kwargs: Any) -> InferenceResult:
        """Generate one Sopro segment as raw mono float32 PCM.

        Omitted ``prompt_tokens`` preserves reference prompts. Results carry the generated
        semantic tokens and configured prompt limit in metadata.
        """
        if self.profile != "wasm-uint8":
            raise CapabilityError(
                f"Native Python Sopro inference is only parity-gated for wasm-uint8, not {self.profile!r}"
            )
        requested = self.providers
        names = (requested,) if isinstance(requested, str) else tuple(requested or ())
        if names not in {("cpu",), ("CPUExecutionProvider",)}:
            raise CapabilityError(
                "Native Python Sopro inference is parity-gated only for CPUExecutionProvider; "
                "open the runtime with provider='cpu'"
            )
        allowed = {
            "voice_state",
            "language",
            "temperature",
            "top_p",
            "top_k",
            "max_seconds",
            "min_seconds",
            "seed",
            "steps",
            "prompt_tokens",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise RuntimeContractError(f"Unsupported Sopro inference option(s): {sorted(unknown)}")
        voice_state = kwargs.pop("voice_state", None)
        prompt_tokens = kwargs.pop("prompt_tokens", None)
        if not isinstance(voice_state, SoproVoiceState):
            raise RuntimeContractError("Sopro inference requires a prepared SoproVoiceState")
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, (Sequence, np.ndarray)):
            raise RuntimeContractError(
                "Sopro inference token_ids must be a one-dimensional integer sequence"
            )
        ids = np.asarray(token_ids)
        model = self._frontend_config().get("model")
        max_text_len = model.get("max_text_len") if isinstance(model, Mapping) else None
        vocab_size = model.get("text_vocab_size") if isinstance(model, Mapping) else None
        if (
            ids.ndim != 1
            or ids.size == 0
            or type(max_text_len) is not int
            or type(vocab_size) is not int
            or ids.size > max_text_len
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= vocab_size)
        ):
            raise RuntimeContractError(
                "Sopro inference token_ids have an invalid rank, length, dtype, or value"
            )
        language = kwargs.pop("language", None)
        if language is not None:
            if not isinstance(language, str):
                raise RuntimeContractError("Sopro language must be auto, en, pt, fr, or de")
            language = language.strip().lower()
            if language == "auto" or not language:
                language = None
            elif language not in {"en", "pt", "fr", "de"}:
                raise CapabilityError(f"Unsupported Sopro language {language!r}")
        text_input = SoproTextInput(
            normalized_text="",
            token_ids=tuple(int(token_id) for token_id in ids),
            language=language,
        )
        return self._synthesize_one_segment(
            text_input,
            voice_state,
            options=kwargs,
            prompt_tokens=prompt_tokens,
        )

    def diagnostics(self) -> RuntimeDiagnostic:
        """Return diagnostics for every graph session created so far."""
        return super().diagnostics()


class SoproRun:
    """Mutable, non-thread-safe state for sequential Sopro text segments."""

    def __init__(
        self,
        adapter: SoproAdapter,
        voice: SoproVoiceState,
        language: str | None,
        options: _SemanticOptions,
    ) -> None:
        self._adapter_ref = weakref.ref(adapter)
        self._adapter_generation = adapter._generation_token
        self._voice = voice
        self._language = language
        self._options = options
        self._carry: np.ndarray | None = None
        self._segment_index = 0
        self._closed = False

    def _adapter(self) -> SoproAdapter:
        adapter = self._adapter_ref()
        if self._closed or adapter is None or adapter._generation_token != self._adapter_generation:
            raise RuntimeContractError("Sopro run is closed or its adapter has been closed")
        return adapter

    @property
    def voice(self) -> SoproVoiceState:
        self._adapter()
        return self._voice

    @property
    def segment_index(self) -> int:
        return self._segment_index

    def infer_segment(self, text: str, *, language: str | None = None) -> InferenceResult:
        """Synthesize one segment and carry its semantic tail into the next."""
        adapter = self._adapter()
        effective_language = self._language if language is None else language
        text_input = adapter.prepare_text(text, effective_language)
        result = adapter._synthesize_one_segment(
            text_input,
            self._voice,
            options=self._options,
            prompt_tokens=self._carry,
        )
        generated = np.asarray(result.metadata.get("generated_semantic_tokens", ()), dtype=np.int64)
        prompt_limit = result.metadata.get("prompt_token_limit", 0)
        if type(prompt_limit) is int and prompt_limit > 0 and generated.size:
            self._carry = generated[-prompt_limit:].copy()
        else:
            self._carry = None
        self._segment_index += 1
        return result

    def stream_segment(
        self, text: str, *, language: str | None = None
    ) -> Generator[SoproStreamChunk]:
        """Synchronously stream raw audio chunks for one text segment."""
        adapter = self._adapter()
        effective_language = self._language if language is None else language
        text_input = adapter.prepare_text(text, effective_language)
        stream = adapter._stream_one_segment(
            text_input,
            self._voice,
            self._options,
            self._carry,
            segment_index=self._segment_index,
        )
        try:
            while True:
                try:
                    chunk = next(stream)
                except StopIteration as completed:
                    self._carry = completed.value
                    self._segment_index += 1
                    return
                if chunk.is_final:
                    try:
                        carry = next(stream)
                    except StopIteration as completed:
                        self._carry = completed.value
                    else:
                        raise RuntimeContractError(
                            f"Sopro stream yielded unexpected data after final chunk: {carry!r}"
                        )
                    self._segment_index += 1
                    yield chunk
                    return
                yield chunk
        finally:
            stream.close()

    def close(self) -> None:
        """Release run-local carry; repeated close calls are safe."""
        self._closed = True
        self._carry = None
