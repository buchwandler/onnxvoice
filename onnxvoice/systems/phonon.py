"""Low-level streaming ONNX runtime for Phonon token IDs to PCM."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import (
    InferenceResult,
    Installation,
    RuntimeDiagnostic,
    TensorSpec,
    validate_safe_component,
)
from .base import SystemAdapter

_GRAPH_FILES = {
    "text_conditioner": "onnx/text_conditioner.onnx",
    "flow_lm_main": "onnx/flow_lm_main.onnx",
    "flow_lm_flow": "onnx/flow_lm_flow.onnx",
    "mimi_decoder": "onnx/mimi_decoder.onnx",
}
_GRAPH_COMPONENTS = tuple(_GRAPH_FILES)
_RUNTIME_KIND = "phonon-streaming-onnx-v1"
_SUPPORTED_DTYPES: dict[str, np.dtype[Any]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}
_STATE_DTYPE_NAMES = {dtype.name: dtype for dtype in _SUPPORTED_DTYPES.values()}
_KV_INPUT_RE = re.compile(r"^past_key_values\.(\d+)\.(key|value)$")
_KV_OUTPUT_RE = re.compile(r"^present\.(\d+)\.(key|value)$")
_STATE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_GENERATION_ALIASES = {
    "temperature_default": ("temperature_default", "temperature"),
    "eos_threshold_default": ("eos_threshold_default", "eos_threshold"),
    "duration_delta_default": ("duration_delta_default", "duration_delta"),
    "frames_after_eos": ("frames_after_eos",),
    "min_frames_before_eos": ("min_frames_before_eos",),
    "max_frames": ("max_frames", "max_frames_default"),
}
_RUNTIME_ALIASES = {
    "sample_rate": ("sample_rate", "sampling_rate"),
    "frame_samples": ("frame_samples",),
    "frame_rate": ("frame_rate",),
    "d_model": ("d_model", "hidden_size"),
    "latent_dim": ("latent_dim",),
    "num_layers": ("num_layers", "flow_layers", "num_flow_layers"),
    "num_heads": ("num_heads", "heads"),
    "head_dim": ("head_dim",),
    "mimi_state": ("mimi_state",),
}


@dataclass(frozen=True, slots=True)
class PhononVoiceState:
    """A validated, reusable projected Phonon voice prompt."""

    name: str
    embeddings: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RuntimeContractError("Phonon voice state name must be a non-empty string")
        embeddings = np.array(self.embeddings, dtype=np.float32, order="C", copy=True)
        if (
            embeddings.ndim != 3
            or embeddings.shape[0] != 1
            or embeddings.shape[1] <= 0
            or embeddings.shape[2] <= 0
        ):
            raise RuntimeContractError(
                "Phonon voice embeddings must have shape [1, frames, d_model]"
            )
        if not np.isfinite(embeddings).all():
            raise RuntimeContractError("Phonon voice embeddings must contain only finite values")
        embeddings.setflags(write=False)
        if not isinstance(self.metadata, Mapping):
            raise RuntimeContractError("Phonon voice metadata must be a mapping")
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PhononStreamChunk:
    """One decoded mono PCM frame produced by :meth:`PhononAdapter.stream`."""

    audio: np.ndarray
    sample_rate: int
    frame_index: int
    eos_detected: bool
    eos_logit: float | None = None

    def __post_init__(self) -> None:
        audio = np.array(self.audio, dtype=np.float32, order="C", copy=True)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise RuntimeContractError("Phonon chunk audio must be finite mono float32")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise RuntimeContractError("Phonon chunk sample_rate must be a positive integer")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise RuntimeContractError("Phonon chunk frame_index must be a non-negative integer")
        if type(self.eos_detected) is not bool:
            raise RuntimeContractError("Phonon chunk eos_detected must be boolean")
        if self.eos_logit is not None and (
            isinstance(self.eos_logit, bool)
            or not isinstance(self.eos_logit, Real)
            or not math.isfinite(float(self.eos_logit))
        ):
            raise RuntimeContractError("Phonon chunk eos_logit must be finite or null")
        audio.setflags(write=False)
        object.__setattr__(self, "audio", audio)
        if self.eos_logit is not None:
            object.__setattr__(self, "eos_logit", float(self.eos_logit))


@dataclass(frozen=True, slots=True)
class _KVBinding:
    input_name: str
    output_name: str
    input_spec: TensorSpec
    output_spec: TensorSpec
    sequence_axis: int
    dtype: np.dtype[Any]
    layer: int
    kind: str


@dataclass(frozen=True, slots=True)
class _MimiStateBinding:
    name: str
    input_name: str
    output_name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    input_spec: TensorSpec
    output_spec: TensorSpec


class PhononAdapter(SystemAdapter):
    """Provider-configured, lazy four-graph Phonon runtime."""

    system = "phonon"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}
        self._config: dict[str, Any] | None = None
        self._runtime: dict[str, Any] = {}
        self._generation: dict[str, Any] = {}
        self._voices: dict[str, dict[str, Any]] = {}
        self._default_voice: str | None = None
        self._mimi_state_specs: tuple[tuple[str, tuple[int, ...], np.dtype[Any]], ...] = ()
        self._mimi_bindings: tuple[_MimiStateBinding, ...] = ()
        self._offset_input_spec: TensorSpec | None = None
        self._offset_output_spec: TensorSpec | None = None
        self._kv_bindings: tuple[_KVBinding, ...] = ()
        self._validated_contracts: set[str] = set()
        self._voice_states: dict[str, PhononVoiceState] = {}

    @staticmethod
    def _find_value(value: Any, keys: Sequence[str]) -> Any:
        if isinstance(value, Mapping):
            for key in keys:
                if key in value:
                    return value[key]
            for child in value.values():
                found = PhononAdapter._find_value(child, keys)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = PhononAdapter._find_value(child, keys)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeContractError(f"Phonon {label} must be a finite number")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise RuntimeContractError(f"Phonon {label} must be a finite number") from exc
        if not math.isfinite(number) or (minimum is not None and number < minimum):
            raise RuntimeContractError(f"Phonon {label} is outside its valid range")
        return number

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if type(value) is not int or value <= 0:
            raise RuntimeContractError(f"Phonon {label} must be a positive integer")
        return value

    @staticmethod
    def _normalize_locale(value: Any) -> tuple[str, str, str]:
        if not isinstance(value, str):
            return "", "", ""
        candidate = value.replace("_", "-")
        if _LANGUAGE_RE.fullmatch(candidate) is None:
            return "", "", ""
        parts = candidate.split("-")
        normalized = [parts[0].lower()]
        for part in parts[1:]:
            if len(part) == 2 and part.isalpha():
                normalized.append(part.upper())
            elif len(part) == 4 and part.isalpha():
                normalized.append(part.title())
            else:
                normalized.append(part.lower())
        locale = "-".join(normalized)
        return normalized[0], locale, locale

    @staticmethod
    def _runtime_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
        value = config.get("runtime")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _runtime_value(
        config: Mapping[str, Any], runtime: Mapping[str, Any], keys: Sequence[str]
    ) -> Any:
        """Read a known runtime field from explicit compatibility sections only."""
        onnx = config.get("onnx")
        sections = (runtime, onnx, config)
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            for key in keys:
                value = section.get(key)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _normalize_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
        config_runtime = PhononAdapter._runtime_section(config)
        kind = config_runtime.get("kind")
        if kind is None:
            kind = config.get("runtime_kind")
        if kind is None:
            kind = config.get("kind")
        if kind is None:
            kind = _RUNTIME_KIND
        if kind != _RUNTIME_KIND:
            raise RuntimeContractError(
                f"Phonon config runtime kind must be {_RUNTIME_KIND!r}, got {kind!r}"
            )
        runtime: dict[str, Any] = {"kind": kind, "graphs": list(_GRAPH_FILES.values())}
        for field_name, aliases in _RUNTIME_ALIASES.items():
            value = PhononAdapter._runtime_value(config, config_runtime, aliases)
            if field_name == "mimi_state":
                if not isinstance(value, (Mapping, list, tuple)) or not value:
                    raise RuntimeContractError(
                        "Phonon config must declare non-empty Mimi decoder state"
                    )
                runtime[field_name] = value
            elif field_name == "frame_rate":
                runtime[field_name] = PhononAdapter._finite_number(value, field_name, minimum=0.0)
                if runtime[field_name] <= 0:
                    raise RuntimeContractError("Phonon frame_rate must be positive")
            else:
                runtime[field_name] = PhononAdapter._positive_int(value, field_name)
        onnx = config.get("onnx")
        declared_graphs = config_runtime.get("graphs")
        if declared_graphs is None and isinstance(onnx, Mapping):
            declared_graphs = onnx.get("graphs")
        if declared_graphs is None:
            declared_graphs = config.get("graphs")
        if declared_graphs is not None:
            if not isinstance(declared_graphs, Sequence) or isinstance(
                declared_graphs, (str, bytes)
            ):
                raise RuntimeContractError("Phonon config graphs must be a sequence")
            if tuple(declared_graphs) != tuple(_GRAPH_FILES.values()):
                raise RuntimeContractError(
                    "Phonon config graph inventory does not match the runtime ABI"
                )
        return runtime

    @staticmethod
    def _normalize_generation(config: Mapping[str, Any]) -> dict[str, Any]:
        raw: Any = config.get("generation")
        if not isinstance(raw, Mapping):
            onnx = config.get("onnx")
            raw = onnx.get("generation") if isinstance(onnx, Mapping) else None
        raw = raw if isinstance(raw, Mapping) else {}
        result: dict[str, Any] = {}
        for field_name, aliases in _GENERATION_ALIASES.items():
            value = PhononAdapter._find_value(raw, aliases)
            if value is None:
                result[field_name] = None
            elif field_name == "temperature_default":
                result[field_name] = PhononAdapter._finite_number(value, field_name, minimum=0.0)
            elif field_name in {"frames_after_eos", "min_frames_before_eos", "max_frames"}:
                minimum = 1 if field_name == "max_frames" else 0
                if type(value) is not int or value < minimum:
                    raise RuntimeContractError(f"Phonon {field_name} is invalid")
                result[field_name] = value
            else:
                result[field_name] = PhononAdapter._finite_number(value, field_name)
        return result

    @staticmethod
    def _parse_mimi_state(raw: Any) -> tuple[tuple[str, tuple[int, ...], np.dtype[Any]], ...]:
        if isinstance(raw, Mapping):
            if "name" in raw and "shape" in raw:
                entries = [raw]
            else:
                entries = []
                for name, value in raw.items():
                    if not isinstance(value, Mapping):
                        raise RuntimeContractError("Phonon Mimi state entries must be objects")
                    entries.append({"name": name, **value})
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            entries = list(raw)
        else:
            raise RuntimeContractError("Phonon Mimi state must be a sequence or mapping")
        result: list[tuple[str, tuple[int, ...], np.dtype[Any]]] = []
        names: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RuntimeContractError(f"Phonon Mimi state entry {index} must be an object")
            name = entry.get("name") or entry.get("id")
            if not isinstance(name, str) or _STATE_NAME_RE.fullmatch(name) is None:
                raise RuntimeContractError(f"Phonon Mimi state entry {index} has an invalid name")
            if name in names:
                raise RuntimeContractError(f"Phonon Mimi state has duplicate name {name!r}")
            shape = entry.get("shape")
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
                raise RuntimeContractError(f"Phonon Mimi state {name!r} must declare a shape")
            if any(type(dimension) is not int or dimension < 0 for dimension in shape):
                raise RuntimeContractError(f"Phonon Mimi state {name!r} has an invalid shape")
            dtype_name = entry.get("dtype")
            if not isinstance(dtype_name, str):
                raise RuntimeContractError(f"Phonon Mimi state {name!r} must declare a dtype")
            dtype = _STATE_DTYPE_NAMES.get(dtype_name)
            if dtype is None:
                raise RuntimeContractError(
                    f"Phonon Mimi state {name!r} has unsupported dtype {dtype_name!r}"
                )
            names.add(name)
            result.append((name, tuple(shape), dtype))
        if not result:
            raise RuntimeContractError("Phonon Mimi state inventory cannot be empty")
        return tuple(result)

    @classmethod
    def _parse_voices(cls, raw: Any) -> dict[str, dict[str, Any]]:
        if isinstance(raw, Mapping):
            entries: list[Any] = []
            for name, value in raw.items():
                if not isinstance(value, Mapping):
                    raise RuntimeContractError("Phonon config voice entries must be objects")
                entries.append({"name": name, **value})
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            entries = list(raw)
        else:
            raise RuntimeContractError("Phonon config must declare bundled voices")
        voices: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RuntimeContractError(f"Phonon config voice entry {index} must be an object")
            name = entry.get("name") or entry.get("id")
            if not isinstance(name, str):
                raise RuntimeContractError(f"Phonon config voice entry {index} has no name")
            try:
                validate_safe_component(name, field_name="Phonon voice name")
            except ValueError as exc:
                raise RuntimeContractError(str(exc)) from exc
            if name in voices:
                raise RuntimeContractError(f"Phonon config has duplicate voice {name!r}")
            display_name = entry.get("display_name", name)
            if not isinstance(display_name, str) or not display_name.strip():
                raise RuntimeContractError(f"Phonon voice {name!r} has an invalid display_name")
            voice_id = entry.get("voice_id")
            if voice_id is not None and (not isinstance(voice_id, str) or not voice_id):
                raise RuntimeContractError(f"Phonon voice {name!r} has an invalid voice_id")
            source_locale = entry.get("locale")
            if source_locale is not None and not isinstance(source_locale, str):
                raise RuntimeContractError(f"Phonon voice {name!r} has an invalid locale")
            language, locale, language_label = cls._normalize_locale(source_locale)
            source_gender = entry.get("gender")
            if source_gender is not None and not isinstance(source_gender, str):
                raise RuntimeContractError(f"Phonon voice {name!r} has an invalid gender")
            gender = (
                source_gender.lower()
                if isinstance(source_gender, str) and source_gender.lower() in {"male", "female"}
                else "unknown"
            )
            frames = entry.get("frames")
            if frames is not None:
                frames = cls._positive_int(frames, f"voices.{name}.frames")
            detail: dict[str, Any] = {
                "id": name,
                "name": display_name,
                "display_name": display_name,
                "language": language,
                "locale": locale,
                "language_label": language_label or language or "unknown",
                "gender": gender,
            }
            if voice_id is not None:
                detail["voice_id"] = voice_id
                detail["upstream_voice_id"] = voice_id
            if frames is not None:
                detail["frames"] = frames
            if source_locale and not locale:
                detail["source_locale"] = source_locale
            if source_gender is not None and gender == "unknown":
                detail["source_gender"] = source_gender
            voices[name] = detail
        if not voices:
            raise RuntimeContractError("Phonon config voice inventory cannot be empty")
        return voices

    def _load_contract(self) -> dict[str, Any]:
        if self._config is not None:
            return self._config
        try:
            artifact = self.installation.require_artifact("config")
        except KeyError as exc:
            raise RuntimeContractError("Phonon installation is missing config.json") from exc
        try:
            config_data = json.loads(artifact.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(f"Could not read Phonon config.json: {exc}") from exc
        if not isinstance(config_data, Mapping):
            raise RuntimeContractError("Phonon config.json must contain an object")
        config = dict(config_data)
        runtime = self._normalize_runtime(config)
        generation = self._normalize_generation(config)
        mimi_specs = self._parse_mimi_state(runtime["mimi_state"])
        voices = self._parse_voices(config.get("voices"))
        raw_default = config.get("default_voice")
        if raw_default is not None and (
            not isinstance(raw_default, str) or raw_default not in voices
        ):
            raise RuntimeContractError("Phonon config default_voice must name a bundled voice")

        catalog_runtime = self.installation.metadata.get("runtime")
        if isinstance(catalog_runtime, Mapping):
            for field_name in (
                "kind",
                "graphs",
                "sample_rate",
                "frame_samples",
                "frame_rate",
                "d_model",
                "latent_dim",
                "num_layers",
                "num_heads",
                "head_dim",
                "mimi_state",
            ):
                if (
                    field_name in catalog_runtime
                    and catalog_runtime[field_name] != runtime[field_name]
                ):
                    raise RuntimeContractError(
                        f"Phonon catalog/config runtime mismatch for {field_name}"
                    )
        catalog_generation = self.installation.metadata.get("generation")
        if isinstance(catalog_generation, Mapping):
            for field_name, value in generation.items():
                if field_name in catalog_generation and catalog_generation[field_name] != value:
                    raise RuntimeContractError(
                        f"Phonon catalog/config generation mismatch for {field_name}"
                    )
        if (
            self.installation.sample_rate is not None
            and self.installation.sample_rate != runtime["sample_rate"]
        ):
            raise RuntimeContractError("Phonon catalog/config sample_rate mismatch")

        catalog_voices = tuple(self.installation.voices)
        metadata_voices = self.installation.metadata.get("voices")
        if (
            not catalog_voices
            and isinstance(metadata_voices, Sequence)
            and not isinstance(metadata_voices, (str, bytes))
        ):
            catalog_voices = tuple(metadata_voices)
        if catalog_voices and catalog_voices != tuple(voices):
            raise RuntimeContractError("Phonon catalog/config voice inventory mismatch")
        catalog_default = self.installation.default_voice
        metadata_default = self.installation.metadata.get("default_voice")
        if catalog_default is not None and catalog_default not in voices:
            raise RuntimeContractError("Phonon installation default_voice is not bundled")
        if isinstance(metadata_default, str) and metadata_default not in voices:
            raise RuntimeContractError("Phonon catalog default_voice is not bundled")
        if (
            raw_default is not None
            and catalog_default is not None
            and raw_default != catalog_default
        ):
            raise RuntimeContractError("Phonon catalog/config default_voice mismatch")
        if (
            raw_default is not None
            and isinstance(metadata_default, str)
            and raw_default != metadata_default
        ):
            raise RuntimeContractError("Phonon catalog/config default_voice mismatch")

        self._runtime = runtime
        self._generation = generation
        self._voices = voices
        self._default_voice = catalog_default or raw_default
        self._mimi_state_specs = mimi_specs
        self._config = config
        return config

    @staticmethod
    def _specs(session: Any, attribute: str, component: str) -> tuple[TensorSpec, ...]:
        specs = getattr(session, attribute, None)
        if not isinstance(specs, (tuple, list)) or not specs:
            raise RuntimeContractError(f"Phonon {component} graph does not expose {attribute}")
        if any(not isinstance(spec, TensorSpec) for spec in specs):
            raise RuntimeContractError(f"Phonon {component} graph has malformed tensor metadata")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise RuntimeContractError(
                f"Phonon {component} graph has duplicate or empty tensor names"
            )
        return tuple(specs)

    @staticmethod
    def _dtype(spec: TensorSpec, component: str) -> np.dtype[Any]:
        dtype = _SUPPORTED_DTYPES.get(spec.ort_type)
        if dtype is None:
            raise RuntimeContractError(
                f"Phonon {component} tensor {spec.name!r} uses unsupported dtype {spec.ort_type!r}"
            )
        return dtype

    @classmethod
    def _require_dtype(cls, spec: TensorSpec, component: str, category: str) -> np.dtype[Any]:
        dtype = cls._dtype(spec, component)
        if category == "floating":
            valid = np.issubdtype(dtype, np.floating)
        elif category == "integer":
            valid = np.issubdtype(dtype, np.integer)
        else:
            raise ValueError(f"Unknown Phonon dtype category {category!r}")
        if not valid:
            raise RuntimeContractError(
                f"Phonon {component} tensor {spec.name!r} must use a {category} dtype"
            )
        return dtype

    @staticmethod
    def _require_names(actual: set[str], expected: set[str], component: str) -> None:
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unexpected {', '.join(extra)}")
            raise RuntimeContractError(
                f"Phonon {component} graph signature mismatch: {'; '.join(details)}"
            )

    @staticmethod
    def _state_name(name: str, pattern: re.Pattern[str], component: str) -> tuple[int, str]:
        match = pattern.fullmatch(name)
        if match is None:
            raise RuntimeContractError(f"Phonon {component} has invalid state tensor name {name!r}")
        return int(match.group(1)), match.group(2)

    @staticmethod
    def _compatible_static_shape(actual: Sequence[Any], expected: Sequence[Any]) -> bool:
        return len(actual) == len(expected) and all(
            not isinstance(left, int) or not isinstance(right, int) or left == right
            for left, right in zip(actual, expected, strict=True)
        )

    def _validate_contract(self, component: str, session: Any) -> None:
        if component in self._validated_contracts:
            return
        inputs = self._specs(session, "input_specs", component)
        outputs = self._specs(session, "output_specs", component)
        input_specs = {spec.name: spec for spec in inputs}
        output_specs = {spec.name: spec for spec in outputs}
        input_names = set(input_specs)
        output_names = set(output_specs)

        if component == "text_conditioner":
            self._require_names(input_names, {"input_ids", "duration_delta"}, component)
            self._require_names(output_names, {"inputs_embeds", "bos_embeds"}, component)
            self._require_dtype(input_specs["input_ids"], component, "integer")
            if self._dtype(input_specs["duration_delta"], component) != np.dtype(np.float32):
                raise RuntimeContractError("Phonon duration_delta input must be float32")
            self._require_dtype(output_specs["inputs_embeds"], component, "floating")
            self._require_dtype(output_specs["bos_embeds"], component, "floating")
            if len(input_specs["input_ids"].shape) != 2:
                raise RuntimeContractError("Phonon text_conditioner input_ids must be rank 2")
            if len(input_specs["duration_delta"].shape) != 1:
                raise RuntimeContractError(
                    "Phonon text_conditioner duration_delta must have shape [1]"
                )
            if len(output_specs["inputs_embeds"].shape) != 3 or len(
                output_specs["bos_embeds"].shape
            ) not in {2, 3}:
                raise RuntimeContractError("Phonon text embeddings have an unsupported rank")
            for spec in (output_specs["inputs_embeds"], output_specs["bos_embeds"]):
                if isinstance(spec.shape[-1], int) and spec.shape[-1] != self._runtime["d_model"]:
                    raise RuntimeContractError(
                        "Phonon text conditioner d_model does not match config"
                    )

        elif component == "flow_lm_main":
            input_state: dict[tuple[int, str], TensorSpec] = {}
            output_state: dict[tuple[int, str], TensorSpec] = {}
            for spec in inputs:
                if spec.name.startswith("past_key_values."):
                    key = self._state_name(spec.name, _KV_INPUT_RE, component)
                    if key in input_state:
                        raise RuntimeContractError(
                            "Phonon flow_lm_main has duplicate past-KV inputs"
                        )
                    input_state[key] = spec
            for spec in outputs:
                if spec.name.startswith("present."):
                    key = self._state_name(spec.name, _KV_OUTPUT_RE, component)
                    if key in output_state:
                        raise RuntimeContractError(
                            "Phonon flow_lm_main has duplicate present-KV outputs"
                        )
                    output_state[key] = spec
            expected_layers = set(range(self._runtime["num_layers"]))
            expected_state = {
                (layer, kind) for layer in expected_layers for kind in ("key", "value")
            }
            if set(input_state) != expected_state or set(output_state) != expected_state:
                raise RuntimeContractError(
                    "Phonon flow_lm_main KV signature must contain configured key/value layers"
                )
            self._require_names(
                input_names,
                {"inputs_embeds", *(spec.name for spec in input_state.values())},
                component,
            )
            self._require_names(
                output_names,
                {"hidden", "eos_logit", *(spec.name for spec in output_state.values())},
                component,
            )
            embed_spec = input_specs["inputs_embeds"]
            if len(embed_spec.shape) != 3:
                raise RuntimeContractError("Phonon flow_lm_main inputs_embeds must be rank 3")
            if (
                isinstance(embed_spec.shape[-1], int)
                and embed_spec.shape[-1] != self._runtime["d_model"]
            ):
                raise RuntimeContractError("Phonon flow_lm_main d_model does not match config")
            self._require_dtype(embed_spec, component, "floating")
            self._require_dtype(output_specs["hidden"], component, "floating")
            self._require_dtype(output_specs["eos_logit"], component, "floating")
            kv_bindings: list[_KVBinding] = []
            for layer, kind in sorted(expected_state):
                input_spec = input_state[(layer, kind)]
                output_spec = output_state[(layer, kind)]
                input_dtype = self._require_dtype(input_spec, component, "floating")
                output_dtype = self._require_dtype(output_spec, component, "floating")
                if input_dtype != output_dtype:
                    raise RuntimeContractError(
                        f"Phonon past/present dtype mismatch for layer {layer} {kind}"
                    )
                if len(input_spec.shape) < 3:
                    raise RuntimeContractError(
                        f"Phonon KV tensor {input_spec.name!r} must have rank >= 3"
                    )
                sequence_axis = len(input_spec.shape) - 2
                input_sequence = input_spec.shape[sequence_axis]
                if isinstance(input_sequence, int):
                    raise RuntimeContractError(
                        f"Phonon past-KV sequence axis must be dynamic: {input_spec.name}"
                    )
                incompatible_shape = not self._compatible_static_shape(
                    input_spec.shape, output_spec.shape
                ) and (
                    len(input_spec.shape) != len(output_spec.shape)
                    or any(
                        index != sequence_axis
                        and isinstance(left, int)
                        and isinstance(right, int)
                        and left != right
                        for index, (left, right) in enumerate(
                            zip(input_spec.shape, output_spec.shape, strict=True)
                        )
                    )
                )
                if incompatible_shape:
                    raise RuntimeContractError("Phonon past/present KV shapes are incompatible")
                kv_bindings.append(
                    _KVBinding(
                        input_spec.name,
                        output_spec.name,
                        input_spec,
                        output_spec,
                        sequence_axis,
                        input_dtype,
                        layer,
                        kind,
                    )
                )
            self._kv_bindings = tuple(kv_bindings)

        elif component == "flow_lm_flow":
            self._require_names(input_names, {"hidden", "noise", "duration_delta"}, component)
            self._require_names(output_names, {"latent", "next_embeds"}, component)
            for name in ("hidden", "noise", "duration_delta"):
                self._require_dtype(input_specs[name], component, "floating")
            if self._dtype(input_specs["noise"], component) != np.dtype(np.float32):
                raise RuntimeContractError("Phonon flow noise input must be float32")
            if self._dtype(input_specs["duration_delta"], component) != np.dtype(np.float32):
                raise RuntimeContractError("Phonon flow duration_delta input must be float32")
            self._require_dtype(output_specs["latent"], component, "floating")
            self._require_dtype(output_specs["next_embeds"], component, "floating")
            if len(input_specs["noise"].shape) < 2:
                raise RuntimeContractError(
                    "Phonon flow noise input must include batch and latent dimensions"
                )
            if (
                isinstance(input_specs["noise"].shape[-1], int)
                and input_specs["noise"].shape[-1] != self._runtime["latent_dim"]
            ):
                raise RuntimeContractError("Phonon flow noise latent_dim does not match config")

        elif component == "mimi_decoder":
            state_inputs = {f"state.{name}" for name, _, _ in self._mimi_state_specs}
            state_outputs = {f"next_state.{name}" for name, _, _ in self._mimi_state_specs}
            self._require_names(input_names, {"latent", "offset", *state_inputs}, component)
            self._require_names(output_names, {"audio", "next_offset", *state_outputs}, component)
            self._require_dtype(input_specs["latent"], component, "floating")
            self._require_dtype(output_specs["audio"], component, "floating")
            if len(input_specs["latent"].shape) < 2:
                raise RuntimeContractError("Phonon Mimi latent input has unsupported rank")
            if (
                isinstance(input_specs["latent"].shape[-1], int)
                and input_specs["latent"].shape[-1] != self._runtime["latent_dim"]
            ):
                raise RuntimeContractError("Phonon Mimi latent_dim does not match config")
            offset_input = input_specs["offset"]
            offset_output = output_specs["next_offset"]
            offset_dtype = self._dtype(offset_input, component)
            if self._dtype(offset_output, component) != offset_dtype:
                raise RuntimeContractError("Phonon Mimi offset input/output dtypes do not match")
            if not self._compatible_static_shape(offset_input.shape, offset_output.shape):
                raise RuntimeContractError("Phonon Mimi offset input/output shapes do not match")
            mimi_bindings: list[_MimiStateBinding] = []
            for name, shape, configured_dtype in self._mimi_state_specs:
                input_name = f"state.{name}"
                output_name = f"next_state.{name}"
                input_spec = input_specs[input_name]
                output_spec = output_specs[output_name]
                input_dtype = self._dtype(input_spec, component)
                if (
                    input_dtype != configured_dtype
                    or self._dtype(output_spec, component) != configured_dtype
                ):
                    raise RuntimeContractError(f"Phonon Mimi state dtype mismatch for {name!r}")
                if not self._compatible_static_shape(input_spec.shape, shape):
                    raise RuntimeContractError(f"Phonon Mimi state shape mismatch for {name!r}")
                if not self._compatible_static_shape(output_spec.shape, shape):
                    raise RuntimeContractError(
                        f"Phonon Mimi next-state shape mismatch for {name!r}"
                    )
                mimi_bindings.append(
                    _MimiStateBinding(
                        name,
                        input_name,
                        output_name,
                        shape,
                        configured_dtype,
                        input_spec,
                        output_spec,
                    )
                )
            self._mimi_bindings = tuple(mimi_bindings)
            self._offset_input_spec = offset_input
            self._offset_output_spec = offset_output
        else:
            raise RuntimeContractError(f"Unknown Phonon graph component {component!r}")
        self._validated_contracts.add(component)

    def _get_session(self, component: str) -> OnnxSession:
        existing = self._sessions.get(component)
        if existing is not None:
            return existing
        if component not in _GRAPH_FILES:
            raise RuntimeContractError(f"Unknown Phonon graph component {component!r}")
        self._load_contract()
        try:
            artifact = self.installation.require_artifact("model", component=component)
        except KeyError as exc:
            raise RuntimeContractError(
                f"Phonon installation is missing graph {component!r}"
            ) from exc
        session = OnnxSession(
            artifact.path,
            component=component,
            providers=self.providers,
            provider_options=self.provider_options,
            session_options=self.session_options,
        )
        try:
            self._validate_contract(component, session)
        except Exception:
            session.close()
            raise
        self._sessions[component] = session
        return session

    @classmethod
    def _cast_feed(
        cls,
        value: Any,
        spec: TensorSpec,
        component: str,
        *,
        category: str | None = None,
    ) -> np.ndarray:
        dtype = cls._dtype(spec, component)
        if category is not None:
            cls._require_dtype(spec, component, category)
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError(
                f"Phonon {component} could not cast feed {spec.name!r} to {dtype.name}"
            ) from exc
        if array.ndim != len(spec.shape):
            raise RuntimeContractError(
                f"Phonon {component} feed {spec.name!r} has rank {array.ndim}, expected {len(spec.shape)}"
            )
        for axis, (actual, expected) in enumerate(zip(array.shape, spec.shape, strict=True)):
            if isinstance(expected, int) and expected != actual:
                raise RuntimeContractError(
                    f"Phonon {component} feed {spec.name!r} axis {axis} has size {actual}, expected {expected}"
                )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise RuntimeContractError(f"Phonon {component} feed {spec.name!r} must be finite")
        return np.ascontiguousarray(array)

    @classmethod
    def _fit_feature(cls, value: Any, spec: TensorSpec, component: str) -> np.ndarray:
        array = np.asarray(value)
        target_rank = len(spec.shape)
        if array.ndim == 2 and target_rank == 3 and array.shape[0] == 1:
            array = array[:, None, :]
        elif array.ndim == 3 and target_rank == 2 and array.shape[0] == 1 and array.shape[1] == 1:
            array = array[:, 0, :]
        return cls._cast_feed(array, spec, component, category="floating")

    @staticmethod
    def _named_outputs(session: Any, values: Any, component: str) -> dict[str, np.ndarray]:
        names = tuple(getattr(session, "output_names", ()))
        if isinstance(values, Mapping):
            named = {str(name): np.asarray(value) for name, value in values.items()}
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) != len(names):
                raise RuntimeContractError(
                    f"Phonon {component} returned {len(values)} outputs for {len(names)} names"
                )
            named = {name: np.asarray(value) for name, value in zip(names, values, strict=True)}
        else:
            raise RuntimeContractError(f"Phonon {component} returned invalid outputs")
        if set(named) != set(names):
            raise RuntimeContractError(
                f"Phonon {component} returned outputs that do not match its signature"
            )
        return named

    @classmethod
    def _run(
        cls, session: Any, feeds: dict[str, np.ndarray], component: str
    ) -> dict[str, np.ndarray]:
        try:
            values = session.run(feeds)
        except Exception as exc:
            raise RuntimeContractError(f"Phonon {component} inference failed: {exc}") from exc
        return cls._named_outputs(session, values, component)

    def _initial_cache_tensor(self, binding: _KVBinding) -> np.ndarray:
        shape: list[int] = []
        rank = len(binding.input_spec.shape)
        for axis, dimension in enumerate(binding.input_spec.shape):
            if axis == binding.sequence_axis:
                shape.append(0)
            elif isinstance(dimension, int):
                shape.append(dimension)
            elif axis == 0:
                shape.append(1)
            elif rank == 4 and axis == 1:
                shape.append(self._runtime["num_heads"])
            elif axis == rank - 1:
                shape.append(self._runtime["head_dim"])
            else:
                shape.append(1)
        return np.zeros(tuple(shape), dtype=binding.dtype)

    def _initial_dynamic_tensor(self, spec: TensorSpec, component: str) -> np.ndarray:
        shape = tuple(dimension if isinstance(dimension, int) else 1 for dimension in spec.shape)
        return np.zeros(shape, dtype=self._dtype(spec, component))

    @staticmethod
    def _normalize_embeddings(value: Any, label: str, d_model: int) -> np.ndarray:
        embeddings = np.asarray(value, dtype=np.float32)
        if embeddings.ndim == 2 and embeddings.shape[0] == 1:
            embeddings = embeddings[:, None, :]
        if (
            embeddings.ndim != 3
            or embeddings.shape[0] != 1
            or embeddings.shape[1] <= 0
            or embeddings.shape[2] != d_model
            or not np.isfinite(embeddings).all()
        ):
            raise RuntimeContractError(
                f"Phonon {label} must be finite [1, sequence, {d_model}] float embeddings"
            )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    @staticmethod
    def _scalar(value: Any, label: str) -> float:
        array = np.asarray(value)
        if array.size != 1 or not np.issubdtype(array.dtype, np.number):
            raise RuntimeContractError(f"Phonon {label} must contain exactly one number")
        number = float(array.reshape(-1)[0])
        if not math.isfinite(number):
            raise RuntimeContractError(f"Phonon {label} must be finite")
        return number

    @staticmethod
    def _audio(value: Any, *, frame_samples: int, component: str) -> np.ndarray:
        try:
            audio = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError(
                f"Phonon {component} audio cannot be converted to float32"
            ) from exc
        if audio.ndim == 0 or any(dimension != 1 for dimension in audio.shape[:-1]):
            raise RuntimeContractError(
                "Phonon Mimi audio must be mono with only singleton batch axes"
            )
        audio = np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)
        if audio.size != frame_samples:
            raise RuntimeContractError(
                f"Phonon Mimi audio frame has {audio.size} samples, expected {frame_samples}"
            )
        if not np.isfinite(audio).all():
            raise RuntimeContractError("Phonon Mimi audio contains NaN or infinity")
        return audio

    def _resolve_voice(self, voice: str | PhononVoiceState | None) -> PhononVoiceState:
        self._load_contract()
        if voice is None:
            if self._default_voice is None:
                raise RuntimeContractError(
                    "Phonon has no default voice; select a bundled voice explicitly"
                )
            voice = self._default_voice
        if isinstance(voice, str):
            return self.prepare_voice(voice)
        if not isinstance(voice, PhononVoiceState):
            raise RuntimeContractError("Phonon voice must be a bundled name or PhononVoiceState")
        detail = self._voices.get(voice.name)
        if detail is None:
            raise RuntimeContractError(f"Unknown Phonon voice state {voice.name!r}")
        if voice.embeddings.shape[2] != self._runtime["d_model"]:
            raise RuntimeContractError("Phonon voice-state d_model does not match installed config")
        expected_frames = detail.get("frames")
        if expected_frames is not None and voice.embeddings.shape[1] != expected_frames:
            raise RuntimeContractError(
                "Phonon voice-state frame count does not match voice metadata"
            )
        return voice

    def prepare_voice(self, name: str | None = None) -> PhononVoiceState:
        """Load a declared little-endian float32 projected voice prompt."""
        self._load_contract()
        selected = self._default_voice if name is None else name
        if selected is None:
            raise RuntimeContractError(
                "Phonon default voice is not declared; select a voice explicitly"
            )
        if not isinstance(selected, str) or selected not in self._voices:
            available = ", ".join(self._voices) or "none"
            raise RuntimeContractError(
                f"Unknown Phonon voice {selected!r}; available voices: {available}"
            )
        cached = self._voice_states.get(selected)
        if cached is not None:
            return cached
        try:
            artifact = self.installation.require_artifact("voice_prompt", component=selected)
        except KeyError as exc:
            raise RuntimeContractError(
                f"Phonon installation is missing voice prompt {selected!r}"
            ) from exc
        if not artifact.path.is_file():
            raise RuntimeContractError(f"Phonon voice prompt {selected!r} is not a regular file")
        try:
            values = np.fromfile(artifact.path, dtype="<f4")
        except OSError as exc:
            raise RuntimeContractError(
                f"Could not load Phonon voice prompt {selected!r}: {exc}"
            ) from exc
        d_model = self._runtime["d_model"]
        if values.size == 0 or values.size % d_model:
            raise RuntimeContractError(
                f"Phonon voice prompt {selected!r} byte count is not divisible by d_model * 4"
            )
        frames = values.size // d_model
        expected_frames = self._voices[selected].get("frames")
        if expected_frames is not None and frames != expected_frames:
            raise RuntimeContractError(
                f"Phonon voice prompt {selected!r} has {frames} frames, expected {expected_frames}"
            )
        embeddings = np.ascontiguousarray(values.reshape(1, frames, d_model), dtype=np.float32)
        if not np.isfinite(embeddings).all():
            raise RuntimeContractError(f"Phonon voice prompt {selected!r} contains NaN or infinity")
        state = PhononVoiceState(selected, embeddings, self._voices[selected])
        self._voice_states[selected] = state
        return state

    def _generation_control(
        self,
        provided: Any,
        field_name: str,
        *,
        kind: str,
        minimum: int | float | None = None,
    ) -> float | int:
        value = self._generation[field_name] if provided is None else provided
        if value is None:
            raise RuntimeContractError(
                f"Phonon {field_name} has no bundle default; provide it explicitly"
            )
        if kind == "number":
            return self._finite_number(value, field_name, minimum=minimum)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RuntimeContractError(f"Phonon {field_name} must be an integer")
        number = int(value)
        if minimum is not None and number < minimum:
            raise RuntimeContractError(f"Phonon {field_name} must be >= {minimum}")
        return number

    def _controls(
        self,
        *,
        duration_delta: Any,
        temperature: Any,
        seed: Any,
        eos_threshold: Any,
        frames_after_eos: Any,
        min_frames_before_eos: Any,
        max_frames: Any,
    ) -> dict[str, Any]:
        duration = self._generation_control(duration_delta, "duration_delta_default", kind="number")
        temp = self._generation_control(
            temperature, "temperature_default", kind="number", minimum=0.0
        )
        threshold = self._generation_control(eos_threshold, "eos_threshold_default", kind="number")
        tail = self._generation_control(
            frames_after_eos, "frames_after_eos", kind="integer", minimum=0
        )
        minimum = self._generation_control(
            min_frames_before_eos,
            "min_frames_before_eos",
            kind="integer",
            minimum=0,
        )
        maximum = self._generation_control(max_frames, "max_frames", kind="integer", minimum=1)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
                raise RuntimeContractError("Phonon seed must be a non-negative integer or null")
            seed = int(seed)
        return {
            "duration_delta": float(duration),
            "temperature": float(temp),
            "seed": seed,
            "eos_threshold": float(threshold),
            "frames_after_eos": int(tail),
            "min_frames_before_eos": int(minimum),
            "max_frames": int(maximum),
        }

    @staticmethod
    def _token_ids(token_ids: Sequence[int]) -> np.ndarray:
        try:
            values = np.asarray(token_ids)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(
                "Phonon token_ids must be a one-dimensional integer sequence"
            ) from exc
        if values.ndim != 1 or values.size == 0 or values.dtype.kind not in {"i", "u"}:
            raise RuntimeContractError(
                "Phonon token_ids must be a non-empty one-dimensional integer sequence"
            )
        if np.any(values < 0):
            raise RuntimeContractError("Phonon token_ids must be non-negative")
        return values

    @staticmethod
    def _resolve_noise_shape(spec: TensorSpec, runtime: Mapping[str, Any]) -> tuple[int, ...]:
        shape: list[int] = []
        rank = len(spec.shape)
        for axis, dimension in enumerate(spec.shape):
            if isinstance(dimension, int):
                shape.append(dimension)
            elif axis == 0:
                shape.append(1)
            elif axis == rank - 1:
                shape.append(int(runtime["latent_dim"]))
            else:
                shape.append(1)
        if not shape or shape[0] != 1 or shape[-1] != runtime["latent_dim"]:
            raise RuntimeContractError(
                "Phonon flow noise shape does not match configured latent_dim"
            )
        if any(dimension <= 0 for dimension in shape):
            raise RuntimeContractError("Phonon flow noise shape must be positive")
        return tuple(shape)

    def _main_step(
        self,
        session: Any,
        embedding: np.ndarray,
        cache: Mapping[str, np.ndarray],
        *,
        capture_eos: bool = True,
    ) -> tuple[np.ndarray, float | None, dict[str, np.ndarray]]:
        input_specs = {
            spec.name: spec for spec in self._specs(session, "input_specs", "flow_lm_main")
        }
        output_specs = {
            spec.name: spec for spec in self._specs(session, "output_specs", "flow_lm_main")
        }
        feeds = {
            "inputs_embeds": self._fit_feature(
                embedding, input_specs["inputs_embeds"], "flow_lm_main"
            ),
        }
        expected_cache = {binding.input_name for binding in self._kv_bindings}
        if set(cache) != expected_cache:
            raise RuntimeContractError("Phonon flow-LM cache does not match graph inputs")
        for binding in self._kv_bindings:
            feeds[binding.input_name] = self._cast_feed(
                cache[binding.input_name], binding.input_spec, "flow_lm_main"
            )
        outputs = self._run(session, feeds, "flow_lm_main")
        hidden = self._fit_feature(outputs["hidden"], output_specs["hidden"], "flow_lm_main")
        if hidden.shape[-1] != self._runtime["d_model"]:
            raise RuntimeContractError("Phonon flow-LM hidden d_model does not match config")
        eos_logit = self._scalar(outputs["eos_logit"], "eos_logit") if capture_eos else None
        next_cache: dict[str, np.ndarray] = {}
        for binding in self._kv_bindings:
            value = self._cast_feed(
                outputs[binding.output_name], binding.output_spec, "flow_lm_main"
            )
            next_cache[binding.input_name] = value
        return hidden, eos_logit, next_cache

    def _one_stream(
        self,
        token_ids: np.ndarray,
        voice: PhononVoiceState,
        controls: Mapping[str, Any],
    ) -> Iterator[PhononStreamChunk]:
        text_session = self._get_session("text_conditioner")
        main_session = self._get_session("flow_lm_main")
        flow_session = self._get_session("flow_lm_flow")
        mimi_session = self._get_session("mimi_decoder")
        text_inputs = {
            spec.name: spec for spec in self._specs(text_session, "input_specs", "text_conditioner")
        }
        flow_inputs = {
            spec.name: spec for spec in self._specs(flow_session, "input_specs", "flow_lm_flow")
        }
        mimi_inputs = {
            spec.name: spec for spec in self._specs(mimi_session, "input_specs", "mimi_decoder")
        }
        main_inputs = {
            spec.name: spec for spec in self._specs(main_session, "input_specs", "flow_lm_main")
        }

        ids_spec = text_inputs["input_ids"]
        ids_dtype = self._dtype(ids_spec, "text_conditioner")
        limits = np.iinfo(ids_dtype)
        if int(token_ids.min()) < limits.min or int(token_ids.max()) > limits.max:
            raise RuntimeContractError(f"Phonon token_ids exceed graph dtype {ids_dtype.name}")
        ids = self._cast_feed(token_ids[None, :], ids_spec, "text_conditioner", category="integer")
        duration = np.asarray([controls["duration_delta"]], dtype=np.float32)
        duration = self._cast_feed(duration, text_inputs["duration_delta"], "text_conditioner")

        cache: dict[str, np.ndarray] = {
            binding.input_name: self._initial_cache_tensor(binding) for binding in self._kv_bindings
        }
        _, _, cache = self._main_step(main_session, voice.embeddings, cache, capture_eos=False)

        conditioned = self._run(
            text_session,
            {"input_ids": ids, "duration_delta": duration},
            "text_conditioner",
        )
        text_embeddings = self._normalize_embeddings(
            conditioned["inputs_embeds"], "text embeddings", self._runtime["d_model"]
        )
        bos_embeddings = self._normalize_embeddings(
            conditioned["bos_embeds"], "BOS embeddings", self._runtime["d_model"]
        )
        text_embeddings = self._fit_feature(
            text_embeddings, main_inputs["inputs_embeds"], "flow_lm_main"
        )
        _, _, cache = self._main_step(main_session, text_embeddings, cache, capture_eos=False)
        current = self._fit_feature(bos_embeddings, main_inputs["inputs_embeds"], "flow_lm_main")

        offset_spec = self._offset_input_spec
        offset_output_spec = self._offset_output_spec
        if offset_spec is None or offset_output_spec is None:
            raise RuntimeContractError("Phonon Mimi offset contract was not initialized")
        offset = self._initial_dynamic_tensor(offset_spec, "mimi_decoder")
        mimi_state = {
            binding.input_name: np.zeros(binding.shape, dtype=binding.dtype)
            for binding in self._mimi_bindings
        }
        rng = np.random.default_rng(controls["seed"])
        noise_spec = flow_inputs["noise"]
        noise_shape = self._resolve_noise_shape(noise_spec, self._runtime)
        eos_frame: int | None = None

        for frame_index in range(controls["max_frames"]):
            hidden, eos_logit, cache = self._main_step(main_session, current, cache)
            if eos_logit is None:
                raise RuntimeContractError("Phonon generation step did not capture eos_logit")
            hidden_feed = self._fit_feature(hidden, flow_inputs["hidden"], "flow_lm_flow")
            if controls["temperature"] == 0:
                noise = np.zeros(noise_shape, dtype=np.float32)
            else:
                scale = math.sqrt(controls["temperature"])
                noise = np.asarray(rng.standard_normal(noise_shape) * scale, dtype=np.float32)
            noise = self._cast_feed(noise, noise_spec, "flow_lm_flow")
            flow_duration = self._cast_feed(duration, flow_inputs["duration_delta"], "flow_lm_flow")
            flow_result = self._run(
                flow_session,
                {"hidden": hidden_feed, "noise": noise, "duration_delta": flow_duration},
                "flow_lm_flow",
            )
            latent = self._fit_feature(flow_result["latent"], mimi_inputs["latent"], "mimi_decoder")
            next_embedding = self._normalize_embeddings(
                flow_result["next_embeds"], "next embedding", self._runtime["d_model"]
            )
            next_embedding = self._fit_feature(
                next_embedding, main_inputs["inputs_embeds"], "flow_lm_main"
            )

            decoder_feeds: dict[str, np.ndarray] = {
                "latent": latent,
                "offset": self._cast_feed(offset, mimi_inputs["offset"], "mimi_decoder"),
            }
            decoder_feeds.update(
                {
                    binding.input_name: self._cast_feed(
                        mimi_state[binding.input_name], binding.input_spec, "mimi_decoder"
                    )
                    for binding in self._mimi_bindings
                }
            )
            decoded = self._run(mimi_session, decoder_feeds, "mimi_decoder")
            audio = self._audio(
                decoded["audio"],
                frame_samples=self._runtime["frame_samples"],
                component="mimi_decoder",
            )
            offset = self._cast_feed(decoded["next_offset"], offset_output_spec, "mimi_decoder")
            for binding in self._mimi_bindings:
                value = self._cast_feed(
                    decoded[binding.output_name], binding.output_spec, "mimi_decoder"
                )
                mimi_state[binding.input_name] = value

            detected = False
            if (
                frame_index >= controls["min_frames_before_eos"]
                and eos_logit > controls["eos_threshold"]
                and eos_frame is None
            ):
                eos_frame = frame_index
                detected = True
            yield PhononStreamChunk(
                audio=audio,
                sample_rate=self._runtime["sample_rate"],
                frame_index=frame_index,
                eos_detected=detected,
                eos_logit=eos_logit,
            )
            current = next_embedding
            if eos_frame is not None and frame_index >= eos_frame + controls["frames_after_eos"]:
                break

    def stream(
        self,
        token_ids: Sequence[int],
        *,
        voice: str | PhononVoiceState | None = None,
        duration_delta: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        eos_threshold: float | None = None,
        frames_after_eos: int | None = None,
        min_frames_before_eos: int | None = None,
        max_frames: int | None = None,
    ) -> Iterator[PhononStreamChunk]:
        """Stream decoded mono PCM for already-tokenized IDs."""
        self._load_contract()
        tokens = self._token_ids(token_ids)
        controls = self._controls(
            duration_delta=duration_delta,
            temperature=temperature,
            seed=seed,
            eos_threshold=eos_threshold,
            frames_after_eos=frames_after_eos,
            min_frames_before_eos=min_frames_before_eos,
            max_frames=max_frames,
        )
        selected_voice = self._resolve_voice(voice)
        yield from self._one_stream(tokens, selected_voice, controls)

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        voice: str | PhononVoiceState | None = None,
        duration_delta: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        eos_threshold: float | None = None,
        frames_after_eos: int | None = None,
        min_frames_before_eos: int | None = None,
        max_frames: int | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Collect :meth:`stream` PCM into one contiguous inference result."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise RuntimeContractError(f"Unexpected Phonon infer option(s): {unexpected}")
        chunks = list(
            self.stream(
                token_ids,
                voice=voice,
                duration_delta=duration_delta,
                temperature=temperature,
                seed=seed,
                eos_threshold=eos_threshold,
                frames_after_eos=frames_after_eos,
                min_frames_before_eos=min_frames_before_eos,
                max_frames=max_frames,
            )
        )
        audio = (
            np.ascontiguousarray(
                np.concatenate([chunk.audio for chunk in chunks]), dtype=np.float32
            )
            if chunks
            else np.empty(0, dtype=np.float32)
        )
        controls = self._controls(
            duration_delta=duration_delta,
            temperature=temperature,
            seed=seed,
            eos_threshold=eos_threshold,
            frames_after_eos=frames_after_eos,
            min_frames_before_eos=min_frames_before_eos,
            max_frames=max_frames,
        )
        eos_index = next((chunk.frame_index for chunk in chunks if chunk.eos_detected), None)
        ended_by_eos = (
            eos_index is not None
            and bool(chunks)
            and chunks[-1].frame_index >= eos_index + controls["frames_after_eos"]
        )
        metadata = {
            "system": self.system,
            "voice": voice.name
            if isinstance(voice, PhononVoiceState)
            else voice or self._default_voice,
            "frames": len(chunks),
            "duration_delta": controls["duration_delta"],
            "temperature": controls["temperature"],
            "seed": controls["seed"],
            "eos_threshold": controls["eos_threshold"],
            "frames_after_eos": controls["frames_after_eos"],
            "min_frames_before_eos": controls["min_frames_before_eos"],
            "max_frames": controls["max_frames"],
            "eos_detected": eos_index is not None,
            "terminated_by": "eos" if ended_by_eos else "max_frames",
            "terminated_by_frame_cap": not ended_by_eos,
            "quality": self.installation.selected_quality
            or self.installation.metadata.get("selected_quality"),
        }
        return InferenceResult(
            audio=audio, sample_rate=self._runtime["sample_rate"], metadata=metadata
        )

    def diagnostics(self) -> RuntimeDiagnostic:
        """Report per-graph sessions and the fixed four-graph runtime layout."""
        base = super().diagnostics()
        return RuntimeDiagnostic(
            system=base.system,
            ref=base.ref,
            layout="phonon-four-graph-streaming-v1",
            sessions=base.sessions,
            properties={
                **base.properties,
                "runtime_kind": _RUNTIME_KIND,
                "graph_components": list(_GRAPH_COMPONENTS),
                "created_graphs": sorted(self._sessions),
                "selected_quality": self.installation.selected_quality,
            },
        )

    def close(self) -> None:
        """Close created graph sessions and allow validated sessions to be reopened."""
        super().close()
        self._validated_contracts.clear()
        self._kv_bindings = ()
        self._mimi_bindings = ()
        self._offset_input_spec = None
        self._offset_output_spec = None
