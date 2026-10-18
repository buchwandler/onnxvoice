"""Cleanroom Chatterbox four-graph ONNX runtime adapter."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, TensorSpec
from .base import SystemAdapter

_STATE_KINDS = ("key", "value")
_DTYPE_NAMES: dict[str, np.dtype[Any]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}


@dataclass(frozen=True, slots=True, init=False)
class ChatterboxVoiceState:
    """Immutable encoder outputs reusable for multiple synthesis calls."""

    cond_emb: np.ndarray
    prompt_token: np.ndarray
    speaker_embeddings: np.ndarray
    speaker_features: np.ndarray

    def __init__(
        self,
        cond_emb: Any,
        prompt_token: Any,
        speaker_embeddings: Any,
        speaker_features: Any,
    ) -> None:
        for name, value, dtype in (
            ("cond_emb", cond_emb, np.float32),
            ("prompt_token", prompt_token, np.int64),
            ("speaker_embeddings", speaker_embeddings, np.float32),
            ("speaker_features", speaker_features, np.float32),
        ):
            array = np.array(value, dtype=dtype, copy=True, order="C")
            if array.size == 0:
                raise RuntimeContractError(f"Chatterbox voice {name} cannot be empty")
            if name == "prompt_token":
                if not np.issubdtype(np.asarray(value).dtype, np.integer):
                    raise RuntimeContractError("Chatterbox prompt speech tokens must be integers")
                if any(int(token) < 0 for token in array.ravel()):
                    raise RuntimeContractError(
                        "Chatterbox prompt speech tokens must be non-negative"
                    )
            elif not np.isfinite(array).all():
                raise RuntimeContractError(f"Chatterbox voice {name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class ChatterboxGenerationConfig:
    """Deterministic generation settings owned by the ONNX adapter."""

    max_new_tokens: int = 1024
    repetition_penalty: float = 1.2

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, (int, float))
            or not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be a positive finite number")


@dataclass(frozen=True, slots=True)
class _KvBinding:
    input_name: str
    output_name: str
    input_spec: TensorSpec
    output_spec: TensorSpec
    sequence_axis: int
    dtype: np.dtype[Any]
    layer: int
    kind: str


class ChatterboxAdapter(SystemAdapter):
    """Lazy, graph-contract-checked adapter for Chatterbox Nano v2 and Turbo."""

    system = "chatterbox"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}
        runtime = installation.metadata.get("runtime") or {}
        self._runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
        self._kv_bindings: tuple[_KvBinding, ...] | None = None
        self._logits_output: str | None = None
        sample_rate = installation.sample_rate
        if sample_rate is None:
            sample_rate = self._runtime.get("sample_rate")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise RuntimeContractError(
                "Chatterbox installation must declare a positive sample rate"
            )
        self._sample_rate: int = sample_rate
        for field in ("start_speech_token", "stop_speech_token", "silence_token"):
            if type(self._runtime.get(field)) is not int or self._runtime[field] < 0:
                raise RuntimeContractError(f"Chatterbox runtime metadata is missing {field}")
        if (
            type(self._runtime.get("trailing_silence_tokens")) is not int
            or self._runtime["trailing_silence_tokens"] < 0
        ):
            raise RuntimeContractError(
                "Chatterbox runtime metadata has invalid trailing_silence_tokens"
            )

    @staticmethod
    def _specs(session: Any, attribute: str, component: str) -> tuple[TensorSpec, ...]:
        specs = getattr(session, attribute, None)
        if not isinstance(specs, (tuple, list)) or not specs:
            raise RuntimeContractError(f"Chatterbox {component} graph does not expose {attribute}")
        if any(not isinstance(spec, TensorSpec) for spec in specs):
            raise RuntimeContractError(
                f"Chatterbox {component} graph has malformed tensor metadata"
            )
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise RuntimeContractError(
                f"Chatterbox {component} graph has duplicate or empty tensor names"
            )
        return tuple(specs)

    @staticmethod
    def _require_names(actual: set[str], required: set[str], component: str) -> None:
        missing = sorted(required - actual)
        if missing:
            raise RuntimeContractError(
                f"Chatterbox {component} graph is missing required inputs: {', '.join(missing)}"
            )

    @staticmethod
    def _dtype(spec: TensorSpec, component: str) -> np.dtype[Any]:
        dtype = _DTYPE_NAMES.get(spec.ort_type)
        if dtype is None:
            raise RuntimeContractError(
                f"Chatterbox {component} graph tensor {spec.name!r} uses unsupported dtype {spec.ort_type!r}"
            )
        return dtype

    @classmethod
    def _require_dtype(cls, spec: TensorSpec, component: str, kind: str) -> None:
        dtype = cls._dtype(spec, component)
        if kind == "floating":
            valid = np.issubdtype(dtype, np.floating)
        elif kind == "integer":
            valid = np.issubdtype(dtype, np.integer)
        elif kind == "mask":
            valid = np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)
        else:
            raise ValueError(f"Unknown tensor dtype category {kind!r}")
        if not valid:
            raise RuntimeContractError(
                f"Chatterbox {component} tensor {spec.name!r} must use a {kind} dtype, got {spec.ort_type!r}"
            )

    @staticmethod
    def _named_outputs(
        session: Any, outputs: Sequence[Any], component: str
    ) -> dict[str, np.ndarray]:
        specs = ChatterboxAdapter._specs(session, "output_specs", component)
        if len(outputs) != len(specs):
            raise RuntimeContractError(
                f"Chatterbox {component} returned {len(outputs)} outputs for {len(specs)} declared outputs"
            )
        result: dict[str, np.ndarray] = {}
        try:
            for spec, value in zip(specs, outputs, strict=True):
                array = np.asarray(value)
                expected_dtype = _DTYPE_NAMES.get(spec.ort_type)
                if expected_dtype is not None and array.dtype != expected_dtype:
                    raise RuntimeContractError(
                        f"Chatterbox {component} output {spec.name!r} dtype {array.dtype} "
                        f"does not match declared {spec.ort_type}"
                    )
                result[spec.name] = array
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(
                f"Chatterbox {component} returned malformed outputs"
            ) from exc
        return result

    @staticmethod
    def _find_named_output(
        outputs: Mapping[str, np.ndarray], candidates: Sequence[str], *, label: str
    ) -> str:
        by_fold = {name.casefold(): name for name in outputs}
        for candidate in candidates:
            if candidate.casefold() in by_fold:
                return by_fold[candidate.casefold()]
        if len(outputs) == 1:
            return next(iter(outputs))
        raise RuntimeContractError(
            f"Chatterbox {label} output is not identifiable from graph metadata: {', '.join(outputs)}"
        )

    @staticmethod
    def _kv_signature(name: str) -> tuple[int, str] | None:
        folded = name.casefold()
        if "past" not in folded and "present" not in folded:
            return None
        kind_matches = re.findall(r"(?:^|[._/])(key|value)(?:$|[._/])", folded)
        layer_match = re.search(r"(?:past|present)[^0-9]*([0-9]+)", folded)
        if not kind_matches or layer_match is None:
            raise RuntimeContractError(f"Unrecognized Chatterbox KV tensor name {name!r}")
        return int(layer_match.group(1)), kind_matches[-1]

    @staticmethod
    def _past_input(name: str) -> bool:
        folded = name.casefold()
        return "past_key_values" in folded or "past." in folded or folded.startswith("past_")

    @staticmethod
    def _sequence_axis(spec: TensorSpec) -> int:
        if len(spec.shape) < 3:
            raise RuntimeContractError(
                f"Chatterbox KV tensor {spec.name!r} must have rank at least 3"
            )
        named = [
            axis
            for axis, dim in enumerate(spec.shape)
            if isinstance(dim, str)
            and any(part in dim.casefold() for part in ("seq", "past", "cache"))
        ]
        if len(named) != 1 or named[0] == 0:
            raise RuntimeContractError(
                f"Chatterbox KV tensor {spec.name!r} has no unique named sequence dimension"
            )
        return named[0]

    def _validate_language_model(self, session: Any) -> None:
        if self._kv_bindings is not None:
            return
        inputs = self._specs(session, "input_specs", "language_model")
        outputs = self._specs(session, "output_specs", "language_model")
        input_names = {spec.name for spec in inputs}
        self._require_names(input_names, {"inputs_embeds", "attention_mask"}, "language_model")
        input_specs = {spec.name: spec for spec in inputs}
        self._require_dtype(input_specs["inputs_embeds"], "language_model", "floating")
        self._require_dtype(input_specs["attention_mask"], "language_model", "mask")
        if "position_ids" in input_specs:
            self._require_dtype(input_specs["position_ids"], "language_model", "integer")
        past_specs = [spec for spec in inputs if self._past_input(spec.name)]
        signatures: dict[tuple[int, str], TensorSpec] = {}
        for spec in past_specs:
            signature = self._kv_signature(spec.name)
            self._require_dtype(spec, "language_model", "floating")
            if signature is None or signature in signatures:
                raise RuntimeContractError("Chatterbox language_model has ambiguous past-KV inputs")
            signatures[signature] = spec
        if not signatures:
            raise RuntimeContractError(
                "Chatterbox language_model graph declares no past-key/value inputs"
            )
        layers = {layer for layer, _ in signatures}
        expected_signatures = {(layer, kind) for layer in layers for kind in _STATE_KINDS}
        if (
            not layers
            or layers != set(range(max(layers) + 1))
            or set(signatures) != expected_signatures
        ):
            raise RuntimeContractError(
                "Chatterbox language_model past-KV inputs must contain complete key/value layers"
            )
        present_specs: dict[tuple[int, str], TensorSpec] = {}
        for spec in outputs:
            if "present" not in spec.name.casefold():
                continue
            signature = self._kv_signature(spec.name)
            if signature is None or signature in present_specs:
                raise RuntimeContractError(
                    "Chatterbox language_model has ambiguous present-KV outputs"
                )
            present_specs[signature] = spec
        if set(signatures) != set(present_specs):
            raise RuntimeContractError(
                "Chatterbox language_model past/present KV layouts do not match"
            )
        bindings: list[_KvBinding] = []
        for signature in sorted(signatures):
            input_spec = signatures[signature]
            output_spec = present_specs[signature]
            sequence_axis = self._sequence_axis(input_spec)
            if self._sequence_axis(output_spec) != sequence_axis:
                raise RuntimeContractError("Chatterbox past/present KV sequence axes differ")
            if (
                isinstance(input_spec.shape[sequence_axis], int)
                and input_spec.shape[sequence_axis] > 0
            ):
                raise RuntimeContractError("Chatterbox past-KV sequence dimension must be dynamic")
            input_dtype = self._dtype(input_spec, "language_model")
            output_dtype = _DTYPE_NAMES.get(output_spec.ort_type)
            if output_dtype != input_dtype:
                raise RuntimeContractError(
                    f"Chatterbox KV input/output dtype mismatch for {input_spec.name!r}"
                )
            bindings.append(
                _KvBinding(
                    input_spec.name,
                    output_spec.name,
                    input_spec,
                    output_spec,
                    sequence_axis,
                    input_dtype,
                    signature[0],
                    signature[1],
                )
            )
        non_kv_outputs = [
            spec.name
            for spec in outputs
            if spec.name not in {item.name for item in present_specs.values()}
        ]
        if not non_kv_outputs:
            raise RuntimeContractError("Chatterbox language_model graph has no logits output")
        logits = self._find_named_output(
            {name: np.empty(0) for name in non_kv_outputs},
            ("logits", "speech_logits"),
            label="language_model logits",
        )
        logits_spec = next(spec for spec in outputs if spec.name == logits)
        self._require_dtype(logits_spec, "language_model", "floating")
        allowed = {
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            *(spec.name for spec in past_specs),
        }
        unexpected_inputs = input_names - allowed
        if unexpected_inputs:
            raise RuntimeContractError(
                "Chatterbox language_model declares unsupported inputs: "
                + ", ".join(sorted(unexpected_inputs))
            )
        self._kv_bindings = tuple(bindings)
        self._logits_output = logits

    def _validate_graph(self, component: str, session: Any) -> None:
        inputs = self._specs(session, "input_specs", component)
        outputs = self._specs(session, "output_specs", component)
        names = {spec.name for spec in inputs}
        output_names = {spec.name for spec in outputs}
        if component == "speech_encoder":
            self._require_names(names, {"audio_values"}, component)
            required_outputs = {
                "cond_emb",
                "prompt_token",
                "speaker_embeddings",
                "speaker_features",
            }
            missing = required_outputs - output_names
            if missing:
                raise RuntimeContractError(
                    "Chatterbox speech_encoder is missing required outputs: "
                    + ", ".join(sorted(missing))
                )
            if names != {"audio_values"}:
                raise RuntimeContractError("Chatterbox speech_encoder declares unsupported inputs")
            input_specs = {spec.name: spec for spec in inputs}
            output_specs = {spec.name: spec for spec in outputs}
            self._require_dtype(input_specs["audio_values"], component, "floating")
            self._require_dtype(output_specs["cond_emb"], component, "floating")
            self._require_dtype(output_specs["prompt_token"], component, "integer")
            self._require_dtype(output_specs["speaker_embeddings"], component, "floating")
            self._require_dtype(output_specs["speaker_features"], component, "floating")
        elif component == "embed_tokens":
            self._require_names(names, {"input_ids"}, component)
            input_specs = {spec.name: spec for spec in inputs}
            output_specs = {spec.name: spec for spec in outputs}
            self._require_dtype(input_specs["input_ids"], component, "integer")
            if "position_ids" in input_specs:
                self._require_dtype(input_specs["position_ids"], component, "integer")
            output_name = self._find_named_output(
                {name: np.empty(0) for name in output_names},
                ("inputs_embeds", "embeddings", "token_embeddings"),
                label="embed_tokens",
            )
            self._require_dtype(output_specs[output_name], component, "floating")
            if names - {"input_ids", "position_ids"}:
                raise RuntimeContractError("Chatterbox embed_tokens declares unsupported inputs")
            self._find_named_output(
                {name: np.empty(0) for name in output_names},
                ("inputs_embeds", "embeddings", "token_embeddings"),
                label="embed_tokens",
            )
        elif component == "language_model":
            self._validate_language_model(session)
        elif component == "conditional_decoder":
            self._require_names(
                names, {"speech_tokens", "speaker_embeddings", "speaker_features"}, component
            )
            input_specs = {spec.name: spec for spec in inputs}
            output_specs = {spec.name: spec for spec in outputs}
            self._require_dtype(input_specs["speech_tokens"], component, "integer")
            self._require_dtype(input_specs["speaker_embeddings"], component, "floating")
            self._require_dtype(input_specs["speaker_features"], component, "floating")
            output_name = self._find_named_output(
                {name: np.empty(0) for name in output_names},
                ("audio", "waveform", "wav", "output"),
                label="decoder audio",
            )
            self._require_dtype(output_specs[output_name], component, "floating")
            if names != {"speech_tokens", "speaker_embeddings", "speaker_features"}:
                raise RuntimeContractError(
                    "Chatterbox conditional_decoder declares unsupported inputs"
                )
            self._find_named_output(
                {name: np.empty(0) for name in output_names},
                ("audio", "waveform", "wav", "output"),
                label="conditional_decoder",
            )
        else:
            raise RuntimeContractError(f"Unknown Chatterbox graph component {component!r}")

    def _get_session(self, component: str) -> OnnxSession:
        existing = self._sessions.get(component)
        if existing is not None:
            return existing
        matches = self.installation.artifacts_for("graph", component=component)
        if len(matches) != 1:
            raise RuntimeContractError(
                f"Chatterbox installation requires exactly one {component} graph; found {len(matches)}"
            )
        session = OnnxSession(
            matches[0].path,
            component=component,
            providers=self.providers,
            provider_options=self.provider_options,
            session_options=self.session_options,
        )
        try:
            self._validate_graph(component, session)
        except Exception:
            session.close()
            raise
        self._sessions[component] = session
        return session

    @staticmethod
    def _run(session: Any, feeds: dict[str, np.ndarray], component: str) -> dict[str, np.ndarray]:
        try:
            values = session.run(feeds)
        except Exception as exc:
            raise RuntimeContractError(f"Chatterbox {component} inference failed: {exc}") from exc
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RuntimeContractError(f"Chatterbox {component} returned invalid outputs")
        return ChatterboxAdapter._named_outputs(session, values, component)

    @staticmethod
    def _cast_feed(value: Any, spec: TensorSpec, component: str) -> np.ndarray:
        return np.ascontiguousarray(
            np.asarray(value, dtype=ChatterboxAdapter._dtype(spec, component))
        )

    @staticmethod
    def _output(
        outputs: Mapping[str, np.ndarray], candidates: Sequence[str], *, label: str
    ) -> np.ndarray:
        name = ChatterboxAdapter._find_named_output(outputs, candidates, label=label)
        return outputs[name]

    def prepare_voice(self, audio_values: np.ndarray) -> ChatterboxVoiceState:
        """Encode normalized, native-rate mono audio once for reusable conditioning."""
        audio = np.asarray(audio_values, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeContractError(
                "Chatterbox reference audio must be non-empty finite mono float32"
            )
        session = self._get_session("speech_encoder")
        specs = {spec.name: spec for spec in self._specs(session, "input_specs", "speech_encoder")}
        audio_spec = specs["audio_values"]
        if len(audio_spec.shape) == 2:
            feed_audio = audio.reshape(1, -1)
        elif len(audio_spec.shape) == 1:
            feed_audio = audio
        else:
            raise RuntimeContractError(
                f"Chatterbox speech_encoder audio_values rank {len(audio_spec.shape)} is unsupported"
            )
        outputs = self._run(
            session,
            {"audio_values": self._cast_feed(feed_audio, audio_spec, "speech_encoder")},
            "speech_encoder",
        )
        return ChatterboxVoiceState(
            cond_emb=self._output(outputs, ("cond_emb",), label="conditioning embedding"),
            prompt_token=self._output(outputs, ("prompt_token",), label="prompt speech token"),
            speaker_embeddings=self._output(
                outputs, ("speaker_embeddings",), label="speaker embeddings"
            ),
            speaker_features=self._output(outputs, ("speaker_features",), label="speaker features"),
        )

    @staticmethod
    def _validate_token_ids(token_ids: Sequence[int]) -> np.ndarray:
        if isinstance(token_ids, (str, bytes)):
            raise RuntimeContractError("Chatterbox token IDs must be a non-empty integer sequence")
        try:
            values = np.asarray(list(token_ids))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError(
                "Chatterbox token IDs must be a non-empty integer sequence"
            ) from exc
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
            raise RuntimeContractError("Chatterbox token IDs must be a non-empty integer sequence")
        if np.any(values < 0):
            raise RuntimeContractError("Chatterbox token IDs must be non-negative")
        return np.ascontiguousarray(values.astype(np.int64, copy=False)[None, :])

    def _embed(self, token_ids: np.ndarray, position: int) -> np.ndarray:
        session = self._get_session("embed_tokens")
        specs = {spec.name: spec for spec in self._specs(session, "input_specs", "embed_tokens")}
        feeds = {"input_ids": self._cast_feed(token_ids, specs["input_ids"], "embed_tokens")}
        if "position_ids" in specs:
            shape = specs["position_ids"].shape
            positions = np.arange(position, position + token_ids.shape[-1], dtype=np.int64)
            positions = positions.reshape((1,) * max(0, len(shape) - 1) + (positions.size,))
            feeds["position_ids"] = self._cast_feed(
                positions, specs["position_ids"], "embed_tokens"
            )
        outputs = self._run(session, feeds, "embed_tokens")
        embeddings = self._output(
            outputs, ("inputs_embeds", "embeddings", "token_embeddings"), label="token embeddings"
        )
        if (
            embeddings.ndim != 3
            or embeddings.shape[0] != 1
            or embeddings.shape[1] != token_ids.shape[-1]
        ):
            raise RuntimeContractError(
                "Chatterbox embed_tokens returned malformed embedding dimensions"
            )
        if not np.isfinite(embeddings).all():
            raise RuntimeContractError("Chatterbox embed_tokens returned non-finite values")
        return np.ascontiguousarray(embeddings)

    @staticmethod
    def _empty_kv(binding: _KvBinding) -> np.ndarray:
        shape: list[int] = []
        for axis, dimension in enumerate(binding.input_spec.shape):
            if axis == 0:
                shape.append(1)
            elif axis == binding.sequence_axis:
                shape.append(0)
            elif isinstance(dimension, int) and dimension > 0:
                shape.append(dimension)
            else:
                raise RuntimeContractError(
                    f"Chatterbox KV input {binding.input_name!r} has an unresolved non-sequence dimension"
                )
        return np.zeros(tuple(shape), dtype=binding.dtype)

    @staticmethod
    def _position_ids(shape: tuple[Any, ...], start: int, length: int) -> np.ndarray:
        if not 1 <= len(shape) <= 3:
            raise RuntimeContractError("Chatterbox position_ids must have rank 1, 2, or 3")
        positions = np.arange(start, start + length, dtype=np.int64)
        return positions.reshape((1,) * (len(shape) - 1) + (length,))

    @staticmethod
    def _present_outputs(
        outputs: Mapping[str, np.ndarray],
        bindings: Sequence[_KvBinding],
        cache: Mapping[str, np.ndarray],
        input_length: int,
    ) -> dict[str, np.ndarray]:
        updated: dict[str, np.ndarray] = {}
        for binding in bindings:
            value = np.asarray(outputs[binding.output_name])
            previous = cache[binding.input_name]
            if value.ndim != previous.ndim:
                raise RuntimeContractError(
                    f"Chatterbox present KV {binding.output_name!r} has invalid rank"
                )
            expected_length = previous.shape[binding.sequence_axis] + input_length
            if value.shape[binding.sequence_axis] != expected_length:
                raise RuntimeContractError(
                    f"Chatterbox present KV {binding.output_name!r} has sequence length "
                    f"{value.shape[binding.sequence_axis]}, expected {expected_length}"
                )
            for axis, (actual, expected) in enumerate(
                zip(value.shape, previous.shape, strict=True)
            ):
                if axis != binding.sequence_axis and actual != expected:
                    raise RuntimeContractError(
                        f"Chatterbox present KV {binding.output_name!r} shape changed"
                    )
            if not np.isfinite(value).all():
                raise RuntimeContractError(
                    f"Chatterbox present KV {binding.output_name!r} is non-finite"
                )
            updated[binding.input_name] = np.ascontiguousarray(value, dtype=binding.dtype)
        return updated

    @staticmethod
    def _penalize(logits: np.ndarray, generated: Sequence[int], penalty: float) -> np.ndarray:
        scores = np.asarray(logits, dtype=np.float32).copy()
        for token in set(generated):
            if token < 0 or token >= scores.size:
                raise RuntimeContractError(
                    "Chatterbox generated token is outside the logits vocabulary"
                )
            if scores[token] < 0:
                scores[token] *= penalty
            else:
                scores[token] /= penalty
        return scores

    def _language_step(
        self,
        embeddings: np.ndarray,
        mask: np.ndarray,
        positions: np.ndarray | None,
        cache: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        session = self._get_session("language_model")
        self._validate_language_model(session)
        specs = {spec.name: spec for spec in self._specs(session, "input_specs", "language_model")}
        feeds = {
            "inputs_embeds": self._cast_feed(embeddings, specs["inputs_embeds"], "language_model"),
            "attention_mask": self._cast_feed(mask, specs["attention_mask"], "language_model"),
            **cache,
        }
        if positions is not None:
            feeds["position_ids"] = self._cast_feed(
                positions, specs["position_ids"], "language_model"
            )
        outputs = self._run(session, feeds, "language_model")
        assert self._logits_output is not None and self._kv_bindings is not None
        logits = np.asarray(outputs[self._logits_output], dtype=np.float32)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[2] < 1:
            raise RuntimeContractError("Chatterbox language_model returned malformed logits")
        if not np.isfinite(logits).all():
            raise RuntimeContractError("Chatterbox language_model returned non-finite logits")
        updated = self._present_outputs(outputs, self._kv_bindings, cache, embeddings.shape[1])
        return logits[0, -1], updated

    def _decode(
        self,
        generated: Sequence[int],
        voice_state: ChatterboxVoiceState,
    ) -> np.ndarray:
        start_token = self._runtime["start_speech_token"]
        stop_token = self._runtime["stop_speech_token"]
        speech = list(generated)
        if speech and speech[0] == start_token:
            speech = speech[1:]
        if speech and speech[-1] == stop_token:
            speech = speech[:-1]
        silence = [self._runtime["silence_token"]] * self._runtime["trailing_silence_tokens"]
        prompt = np.asarray(voice_state.prompt_token, dtype=np.int64).reshape(-1).tolist()
        decoder_tokens = np.asarray([prompt + speech + silence], dtype=np.int64)
        session = self._get_session("conditional_decoder")
        specs = {
            spec.name: spec for spec in self._specs(session, "input_specs", "conditional_decoder")
        }
        feeds = {
            "speech_tokens": self._cast_feed(
                decoder_tokens, specs["speech_tokens"], "conditional_decoder"
            ),
            "speaker_embeddings": self._cast_feed(
                voice_state.speaker_embeddings, specs["speaker_embeddings"], "conditional_decoder"
            ),
            "speaker_features": self._cast_feed(
                voice_state.speaker_features, specs["speaker_features"], "conditional_decoder"
            ),
        }
        outputs = self._run(session, feeds, "conditional_decoder")
        audio = self._output(outputs, ("audio", "waveform", "wav", "output"), label="decoder audio")
        audio = np.asarray(audio, dtype=np.float32)
        while audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeContractError("Chatterbox conditional_decoder returned invalid mono audio")
        return np.ascontiguousarray(audio, dtype=np.float32)

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        voice_state: ChatterboxVoiceState | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Generate speech greedily from model-ready text token IDs and prepared voice state."""
        if kwargs:
            raise TypeError(
                f"Unsupported Chatterbox inference options: {', '.join(sorted(kwargs))}"
            )
        if not isinstance(voice_state, ChatterboxVoiceState):
            raise RuntimeContractError("Chatterbox inference requires a ChatterboxVoiceState")
        ids = self._validate_token_ids(token_ids)
        config = ChatterboxGenerationConfig(
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else self._runtime.get("default_max_new_tokens", 1024)
            ),
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else self._runtime.get("default_repetition_penalty", 1.2)
            ),
        )
        self._get_session("embed_tokens")
        lm_session = self._get_session("language_model")
        self._validate_language_model(lm_session)
        assert self._kv_bindings is not None
        cache = {binding.input_name: self._empty_kv(binding) for binding in self._kv_bindings}
        generated = [self._runtime["start_speech_token"]]
        current_ids = ids
        mask: np.ndarray | None = None
        position_offset = 0
        stopped = False
        cond = np.asarray(voice_state.cond_emb, dtype=np.float32)
        for step in range(config.max_new_tokens):
            embeddings = self._embed(current_ids, position_offset)
            if step == 0:
                if cond.ndim != 3 or cond.shape[0] != 1 or not np.isfinite(cond).all():
                    raise RuntimeContractError(
                        "Chatterbox conditioning embedding has invalid shape or values"
                    )
                target_dtype = embeddings.dtype
                if cond.shape[-1] != embeddings.shape[-1]:
                    raise RuntimeContractError(
                        "Chatterbox conditioning and token embedding widths differ"
                    )
                first = np.asarray(cond, dtype=target_dtype)
                embeddings = np.concatenate((first, embeddings), axis=1)
                mask = np.ones((1, embeddings.shape[1]), dtype=np.int64)
                position_offset = embeddings.shape[1]
            else:
                assert mask is not None
                mask = np.concatenate(
                    (mask, np.ones((mask.shape[0], 1), dtype=mask.dtype)), axis=-1
                )
            lm_specs = {
                spec.name: spec for spec in self._specs(lm_session, "input_specs", "language_model")
            }
            positions = None
            if "position_ids" in lm_specs:
                positions = self._position_ids(
                    lm_specs["position_ids"].shape,
                    0 if step == 0 else position_offset,
                    embeddings.shape[1],
                )
            logits, cache = self._language_step(embeddings, mask, positions, cache)
            scores = self._penalize(logits, generated, float(config.repetition_penalty))
            next_token = int(np.argmax(scores))
            generated.append(next_token)
            if next_token == self._runtime["stop_speech_token"]:
                stopped = True
                break
            current_ids = np.asarray([[next_token]], dtype=np.int64)
            if step > 0:
                position_offset += embeddings.shape[1]
        audio = self._decode(generated, voice_state)
        return InferenceResult(
            audio=audio,
            sample_rate=self._sample_rate,
            outputs={"speech_tokens": np.asarray(generated, dtype=np.int64)},
            metadata={
                "profile": self._runtime.get("profile")
                or self.installation.metadata.get("selected_quality"),
                "generated_tokens": len(generated) - 1,
                "stopped_on_eos": stopped,
                "max_new_tokens": config.max_new_tokens,
                "repetition_penalty": float(config.repetition_penalty),
                "sample_rate": self._sample_rate,
            },
        )
