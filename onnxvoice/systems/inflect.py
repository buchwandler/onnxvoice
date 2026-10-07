from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, cast

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, TensorSpec
from .base import SystemAdapter

_EXPECTED_DURATION_INPUTS = {"tokens", "lengths", "length_scale"}
_EXPECTED_DURATION_OUTPUTS = {"m_p_exp", "logs_p_exp", "y_mask"}
_EXPECTED_DECODE_INPUTS = {
    "m_p_exp",
    "logs_p_exp",
    "y_mask",
    "zp_noise",
    "noise_scale",
}
_EXPECTED_DECODE_OUTPUTS = {"waveform"}
_INTEGER_DTYPES = {
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
}
_FLOAT_DTYPES = {
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
}


class InflectAdapter(SystemAdapter):
    """Execute model-ready Inflect v2 token IDs with the split ONNX runtime."""

    system = "inflect"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}

    def _session_for(self, role: str) -> OnnxSession:
        session = self._sessions.get(role)
        if session is not None:
            return session
        try:
            model_path = self.installation.require_artifact(role).path
        except KeyError as exc:
            raise CapabilityError(f"Inflect installation is missing its {role!r} artifact") from exc
        session = OnnxSession(
            model_path,
            component=role,
            providers=self.providers,
            provider_options=self.provider_options,
            session_options=self.session_options,
        )
        self._sessions[role] = session
        return session

    @property
    def duration_session(self) -> OnnxSession:
        return self._session_for("duration")

    @property
    def decode_session(self) -> OnnxSession:
        return self._session_for("decode")

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
        **kwargs: Any,
    ) -> InferenceResult:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Inflect inference got unexpected arguments: {names}")
        speed_value = self._control(speed, "speed", minimum=0.5, maximum=2.0)
        variation_value = self._control(variation, "variation", minimum=0.0, maximum=1.0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeContractError("Inflect seed must be a Python integer, not bool")
        try:
            rng = np.random.default_rng(seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError("Inflect seed must be a valid non-negative RNG seed") from exc
        tokens = self._token_values(token_ids)
        sample_rate = self._sample_rate()

        # Validate both graphs before executing either one, so a bad decoder does not
        # produce misleading partial work or consume runtime resources unnecessarily.
        duration_input_specs, duration_output_names, duration_output_specs = self._validate_duration_graph()
        decode_input_specs, decode_output_names, decode_output_specs = self._validate_decode_graph()
        token_input = self._integer_feed(tokens, duration_input_specs["tokens"], "tokens", rank=2)
        length_input = self._integer_feed(
            [len(tokens)], duration_input_specs["lengths"], "lengths", rank=1
        )
        length_scale = self._scalar_feed(
            1.0 / speed_value, duration_input_specs["length_scale"], "length_scale"
        )

        duration_values = self.duration_session.run(
            {"tokens": token_input, "lengths": length_input, "length_scale": length_scale}
        )
        duration_named = self._named_outputs(
            duration_values, duration_output_names, "duration"
        )
        intermediates = {
            name: self._finite_float_array(duration_named[name], f"duration output {name!r}")
            for name in ("m_p_exp", "logs_p_exp", "y_mask")
        }
        shapes = {value.shape for value in intermediates.values()}
        if len(shapes) != 1:
            raise RuntimeContractError(
                "Inflect duration outputs m_p_exp, logs_p_exp, and y_mask must have matching shapes"
            )
        m_p_exp = intermediates["m_p_exp"]
        for name, value in intermediates.items():
            self._validate_declared_shape(value, duration_output_specs[name], name)
            self._validate_declared_shape(value, decode_input_specs[name], name)

        zp_noise = rng.standard_normal(m_p_exp.shape, dtype=np.float32)
        noise_spec = decode_input_specs["zp_noise"]
        self._validate_declared_shape(zp_noise, noise_spec, "zp_noise")
        noise_dtype = self._fp32_dtype(noise_spec, "zp_noise")
        zp_noise = zp_noise.astype(noise_dtype, copy=False)
        noise_scale = self._scalar_feed(
            variation_value, decode_input_specs["noise_scale"], "noise_scale"
        )
        decode_feeds = {
            **intermediates,
            "zp_noise": zp_noise,
            "noise_scale": noise_scale,
        }
        decode_values = self.decode_session.run(decode_feeds)
        waveform_named = self._named_outputs(decode_values, decode_output_names, "decode")
        audio_values = self._finite_float_array(waveform_named["waveform"], "waveform")
        self._validate_declared_shape(audio_values, decode_output_specs["waveform"], "waveform")
        audio = np.squeeze(audio_values).astype(np.float32, copy=False)
        if audio.ndim == 0:
            audio = audio.reshape(1)
        if audio.ndim != 1:
            raise RuntimeContractError("Inflect waveform must resolve to an unambiguous mono signal")
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise RuntimeContractError("Inflect waveform must be non-empty and finite")

        return InferenceResult(
            audio=audio,
            sample_rate=sample_rate,
            metadata={
                "system": self.system,
                "speed": speed_value,
                "variation": variation_value,
                "seed": seed,
            },
        )

    def _sample_rate(self) -> int:
        sample_rate = self.installation.sample_rate
        if sample_rate is None:
            return 24000
        if type(sample_rate) is not int or sample_rate != 24000:
            raise RuntimeContractError("Inflect v2 requires a 24000 Hz sample rate")
        return sample_rate

    @staticmethod
    def _control(value: Any, name: str, *, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeContractError(f"Inflect {name} must be a finite real number")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise RuntimeContractError(f"Inflect {name} must be a finite real number") from None
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise RuntimeContractError(
                f"Inflect {name} must be finite and between {minimum} and {maximum}"
            )
        return number

    @staticmethod
    def _token_values(token_ids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(token_ids, (str, bytes)):
            raise RuntimeContractError("Inflect token_ids must be a non-empty integer sequence")
        try:
            values = tuple(token_ids)
        except TypeError as exc:
            raise RuntimeContractError(
                "Inflect token_ids must be a non-empty integer sequence"
            ) from exc
        if not values or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in values
        ):
            raise RuntimeContractError("Inflect token_ids must be a non-empty integer sequence")
        return tuple(int(value) for value in values)

    def _validate_duration_graph(
        self,
    ) -> tuple[dict[str, TensorSpec], tuple[str, ...], dict[str, TensorSpec]]:
        session = self.duration_session
        self._validate_names(session.input_names, _EXPECTED_DURATION_INPUTS, "duration input")
        output_names = self._validate_names(
            session.output_names, _EXPECTED_DURATION_OUTPUTS, "duration output"
        )
        inputs = self._spec_map(session.input_specs, _EXPECTED_DURATION_INPUTS, "duration input")
        outputs = self._spec_map(
            session.output_specs, _EXPECTED_DURATION_OUTPUTS, "duration output"
        )
        self._integer_dtype(inputs["tokens"], "tokens")
        self._integer_dtype(inputs["lengths"], "lengths")
        self._fp32_dtype(inputs["length_scale"], "length_scale")
        self._validate_spec_rank(inputs["tokens"], 2, "tokens")
        self._validate_spec_rank(inputs["lengths"], 1, "lengths")
        self._validate_spec_rank(inputs["length_scale"], (0, 1), "length_scale")
        for name, spec in outputs.items():
            self._fp32_dtype(spec, name)
        self._validate_compatible_specs(outputs, tuple(_EXPECTED_DURATION_OUTPUTS), "duration outputs")
        return inputs, output_names, outputs

    def _validate_decode_graph(
        self,
    ) -> tuple[dict[str, TensorSpec], tuple[str, ...], dict[str, TensorSpec]]:
        session = self.decode_session
        self._validate_names(session.input_names, _EXPECTED_DECODE_INPUTS, "decode input")
        output_names = self._validate_names(
            session.output_names, _EXPECTED_DECODE_OUTPUTS, "decode output"
        )
        inputs = self._spec_map(session.input_specs, _EXPECTED_DECODE_INPUTS, "decode input")
        outputs = self._spec_map(session.output_specs, _EXPECTED_DECODE_OUTPUTS, "decode output")
        for name, spec in inputs.items():
            self._fp32_dtype(spec, name)
        self._fp32_dtype(outputs["waveform"], "waveform")
        self._validate_compatible_specs(
            inputs, ("m_p_exp", "logs_p_exp", "y_mask", "zp_noise"), "decode latent inputs"
        )
        self._validate_spec_rank(inputs["noise_scale"], (0, 1), "noise_scale")
        return inputs, output_names, outputs

    @staticmethod
    def _validate_names(
        names: Sequence[str], expected: set[str], label: str
    ) -> tuple[str, ...]:
        actual = tuple(names)
        if len(actual) != len(expected) or set(actual) != expected:
            missing = sorted(expected - set(actual))
            unexpected = sorted(set(actual) - expected)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise RuntimeContractError(
                f"Inflect {label} contract mismatch ({'; '.join(details)})"
            )
        return actual

    @staticmethod
    def _spec_map(
        specs: Sequence[TensorSpec], expected: set[str], label: str
    ) -> dict[str, TensorSpec]:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs) or set(by_name) != expected:
            raise RuntimeContractError(f"Inflect {label} tensor specifications are incomplete")
        return by_name

    @staticmethod
    def _integer_dtype(spec: TensorSpec, name: str) -> np.dtype[Any]:
        try:
            return cast(np.dtype[Any], _INTEGER_DTYPES[spec.ort_type.lower()])
        except KeyError as exc:
            raise RuntimeContractError(
                f"Inflect tensor {name!r} must use an integer type, got {spec.ort_type!r}"
            ) from exc

    @staticmethod
    def _float_dtype(spec: TensorSpec, name: str) -> np.dtype[Any]:
        try:
            return cast(np.dtype[Any], _FLOAT_DTYPES[spec.ort_type.lower()])
        except KeyError as exc:
            raise RuntimeContractError(
                f"Inflect tensor {name!r} must use a floating type, got {spec.ort_type!r}"
            ) from exc

    @classmethod
    def _fp32_dtype(cls, spec: TensorSpec, name: str) -> np.dtype[Any]:
        dtype = cls._float_dtype(spec, name)
        if dtype != np.dtype(np.float32):
            raise RuntimeContractError(f"Inflect tensor {name!r} must use fp32 precision")
        return dtype

    @staticmethod
    def _validate_spec_rank(spec: TensorSpec, ranks: int | tuple[int, ...], name: str) -> None:
        allowed = (ranks,) if isinstance(ranks, int) else ranks
        if len(spec.shape) not in allowed:
            expected = " or ".join(str(rank) for rank in allowed)
            raise RuntimeContractError(
                f"Inflect tensor {name!r} must have rank {expected}, got {len(spec.shape)}"
            )

    @staticmethod
    def _validate_compatible_specs(
        specs: Mapping[str, TensorSpec], names: tuple[str, ...], label: str
    ) -> None:
        shapes = [specs[name].shape for name in names]
        rank = len(shapes[0])
        if any(len(shape) != rank for shape in shapes[1:]):
            raise RuntimeContractError(f"Inflect {label} must have compatible tensor ranks")
        for axis in range(rank):
            dimensions = [shape[axis] for shape in shapes]
            static = {
                dimension
                for dimension in dimensions
                if isinstance(dimension, int) and not isinstance(dimension, bool)
            }
            if len(static) > 1:
                raise RuntimeContractError(f"Inflect {label} have incompatible declared shapes")

    @staticmethod
    def _validate_declared_shape(array: np.ndarray, spec: TensorSpec, name: str) -> None:
        if array.ndim != len(spec.shape):
            raise RuntimeContractError(
                f"Inflect tensor {name!r} must have rank {len(spec.shape)}, got {array.ndim}"
            )
        for axis, (actual, declared) in enumerate(zip(array.shape, spec.shape, strict=True)):
            if isinstance(declared, int) and not isinstance(declared, bool) and declared != actual:
                raise RuntimeContractError(
                    f"Inflect tensor {name!r} axis {axis} must have size {declared}, got {actual}"
                )

    @classmethod
    def _integer_feed(
        cls, values: Sequence[int], spec: TensorSpec, name: str, *, rank: int
    ) -> np.ndarray:
        dtype = cls._integer_dtype(spec, name)
        if len(spec.shape) != rank:
            raise RuntimeContractError(f"Inflect tensor {name!r} must have rank {rank}")
        bounds = np.iinfo(dtype)
        if any(value < bounds.min or value > bounds.max for value in values):
            raise RuntimeContractError(
                f"Inflect {name} values do not fit graph dtype {dtype.name}"
            )
        feed = np.asarray(values, dtype=dtype)
        if name == "tokens":
            feed = feed[None, :]
        cls._validate_declared_shape(feed, spec, name)
        if name == "lengths" and feed.shape[0] != 1:
            raise RuntimeContractError("Inflect lengths must have batch size 1")
        if name == "tokens" and feed.shape[0] != 1:
            raise RuntimeContractError("Inflect tokens must have batch size 1")
        return feed

    @classmethod
    def _scalar_feed(cls, value: float, spec: TensorSpec, name: str) -> np.ndarray:
        dtype = cls._float_dtype(spec, name)
        if len(spec.shape) == 0:
            feed = np.asarray(value, dtype=dtype)
        elif len(spec.shape) == 1 and (
            spec.shape[0] is None
            or isinstance(spec.shape[0], str)
            or (type(spec.shape[0]) is int and spec.shape[0] == 1)
        ):
            feed = np.asarray([value], dtype=dtype)
        else:
            raise RuntimeContractError(
                f"Inflect tensor {name!r} must be scalar or have a compatible one-element shape"
            )
        cls._validate_declared_shape(feed, spec, name)
        if not np.all(np.isfinite(feed)):
            raise RuntimeContractError(f"Inflect {name} is not finite in the graph dtype")
        return feed

    @staticmethod
    def _named_outputs(
        values: Sequence[Any], names: Sequence[str], component: str
    ) -> dict[str, Any]:
        try:
            return dict(zip(names, values, strict=True))
        except ValueError as exc:
            raise RuntimeContractError(
                f"Inflect {component} graph returned an unexpected number of outputs"
            ) from exc

    @staticmethod
    def _finite_float_array(value: Any, label: str) -> np.ndarray:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(f"Inflect {label} must be a finite numeric tensor") from exc
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
            or not np.all(np.isfinite(array))
        ):
            raise RuntimeContractError(f"Inflect {label} must be a finite real numeric tensor")
        with np.errstate(over="ignore", invalid="ignore"):
            result = array.astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise RuntimeContractError(f"Inflect {label} is not finite as float32")
        return result
