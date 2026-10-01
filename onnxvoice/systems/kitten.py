from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any, cast

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, TensorSpec
from .base import SystemAdapter

_UPSTREAM_TAIL_TRIM_SAMPLES = 5000
_EXPECTED_INPUTS = {"input_ids", "style", "speed"}
_TENSOR_DTYPES = {
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
}


class KittenAdapter(SystemAdapter):
    """Execute model-ready Kitten token IDs with caller-selected style and speed."""

    system = "kitten"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)

    @property
    def session(self) -> OnnxSession:
        if self._session is None:
            try:
                model_path = self.installation.require_artifact("model").path
            except KeyError as exc:
                raise CapabilityError("Kitten installation is missing its model artifact") from exc
            self._session = OnnxSession(
                model_path,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._session

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: np.ndarray | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> InferenceResult:
        if style is None:
            raise TypeError("Kitten inference requires a style tensor")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Kitten inference got unexpected arguments: {names}")
        sample_rate = self._sample_rate()
        specs = self._validate_graph_contract()
        ids = self._tokens(token_ids, specs["input_ids"])
        style_input = self._style(style, specs["style"])
        speed_input = self._speed(speed, specs["speed"])

        outputs = self.session.run({"input_ids": ids, "style": style_input, "speed": speed_input})
        if not outputs:
            raise RuntimeContractError("Kitten model returned no outputs")
        raw_audio = np.asarray(outputs[0])
        if raw_audio.ndim == 0 or raw_audio.shape[-1] <= _UPSTREAM_TAIL_TRIM_SAMPLES:
            raise RuntimeContractError(
                "Kitten model output is too short for the required 5000-sample tail trim"
            )
        audio = np.squeeze(raw_audio[..., :-_UPSTREAM_TAIL_TRIM_SAMPLES])
        if audio.ndim != 1:
            raise RuntimeContractError("Kitten audio output must be mono and one-dimensional")
        if not np.issubdtype(audio.dtype, np.number) or np.issubdtype(
            audio.dtype, np.complexfloating
        ):
            raise RuntimeContractError("Kitten audio output must contain real numeric samples")
        if not np.all(np.isfinite(audio)):
            raise RuntimeContractError("Kitten model returned non-finite audio")
        audio = audio.astype(np.float32, copy=False)
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise RuntimeContractError("Kitten audio output must be non-empty and finite float32")

        return InferenceResult(
            audio=audio,
            sample_rate=sample_rate,
            metadata={
                "system": self.system,
                "speed": float(speed),
                "tail_trim_samples": _UPSTREAM_TAIL_TRIM_SAMPLES,
            },
        )

    def _sample_rate(self) -> int:
        sample_rate = self.installation.sample_rate
        if sample_rate is None:
            return 24000
        if sample_rate != 24000:
            raise RuntimeContractError("Kitten v0.8 requires a 24000 Hz sample rate")
        return sample_rate

    def _validate_graph_contract(self) -> dict[str, TensorSpec]:
        names = set(self.session.input_names)
        if names != _EXPECTED_INPUTS:
            missing = sorted(_EXPECTED_INPUTS - names)
            unexpected = sorted(names - _EXPECTED_INPUTS)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise RuntimeContractError(
                f"Kitten graph input contract mismatch ({'; '.join(details)})"
            )

        specs = {spec.name: spec for spec in self.session.input_specs}
        if set(specs) != _EXPECTED_INPUTS:
            raise RuntimeContractError("Kitten graph input specifications are incomplete")
        return specs

    @staticmethod
    def _dtype(spec: TensorSpec, name: str) -> np.dtype[Any]:
        try:
            return cast(np.dtype[Any], _TENSOR_DTYPES[spec.ort_type.lower()])
        except KeyError as exc:
            raise RuntimeContractError(
                f"Kitten input {name!r} has unsupported tensor type {spec.ort_type!r}"
            ) from exc

    @staticmethod
    def _validate_shape(array: np.ndarray, spec: TensorSpec, name: str) -> None:
        if array.ndim != len(spec.shape):
            raise RuntimeContractError(
                f"Kitten input {name!r} must have rank {len(spec.shape)}, got {array.ndim}"
            )
        for axis, (actual, declared) in enumerate(zip(array.shape, spec.shape, strict=True)):
            if isinstance(declared, int) and not isinstance(declared, bool) and declared != actual:
                raise RuntimeContractError(
                    f"Kitten input {name!r} axis {axis} must have size {declared}, got {actual}"
                )

    @classmethod
    def _tokens(cls, token_ids: Sequence[int], spec: TensorSpec) -> np.ndarray:
        dtype = cls._dtype(spec, "input_ids")
        if not np.issubdtype(dtype, np.integer):
            raise RuntimeContractError("Kitten input 'input_ids' must use an integer tensor type")
        if isinstance(token_ids, (str, bytes)):
            raise RuntimeContractError("Kitten token_ids must be a non-empty integer sequence")
        try:
            raw_values = list(token_ids)
        except TypeError as exc:
            raise RuntimeContractError(
                "Kitten token_ids must be a non-empty integer sequence"
            ) from exc
        if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
            raise RuntimeContractError("Kitten token_ids must be a non-empty integer sequence")
        try:
            values = np.asarray(raw_values)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError(
                "Kitten token_ids must be a non-empty integer sequence"
            ) from exc
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
            raise RuntimeContractError("Kitten token_ids must be a non-empty integer sequence")
        bounds = np.iinfo(dtype)
        if np.any(values < bounds.min) or np.any(values > bounds.max):
            raise RuntimeContractError(f"Kitten token_ids do not fit graph dtype {dtype.name}")
        if len(spec.shape) != 2:
            raise RuntimeContractError("Kitten input 'input_ids' must have rank 2")
        batched = values.astype(dtype, copy=False)[None, :]
        if batched.shape[0] != 1:
            raise RuntimeContractError("Kitten input 'input_ids' batch size must be 1")
        cls._validate_shape(batched, spec, "input_ids")
        return batched

    @classmethod
    def _style(cls, style: np.ndarray, spec: TensorSpec) -> np.ndarray:
        dtype = cls._dtype(spec, "style")
        if not np.issubdtype(dtype, np.floating):
            raise RuntimeContractError("Kitten input 'style' must use a floating tensor type")
        try:
            values = np.asarray(style)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError("Kitten style must be a finite floating array") from exc
        if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
            raise RuntimeContractError("Kitten style must be a finite floating array")
        if values.ndim < 2 or values.shape[0] != 1:
            raise RuntimeContractError("Kitten style must have a batch dimension of 1")
        if len(spec.shape) < 2:
            raise RuntimeContractError("Kitten input 'style' must have rank 2 or greater")
        cls._validate_shape(values, spec, "style")
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = values.astype(dtype, copy=False)
        if not np.all(np.isfinite(normalized)):
            raise RuntimeContractError("Kitten style is not finite in the graph tensor dtype")
        return normalized

    @classmethod
    def _speed(cls, speed: float, spec: TensorSpec) -> np.ndarray:
        if isinstance(speed, bool) or not isinstance(speed, Real):
            raise RuntimeContractError("Kitten speed must be a finite positive number")
        speed_value = float(speed)
        if not math.isfinite(speed_value) or speed_value <= 0:
            raise RuntimeContractError("Kitten speed must be a finite positive number")
        dtype = cls._dtype(spec, "speed")
        if not np.issubdtype(dtype, np.floating):
            raise RuntimeContractError("Kitten input 'speed' must use a floating tensor type")
        if len(spec.shape) != 1:
            raise RuntimeContractError("Kitten input 'speed' must have rank 1")
        speed_input = np.asarray([speed_value], dtype=dtype)
        cls._validate_shape(speed_input, spec, "speed")
        if not np.all(np.isfinite(speed_input)) or speed_input[0] <= 0:
            raise RuntimeContractError(
                "Kitten speed must be finite and positive in the graph dtype"
            )
        return speed_input
