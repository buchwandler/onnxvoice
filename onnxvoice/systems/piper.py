from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, TensorSpec
from .base import SystemAdapter


class PiperAdapter(SystemAdapter):
    system = "piper"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._config_cache: dict[str, Any] | None = None

    @property
    def session(self) -> OnnxSession:
        if self._session is None:
            self._session = OnnxSession(
                self.installation.artifact("model").path,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._session

    @property
    def requires_speaker_id(self) -> bool:
        return "sid" in self.session.input_names

    def _config(self) -> dict[str, Any]:
        if self._config_cache is not None:
            return self._config_cache
        try:
            path = self.installation.artifact("config").path
        except KeyError:
            self._config_cache = {}
            return self._config_cache
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(f"Could not read Piper config {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeContractError("Piper config must contain an object")
        self._config_cache = config
        return config

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        speaker_id: int | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_w: float = 0.8,
        **_: Any,
    ) -> InferenceResult:
        token_values = list(token_ids)
        names = set(self.session.input_names)
        required = {"input", "input_lengths", "scales"}
        missing = required - names
        if missing:
            raise RuntimeContractError(
                f"Piper model is missing expected inputs: {', '.join(sorted(missing))}"
            )
        token_dtype = self._numpy_dtype(self._input_spec("input"), default=np.int64)
        length_dtype = self._numpy_dtype(self._input_spec("input_lengths"), default=token_dtype)
        scale_dtype = self._numpy_dtype(self._input_spec("scales"), default=np.float32)
        ids = np.asarray([token_values], dtype=token_dtype)
        inputs: dict[str, np.ndarray] = {
            "input": ids,
            "input_lengths": np.asarray([ids.shape[1]], dtype=length_dtype),
            "scales": np.asarray([noise_scale, length_scale, noise_w], dtype=scale_dtype),
        }
        if "sid" in names:
            if speaker_id is None:
                raise RuntimeContractError(
                    "Model requires input 'sid'; speaker_id was not provided."
                )
            inputs["sid"] = np.asarray(
                [speaker_id], dtype=self._numpy_dtype(self._input_spec("sid"), default=np.int64)
            )
        elif speaker_id is not None:
            raise RuntimeContractError("speaker_id was provided, but model has no input 'sid'.")
        outputs = self.session.run(inputs)
        output_names = tuple(getattr(self.session, "output_names", ()))
        if len(output_names) != len(outputs):
            output_names = tuple(f"output_{index}" for index in range(len(outputs)))
        named_outputs = {
            name: np.asarray(value) for name, value in zip(output_names, outputs, strict=True)
        }
        audio_name = output_names[0]
        audio = self._canonical_audio(named_outputs[audio_name])
        auxiliary = {name: value for name, value in named_outputs.items() if name != audio_name}
        timings = next(
            (
                value
                for name, value in auxiliary.items()
                if any(token in name.lower() for token in ("timing", "duration", "timestamp"))
            ),
            None,
        )
        config = self._config()
        sample_rate = int(
            self.installation.sample_rate
            or (config.get("audio") or {}).get("sample_rate")
            or config.get("sample_rate")
            or 22050
        )
        return InferenceResult(
            audio=audio,
            sample_rate=sample_rate,
            timings=timings,
            outputs=auxiliary,
            metadata={"system": "piper", "speaker_id": speaker_id},
        )

    def _input_spec(self, name: str) -> TensorSpec | None:
        for spec in getattr(self.session, "input_specs", ()):
            if spec.name == name:
                return spec
        return None

    @staticmethod
    def _numpy_dtype(spec: TensorSpec | None, *, default: Any) -> np.dtype[Any]:
        if spec is None:
            return np.dtype(default)
        dtype_name = spec.ort_type.lower()
        for marker, dtype in (
            ("int64", np.int64),
            ("int32", np.int32),
            ("int16", np.int16),
            ("float16", np.float16),
            ("float", np.float32),
            ("double", np.float64),
        ):
            if marker in dtype_name:
                return np.dtype(dtype)
        raise RuntimeContractError(f"Unsupported Piper tensor type {spec.ort_type!r}")

    @staticmethod
    def _canonical_audio(value: np.ndarray) -> np.ndarray:
        audio = np.asarray(value)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Piper audio output must be one-dimensional")
        return audio.astype(np.float32, copy=False)
