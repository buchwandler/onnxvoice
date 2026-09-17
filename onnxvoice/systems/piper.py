from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import AudioResult, Installation, TensorSpec
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
            )
        return self._session

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
        tokens: Sequence[int],
        *,
        speaker: int | str | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_w: float = 0.8,
        **_: Any,
    ) -> AudioResult:
        token_values = list(tokens)
        names = set(self.session.input_names)
        if "input" not in names:
            raise RuntimeContractError("Piper model is missing expected input 'input'")
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
            inputs["sid"] = np.asarray(
                [self._resolve_speaker(speaker)],
                dtype=self._numpy_dtype(self._input_spec("sid"), default=np.int64),
            )
        missing = set(inputs) - names
        if missing:
            raise RuntimeContractError(
                f"Piper model is missing expected inputs: {', '.join(sorted(missing))}"
            )
        outputs = self.session.run(inputs)
        audio = self._canonical_audio(np.asarray(outputs[0]))
        config = self._config()
        sample_rate = int(
            self.installation.sample_rate
            or (config.get("audio") or {}).get("sample_rate")
            or config.get("sample_rate")
            or 22050
        )
        return AudioResult(
            audio=audio,
            sample_rate=sample_rate,
            metadata={"system": "piper", "speaker": speaker},
        )

    def _resolve_speaker(self, speaker: int | str | None) -> int:
        config = self._config()
        num_speakers = int(config.get("num_speakers") or self.installation.metadata.get("num_speakers") or 1)
        speaker_map = config.get("speaker_id_map") or self.installation.metadata.get("speaker_id_map") or {}
        if not isinstance(speaker_map, dict):
            raise RuntimeContractError("Piper speaker_id_map must be an object")
        if speaker is None:
            if num_speakers == 1:
                return 0
            raise RuntimeContractError("A speaker must be provided for a multi-speaker Piper model")
        if isinstance(speaker, str):
            if speaker not in speaker_map:
                raise RuntimeContractError(f"Unknown Piper speaker {speaker!r}")
            speaker_id = speaker_map[speaker]
        else:
            speaker_id = speaker
        if not isinstance(speaker_id, int) or not 0 <= speaker_id < num_speakers:
            raise RuntimeContractError(
                f"Piper speaker id {speaker_id!r} is outside the range 0..{num_speakers - 1}"
            )
        return speaker_id

    def _input_spec(self, name: str) -> TensorSpec | None:
        for spec in getattr(self.session, "input_specs", ()):
            if spec.name == name:
                return spec
        return None

    @staticmethod
    def _numpy_dtype(spec: TensorSpec | None, *, default: np.dtype[Any]) -> np.dtype[Any]:
        if spec is None:
            return default
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
        audio = np.squeeze(value)
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Piper audio output must be one-dimensional")
        return audio.astype(np.float32, copy=False)
