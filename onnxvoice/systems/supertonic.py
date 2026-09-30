from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult
from .base import SystemAdapter


class SupertonicAdapter(SystemAdapter):
    system = "supertonic"
    COMPONENTS = (
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
    )
    _MODEL_INPUTS = {
        "duration_predictor": {"text_ids", "style_dp", "text_mask"},
        "text_encoder": {"text_ids", "style_ttl", "text_mask"},
        "vector_estimator": {
            "noisy_latent",
            "text_emb",
            "style_ttl",
            "text_mask",
            "latent_mask",
            "current_step",
            "total_step",
        },
        "vocoder": {"latent"},
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}
        self._config_cache: dict[str, Any] | None = None
        self._settings_cache: tuple[int, int, int, int] | None = None

    def _get_session(self, component: str) -> OnnxSession:
        if component not in self.COMPONENTS:
            raise CapabilityError(f"Unknown Supertonic model component {component!r}")
        if component not in self._sessions:
            try:
                path = self.installation.artifact("model", component=component).path
            except KeyError as exc:
                raise CapabilityError(
                    f"Supertonic installation is missing model component {component!r}"
                ) from exc
            self._sessions[component] = OnnxSession(
                path,
                component=component,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._sessions[component]

    def _config(self) -> dict[str, Any]:
        if self._config_cache is None:
            try:
                path = self.installation.artifact("config").path
            except KeyError as exc:
                raise CapabilityError("Supertonic installation is missing config artifact") from exc
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeContractError(
                    f"Could not read Supertonic config {path}: {exc}"
                ) from exc
            if not isinstance(config, dict):
                raise RuntimeContractError("Supertonic config must contain an object")
            self._config_cache = config
        return self._config_cache

    @staticmethod
    def _required_positive_int(config: Mapping[str, Any], section: str, key: str) -> int:
        value = config.get(section)
        path = f"{section}.{key}"
        if not isinstance(value, Mapping) or key not in value:
            raise RuntimeContractError(f"Supertonic config is missing required key {path}")
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise RuntimeContractError(f"Supertonic config value {path} must be a positive integer")
        return number

    def _settings(self) -> tuple[int, int, int, int]:
        if self._settings_cache is None:
            config = self._config()
            sample_rate = self._required_positive_int(config, "ae", "sample_rate")
            base_chunk_size = self._required_positive_int(config, "ae", "base_chunk_size")
            chunk_compress_factor = self._required_positive_int(
                config, "ttl", "chunk_compress_factor"
            )
            latent_dim = self._required_positive_int(config, "ttl", "latent_dim")
            installed_rate = self.installation.sample_rate
            if installed_rate is not None and installed_rate != sample_rate:
                raise RuntimeContractError(
                    f"Installation sample rate {installed_rate} disagrees with Supertonic config "
                    f"sample rate {sample_rate}"
                )
            self._settings_cache = (
                sample_rate,
                base_chunk_size,
                chunk_compress_factor,
                latent_dim,
            )
        return self._settings_cache

    @staticmethod
    def _validate_arguments(steps: Any, speed: Any, seed: Any) -> tuple[int, float, int | None]:
        if isinstance(steps, bool) or not isinstance(steps, Integral):
            raise RuntimeContractError("steps must be an integer from 1 through 100")
        steps_value = int(steps)
        if not 1 <= steps_value <= 100:
            raise RuntimeContractError("steps must be an integer from 1 through 100")
        if isinstance(speed, bool) or not isinstance(speed, Real):
            raise RuntimeContractError("speed must be a finite number from 0.7 through 2.0")
        speed_value = float(speed)
        if not math.isfinite(speed_value) or not 0.7 <= speed_value <= 2.0:
            raise RuntimeContractError("speed must be a finite number from 0.7 through 2.0")
        seed_value = None
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise RuntimeContractError("seed must be an integer or None")
            seed_value = int(seed)
            if not 0 <= seed_value <= 0xFFFFFFFF:
                raise RuntimeContractError("seed must be between 0 and 4294967295")
        return steps_value, speed_value, seed_value

    @staticmethod
    def _tokens(token_ids: Sequence[int]) -> np.ndarray:
        if isinstance(token_ids, (str, bytes)):
            raise RuntimeContractError("token_ids must be one non-empty integer sequence")
        try:
            values = np.asarray(list(token_ids))
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError("token_ids must be one non-empty integer sequence") from exc
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
            raise RuntimeContractError("token_ids must be one non-empty integer sequence")
        return values.astype(np.int64, copy=False)[None, :]

    @staticmethod
    def _input_array(
        value: Any,
        *,
        name: str,
        dtype: np.dtype[Any],
        shape: tuple[int, ...] | None = None,
        batch: bool = False,
    ) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError(f"{name} must be a valid {dtype.name} array") from exc
        if shape is not None and array.shape != shape:
            raise RuntimeContractError(f"{name} must have shape {shape}, got {array.shape}")
        if batch and (array.ndim < 2 or array.shape[0] != 1):
            raise RuntimeContractError(f"{name} must have a batch dimension of 1")
        if not np.all(np.isfinite(array)):
            raise RuntimeContractError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _session_input_names(session: Any) -> set[str] | None:
        names = getattr(session, "input_names", None)
        if names is not None:
            return set(names)
        get_inputs = getattr(session, "get_inputs", None)
        if get_inputs is not None:
            return {str(value.name) for value in get_inputs()}
        return None

    @classmethod
    def _validate_session_inputs(cls, session: Any, component: str) -> None:
        names = cls._session_input_names(session)
        if names is None:
            return
        expected = cls._MODEL_INPUTS[component]
        if names != expected:
            missing = sorted(expected - names)
            unexpected = sorted(names - expected)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise RuntimeContractError(
                f"Supertonic {component} input contract mismatch ({'; '.join(details)})"
            )

    @staticmethod
    def _run(session: Any, inputs: dict[str, np.ndarray]) -> list[Any]:
        try:
            return list(session.run(inputs))
        except TypeError:
            return list(session.run(None, inputs))

    @staticmethod
    def _first_output(session: Any, inputs: dict[str, np.ndarray], component: str) -> np.ndarray:
        SupertonicAdapter._validate_session_inputs(session, component)
        outputs = SupertonicAdapter._run(session, inputs)
        if not outputs:
            raise RuntimeContractError(f"Supertonic {component} returned no outputs")
        value = np.asarray(outputs[0])
        if not np.all(np.isfinite(value)):
            raise RuntimeContractError(f"Supertonic {component} returned non-finite output")
        return value

    @staticmethod
    def _latent_mask(wav_lengths: np.ndarray, chunk_size: int) -> np.ndarray:
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        max_length = int(latent_lengths.max())
        indices = np.arange(max_length)
        mask = (indices < latent_lengths[:, None]).astype(np.float32)
        return mask.reshape(-1, 1, max_length)

    @staticmethod
    def _canonical_audio(value: np.ndarray) -> np.ndarray:
        audio = np.asarray(value, dtype=np.float32).squeeze()
        if audio.ndim == 0 and audio.size == 1:
            audio = audio.reshape(1)
        if audio.ndim != 1 or audio.size == 0:
            raise RuntimeContractError("Supertonic vocoder audio must be a non-empty vector")
        if not np.all(np.isfinite(audio)):
            raise RuntimeContractError("Supertonic vocoder returned non-finite audio")
        return audio

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        text_mask: Any = None,
        style_ttl: Any = None,
        style_dp: Any = None,
        steps: int = 5,
        speed: float = 1.05,
        seed: int | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        if kwargs:
            raise RuntimeContractError(
                "Unsupported Supertonic inference arguments: " + ", ".join(sorted(kwargs))
            )
        if text_mask is None or style_ttl is None or style_dp is None:
            raise RuntimeContractError("text_mask, style_ttl, and style_dp are required")
        steps, speed, seed = self._validate_arguments(steps, speed, seed)
        text_ids = self._tokens(token_ids)
        text_mask_value = self._input_array(
            text_mask,
            name="text_mask",
            dtype=np.dtype(np.float32),
            shape=(1, 1, text_ids.shape[1]),
        )
        ttl_style = self._input_array(
            style_ttl, name="style_ttl", dtype=np.dtype(np.float32), batch=True
        )
        dp_style = self._input_array(
            style_dp, name="style_dp", dtype=np.dtype(np.float32), batch=True
        )
        sample_rate, base_chunk_size, chunk_compress_factor, latent_dim = self._settings()

        duration = self._first_output(
            self._get_session("duration_predictor"),
            {"text_ids": text_ids, "style_dp": dp_style, "text_mask": text_mask_value},
            "duration_predictor",
        ).astype(np.float32, copy=False)
        if duration.size != 1 or np.any(duration <= 0):
            raise RuntimeContractError("Supertonic duration must contain one positive value")
        duration = duration / np.float32(speed)
        if not np.all(np.isfinite(duration)) or np.any(duration <= 0):
            raise RuntimeContractError("Supertonic duration must contain one finite positive value")

        text_emb = self._first_output(
            self._get_session("text_encoder"),
            {"text_ids": text_ids, "style_ttl": ttl_style, "text_mask": text_mask_value},
            "text_encoder",
        )
        chunk_size = base_chunk_size * chunk_compress_factor
        wav_lengths = (duration * sample_rate).astype(np.int64).reshape(-1)
        latent_length = math.ceil(float(duration.max()) * sample_rate / chunk_size)
        latent_width = latent_dim * chunk_compress_factor
        rng = np.random.RandomState(seed)
        noisy_latent = rng.randn(1, latent_width, latent_length).astype(np.float32)
        latent_mask = self._latent_mask(wav_lengths, chunk_size)
        try:
            noisy_latent = noisy_latent * latent_mask
        except ValueError as exc:
            raise RuntimeContractError(
                "Supertonic duration produced an incompatible latent mask shape"
            ) from exc

        for step in range(steps):
            current_step = np.asarray([step], dtype=np.float32)
            total_step = np.asarray([steps], dtype=np.float32)
            latent = self._first_output(
                self._get_session("vector_estimator"),
                {
                    "noisy_latent": noisy_latent,
                    "text_emb": text_emb,
                    "style_ttl": ttl_style,
                    "text_mask": text_mask_value,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step,
                },
                "vector_estimator",
            )
            if latent.shape != noisy_latent.shape:
                raise RuntimeContractError(
                    "Supertonic vector_estimator output shape does not match noisy_latent"
                )
            noisy_latent = latent

        audio = self._canonical_audio(
            self._first_output(self._get_session("vocoder"), {"latent": noisy_latent}, "vocoder")
        )
        return InferenceResult(
            audio=audio,
            sample_rate=sample_rate,
            timings=duration,
            outputs={"duration": duration},
            metadata={
                "system": "supertonic",
                "layout": "supertonic-3-v1",
                "steps": steps,
                "speed": speed,
                "seed": seed,
            },
        )

    def diagnostics(self) -> Any:
        for component in self.COMPONENTS:
            self._get_session(component)
        return super().diagnostics()
