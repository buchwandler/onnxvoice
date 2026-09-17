from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import AudioResult, TensorSpec
from .base import SystemAdapter


class KokoroAdapter(SystemAdapter):
    system = "kokoro"

    @property
    def session(self) -> OnnxSession:
        if self._session is None:
            models = [artifact for artifact in self.installation.artifacts if artifact.role == "model"]
            runtime = self.installation.metadata.get("runtime") or {}
            if len(models) != 1 or runtime.get("layout") in {"split", "multi"}:
                raise CapabilityError(
                    "Split or multi-component Kokoro layouts are not supported by this adapter"
                )
            self._session = OnnxSession(
                models[0].path,
                providers=self.providers,
                provider_options=self.provider_options,
            )
        return self._session

    def _voice_style(self, voice: str, token_count: int, dtype: np.dtype[Any]) -> np.ndarray:
        if self.installation.voices and voice not in self.installation.voices:
            raise RuntimeContractError(
                f"Unknown Kokoro voice {voice!r}. Available: {', '.join(self.installation.voices[:12])}"
            )
        try:
            voices_path = self.installation.artifact("voices").path
        except KeyError as exc:
            raise RuntimeContractError("Kokoro installation has no voices archive") from exc
        with np.load(voices_path, allow_pickle=False) as archive:
            if voice not in archive.files:
                raise RuntimeContractError(
                    f"Unknown Kokoro voice {voice!r}. Available: {', '.join(archive.files[:12])}"
                )
            style = np.asarray(archive[voice])
        if style.ndim == 1:
            selected = style
        else:
            index = min(max(token_count - 1, 0), style.shape[0] - 1)
            selected = style[index]
        if selected.ndim == 1:
            selected = selected[None, :]
        return np.asarray(selected, dtype=dtype)

    def infer(
        self,
        tokens: Sequence[int],
        *,
        voice: str | None = None,
        style: np.ndarray | None = None,
        speed: float = 1.0,
        **_: Any,
    ) -> AudioResult:
        token_values = list(tokens)
        voice = voice or self.installation.default_voice
        names = set(self.session.input_names)
        token_name = "input_ids" if "input_ids" in names else "tokens"
        style_name = "ref_s" if "ref_s" in names else "style"
        token_spec = self._input_spec(token_name)
        token_dtype = self._numpy_dtype(token_spec, default=np.int64)
        self._validate_token_limit(token_spec, len(token_values))
        padded = np.asarray([[0, *token_values, 0]], dtype=token_dtype)

        style_spec = self._input_spec(style_name)
        style_dtype = self._numpy_dtype(style_spec, default=np.float32)
        if style is None:
            if not voice:
                raise RuntimeContractError("A Kokoro voice must be provided")
            style_value = self._voice_style(voice, len(token_values), style_dtype)
        else:
            style_value = np.asarray(style, dtype=style_dtype)
            if style_value.ndim == 1:
                style_value = style_value[None, :]

        inputs: dict[str, np.ndarray] = {
            token_name: padded,
            style_name: style_value,
        }
        if "speed" in names:
            inputs["speed"] = np.asarray([speed], dtype=self._numpy_dtype(self._input_spec("speed"), default=np.float32))
        missing = {token_name, style_name} - names
        if missing:
            raise RuntimeContractError(
                f"Kokoro model is missing expected inputs: {', '.join(sorted(missing))}"
            )
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
        metadata: dict[str, Any] = {
            "system": "kokoro",
            "voice": voice,
            "speed": speed,
            "output_names": output_names,
        }
        return AudioResult(
            audio=audio,
            sample_rate=int(self.installation.sample_rate or 24000),
            metadata=metadata,
            timings=timings,
            outputs=auxiliary,
        )

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
        raise RuntimeContractError(f"Unsupported Kokoro tensor type {spec.ort_type!r}")

    @staticmethod
    def _validate_token_limit(spec: TensorSpec | None, token_count: int) -> None:
        if spec is None or not spec.shape:
            return
        maximum = spec.shape[-1]
        if isinstance(maximum, int) and maximum > 0 and token_count + 2 > maximum:
            raise RuntimeContractError(
                f"Kokoro token sequence has {token_count} tokens, maximum is {maximum - 2}"
            )

    @staticmethod
    def _canonical_audio(value: np.ndarray) -> np.ndarray:
        audio = np.asarray(value)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Kokoro audio output must be one-dimensional")
        return audio.astype(np.float32, copy=False)
