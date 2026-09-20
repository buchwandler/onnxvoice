from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, TensorSpec
from .base import SystemAdapter
from .kokoro_split import SplitKokoroRuntime


class KokoroAdapter(SystemAdapter):
    system = "kokoro"

    def __init__(self, *args: Any, session_factory: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._split_runtime: SplitKokoroRuntime | None = None
        self._split_session_factory = session_factory

    def _uses_split_runtime(self) -> bool:
        runtime = self.installation.metadata.get("runtime") or {}
        return runtime.get("layout") in {"split", "multi", "split-onnx-v1"}

    def _split(self) -> SplitKokoroRuntime:
        if self._split_runtime is None:
            self._split_runtime = SplitKokoroRuntime(
                self.installation,
                session_factory=self._split_session_factory,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._split_runtime

    @property
    def session(self) -> OnnxSession:
        if self._session is None:
            models = [
                artifact for artifact in self.installation.artifacts if artifact.role == "model"
            ]
            runtime = self.installation.metadata.get("runtime") or {}
            if len(models) != 1 or runtime.get("layout") in {"split", "multi", "split-onnx-v1"}:
                raise CapabilityError(
                    "Split or multi-component Kokoro layouts do not expose a single session"
                )
            self._session = OnnxSession(
                models[0].path,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._session

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: np.ndarray | Sequence[float] | None = None,
        speed: float = 1.0,
        seed: int = 1234,
        **_: Any,
    ) -> InferenceResult:
        if self._uses_split_runtime():
            if style is None:
                raise RuntimeContractError("A model-ready Kokoro style must be provided")
            return self._split().infer(token_ids, style=style, speed=speed, seed=seed)
        if style is None:
            raise RuntimeContractError("A model-ready Kokoro style must be provided")
        token_values = list(token_ids)
        names = set(self.session.input_names)
        token_name = "input_ids" if "input_ids" in names else "tokens"
        style_name = "ref_s" if "ref_s" in names else "style"
        token_spec = self._input_spec(token_name)
        token_dtype = self._numpy_dtype(token_spec, default=np.int64)
        self._validate_token_limit(token_spec, len(token_values))
        padded = np.asarray([[0, *token_values, 0]], dtype=token_dtype)

        style_spec = self._input_spec(style_name)
        style_dtype = self._numpy_dtype(style_spec, default=np.float32)
        style_value = np.asarray(style, dtype=style_dtype)
        if style_value.ndim == 1:
            style_value = style_value[None, :]

        inputs: dict[str, np.ndarray] = {token_name: padded, style_name: style_value}
        if "speed" in names:
            inputs["speed"] = np.asarray(
                [speed], dtype=self._numpy_dtype(self._input_spec("speed"), default=np.float32)
            )
        missing = {token_name, style_name} - names
        if missing:
            raise RuntimeContractError(
                f"Kokoro model is missing expected inputs: {', '.join(sorted(missing))}"
            )
        outputs = self.session.run(inputs)
        output_names = tuple(getattr(self.session, "output_names", ()))
        if not outputs:
            raise RuntimeContractError("Kokoro model returned no outputs")
        if len(output_names) != len(outputs):
            output_names = tuple(f"output_{index}" for index in range(len(outputs)))
        named_outputs = {
            name: np.asarray(value) for name, value in zip(output_names, outputs, strict=True)
        }
        audio_name = output_names[0]
        audio = self._canonical_audio(named_outputs[audio_name])
        auxiliary = {name: value for name, value in named_outputs.items() if name != audio_name}
        timings = self._select_timings(auxiliary)

        return InferenceResult(
            audio=audio,
            sample_rate=int(self.installation.sample_rate or 24000),
            timings=timings,
            outputs=auxiliary,
            metadata={"system": "kokoro", "speed": speed},
        )

    def close(self) -> None:
        if self._split_runtime is not None:
            self._split_runtime.close()
            self._split_runtime = None
        super().close()

    def diagnostics(self) -> Any:
        if self._split_runtime is not None:
            return self._split_runtime.diagnostics()
        return super().diagnostics()

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

    def _select_timings(self, auxiliary: dict[str, np.ndarray]) -> np.ndarray | None:
        declared = self.installation.timing_output
        if declared and declared in auxiliary:
            return auxiliary[declared]
        for name in ("pred_dur", "duration", "durations", "timing", "timestamp"):
            if name in auxiliary:
                return auxiliary[name]
        for prefix in ("duration", "timing", "timestamp"):
            for name in sorted(auxiliary):
                if name.lower().startswith(prefix):
                    return auxiliary[name]
        return None

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
