from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..types import InferenceResult, Installation, TensorSpec


def infer_static_kokoro(
    installation: Installation,
    session: Any,
    token_ids: Sequence[int],
    *,
    style: np.ndarray | Sequence[float] | None,
    speed: float,
) -> InferenceResult:
    """Run the shared static Kokoro tokens, style, and speed graph contract."""
    if style is None:
        raise RuntimeContractError("A model-ready Kokoro style must be provided")

    token_values = list(token_ids)
    names = set(session.input_names)
    token_name = "input_ids" if "input_ids" in names else "tokens"
    style_name = "ref_s" if "ref_s" in names else "style"
    token_spec = _input_spec(session, token_name)
    token_dtype = _numpy_dtype(token_spec, default=np.int64)
    _validate_token_limit(token_spec, len(token_values))
    padded = np.asarray([[0, *token_values, 0]], dtype=token_dtype)

    style_spec = _input_spec(session, style_name)
    style_dtype = _numpy_dtype(style_spec, default=np.float32)
    style_value = np.asarray(style, dtype=style_dtype)
    if style_value.ndim == 1:
        style_value = style_value[None, :]

    inputs: dict[str, np.ndarray] = {token_name: padded, style_name: style_value}
    if "speed" in names:
        inputs["speed"] = np.asarray(
            [speed], dtype=_numpy_dtype(_input_spec(session, "speed"), default=np.float32)
        )
    missing = {token_name, style_name} - names
    if missing:
        raise RuntimeContractError(
            f"Kokoro model is missing expected inputs: {', '.join(sorted(missing))}"
        )

    outputs = session.run(inputs)
    output_names = tuple(getattr(session, "output_names", ()))
    if not outputs:
        raise RuntimeContractError("Kokoro model returned no outputs")
    if len(output_names) != len(outputs):
        output_names = tuple(f"output_{index}" for index in range(len(outputs)))
    named_outputs = {
        name: np.asarray(value) for name, value in zip(output_names, outputs, strict=True)
    }
    audio_name = output_names[0]
    audio = _canonical_audio(named_outputs[audio_name])
    auxiliary = {name: value for name, value in named_outputs.items() if name != audio_name}
    timings = _select_timings(installation, auxiliary)

    return InferenceResult(
        audio=audio,
        sample_rate=int(installation.sample_rate or 24000),
        timings=timings,
        outputs=auxiliary,
        metadata={"system": "kokoro", "speed": speed},
    )


def _input_spec(session: Any, name: str) -> TensorSpec | None:
    for spec in getattr(session, "input_specs", ()):
        if spec.name == name:
            return spec
    return None


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


def _validate_token_limit(spec: TensorSpec | None, token_count: int) -> None:
    if spec is None or not spec.shape:
        return
    maximum = spec.shape[-1]
    if isinstance(maximum, int) and maximum > 0 and token_count + 2 > maximum:
        raise RuntimeContractError(
            f"Kokoro token sequence has {token_count} tokens, maximum is {maximum - 2}"
        )


def _select_timings(
    installation: Installation, auxiliary: dict[str, np.ndarray]
) -> np.ndarray | None:
    declared = installation.timing_output
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


def _canonical_audio(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value)
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        if audio.ndim == 2 and 1 in audio.shape:
            audio = audio.reshape(-1)
        else:
            raise RuntimeContractError("Kokoro audio output must be one-dimensional")
    return audio.astype(np.float32, copy=False)
