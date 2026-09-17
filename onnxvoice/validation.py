from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import IntegrityError
from .runtime import OnnxSession
from .store import AssetStore
from .types import AudioResult, Installation


@dataclass(frozen=True, slots=True)
class ValidationReport:
    ok: bool
    checks: tuple[str, ...]


def verify_installation(store: AssetStore, installation: Installation) -> ValidationReport:
    store.verify(installation)
    return ValidationReport(True, ("files", "size", "sha256"))


def validate_onnx(
    path: str | Path, *, providers: tuple[str, ...] = ("CPUExecutionProvider",)
) -> ValidationReport:
    session = OnnxSession(path, providers=providers)
    _ = session.input_names
    _ = session.output_names
    session.close()
    return ValidationReport(True, ("onnx-load", "inputs", "outputs"))


def validate_audio(result: AudioResult) -> ValidationReport:
    audio = np.asarray(result.audio)
    if audio.size == 0:
        raise IntegrityError("Inference returned empty audio")
    if not np.issubdtype(audio.dtype, np.number):
        raise IntegrityError(f"Inference returned non-numeric audio dtype: {audio.dtype}")
    if not np.all(np.isfinite(audio)):
        raise IntegrityError("Inference returned NaN or infinite samples")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    if peak == 0.0 or rms == 0.0:
        raise IntegrityError("Inference returned silent audio")
    return ValidationReport(True, ("numeric", "finite", "non-silent"))
