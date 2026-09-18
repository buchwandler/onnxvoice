from __future__ import annotations

import numpy as np
import pytest

from onnxvoice.errors import IntegrityError
from onnxvoice.types import InferenceResult
from onnxvoice.validation import validate_audio


def test_audio_validation_accepts_finite_signal():
    report = validate_audio(InferenceResult(np.array([0.1, -0.2], dtype=np.float32), 24000))
    assert report.ok


def test_audio_validation_rejects_silence():
    with pytest.raises(IntegrityError, match="silent"):
        validate_audio(InferenceResult(np.zeros(10, dtype=np.float32), 24000))
