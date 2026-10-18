from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from onnxvoice import OnnxVoice


@pytest.mark.integration
@pytest.mark.resource_heavy
def test_bruno_cpu_real_bundle_smoke_and_optional_parity() -> None:
    raw_bundle = os.environ.get("ONNXVOICE_KITTEN2_BUNDLE")
    if not raw_bundle:
        pytest.skip("set ONNXVOICE_KITTEN2_BUNDLE to a complete KittenTTS 2 bundle")
    bundle_path = Path(raw_bundle)
    if bundle_path.is_dir():
        bundle_path /= "bundle.json"

    runtime = OnnxVoice.open_local(
        system="kitten2",
        files={"bundle": bundle_path},
        providers="cpu",
    )
    try:
        result = runtime.infer_text(
            "Hello from KittenTTS 2.",
            voice="Bruno",
            preset="stable",
            seed=7,
            max_new_tokens=64,
        )
        audio = result.audio
        assert result.sample_rate == 24000
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert float(np.max(np.abs(audio))) > 1e-5
        assert float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) > 1e-6
        reference_path = os.environ.get("ONNXVOICE_KITTEN2_REFERENCE_AUDIO")
        if reference_path:
            reference_audio = np.load(reference_path, allow_pickle=False)
            assert reference_audio.shape == audio.shape
            np.testing.assert_allclose(audio, reference_audio, rtol=1e-5, atol=1e-5)
    finally:
        runtime.close()
