from __future__ import annotations

import os

import numpy as np
import pytest

from onnxvoice import NotInstalledError, OnnxVoice


@pytest.mark.integration
@pytest.mark.resource_heavy
@pytest.mark.parametrize(
    ("bundle", "model_type", "quality"),
    [
        ("voice-design-1.7b", "voice_design", "fp32"),
        ("voice-design-1.7b", "voice_design", "int4"),
        ("base-0.6b", "base", "fp32"),
        ("base-0.6b", "base", "int4"),
    ],
)
def test_qwen3_real_bundle_cpu_inference(bundle: str, model_type: str, quality: str) -> None:
    """Exercise actual WaveKat graphs when an offline managed bundle is installed."""
    pytest.importorskip("onnxruntime")
    manager = OnnxVoice(
        cache_dir=os.environ.get("ONNXVOICE_TEST_CACHE"),
        offline=True,
    )
    try:
        installation = manager.resolve(f"qwen3:{bundle}", quality=quality)
    except NotInstalledError:
        pytest.skip(f"qwen3:{bundle}:{quality} is not installed in the offline OnnxVoice store")

    adapter = manager.open(installation, providers="cpu")
    try:
        assert adapter.model_type == model_type
        config = adapter.config
        # The integration path intentionally validates the model/runtime bridge with
        # ABI-valid token IDs; text quality and frontend tokenization are covered elsewhere.
        text_ids = (
            config.im_start_token_id,
            1,
            2,
            3,
            4,
            config.im_end_token_id,
            5,
            config.im_start_token_id,
            6,
            7,
        )
        generation = {
            "language": "english",
            "temperature": 0.8,
            "top_k": 1,
            "repetition_penalty": 1.0,
            "max_new_tokens": 2,
            "seed": 17,
        }
        if model_type == "voice_design":
            result = adapter.infer(
                text_ids,
                instruction_token_ids=(config.im_start_token_id, 1, 2, config.im_end_token_id, 3),
                **generation,
            )
        else:
            sample_rate = config.sample_rate
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            reference_audio = np.asarray(0.1 * np.sin(2 * np.pi * 220 * time), dtype=np.float32)
            reference_ids = (
                config.im_start_token_id,
                1,
                2,
                3,
                4,
                config.im_end_token_id,
                5,
            )
            voice = adapter.prepare_voice(
                reference_audio,
                sample_rate=sample_rate,
                reference_token_ids=reference_ids,
            )
            result = adapter.infer(text_ids, voice_state=voice, **generation)

        assert result.sample_rate == config.sample_rate == 24_000
        assert result.audio.dtype == np.float32
        assert result.audio.ndim == 1 and result.audio.size > 0
        assert np.isfinite(result.audio).all()
        assert result.outputs["codes"].shape[1] == 16
    finally:
        adapter.close()
