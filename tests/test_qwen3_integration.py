from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from onnxvoice import NotInstalledError, OnnxVoice

_REQUIRE_REAL = os.environ.get("ONNXVOICE_REQUIRE_QWEN3_REAL") == "1"


def _require_runtime() -> None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        message = (
            "onnxruntime is required for the Qwen3 release gate"
            if _REQUIRE_REAL
            else "onnxruntime is not installed"
        )
        if _REQUIRE_REAL:
            pytest.fail(message)
        pytest.skip(message)


def _resolve_or_skip(manager, ref: str, quality: str):
    try:
        return manager.resolve(ref, quality=quality)
    except NotInstalledError:
        message = f"{ref}:{quality} is not installed in the offline OnnxVoice store"
        if _REQUIRE_REAL:
            pytest.fail(message)
        pytest.skip(message)


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
    _require_runtime()
    manager = OnnxVoice(
        cache_dir=os.environ.get("ONNXVOICE_TEST_CACHE"),
        offline=True,
    )
    installation = _resolve_or_skip(manager, f"qwen3:{bundle}", quality)

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
        assert np.max(np.abs(result.audio)) > 1e-6
        rms = float(np.sqrt(np.mean(np.square(result.audio, dtype=np.float64))))
        assert rms > 1e-7
        assert "CPUExecutionProvider" in result.metadata["providers_active"]
        assert result.outputs["codes"].shape[1] == 16
    finally:
        adapter.close()


def test_missing_runtime_skips_in_developer_mode(monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_REQUIRE_REAL", False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(pytest.skip.Exception, match="onnxruntime is not installed"):
        _require_runtime()


def test_missing_runtime_fails_in_release_mode(monkeypatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_REQUIRE_REAL", True)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(pytest.fail.Exception, match="onnxruntime is required"):
        _require_runtime()


def test_missing_profile_skips_in_developer_mode(monkeypatch) -> None:
    class MissingBundleManager:
        def resolve(self, ref: str, quality: str) -> None:
            raise NotInstalledError(f"{ref}:{quality}")

    monkeypatch.setattr(sys.modules[__name__], "_REQUIRE_REAL", False)

    with pytest.raises(pytest.skip.Exception, match="qwen3:voice-design-1.7b:int4"):
        _resolve_or_skip(MissingBundleManager(), "qwen3:voice-design-1.7b", "int4")


def test_missing_profile_fails_in_release_mode(monkeypatch) -> None:
    class MissingBundleManager:
        def resolve(self, ref: str, quality: str) -> None:
            raise NotInstalledError(f"{ref}:{quality}")

    monkeypatch.setattr(sys.modules[__name__], "_REQUIRE_REAL", True)

    with pytest.raises(pytest.fail.Exception, match="qwen3:voice-design-1.7b:int4"):
        _resolve_or_skip(MissingBundleManager(), "qwen3:voice-design-1.7b", "int4")
