"""Tests for Pocket local runtime opening with files parameter."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnxvoice import OnnxVoice


class TestOpenLocalFiles:
    """Test open_local() with files parameter."""

    def test_files_parameter_basic(self, tmp_path: Path) -> None:
        """Test basic files parameter usage."""
        # Create dummy model file
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy onnx content")

        # Pocket adapter should accept files parameter
        runtime = OnnxVoice.open_local(
            system="pocket",
            files={
                "flow_lm_main": str(model_file),
            },
            sample_rate=24000,
        )
        assert runtime is not None
        runtime.close()

    def test_files_and_model_mutually_exclusive(self, tmp_path: Path) -> None:
        """Test that files and model parameters are mutually exclusive."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Use model or files, not both"):
            OnnxVoice.open_local(
                system="pocket",
                model=str(model_file),
                files={"flow_lm_main": str(model_file)},
            )

    def test_files_and_artifacts_mutually_exclusive(self, tmp_path: Path) -> None:
        """Test that files and artifacts parameters are mutually exclusive."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Use artifacts or files, not both"):
            OnnxVoice.open_local(
                system="pocket",
                artifacts={"flow_lm_main": str(model_file)},
                files={"flow_lm_main": str(model_file)},
            )

    def test_files_with_config_raises(self, tmp_path: Path) -> None:
        """Test that config cannot be passed with files."""
        model_file = tmp_path / "model.onnx"
        config_file = tmp_path / "config.json"
        model_file.write_bytes(b"dummy")
        config_file.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="config must be included in files"):
            OnnxVoice.open_local(
                system="pocket",
                files={"flow_lm_main": str(model_file)},
                config=str(config_file),
            )

    def test_files_with_voices_raises(self, tmp_path: Path) -> None:
        """Test that voices cannot be passed with files."""
        model_file = tmp_path / "model.onnx"
        voices_file = tmp_path / "voices.bin"
        model_file.write_bytes(b"dummy")
        voices_file.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="voices must be included in files"):
            OnnxVoice.open_local(
                system="pocket",
                files={"flow_lm_main": str(model_file)},
                voices=str(voices_file),
            )

    def test_files_with_metadata(self, tmp_path: Path) -> None:
        """Test that metadata can be passed with files."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        # Pocket adapter should accept metadata
        runtime = OnnxVoice.open_local(
            system="pocket",
            files={"flow_lm_main": str(model_file)},
            metadata={"custom_key": "custom_value"},
            sample_rate=24000,
        )
        assert runtime is not None
        runtime.close()

    def test_files_with_artifact_metadata(self, tmp_path: Path) -> None:
        """Test that artifact_metadata can be passed with files."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        # Pocket adapter should accept artifact_metadata
        runtime = OnnxVoice.open_local(
            system="pocket",
            files={"flow_lm_main": str(model_file)},
            artifact_metadata={"flow_lm_main": {"custom": "value"}},
            sample_rate=24000,
        )
        assert runtime is not None
        runtime.close()

    def test_backward_compatible_model_parameter(self, tmp_path: Path) -> None:
        """Test that old model parameter still works."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        # Piper adapter should accept model parameter (backward compatible)
        # This will succeed because piper adapter doesn't validate ONNX content at init
        runtime = OnnxVoice.open_local(
            system="piper",
            model=str(model_file),
        )
        assert runtime is not None
        runtime.close()

    def test_backward_compatible_artifacts_parameter(self, tmp_path: Path) -> None:
        """Test that old artifacts parameter still works."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        # Piper adapter should accept artifacts parameter (backward compatible)
        runtime = OnnxVoice.open_local(
            system="piper",
            artifacts={"model": str(model_file)},
        )
        assert runtime is not None
        runtime.close()
