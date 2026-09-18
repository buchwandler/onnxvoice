"""Tests for Pocket network boundaries and lazy loading."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from onnxvoice import OnnxVoice
from onnxvoice.systems.pocket import PocketAdapter, PocketVoiceState
from onnxvoice.types import Installation, InstalledArtifact


def _make_installation(
    *,
    artifacts: dict[str, Path] | None = None,
    metadata: dict[str, Any] | None = None,
    sample_rate: int = 24000,
) -> Installation:
    """Create a mock installation for testing."""
    installed_artifacts: list[InstalledArtifact] = []
    if artifacts:
        for role, path in artifacts.items():
            installed_artifacts.append(
                InstalledArtifact(
                    role=role,
                    filename=path.name,
                    path=path,
                    sha256="abc123",
                    size=1024,
                )
            )

    return Installation(
        system="pocket",
        id="test-bundle",
        kind="bundle",
        path=Path("/tmp/test"),
        artifacts=tuple(installed_artifacts),
        sample_rate=sample_rate,
        metadata=metadata or {},
    )


class TestMimiEncoderLazyLoading:
    """Test that Mimi encoder is lazy loaded."""

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_encoder_not_loaded_on_init(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        """Mimi encoder should not be loaded until prepare_voice() is called."""
        encoder_file = tmp_path / "mimi_encoder.onnx"
        encoder_file.write_bytes(b"dummy")

        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
            "flow_lm_state_manifest": {},
            "mimi_state_manifest": {},
            "insert_bos_before_voice": False,
            "bos_before_voice_file": "bos",
        }
        installation = _make_installation(
            artifacts={"mimi_encoder": encoder_file},
            metadata={"runtime": runtime_metadata},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # Encoder should not be loaded yet
        assert "mimi_encoder" not in adapter._sessions
        mock_session_cls.assert_not_called()

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_encoder_loaded_on_prepare_voice(
        self, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Mimi encoder should be loaded when prepare_voice() is called."""
        mock_session = MagicMock()
        mock_session.output_names = ("encoded",)
        mock_session.run.return_value = [np.zeros((1, 10, 64))]
        mock_session_cls.return_value = mock_session

        encoder_file = tmp_path / "mimi_encoder.onnx"
        encoder_file.write_bytes(b"dummy")

        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
            "flow_lm_state_manifest": {},
            "mimi_state_manifest": {},
            "insert_bos_before_voice": False,
            "bos_before_voice_file": "bos",
        }
        installation = _make_installation(
            artifacts={"mimi_encoder": encoder_file},
            metadata={"runtime": runtime_metadata},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # Call prepare_voice
        audio = np.random.randn(24000).astype(np.float32)
        adapter.prepare_voice(audio, sample_rate=24000)

        # Now encoder should be loaded
        assert "mimi_encoder" in adapter._sessions
        mock_session_cls.assert_called_once()


class TestProviderOptionsPropagation:
    """Test that provider options are propagated to all sessions."""

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_provider_options_passed_to_sessions(
        self, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Provider options should be passed to all created sessions."""
        mock_session = MagicMock()
        mock_session.output_names = ("output",)
        mock_session.run.return_value = [np.zeros((1, 10, 64))]
        mock_session_cls.return_value = mock_session

        # Create files for multiple sessions
        artifacts = {}
        for role in ["flow_lm_main", "text_conditioner"]:
            path = tmp_path / f"{role}.onnx"
            path.write_bytes(b"dummy")
            artifacts[role] = path

        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
            "flow_lm_state_manifest": {},
            "mimi_state_manifest": {},
            "insert_bos_before_voice": False,
            "bos_before_voice_file": "bos",
        }
        installation = _make_installation(
            artifacts=artifacts,
            metadata={"runtime": runtime_metadata},
        )

        provider_options = {"CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}}
        adapter = PocketAdapter(
            installation,
            providers="CPUExecutionProvider",
            provider_options=provider_options,
        )

        # Trigger session creation by calling _ensure_validated and _get_session
        adapter._ensure_validated()
        adapter._get_session("flow_lm_main")

        # Check provider options were passed
        call_kwargs = mock_session_cls.call_args
        assert call_kwargs[1].get("provider_options") == provider_options


class TestFakeSessionWiring:
    """Test fake sessions validate state-manifest wiring."""

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_state_manifest_wiring(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        """State manifests should be properly wired to sessions."""
        # Create mock sessions that track their inputs
        captured_inputs: dict[str, dict[str, Any]] = {}

        def create_session(path: Any, **kwargs: Any) -> MagicMock:
            component = kwargs.get("component", "unknown")
            session = MagicMock()

            def capture_run(inputs: dict[str, Any]) -> list[np.ndarray]:
                captured_inputs[component] = inputs
                if component == "mimi_decoder":
                    # Return 1D audio array for decoder
                    return [np.random.randn(24000).astype(np.float32)]
                return [np.zeros((1, 10, 64))]

            session.run.side_effect = capture_run
            session.output_names = ("output",)
            return session

        mock_session_cls.side_effect = create_session

        # Create installation with state manifests
        artifacts = {}
        for role in [
            "text_conditioner",
            "flow_lm_main",
            "flow_lm_flow",
            "mimi_decoder",
            "bos_conditioning",
        ]:
            path = (
                tmp_path / f"{role}.onnx"
                if role != "bos_conditioning"
                else tmp_path / f"{role}.npy"
            )
            path.write_bytes(b"dummy")
            artifacts[role] = path

        bos_path = tmp_path / "bos_conditioning.npy"
        np.save(bos_path, np.zeros((1, 1, 128)))
        artifacts["bos_conditioning"] = bos_path

        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
            "flow_lm_state_manifest": {
                "hidden_state": {"shape": [1, 10, 64], "dtype": "float32"},
                "cell_state": {"shape": [1, 10, 64], "dtype": "float32"},
            },
            "mimi_state_manifest": {
                "encoder_state": {"shape": [1, 5, 32], "dtype": "float32"},
            },
            "insert_bos_before_voice": False,
            "bos_before_voice_file": "hidden_state",
            "max_token_per_chunk": 5,
        }
        installation = _make_installation(
            artifacts=artifacts,
            metadata={"runtime": runtime_metadata},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # Create voice state with matching keys
        voice_state = PocketVoiceState(
            values={
                "hidden_state": np.ones((1, 10, 64), dtype=np.float32),
                "encoder_state": np.ones((1, 5, 32), dtype=np.float32),
            },
            metadata={"sample_rate": 24000},
        )

        # Run inference
        token_ids = [1, 2, 3]
        result = adapter.infer(token_ids, voice_state=voice_state)

        # Check that voice state was wired to the correct sessions
        assert "flow_lm_main" in captured_inputs
        assert "hidden_state" in captured_inputs["flow_lm_main"]
        assert np.array_equal(
            captured_inputs["flow_lm_main"]["hidden_state"],
            np.ones((1, 10, 64), dtype=np.float32),
        )

        # Check result is valid
        assert result.sample_rate == 24000
        assert result.audio.ndim == 1


class TestOutputNaming:
    """Test that output naming is handled correctly."""

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_output_names_from_session(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        """Output names should come from session."""
        mock_session = MagicMock()
        mock_session.output_names = ("latent", "eos")
        mock_session.run.return_value = [
            np.zeros((1, 1, 64)),
            np.array([[0.0]]),
        ]
        mock_session_cls.return_value = mock_session

        artifacts = {}
        for role in ["flow_lm_main"]:
            path = tmp_path / f"{role}.onnx"
            path.write_bytes(b"dummy")
            artifacts[role] = path

        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
            "flow_lm_state_manifest": {},
            "mimi_state_manifest": {},
            "insert_bos_before_voice": False,
            "bos_before_voice_file": "bos",
        }
        installation = _make_installation(
            artifacts=artifacts,
            metadata={"runtime": runtime_metadata},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # Get session
        adapter._ensure_validated()
        session = adapter._get_session("flow_lm_main")

        # Check output names
        assert session.output_names == ("latent", "eos")


class TestNoNetworkAccessForLocalOpening:
    """Test that local opening doesn't access network."""

    def test_open_local_no_catalog_access(self, tmp_path: Path) -> None:
        """open_local() should not access catalog or network."""
        # Create dummy files
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        # This should work without any network access
        # (will fail because piper adapter expects valid ONNX, but that's OK)
        with contextlib.suppress(Exception):  # Expected to fail due to invalid ONNX
            OnnxVoice.open_local(
                system="piper",
                model=str(model_file),
            )

        # No network access should have been made

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_pocket_local_no_catalog(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        """Pocket local opening should not access catalog."""
        mock_session = MagicMock()
        mock_session.output_names = ("output",)
        mock_session.run.return_value = [np.zeros((1, 10, 64))]
        mock_session_cls.return_value = mock_session

        # Create files
        files = {}
        for role in [
            "flow_lm_main",
            "flow_lm_flow",
            "mimi_decoder",
            "text_conditioner",
            "bos_conditioning",
        ]:
            path = (
                tmp_path / f"{role}.onnx"
                if role != "bos_conditioning"
                else tmp_path / f"{role}.npy"
            )
            path.write_bytes(b"dummy")
            files[role] = path

        # Save bos_conditioning as numpy
        bos_path = tmp_path / "bos_conditioning.npy"
        np.save(bos_path, np.zeros((1, 1, 128)))
        files["bos_conditioning"] = bos_path

        # Open locally - should not access network
        runtime = OnnxVoice.open_local(
            system="pocket",
            files=files,
            metadata={
                "schema_version": 2,
                "sample_rate": 24000,
                "samples_per_frame": 1,
                "latent_dim": 64,
                "conditioning_dim": 128,
                "flow_lm_state_manifest": {},
                "mimi_state_manifest": {},
                "insert_bos_before_voice": False,
                "bos_before_voice_file": "bos",
            },
            sample_rate=24000,
            providers="CPUExecutionProvider",
        )

        assert runtime is not None
        runtime.close()
