"""Tests for PocketAdapter multi-session management and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.pocket import (
    POCKET_REQUIRED_METADATA_FIELDS,
    PocketAdapter,
    PocketVoiceState,
)
from onnxvoice.types import InferenceResult, Installation, InstalledArtifact


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


class TestPocketVoiceState:
    """Test PocketVoiceState dataclass."""

    def test_basic_creation(self) -> None:
        values = {"encoder_output": np.zeros((1, 10, 64))}
        state = PocketVoiceState(values=values)
        assert "encoder_output" in state.values
        assert state.metadata == {}

    def test_with_metadata(self) -> None:
        values = {"encoder_output": np.zeros((1, 10, 64))}
        metadata = {"sample_rate": 24000}
        state = PocketVoiceState(values=values, metadata=metadata)
        assert state.metadata["sample_rate"] == 24000

    def test_frozen(self) -> None:
        state = PocketVoiceState(values={"test": np.array([1, 2, 3])})
        with pytest.raises(AttributeError):
            state.values = {}  # type: ignore


class TestPocketAdapterInit:
    """Test PocketAdapter initialization."""

    def test_basic_init(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)
        assert adapter.system == "pocket"
        assert adapter._sessions == {}
        assert adapter._validated is False

    def test_with_providers(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")
        assert adapter.providers == "CPUExecutionProvider"


class TestPocketAdapterValidation:
    """Test PocketAdapter bundle metadata validation."""

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        installation = _make_installation(metadata={})
        adapter = PocketAdapter(installation)

        with pytest.raises(RuntimeContractError, match="not found"):
            adapter._validate_bundle_metadata()

    def test_runtime_metadata_fallback(self, tmp_path: Path) -> None:
        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
        }
        installation = _make_installation(metadata={"runtime": runtime_metadata})
        adapter = PocketAdapter(installation)

        result = adapter._validate_bundle_metadata()
        assert result["schema_version"] == 2
        assert result["sample_rate"] == 24000

    def test_validated_metadata_cached(self, tmp_path: Path) -> None:
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
        installation = _make_installation(metadata={"runtime": runtime_metadata})
        adapter = PocketAdapter(installation)

        # First call validates
        result1 = adapter._ensure_validated()
        assert adapter._validated is True

        # Second call returns cached
        result2 = adapter._ensure_validated()
        assert result1 is result2

    def test_missing_required_fields_raises(self, tmp_path: Path) -> None:
        runtime_metadata = {
            "schema_version": 2,
            # Missing required fields
        }
        installation = _make_installation(metadata={"runtime": runtime_metadata})
        adapter = PocketAdapter(installation)

        with pytest.raises(RuntimeContractError, match="missing required fields"):
            adapter._ensure_validated()

    def test_all_required_fields_present(self, tmp_path: Path) -> None:
        runtime_metadata = {field: f"value_{field}" for field in POCKET_REQUIRED_METADATA_FIELDS}
        installation = _make_installation(metadata={"runtime": runtime_metadata})
        adapter = PocketAdapter(installation)

        adapter._ensure_validated()
        assert adapter._validated is True


class TestPocketAdapterSessions:
    """Test PocketAdapter multi-session management."""

    def test_get_session_lazy_creation(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"dummy")

        installation = _make_installation(
            artifacts={"flow_lm_main": model_file},
            metadata={"runtime": {"schema_version": 2, "sample_rate": 24000}},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # Session should not exist yet
        assert "flow_lm_main" not in adapter._sessions

    def test_multiple_sessions(self, tmp_path: Path) -> None:
        files = {}
        for role in ["flow_lm_main", "flow_lm_flow", "mimi_decoder", "text_conditioner"]:
            path = tmp_path / f"{role}.onnx"
            path.write_bytes(b"dummy")
            files[role] = path

        installation = _make_installation(
            artifacts=files,
            metadata={"runtime": {"schema_version": 2, "sample_rate": 24000}},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # All sessions should be lazily created
        assert len(adapter._sessions) == 0

    def test_close_clears_sessions(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        # Manually add a mock session
        mock_session = MagicMock()
        adapter._sessions["test"] = mock_session

        adapter.close()
        assert adapter._sessions == {}
        mock_session.close.assert_called_once()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        mock_session = MagicMock()
        adapter._sessions["test"] = mock_session

        adapter.close()
        adapter.close()  # Should not raise

    def test_context_manager(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        with adapter as ctx:
            assert ctx is adapter

    def test_owned_sessions_returns_all(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        mock_session1 = MagicMock()
        mock_session2 = MagicMock()
        adapter._sessions["s1"] = mock_session1
        adapter._sessions["s2"] = mock_session2

        owned = adapter._owned_sessions()
        assert len(owned) == 2
        assert mock_session1 in owned
        assert mock_session2 in owned


class TestPocketAdapterPrepareVoice:
    """Test PocketAdapter.prepare_voice()."""

    def test_prepare_voice_requires_bundle_metadata(self, tmp_path: Path) -> None:
        installation = _make_installation(metadata={})
        adapter = PocketAdapter(installation)

        with pytest.raises(RuntimeContractError, match="not found"):
            adapter.prepare_voice(np.zeros(24000), sample_rate=24000)

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_prepare_voice_with_mock_session(
        self, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        # Setup mock session
        mock_session = MagicMock()
        mock_session.output_names = ("encoded",)
        mock_session.run.return_value = [np.zeros((1, 10, 64))]
        mock_session_cls.return_value = mock_session

        # Create installation with mimi_encoder artifact
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

        audio = np.random.randn(24000).astype(np.float32)
        state = adapter.prepare_voice(audio, sample_rate=24000)

        assert isinstance(state, PocketVoiceState)
        assert "encoded" in state.values
        assert state.metadata["sample_rate"] == 24000

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_prepare_voice_reshapes_1d_audio(
        self, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
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

        # 1D audio should be reshaped to [1, 1, samples]
        audio = np.random.randn(24000).astype(np.float32)
        adapter.prepare_voice(audio, sample_rate=24000)

        call_args = mock_session.run.call_args
        input_audio = call_args[0][0]["audio"]
        assert input_audio.shape == (1, 1, 24000)


class TestPocketAdapterInfer:
    """Test PocketAdapter.infer()."""

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_infer_with_mock_sessions(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        # Setup mock sessions
        mock_sessions: dict[str, MagicMock] = {}

        def create_session(path: Any, **kwargs: Any) -> MagicMock:
            component = kwargs.get("component", "unknown")
            session = MagicMock()
            session.output_names = ("output",)
            if component == "text_conditioner":
                session.run.return_value = [np.zeros((1, 10, 128))]
            elif component == "flow_lm_main":
                session.run.return_value = [np.zeros((1, 1, 64))]
                session.output_names = ("latent",)
            elif component == "flow_lm_flow":
                session.run.return_value = [np.zeros((1, 10, 64))]
                session.output_names = ("spectral",)
            elif component == "mimi_decoder":
                session.run.return_value = [np.random.randn(1, 1, 24000).astype(np.float32)]
            else:
                session.run.return_value = [np.zeros((1, 10, 64))]
            mock_sessions[component] = session
            return session

        mock_session_cls.side_effect = create_session

        # Create installation with all required artifacts
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

        # Save bos_conditioning as numpy file
        bos_path = tmp_path / "bos_conditioning.npy"
        np.save(bos_path, np.zeros((1, 1, 128)))
        artifacts["bos_conditioning"] = bos_path

        runtime_metadata = {
            "schema_version": 2,
            "sample_rate": 24000,
            "samples_per_frame": 1,
            "latent_dim": 64,
            "conditioning_dim": 128,
            "flow_lm_state_manifest": {"state": {"shape": [1, 10, 64], "dtype": "float32"}},
            "mimi_state_manifest": {"state": {"shape": [1, 10, 64], "dtype": "float32"}},
            "insert_bos_before_voice": True,
            "bos_before_voice_file": "state",
            "max_token_per_chunk": 10,
        }
        installation = _make_installation(
            artifacts=artifacts,
            metadata={"runtime": runtime_metadata},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        token_ids = [1, 2, 3, 4, 5]
        result = adapter.infer(token_ids, temperature=0.5, lsd_steps=2)

        assert isinstance(result, InferenceResult)
        assert result.sample_rate == 24000
        assert result.metadata["system"] == "pocket"
        assert result.metadata["temperature"] == 0.5
        assert result.metadata["lsd_steps"] == 2

    def test_infer_requires_bundle_metadata(self, tmp_path: Path) -> None:
        installation = _make_installation(metadata={})
        adapter = PocketAdapter(installation)

        with pytest.raises(RuntimeContractError, match="not found"):
            adapter.infer([1, 2, 3])

    @patch("onnxvoice.systems.pocket.OnnxSession")
    def test_infer_with_voice_state(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        # Setup mock sessions
        def create_session(path: Any, **kwargs: Any) -> MagicMock:
            component = kwargs.get("component", "unknown")
            session = MagicMock()
            if component == "text_conditioner":
                session.run.return_value = [np.zeros((1, 5, 128))]
                session.output_names = ("conditioning",)
            elif component == "flow_lm_main":
                session.run.return_value = [np.zeros((1, 1, 64))]
                session.output_names = ("latent",)
            elif component == "flow_lm_flow":
                session.run.return_value = [np.zeros((1, 5, 64))]
                session.output_names = ("spectral",)
            elif component == "mimi_decoder":
                session.run.return_value = [np.random.randn(1, 1, 12000).astype(np.float32)]
                session.output_names = ("audio",)
            else:
                session.run.return_value = [np.zeros((1, 5, 64))]
                session.output_names = ("output",)
            return session

        mock_session_cls.side_effect = create_session

        # Create installation
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
            "flow_lm_state_manifest": {"voice": {"shape": [1, 5, 64], "dtype": "float32"}},
            "mimi_state_manifest": {},
            "insert_bos_before_voice": False,
            "bos_before_voice_file": "voice",
            "max_token_per_chunk": 5,
        }
        installation = _make_installation(
            artifacts=artifacts,
            metadata={"runtime": runtime_metadata},
        )
        adapter = PocketAdapter(installation, providers="CPUExecutionProvider")

        # Create voice state
        voice_state = PocketVoiceState(
            values={"voice": np.ones((1, 5, 64), dtype=np.float32)},
            metadata={"sample_rate": 24000},
        )

        token_ids = [1, 2, 3]
        result = adapter.infer(token_ids, voice_state=voice_state)

        assert isinstance(result, InferenceResult)
        assert result.sample_rate == 24000


class TestPocketAdapterInitializeState:
    """Test PocketAdapter._initialize_state()."""

    def test_initialize_from_dict_manifest(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        manifest = {
            "hidden": {"shape": [1, 10, 64], "dtype": "float32"},
            "cell": {"shape": [1, 10, 64], "dtype": "float32"},
        }

        state = adapter._initialize_state(manifest, batch_size=1)

        assert "hidden" in state
        assert "cell" in state
        assert state["hidden"].shape == (1, 10, 64)
        assert state["hidden"].dtype == np.float32

    def test_initialize_from_list_manifest(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        manifest = {
            "simple": [1, 10, 64],
        }

        state = adapter._initialize_state(manifest, batch_size=1)

        assert "simple" in state
        assert state["simple"].shape == (1, 10, 64)

    def test_initialize_with_batch_dimension(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        manifest = {
            "hidden": {"shape": [None, 10, 64], "dtype": "float32"},
        }

        state = adapter._initialize_state(manifest, batch_size=2)

        assert state["hidden"].shape == (2, 10, 64)

    def test_initialize_empty_manifest(self, tmp_path: Path) -> None:
        installation = _make_installation()
        adapter = PocketAdapter(installation)

        state = adapter._initialize_state({}, batch_size=1)
        assert state == {}
