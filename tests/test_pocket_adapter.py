"""Tests for PocketAdapter multi-session management and inference."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
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

    def test_flow_state_representation_is_distinct(self) -> None:
        flow_state = {"cache": np.ones((1, 2), dtype=np.float32)}
        state = PocketVoiceState(flow_state=flow_state)
        assert state.embeddings is None
        assert state.flow_state is not None
        assert state.values == {}

    def test_rejects_both_conditioning_representations(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            PocketVoiceState(
                np.ones((1, 2, 3), dtype=np.float32),
                flow_state={"cache": np.ones((1,), dtype=np.float32)},
            )


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


@patch("onnxvoice.systems.pocket.OnnxSession")
def test_v2_runtime_uses_prefix_per_frame_flow_and_stateful_decode(
    mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    sessions: dict[str, MagicMock] = {}
    main_calls = 0
    main_inputs: list[dict[str, np.ndarray]] = []

    def create_session(path: Any, **kwargs: Any) -> MagicMock:
        nonlocal main_calls
        role = kwargs["component"]
        session = MagicMock()
        if role == "text_conditioner":
            session.input_names = ("token_ids",)
            session.output_names = ("embeddings",)
            session.run.return_value = [np.ones((1, 2, 4), dtype=np.float32)]
        elif role == "flow_lm_main":
            session.input_names = ("sequence", "text_embeddings")
            session.output_names = ("conditioning", "eos")

            def main_run(inputs: dict[str, Any]) -> list[np.ndarray]:
                nonlocal main_calls
                main_calls += 1
                main_inputs.append(
                    {name: np.array(value, copy=True) for name, value in inputs.items()}
                )
                eos = np.asarray([1.0 if main_calls == 4 else 0.0], dtype=np.float32)
                return [np.ones((1, 1, 3), dtype=np.float32), eos]

            session.run.side_effect = main_run
        elif role == "flow_lm_flow":
            session.input_names = ("c", "s", "t", "x")
            session.output_names = ("velocity",)
            session.run.return_value = [np.zeros((1, 1, 3), dtype=np.float32)]
        elif role == "mimi_decoder":
            session.input_names = ("latent",)
            session.output_names = ("audio",)
            session.run.return_value = [np.ones((1, 1, 8), dtype=np.float32)]
        sessions[role] = session
        return session

    mock_session_cls.side_effect = create_session

    artifacts: dict[str, Path] = {}
    for role in ("text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"):
        path = tmp_path / f"{role}.onnx"
        path.write_bytes(b"dummy")
        artifacts[role] = path
    bos = tmp_path / "bos.npy"
    np.save(bos, np.zeros((1, 1, 4), dtype=np.float32))
    artifacts["bos_conditioning"] = bos
    metadata = {
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1,
        "latent_dim": 3,
        "conditioning_dim": 4,
        "flow_lm_state_manifest": [],
        "mimi_state_manifest": [],
        "insert_bos_before_voice": False,
        "bos_before_voice_file": "bos",
        "decoder_chunk_frames": 1,
    }
    adapter = PocketAdapter(_make_installation(artifacts=artifacts, metadata={"runtime": metadata}))
    voice = PocketVoiceState(np.zeros((1, 1, 4), dtype=np.float32), 24000)
    result = adapter.infer([1, 2], voice_state=voice, max_frames=5, frames_after_eos=0)

    assert result.audio.shape == (16,)
    assert result.sample_rate == 24000
    assert np.array_equal(main_inputs[0]["text_embeddings"], np.zeros((1, 1, 4), dtype=np.float32))
    assert np.array_equal(main_inputs[1]["text_embeddings"], np.ones((1, 2, 4), dtype=np.float32))
    assert main_calls == 4  # two prefix calls plus two generated frames
    assert sessions["flow_lm_flow"].run.call_count == 2
    assert sessions["mimi_decoder"].run.call_count == 2
    assert "token_ids" in sessions["text_conditioner"].run.call_args.args[0]
    assert set(sessions["flow_lm_flow"].run.call_args.args[0]) == {"c", "s", "t", "x"}


def test_predefined_flow_state_uses_text_prefix_and_defensive_copy(tmp_path: Path) -> None:
    from onnxvoice.systems.pocket import parse_state_manifest

    text_embeddings = np.full((1, 2, 4), 2.0, dtype=np.float32)
    text = MagicMock()
    text.input_names = ("token_ids",)
    text.output_names = ("embeddings",)
    text.run.return_value = [text_embeddings]

    main_inputs: list[dict[str, np.ndarray]] = []
    main_state_inputs: list[np.ndarray] = []
    main = MagicMock()
    main.input_names = ("sequence", "text_embeddings", "state_0")
    main.output_names = ("conditioning", "eos", "out_state_0")

    def run_main(inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        main_state_inputs.append(inputs["state_0"])
        main_inputs.append({name: np.array(value, copy=True) for name, value in inputs.items()})
        return [
            np.ones((1, 1, 3), dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray(inputs["state_0"]) + 1.0,
        ]

    main.run.side_effect = run_main

    flow = MagicMock()
    flow.input_names = ("c", "s", "t", "x")
    flow.output_names = ("velocity",)
    flow.run.return_value = [np.zeros((1, 1, 3), dtype=np.float32)]

    decoder = MagicMock()
    decoder.input_names = ("latent",)
    decoder.output_names = ("audio",)
    decoder.run.return_value = [np.ones((1, 1, 8), dtype=np.float32)]

    sessions = {
        "text_conditioner": text,
        "flow_lm_main": main,
        "flow_lm_flow": flow,
        "mimi_decoder": decoder,
    }
    adapter = PocketAdapter(_make_installation())
    adapter._flow_specs = parse_state_manifest(
        [
            {
                "index": 0,
                "input_name": "state_0",
                "output_name": "out_state_0",
                "module": "module",
                "key": "cache",
                "shape": [1],
                "dtype": "float32",
                "fill": "zeros",
            }
        ],
        name="flow_lm",
    )
    metadata = {
        "sample_rate": 24000,
        "samples_per_frame": 1,
        "latent_dim": 3,
        "conditioning_dim": 4,
        "decoder_chunk_frames": 1,
    }
    imported_state = np.asarray([5.0], dtype=np.float32)
    voice_state = PocketVoiceState(flow_state={"state_0": imported_state})

    with (
        patch.object(adapter, "_ensure_validated", return_value=metadata),
        patch.object(adapter, "_get_session", side_effect=sessions.__getitem__),
    ):
        result = adapter.infer([1], voice_state=voice_state, frames_after_eos=0)

    assert result.audio.shape == (8,)
    assert len(main_inputs) == 2
    assert np.array_equal(main_inputs[0]["text_embeddings"], text_embeddings)
    assert np.array_equal(main_inputs[0]["state_0"], [5.0])
    assert np.array_equal(main_inputs[1]["state_0"], [6.0])
    assert not np.shares_memory(main_state_inputs[0], imported_state)
    assert np.array_equal(voice_state.flow_state["state_0"], [5.0])


def test_generate_latents_matches_rank_two_flow_inputs() -> None:
    adapter = PocketAdapter(_make_installation())
    main = MagicMock()
    main.output_names = ("conditioning", "eos_logit")
    main_inputs: list[dict[str, np.ndarray]] = []
    main.run.side_effect = lambda inputs: (
        main_inputs.append(inputs)
        or [
            np.ones((1, 4), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        ]
    )
    flow = MagicMock()
    flow.input_specs = tuple(
        SimpleNamespace(name=name, shape=shape)
        for name, shape in (
            ("c", ("batch", 4)),
            ("s", ("batch", 1)),
            ("t", ("batch", 1)),
            ("x", ("batch", 3)),
        )
    )
    flow.output_names = ("flow_dir",)
    flow_inputs: list[dict[str, np.ndarray]] = []
    flow.run.side_effect = lambda inputs: (
        flow_inputs.append(inputs) or [np.zeros((1, 3), dtype=np.float32)]
    )

    latents, eos_detected, frame_count = adapter._generate_latents(
        main,
        flow,
        {},
        temperature=0.7,
        lsd_steps=1,
        max_frames=2,
        frames_after_eos=None,
        latent_dim=3,
        conditioning_dim=4,
    )

    assert latents.shape == (1, 2, 3)
    assert not eos_detected
    assert frame_count == 2
    assert [inputs["x"].shape for inputs in flow_inputs] == [(1, 3), (1, 3)]
    assert [inputs["c"].shape for inputs in flow_inputs] == [(1, 4), (1, 4)]
    assert [inputs["s"].shape for inputs in flow_inputs] == [(1, 1), (1, 1)]
    assert [inputs["t"].shape for inputs in flow_inputs] == [(1, 1), (1, 1)]
    assert [inputs["sequence"].shape for inputs in main_inputs] == [(1, 1, 3)] * 2


def _predefined_adapter(
    tmp_path: Path,
    *,
    voices: tuple[str, ...] = ("alba",),
    cache_dir: Path | None = None,
    offline: bool = False,
    conditioning_dim: int = 4,
) -> PocketAdapter:
    bundle_metadata = {
        "bundle_name": "english_2026-04",
        "language": "en",
        "schema_version": 2,
        "sample_rate": 24000,
        "samples_per_frame": 1920,
        "latent_dim": 3,
        "conditioning_dim": conditioning_dim,
        "flow_lm_state_manifest": [
            {
                "index": 0,
                "input_name": "state_0",
                "output_name": "out_state_0",
                "module": "transformer.layers.0.self_attn",
                "key": "cache",
                "shape": [2, 1, 4, 1, 2],
                "dtype": "float32",
                "fill": "nan",
            },
            {
                "index": 1,
                "input_name": "state_1",
                "output_name": "out_state_1",
                "module": "transformer.layers.0.self_attn",
                "key": "current_end",
                "shape": [0],
                "dtype": "float32",
                "fill": "empty",
            },
            {
                "index": 2,
                "input_name": "state_2",
                "output_name": "out_state_2",
                "module": "transformer.layers.0.self_attn",
                "key": "step",
                "shape": [1],
                "dtype": "int64",
                "fill": "zeros",
            },
        ],
        "mimi_state_manifest": [],
        "insert_bos_before_voice": False,
        "bos_before_voice_file": "bos_before_voice.npy",
        "predefined_voices": list(voices),
    }
    metadata_path = tmp_path / "bundle.json"
    metadata_path.write_text(json.dumps(bundle_metadata), encoding="utf-8")
    installation = _make_installation(
        artifacts={"bundle_metadata": metadata_path},
        metadata={"predefined_voice_names": list(voices)},
    )
    return PocketAdapter(
        installation,
        voice_cache_dir=cache_dir or tmp_path / "voice-cache",
        offline=offline,
    )


def _predefined_model_tensors() -> dict[str, np.ndarray]:
    return {
        "transformer.layers.0.self_attn/cache": np.ones((2, 1, 3, 1, 2), dtype=np.float32),
        "transformer.layers.0.self_attn/current_end": np.zeros(3, dtype=np.float32),
        "unused/module_tensor": np.ones(1, dtype=np.float32),
    }


def _explicit_voice_state_record(payload: bytes) -> dict[str, Any]:
    return {
        "name": "alba",
        "compatible_bundle": "english_2026-04",
        "source": {
            "provider": "huggingface",
            "repository": "kyutai/pocket-tts",
            "revision": "d" * 40,
            "path": "languages/english_2026-04/embeddings/alba.safetensors",
        },
        "access": {"gated": True, "distributable": False, "license": "cc-by-4.0"},
        "format": "safetensors",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": None,
        "resolver": "huggingface",
    }


def _write_legacy_voice_state_cache(cache_dir: Path, payload: bytes) -> Path:
    key = {
        "system": "pocket",
        "asset_kind": "predefined_voice_state",
        "model_repo": "kyutai/pocket-tts",
        "model_revision": "d" * 40,
        "asset_path": "languages/english_2026-04/embeddings/alba.safetensors",
        "bundle_id": "english_2026-04",
        "voice_name": "alba",
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    record_dir = cache_dir / digest[:2] / digest
    record_dir.mkdir(parents=True)
    state_path = record_dir / "alba.safetensors"
    state_path.write_bytes(payload)
    (record_dir / "alba.json").write_text(
        json.dumps(
            {
                "key": key,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return state_path


def test_predefined_voice_download_uses_pinned_bundle_asset_and_runtime_cache(
    tmp_path: Path,
) -> None:
    from onnxvoice.systems.pocket import (
        PREDEFINED_VOICE_REPOSITORY,
        PREDEFINED_VOICE_REVISION,
    )

    source = tmp_path / "download.safetensors"
    source.write_bytes(b"fake safetensors payload")
    adapter = _predefined_adapter(tmp_path)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            return_value=source,
        ) as download,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ) as load,
    ):
        state = adapter.prepare_predefined_voice("alba")
        assert adapter.prepare_predefined_voice("alba") is state

    download.assert_called_once()
    source = download.call_args.args[0]
    assert source.repository == PREDEFINED_VOICE_REPOSITORY
    assert source.revision == PREDEFINED_VOICE_REVISION
    assert source.path == "languages/english_2026-04/embeddings/alba.safetensors"
    assert source.gated is True
    assert download.call_args.kwargs["offline"] is False
    assert not download.call_args.kwargs["local_dir"].exists()
    assert not (adapter._voice_cache_dir / "huggingface").exists()
    load.assert_called_once()
    assert state.embeddings is None
    assert state.flow_state is not None
    assert state.flow_state["state_0"].shape == (2, 1, 4, 1, 2)
    assert np.array_equal(
        state.flow_state["state_0"][:, :, :3], np.ones((2, 1, 3, 1, 2), np.float32)
    )
    assert np.isnan(state.flow_state["state_0"][:, :, 3:]).all()
    assert state.flow_state["state_1"].shape == (0,)
    assert np.array_equal(state.flow_state["state_2"], [3])
    assert state.sample_rate == 24000
    assert state.metadata["bundle_id"] == "english_2026-04"
    assert state.metadata["voice_name"] == "alba"
    assert state.metadata["kind"] == "predefined_flow_state"
    assert state.metadata["model_revision"] == PREDEFINED_VOICE_REVISION
    assert "cache_path" not in state.metadata
    assert "token" not in state.metadata


def test_predefined_voice_uses_declared_separate_huggingface_source(
    tmp_path: Path,
) -> None:
    remote_file = tmp_path / "voice.safetensors"
    payload = b"fake safetensors payload"
    remote_file.write_bytes(payload)
    adapter = _predefined_adapter(tmp_path)
    source_record = {
        "name": "alba",
        "compatible_bundle": "english_2026-04",
        "source": {
            "provider": "huggingface",
            "repository": "kyutai/pocket-tts",
            "revision": "d" * 40,
            "path": "languages/english_2026-04/embeddings/alba.safetensors",
        },
        "access": {"gated": True, "distributable": False, "license": "cc-by-4.0"},
        "format": "safetensors",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": None,
        "resolver": "huggingface",
    }
    adapter.installation.metadata["voice_states"] = [source_record]
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            return_value=remote_file,
        ) as download,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ),
    ):
        state = adapter.prepare_predefined_voice("alba")

    source = download.call_args.args[0]
    assert source.repository == "kyutai/pocket-tts"
    assert source.revision == "d" * 40
    assert source.path == source_record["source"]["path"]
    assert source.gated is True
    cached_record = json.loads(
        next(adapter._voice_cache_dir.rglob("alba.json")).read_text(encoding="utf-8")
    )
    assert cached_record["key"]["expected_size"] == len(payload)
    assert cached_record["key"]["expected_sha256"] == hashlib.sha256(payload).hexdigest()
    assert state.metadata["model_repo"] == "kyutai/pocket-tts"
    assert state.metadata["model_revision"] == "d" * 40


def test_pinned_explicit_voice_reuses_legacy_cache_offline(tmp_path: Path) -> None:
    payload = b"cached pinned state"
    cache_dir = tmp_path / "voice-cache"
    legacy_state = _write_legacy_voice_state_cache(cache_dir, payload)
    adapter = _predefined_adapter(tmp_path, cache_dir=cache_dir, offline=True)
    adapter.installation.metadata["voice_states"] = [_explicit_voice_state_record(payload)]

    with (
        patch("onnxvoice.systems.pocket.download_huggingface_file") as download,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ),
    ):
        state = adapter.prepare_predefined_voice("alba")

    download.assert_not_called()
    assert state.metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    assert legacy_state.is_file()


def test_stale_legacy_voice_state_is_not_reused_for_changed_integrity_pin(
    tmp_path: Path,
) -> None:
    old_payload = b"old pinned state"
    new_payload = b"new pinned state with different size"
    cache_dir = tmp_path / "voice-cache"
    _write_legacy_voice_state_cache(cache_dir, old_payload)
    adapter = _predefined_adapter(tmp_path, cache_dir=cache_dir)
    adapter.installation.metadata["voice_states"] = [_explicit_voice_state_record(new_payload)]
    remote_file = tmp_path / "new-state.safetensors"
    remote_file.write_bytes(new_payload)

    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            return_value=remote_file,
        ) as download,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ),
    ):
        state = adapter.prepare_predefined_voice("alba")

    download.assert_called_once()
    assert state.metadata["sha256"] == hashlib.sha256(new_payload).hexdigest()


def test_concurrent_predefined_voice_requests_share_one_atomic_download(tmp_path: Path) -> None:
    source = tmp_path / "download.safetensors"
    source.write_bytes(b"concurrent voice state")
    cache_dir = tmp_path / "voice-cache"
    adapters = [
        _predefined_adapter(tmp_path, cache_dir=cache_dir),
        _predefined_adapter(tmp_path, cache_dir=cache_dir),
    ]
    ready = Barrier(2)
    download_started = Event()
    finish_download = Event()

    def download(*args, **kwargs):
        download_started.set()
        assert finish_download.wait(timeout=5)
        return source

    def prepare(adapter):
        ready.wait(timeout=5)
        return adapter.prepare_predefined_voice("alba")

    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            side_effect=download,
        ) as download_mock,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ) as load,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [executor.submit(prepare, adapter) for adapter in adapters]
        assert download_started.wait(timeout=5)
        finish_download.set()
        states = [future.result(timeout=5) for future in futures]

    download_mock.assert_called_once()
    assert len(states) == 2
    assert load.call_count == 2


def test_predefined_voice_uses_persistent_cache_offline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "download.safetensors"
    source.write_bytes(b"fake safetensors payload")
    cache_dir = tmp_path / "voice-cache"
    online = _predefined_adapter(tmp_path, cache_dir=cache_dir)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            return_value=source,
        ) as download,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ),
    ):
        online_state = online.prepare_predefined_voice("alba")

    offline = _predefined_adapter(tmp_path, cache_dir=cache_dir, offline=True)
    with (
        patch("onnxvoice.systems.pocket.download_huggingface_file") as download_offline,
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ),
    ):
        offline_state = offline.prepare_predefined_voice("alba")

    download.assert_called_once()
    download_offline.assert_not_called()
    assert offline_state.flow_state is not None
    assert online_state.flow_state is not None
    for key in online_state.flow_state:
        assert np.array_equal(
            offline_state.flow_state[key], online_state.flow_state[key], equal_nan=True
        )


def test_predefined_voice_rejects_unknown_name_before_download(tmp_path: Path) -> None:
    from onnxvoice.errors import RuntimeContractError

    adapter = _predefined_adapter(tmp_path)
    with (
        patch("onnxvoice.systems.pocket.download_huggingface_file") as download,
        pytest.raises(RuntimeContractError, match="Available voices: alba"),
    ):
        adapter.prepare_predefined_voice("abla")
    download.assert_not_called()


def test_predefined_voice_rejects_bundle_catalog_name_mismatch(tmp_path: Path) -> None:
    from onnxvoice.errors import RuntimeContractError

    adapter = _predefined_adapter(tmp_path)
    adapter.installation.metadata["predefined_voice_names"] = ["other"]
    with pytest.raises(RuntimeContractError, match="catalog disagree"):
        _ = adapter.predefined_voices


def test_predefined_voice_rejects_malformed_model_state_key(tmp_path: Path) -> None:
    from onnxvoice.errors import PredefinedVoiceIntegrityError

    source = tmp_path / "download.safetensors"
    source.write_bytes(b"fake safetensors payload")
    adapter = _predefined_adapter(tmp_path)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            return_value=source,
        ),
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value={"not-a-namespaced-key": np.ones(1, dtype=np.float32)},
        ),
        pytest.raises(PredefinedVoiceIntegrityError, match="invalid tensor key"),
    ):
        adapter.prepare_predefined_voice("alba")


def test_predefined_voice_gated_access_is_distinct(tmp_path: Path) -> None:
    from onnxvoice.errors import PredefinedVoiceAccessError

    class GatedRepoError(Exception):
        pass

    adapter = _predefined_adapter(tmp_path)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            side_effect=GatedRepoError("private error detail"),
        ),
        pytest.raises(
            PredefinedVoiceAccessError,
            match="requires access to kyutai/pocket-tts",
        ) as caught,
    ):
        adapter.prepare_predefined_voice("alba")
    assert isinstance(caught.value.__cause__, GatedRepoError)
    assert "private error detail" not in str(caught.value)


def test_predefined_voice_missing_asset_is_distinct(tmp_path: Path) -> None:
    from onnxvoice.errors import PredefinedVoiceNotFoundError

    class EntryNotFoundError(Exception):
        pass

    adapter = _predefined_adapter(tmp_path)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            side_effect=EntryNotFoundError(),
        ),
        pytest.raises(PredefinedVoiceNotFoundError, match="not present for bundle"),
    ):
        adapter.prepare_predefined_voice("alba")


def test_predefined_voice_offline_cache_miss_is_distinct(tmp_path: Path) -> None:
    from onnxvoice.errors import OfflineError

    class LocalEntryNotFoundError(Exception):
        pass

    adapter = _predefined_adapter(tmp_path, offline=True)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            side_effect=LocalEntryNotFoundError(),
        ) as download,
        pytest.raises(OfflineError, match="not available in the local cache"),
    ):
        adapter.prepare_predefined_voice("alba")
    assert download.call_args.kwargs["offline"] is True


def test_predefined_voice_cached_corruption_is_reported(tmp_path: Path) -> None:
    from onnxvoice.errors import PredefinedVoiceIntegrityError

    source = tmp_path / "download.safetensors"
    source.write_bytes(b"fake safetensors payload")
    cache_dir = tmp_path / "voice-cache"
    online = _predefined_adapter(tmp_path, cache_dir=cache_dir)
    with (
        patch(
            "onnxvoice.systems.pocket.download_huggingface_file",
            return_value=source,
        ),
        patch(
            "onnxvoice.systems.pocket._load_safetensors",
            return_value=_predefined_model_tensors(),
        ),
    ):
        online.prepare_predefined_voice("alba")
    cached_state = next(cache_dir.rglob("alba.safetensors"))
    cached_state.write_bytes(b"x" * cached_state.stat().st_size)

    offline = _predefined_adapter(tmp_path, cache_dir=cache_dir, offline=True)
    with (
        patch("onnxvoice.systems.pocket.download_huggingface_file") as download,
        pytest.raises(PredefinedVoiceIntegrityError, match="failed integrity validation"),
    ):
        offline.prepare_predefined_voice("alba")
    download.assert_not_called()
