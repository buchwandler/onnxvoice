from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice import OnnxVoice, registered_systems
from onnxvoice.systems import InflectAdapter


def _model_files(tmp_path: Path) -> tuple[Path, Path]:
    duration = tmp_path / "duration.onnx"
    decode = tmp_path / "decode.onnx"
    duration.write_bytes(b"duration graph")
    decode.write_bytes(b"decode graph")
    return duration, decode


def test_inflect_adapter_is_registered() -> None:
    assert "inflect" in registered_systems()
    assert InflectAdapter.system == "inflect"


def test_inflect_open_local_accepts_two_artifact_roles(tmp_path: Path) -> None:
    duration, decode = _model_files(tmp_path)
    runtime = OnnxVoice.open_local(
        system="inflect",
        artifacts={"duration": duration, "decode": decode},
        sample_rate=24000,
        metadata={"runtime": {"profile": "inflect-v2-split-v1", "layout": "split"}},
    )

    try:
        assert isinstance(runtime, InflectAdapter)
        assert runtime.installation.artifact("duration").path == duration.resolve()
        assert runtime.installation.artifact("decode").path == decode.resolve()
        assert runtime.installation.metadata["external"] is True
        assert runtime.installation.metadata["managed"] is False
    finally:
        runtime.close()


def test_inflect_open_local_preserves_split_runtime_metadata(tmp_path: Path) -> None:
    duration, decode = _model_files(tmp_path)
    metadata = {"default_voice": "default", "voices": ["default"]}
    runtime = OnnxVoice.open_local(
        system="inflect",
        artifacts={"duration": duration, "decode": decode},
        sample_rate=24000,
        metadata=metadata,
        runtime={"profile": "inflect-v2-split-v1", "layout": "split", "precision": "fp32"},
    )

    try:
        assert runtime.installation.sample_rate == 24000
        assert runtime.installation.metadata["runtime"] == {
            "profile": "inflect-v2-split-v1",
            "layout": "split",
            "precision": "fp32",
        }
        assert runtime.installation.metadata["default_voice"] == "default"
        assert runtime.installation.metadata["voices"] == ["default"]
    finally:
        runtime.close()


def test_inflect_open_local_provider_options_reach_both_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duration, decode = _model_files(tmp_path)
    provider_options = [{"device_id": "3"}]
    session_options = object()
    created: list[tuple[Path, dict[str, Any]]] = []

    def create_session(path: Path, **kwargs: Any) -> object:
        created.append((path, kwargs))
        return object()

    monkeypatch.setattr("onnxvoice.systems.inflect.OnnxSession", create_session)
    runtime = OnnxVoice.open_local(
        system="inflect",
        artifacts={"duration": duration, "decode": decode},
        sample_rate=24000,
        providers=["CPUExecutionProvider"],
        provider_options=provider_options,
        session_options=session_options,
    )
    try:
        assert created == []
        _ = runtime.duration_session
        _ = runtime.decode_session
        assert [path for path, _ in created] == [duration.resolve(), decode.resolve()]
        for path, options in created:
            assert options == {
                "component": "duration" if path == duration.resolve() else "decode",
                "providers": ["CPUExecutionProvider"],
                "provider_options": provider_options,
                "session_options": session_options,
            }
    finally:
        runtime.close()


def test_inflect_managed_install_verifies_both_artifact_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "inflect_models.json"
    catalog = json.loads(fixture.read_text(encoding="utf-8"))
    payloads: dict[str, bytes] = {}
    for model in catalog["models"].values():
        for artifact in model["artifacts"]:
            payload = f"fixture artifact: {artifact['role']}".encode()
            payloads[artifact["filename"]] = payload
            artifact["size"] = len(payload)
            artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    catalog_path = tmp_path / "models.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    def download(source: Any, *, local_dir: Path, offline: bool) -> Path:
        assert not offline
        destination = local_dir / Path(source.path).name
        destination.write_bytes(payloads[destination.name])
        return destination

    monkeypatch.setattr("onnxvoice.store.download_huggingface_file", download)
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"inflect": str(catalog_path)},
    )
    installation = manager.install("inflect:nano-v2")
    assert {artifact.role for artifact in installation.artifacts} == {"duration", "decode"}
    assert installation.require_artifact("duration").path.read_bytes() == payloads["duration.onnx"]
    assert installation.require_artifact("decode").path.read_bytes() == payloads["decode.onnx"]
    manager.store.verify(installation)

    runtime = manager.open(installation, providers="cpu")
    try:
        assert isinstance(runtime, InflectAdapter)
        assert runtime.installation.ref == "inflect:nano-v2"
    finally:
        runtime.close()
