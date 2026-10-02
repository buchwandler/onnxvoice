from __future__ import annotations

import pytest

import onnxvoice
from onnxvoice.errors import CapabilityError
from onnxvoice.manager import OnnxVoice


def test_local_runtime_does_not_mutate_shared_cache(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    cache = tmp_path / "cache"
    manager = OnnxVoice(cache_dir=cache)
    before = sorted(path.relative_to(cache) for path in cache.rglob("*"))

    adapter = manager.load_local(system="piper", model=model)

    after = sorted(path.relative_to(cache) for path in cache.rglob("*"))
    assert adapter.installation.metadata["managed"] is False
    assert adapter.installation.artifact("model").path == model
    assert after == before


def test_module_local_runtime_does_not_create_default_cache(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    cache = tmp_path / "unused-cache"
    monkeypatch.setenv("ONNXVOICE_CACHE_DIR", str(cache))

    adapter = onnxvoice.load_local(system="piper", model=model)

    assert adapter.installation.artifact("model").path == model
    assert not cache.exists()


def test_local_kokoro_cloning_accepts_generic_role_component_artifacts(tmp_path):
    components = (
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
    )
    artifacts = {}
    for component in components:
        path = tmp_path / f"{component}.onnx"
        path.write_bytes(b"model")
        artifacts[f"model:{component}"] = path
    source_params = tmp_path / "source_params.npz"
    source_params.write_bytes(b"parameters")
    artifacts["metadata:source_params"] = source_params
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    artifacts["config"] = config

    adapter = OnnxVoice.open_local(
        system="kokoro",
        artifacts=artifacts,
        runtime={"layout": "cloning-onnx-v1", "voice_mode": "reference"},
        sample_rate=24000,
    )

    model_artifacts = adapter.installation.artifacts_for(role="model")
    assert {artifact.component for artifact in model_artifacts} == set(components)
    assert (
        adapter.installation.artifact("metadata", component="source_params").path == source_params
    )
    assert adapter.installation.voices == ()
    assert adapter.installation.default_voice is None
    assert adapter._uses_cloning_runtime()


def _inno_capability():
    return {
        "id": "inno-v0.2",
        "kind": "kokoro-voicepack-tuner",
        "input": "reference-audio",
        "transcript_required": False,
        "min_seconds": 3.0,
        "recommended_seconds": 5.0,
        "max_seconds": 30.0,
        "output": {"format": "kokoro-voicepack-v1", "shape": [510, 1, 256], "dtype": "float32"},
        "model_component": "inno_voicepack",
        "metadata_component": "inno_tuner",
    }


def test_local_kokoro_inno_artifacts_are_optional_role_components(tmp_path):
    base_model = tmp_path / "base.onnx"
    base_model.write_bytes(b"base")
    inno_model = tmp_path / "inno.onnx"
    inno_model.write_bytes(b"inno")
    tuner_metadata = tmp_path / "inno.npz"
    tuner_metadata.write_bytes(b"metadata")

    adapter = OnnxVoice.open_local(
        system="kokoro",
        artifacts={
            "model:inno_voicepack": inno_model,
            "metadata:inno_tuner": tuner_metadata,
            "model": base_model,
        },
        runtime={"layout": "single-onnx-v1", "voice_enrollers": [_inno_capability()]},
        sample_rate=24000,
    )

    assert adapter.installation.id == "local-base"
    assert (
        next(
            artifact
            for artifact in adapter.installation.artifacts_for(role="model")
            if artifact.component is None
        ).path
        == base_model
    )
    assert adapter.installation.artifact("model", component="inno_voicepack").path == inno_model
    assert adapter.installation.artifact("metadata", component="inno_tuner").path == tuner_metadata
    assert adapter._session is None


def test_local_kokoro_rejects_incomplete_declared_inno_capability(tmp_path):
    base_model = tmp_path / "base.onnx"
    base_model.write_bytes(b"base")

    with pytest.raises(CapabilityError, match="missing a required artifact"):
        OnnxVoice.open_local(
            system="kokoro",
            artifacts={"model": base_model},
            runtime={"layout": "single-onnx-v1", "voice_enrollers": [_inno_capability()]},
            sample_rate=24000,
        )
