from __future__ import annotations

import onnxvoice
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
    assert adapter.installation.artifact("metadata", component="source_params").path == source_params
    assert adapter.installation.voices == ()
    assert adapter.installation.default_voice is None
    assert adapter._uses_cloning_runtime()
