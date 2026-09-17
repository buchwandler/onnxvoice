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
