from __future__ import annotations

import urllib.request

import pytest

import onnxvoice
from onnxvoice.errors import NotInstalledError
from onnxvoice.manager import OnnxVoice


def _fail_network(*args, **kwargs):
    raise AssertionError("unexpected network access")


def test_local_runtime_paths_never_use_network(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fail_network)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    manager = OnnxVoice(cache_dir=tmp_path / "cache")
    installation = manager.import_model(system="piper", item_id="voice", model=model)

    runtime = manager.open(installation.ref)
    assert runtime.installation.ref == installation.ref
    assert manager.installed("piper")[0].ref == installation.ref
    assert manager.where(installation.ref) == installation.path
    local = manager.open_local(system="piper", model=model)
    assert local.installation.metadata["managed"] is False


def test_open_missing_is_local_only_and_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fail_network)
    manager = OnnxVoice(cache_dir=tmp_path / "cache")

    with pytest.raises(NotInstalledError, match="Not installed: piper:missing"):
        manager.open("piper:missing")


def test_load_download_true_is_rejected_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fail_network)
    manager = OnnxVoice(cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="obsolete"):
        manager.load("piper:missing", download=True)


def test_module_exports_open_and_open_local(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fail_network)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")

    runtime = onnxvoice.open_local(system="piper", model=model)
    assert runtime.installation.metadata["managed"] is False


def test_module_load_download_true_does_not_create_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("ONNXVOICE_CACHE_DIR", str(cache))
    with pytest.raises(ValueError, match="obsolete"):
        onnxvoice.load("piper:missing", download=True)
    assert not cache.exists()
