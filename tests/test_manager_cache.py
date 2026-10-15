from __future__ import annotations

import pytest

from onnxvoice.errors import OfflineError
from onnxvoice.manager import OnnxVoice


def test_environment_cache_root_is_shared_by_store_and_catalog(tmp_path, monkeypatch):
    root = tmp_path / "custom"
    monkeypatch.setenv("ONNXVOICE_CACHE_DIR", str(root))

    manager = OnnxVoice()

    assert manager.store.root == root
    assert manager.catalog.cache_dir == root / "catalogs"
    assert manager.store.offline is False
    assert manager.catalog.offline is False


def test_explicit_cache_root_overrides_environment_and_preserves_offline(tmp_path, monkeypatch):
    environment_root = tmp_path / "environment"
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("ONNXVOICE_CACHE_DIR", str(environment_root))

    manager = OnnxVoice(cache_dir=explicit_root, offline=True)

    assert manager.store.root == explicit_root
    assert manager.catalog.cache_dir == explicit_root / "catalogs"
    assert manager.store.offline is True
    assert manager.catalog.offline is True
    with pytest.raises(OfflineError, match="requires network access"):
        manager.catalog._read_source("https://example.invalid/catalog.json")

    with pytest.raises(OfflineError, match="Network access disabled"):
        manager.store._download(
            "https://example.invalid/model.bin",
            tmp_path / "download.bin",
            expected_size=None,
            ref="test:one",
            artifact="model.bin",
            role="model",
            target="model.bin",
            progress=None,
        )
