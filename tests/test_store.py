from __future__ import annotations

import hashlib

from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, CatalogItem


def test_store_deduplicates_content(tmp_path):
    payload = b"same-model-bytes"
    source_a = tmp_path / "a.onnx"
    source_b = tmp_path / "b.onnx"
    source_a.write_bytes(payload)
    source_b.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    store = AssetStore(tmp_path / "cache")
    first = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(Artifact("model", "a.onnx", source_a.as_uri(), len(payload), sha),),
    )
    second = CatalogItem(
        system="test",
        id="two",
        kind="model",
        artifacts=(Artifact("model", "b.onnx", source_b.as_uri(), len(payload), sha),),
    )

    store.install(first)
    store.install(second)

    blobs = [p for p in store.blobs.glob("*/*") if p.is_file()]
    assert len(blobs) == 1
    assert store.get("test", "one").artifact("model").path.read_bytes() == payload
    assert store.get("test", "two").artifact("model").path.read_bytes() == payload


def test_remove_then_gc(tmp_path):
    payload = b"model"
    source = tmp_path / "voice.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
    )
    store = AssetStore(tmp_path / "cache")
    store.install(item)
    store.remove("test", "one")
    assert store.gc() == 1
