from __future__ import annotations

import hashlib
import os

import pytest

from onnxvoice.errors import IntegrityError
from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, CatalogItem


def test_canonical_pocket_install_requires_integrity(tmp_path):
    source = tmp_path / "bundle.json"
    source.write_bytes(b"bundle")
    item = CatalogItem(
        system="pocket",
        id="bundle",
        kind="bundle",
        artifacts=(Artifact("bundle_metadata", source.name, source.as_uri()),),
        metadata={"canonical_catalog": True},
    )
    store = AssetStore(tmp_path / "cache")
    with pytest.raises(IntegrityError, match="no positive size"):
        store.install(item)


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
    report = store.gc()
    assert report.removed_blobs == 1


def test_install_falls_back_to_copy_when_os_link_is_unavailable(tmp_path, monkeypatch):
    payload = b"model"
    source = tmp_path / "voice.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system="test",
        id="copy-fallback",
        kind="model",
        artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
    )
    store = AssetStore(tmp_path / "cache")

    monkeypatch.delattr(os, "link", raising=False)

    installation = store.install(item)

    assert installation.artifact("model").path.read_bytes() == payload
    store.verify(installation)


@pytest.mark.parametrize(
    "error",
    [
        OSError("hard links unavailable"),
        NotImplementedError("hard links unavailable"),
    ],
)
def test_install_falls_back_when_hard_link_operation_is_unsupported(
    tmp_path,
    monkeypatch,
    error,
):
    payload = b"model"
    source = tmp_path / "voice.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system="test",
        id="copy-fallback-error",
        kind="model",
        artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
    )
    store = AssetStore(tmp_path / "cache")

    def fail_link(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(os, "link", fail_link, raising=False)

    installation = store.install(item)

    assert installation.artifact("model").path.read_bytes() == payload
    store.verify(installation)


def test_new_install_persists_timing_metadata(tmp_path):
    payload = b"model-with-timing"
    source = tmp_path / "timed.onnx"
    source.write_bytes(payload)
    item = CatalogItem(
        system="kokoro",
        id="v1.0",
        kind="model",
        artifacts=(
            Artifact(
                "model",
                source.name,
                source.as_uri(),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ),
        ),
        metadata={"runtime": {"timings_output": "durations"}},
    )
    store = AssetStore(tmp_path / "cache")

    store.install(item)
    reloaded = store.get("kokoro", "v1.0")

    assert reloaded.timing_output == "durations"
