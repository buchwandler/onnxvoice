from __future__ import annotations

import json

import numpy as np
import pytest

from onnxvoice.errors import UnsafePathError
from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, CatalogItem, InferenceResult


def _item(source, *, item_id="voice"):
    return CatalogItem(
        system="test",
        id=item_id,
        kind="model",
        artifacts=(
            Artifact(
                role="model",
                filename="models/voice.onnx",
                url=source.as_uri(),
                size=source.stat().st_size,
                component="acoustic",
                format="onnx",
                metadata={"layout": "single", "version": 3},
            ),
        ),
        metadata={"source_revision": "a" * 40},
    )


def test_manifest_round_trip_preserves_runtime_metadata(tmp_path):
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    store = AssetStore(tmp_path / "cache")

    installation = store.install(_item(source))

    artifact = installation.artifact("model")
    assert artifact.component == "acoustic"
    assert artifact.format == "onnx"
    assert artifact.metadata == {"layout": "single", "version": 3}
    assert installation.metadata["source_revision"] == "a" * 40
    assert artifact.path.read_bytes() == b"model"
    assert json.loads((installation.path / "manifest.json").read_text())["schema"] == 3


def test_schema_one_manifest_remains_readable(tmp_path):
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    store = AssetStore(tmp_path / "cache")
    installation = store.install(_item(source))
    manifest_path = installation.path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["schema"] = 1
    for artifact in data["artifacts"]:
        artifact.pop("component", None)
        artifact.pop("format", None)
        artifact.pop("metadata", None)
    manifest_path.write_text(json.dumps(data))

    migrated = store.get("test", "voice")

    assert migrated.artifact("model").component is None
    assert migrated.artifact("model").metadata == {}




def test_schema_two_still_loads(tmp_path):
    """Schema 2 manifests remain readable after the bump to schema 3."""
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    store = AssetStore(tmp_path / "cache")
    installation = store.install(_item(source))
    manifest_path = installation.path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    # Downgrade to schema 2 by removing storage_id
    data["schema"] = 2
    data.pop("storage_id", None)
    manifest_path.write_text(json.dumps(data))

    migrated = store.get("test", "voice")
    assert migrated.id == "voice"
    assert migrated.storage_id is None


def test_schema_three_round_trip_separates_id_and_storage_id(tmp_path):
    """Schema 3 manifest preserves canonical id and storage_id separately."""
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    store = AssetStore(tmp_path / "cache")
    item = _item(source)
    # Add cache_id to simulate explicit selection
    item = CatalogItem(
        system=item.system,
        id=item.id,
        kind=item.kind,
        artifacts=item.artifacts,
        metadata={**item.metadata, "cache_id": "voice--sel-abcd1234"},
    )
    installation = store.install(item)

    assert installation.id == "voice"
    # storage_id is set because cache_id was provided
    assert installation.storage_id == "voice--sel-abcd1234"
    # Verify manifest has both fields
    manifest = json.loads((installation.path / "manifest.json").read_text())
    assert manifest["id"] == "voice"
    assert "storage_id" in manifest


def test_legacy_distribution_cache_manifest_recovers_canonical_id(tmp_path):
    """Legacy schema 2 with --dist- hashed id recovers canonical prefix."""
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    store = AssetStore(tmp_path / "cache")
    installation = store.install(_item(source))
    manifest_path = installation.path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    # Simulate legacy schema 2 with hashed storage id as canonical id
    data["schema"] = 2
    data["id"] = "voice--dist-abcd1234ef015678"
    data.pop("storage_id", None)
    data["metadata"] = {
        **data.get("metadata", {}),
        "cache_id": "voice--dist-abcd1234ef015678",
        "selected_distribution": "mobile",
    }
    manifest_path.write_text(json.dumps(data))

    # Rename directory to match the hashed id
    old_dir = installation.path
    new_dir = old_dir.parent / "voice--dist-abcd1234ef015678"
    old_dir.rename(new_dir)

    migrated = store.get("test", "voice--dist-abcd1234ef015678")
    assert migrated.id == "voice"
    assert migrated.storage_id == "voice--dist-abcd1234ef015678"
def test_install_rejects_unsafe_paths(tmp_path):
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    store = AssetStore(tmp_path / "cache")

    with pytest.raises(UnsafePathError):
        store.install(_item(source, item_id="../escape"))
    with pytest.raises(UnsafePathError):
        store.install(
            CatalogItem(
                system="test",
                id="voice",
                kind="model",
                artifacts=(Artifact("model", "../escape.onnx", source.as_uri()),),
            )
        )


def test_inference_result_is_canonical_float32_mono():
    result = InferenceResult(np.array([1, 2], dtype=np.float64), 24000)
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1

    with pytest.raises(ValueError, match="one-dimensional"):
        InferenceResult(np.array([[1, 2]], dtype=np.float64), 24000)
