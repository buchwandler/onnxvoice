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
    assert json.loads((installation.path / "manifest.json").read_text())["schema"] == 2


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
