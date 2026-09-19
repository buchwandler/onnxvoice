"""Tests for cache usage reporting and GC byte tracking."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, CatalogItem


# ---------------------------------------------------------------------------
# Cache usage tests
# ---------------------------------------------------------------------------


class TestCacheUsage:
    def _make_item(self, system="test", item_id="one", content=b"model-data"):
        sha = hashlib.sha256(content).hexdigest()
        return CatalogItem(
            system=system,
            id=item_id,
            kind="model",
            artifacts=(Artifact("model", "model.onnx", f"file:///dev/null", len(content), sha),),
        )

    def test_empty_cache_usage(self, tmp_path):
        store = AssetStore(tmp_path / "cache")
        usage = store.usage()
        assert usage.installation_count == 0
        assert usage.install_logical_bytes == 0
        assert usage.orphan_blob_count == 0

    def test_install_logical_bytes(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
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
        usage = store.usage()
        assert usage.installation_count == 1
        assert usage.install_logical_bytes == 1000

    def test_blob_apparent_bytes(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
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
        usage = store.usage()
        assert usage.blob_apparent_bytes == 1000

    def test_unique_file_bytes_no_double_count(self, tmp_path):
        """Hard-linked files should be counted once in unique_file_bytes."""
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
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
        usage = store.usage()

        # If hard links are supported, unique should equal blob size
        # If copy fallback was used, unique might be 2x
        blob_path = store.blobs / sha[:2] / sha
        inst_path = store.installs / "test" / "one" / source.name

        if blob_path.exists() and inst_path.exists():
            blob_stat = blob_path.stat()
            inst_stat = inst_path.stat()
            same_inode = (
                blob_stat.st_ino == inst_stat.st_ino and blob_stat.st_dev == inst_stat.st_dev
            )
            if same_inode:
                assert usage.unique_file_bytes == 1000
            else:
                # Copy fallback: both counted separately
                assert usage.unique_file_bytes == 2000

    def test_orphan_blob_detection(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
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

        usage = store.usage()
        assert usage.orphan_blob_count == 1
        assert usage.orphan_blob_bytes == 1000
        assert usage.installation_count == 0


# ---------------------------------------------------------------------------
# GC report tests
# ---------------------------------------------------------------------------


class TestGcReport:
    def test_gc_returns_report(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
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
        assert report.removed_bytes == 1000

    def test_gc_no_orphans(self, tmp_path):
        store = AssetStore(tmp_path / "cache")
        report = store.gc()
        assert report.removed_blobs == 0
        assert report.removed_bytes == 0

    def test_gc_preserves_referenced_blobs(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
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

        # Don't remove the installation - blob should be preserved
        report = store.gc()
        assert report.removed_blobs == 0

    def test_gc_shared_blob_not_removed(self, tmp_path):
        """A blob shared by two installations should not be GC'd when only one is removed."""
        payload = b"shared-data"
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        artifact = Artifact("model", source.name, source.as_uri(), len(payload), sha)

        item1 = CatalogItem(system="test", id="one", kind="model", artifacts=(artifact,))
        item2 = CatalogItem(system="test", id="two", kind="model", artifacts=(artifact,))

        store = AssetStore(tmp_path / "cache")
        store.install(item1)
        store.install(item2)

        # Remove only one
        store.remove("test", "one")
        report = store.gc()
        assert report.removed_blobs == 0  # Still referenced by "two"

        # Now remove the other
        store.remove("test", "two")
        report = store.gc()
        assert report.removed_blobs == 1
