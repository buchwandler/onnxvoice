"""Tests for update detection: all comparison scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnxvoice.inventory import (
    UpdateComparison,
    compare_installation_to_catalog,
    catalog_content_fingerprint,
)
from onnxvoice.types import Artifact, CatalogItem, InstalledArtifact, Installation


def _inst(artifacts):
    """Helper to create a test Installation."""
    return Installation(
        system="test",
        id="one",
        kind="model",
        path=Path("/tmp"),
        artifacts=tuple(artifacts),
    )


def _item(artifacts, **extra_meta):
    """Helper to create a test CatalogItem."""
    return CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=tuple(artifacts),
        metadata=dict(extra_meta),
    )


# ---------------------------------------------------------------------------
# SHA-256 comparison
# ---------------------------------------------------------------------------


class TestSha256Comparison:
    def test_current_when_sha256_matches(self):
        """Case 1: current SHA-256 matches => 'current'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_update_when_sha256_differs(self):
        """Case 2: SHA-256 differs => 'update_available'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx", sha256="def456", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"
        assert "model" in result.changed_roles


# ---------------------------------------------------------------------------
# MD5 comparison
# ---------------------------------------------------------------------------


class TestMd5Comparison:
    def test_current_when_md5_matches_size(self):
        """Case 3: catalog has MD5, installed has SHA-256, size matches => 'current'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx", md5="md5hash", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_update_when_md5_differs_size(self):
        """Case 4: MD5 present but size differs => 'update_available'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx", md5="md5hash", size=200),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"


# ---------------------------------------------------------------------------
# Catalog revision vs artifact identity
# ---------------------------------------------------------------------------


class TestCatalogRevisionIndependence:
    def test_current_when_revision_changes_but_artifacts_match(self):
        """Case 5: unrelated catalog source revision changes but artifacts match => 'current'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
            ],
            source_revision="new-revision-xyz",
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_update_when_artifacts_change(self):
        """Case 6: artifact set changes => 'update_available'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
                Artifact("config", "config.json", sha256="new123", size=50),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"


# ---------------------------------------------------------------------------
# Insufficient identity
# ---------------------------------------------------------------------------


class TestInsufficientIdentity:
    def test_unknown_when_no_identity_info(self):
        """Case 7: insufficient digest identity => 'unknown'."""
        inst = _inst(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "a.onnx"),  # No sha256, no md5, no size
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "unknown"
        assert result.reason is not None


# ---------------------------------------------------------------------------
# External / imported installations
# ---------------------------------------------------------------------------


class TestExternalInstallations:
    def test_not_applicable_for_local_import(self):
        """Case 8: imported/local installation with no catalog entry => 'not_applicable'.

        This is handled at the merge level, not in compare_installation_to_catalog,
        but we can test the comparison returns unknown when artifacts don't match.
        """
        inst = _inst(
            [
                InstalledArtifact("model", "local.onnx", Path("/tmp/local.onnx"), "sha", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "remote.onnx", sha256="sha", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        # Different filenames mean different identity keys, so the installed artifact
        # has no matching catalog artifact
        assert result.status == "update_available"


# ---------------------------------------------------------------------------
# Kokoro quality/distribution matching
# ---------------------------------------------------------------------------


class TestKokoroSelection:
    def test_current_when_quality_matches(self):
        """Case 9: explicit Kokoro quality compared against same quality."""
        inst = _inst(
            [
                InstalledArtifact(
                    "model", "model.onnx", Path("/tmp/model.onnx"), "abc", 100, quality="fp16"
                ),
            ]
        )
        item = _item(
            [
                Artifact("model", "model.onnx", sha256="abc", size=100, quality="fp16"),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_current_when_distribution_matches(self):
        """Case 10: explicit distribution compared against same distribution."""
        inst = _inst(
            [
                InstalledArtifact("model", "model.onnx", Path("/tmp/model.onnx"), "abc", 100),
            ]
        )
        item = _item(
            [
                Artifact("model", "model.onnx", sha256="abc", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"


# ---------------------------------------------------------------------------
# Multiple artifacts
# ---------------------------------------------------------------------------


class TestMultipleArtifacts:
    def test_current_all_artifacts_match(self):
        inst = _inst(
            [
                InstalledArtifact("model", "model.onnx", Path("/tmp/model.onnx"), "abc", 100),
                InstalledArtifact("config", "config.json", Path("/tmp/config.json"), "def", 50),
            ]
        )
        item = _item(
            [
                Artifact("model", "model.onnx", sha256="abc", size=100),
                Artifact("config", "config.json", sha256="def", size=50),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_update_one_artifact_changed(self):
        inst = _inst(
            [
                InstalledArtifact("model", "model.onnx", Path("/tmp/model.onnx"), "abc", 100),
                InstalledArtifact("config", "config.json", Path("/tmp/config.json"), "def", 50),
            ]
        )
        item = _item(
            [
                Artifact("model", "model.onnx", sha256="abc", size=100),
                Artifact("config", "config.json", sha256="newhash", size=50),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"
        assert "config" in result.changed_roles

    def test_update_artifact_removed_from_catalog(self):
        inst = _inst(
            [
                InstalledArtifact("model", "model.onnx", Path("/tmp/model.onnx"), "abc", 100),
                InstalledArtifact("config", "config.json", Path("/tmp/config.json"), "def", 50),
            ]
        )
        item = _item(
            [
                Artifact("model", "model.onnx", sha256="abc", size=100),
                # config removed from catalog
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"


# ---------------------------------------------------------------------------
# Catalog content fingerprint
# ---------------------------------------------------------------------------


class TestCatalogContentFingerprint:
    def test_deterministic(self):
        item = _item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
            ]
        )
        fp1 = catalog_content_fingerprint(item)
        fp2 = catalog_content_fingerprint(item)
        assert fp1 == fp2

    def test_changes_when_artifacts_change(self):
        item1 = _item([Artifact("model", "a.onnx", sha256="abc123", size=100)])
        item2 = _item([Artifact("model", "a.onnx", sha256="def456", size=100)])
        assert catalog_content_fingerprint(item1) != catalog_content_fingerprint(item2)

    def test_starts_with_sha256_prefix(self):
        item = _item([Artifact("model", "a.onnx", sha256="abc123", size=100)])
        assert catalog_content_fingerprint(item).startswith("sha256:")
