"""Cross-repository regression for the canonical Pocket catalog contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnxvoice.catalog_tools.pocket import load_catalog

REPOSITORY_CATALOG = (
    Path(__file__).resolve().parents[2] / "pocket-onnx-bundles" / "catalog" / "bundles.json"
)
REPOSITORY_SOURCE = (
    Path(__file__).resolve().parents[2] / "pocket-onnx-bundles" / "catalog" / "source.json"
)


@pytest.mark.skipif(
    not REPOSITORY_CATALOG.exists(), reason="pocket-onnx-bundles checkout is not present"
)
def test_committed_pocket_catalog_is_the_canonical_object_map() -> None:
    catalog = load_catalog(REPOSITORY_CATALOG)
    source = json.loads(REPOSITORY_SOURCE.read_text(encoding="utf-8"))

    assert catalog["kind"] == "pocket-onnx-bundle-catalog"
    assert isinstance(catalog["bundles"], dict)
    assert source == {"schema": 1, **catalog["source"]}
    assert len(catalog["bundles"]) == catalog["source"]["bundle_count"]
    artifacts = [
        artifact for bundle in catalog["bundles"].values() for artifact in bundle["artifacts"]
    ]
    assert artifacts
    assert all(artifact["size"] > 0 for artifact in artifacts)
    assert all(len(artifact["sha256"]) == 64 for artifact in artifacts)
