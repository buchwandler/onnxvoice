"""Tests for the canonical Pocket catalog builder and verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from onnxvoice.catalog_tools.pocket import (
    REQUIRED_ONNX_ROLES,
    STATIC_ROLES,
    VALID_ROLES,
    CatalogError,
    _extract_bundle_id,
    _safe_name,
    build_catalog,
    huggingface_resolve_url,
    load_catalog,
    verify_catalog,
)

REVISION = "a" * 40
REPOSITORY = "KevinAHM/pocket-tts-onnx"
BASE = "onnx/english_2026-04"


def _artifact(role: str, filename: str, quality: str | None = None) -> dict[str, Any]:
    path = f"{BASE}/{filename}"
    return {
        "role": role,
        "quality": quality,
        "format": Path(filename).suffix.removeprefix(".") or "binary",
        "filename": filename,
        "path": path,
        "url": huggingface_resolve_url(REPOSITORY, REVISION, path),
        "size": 100,
        "sha256": "a" * 64,
    }


def _artifacts() -> list[dict[str, Any]]:
    result = [
        _artifact("bundle_metadata", "bundle.json"),
        _artifact("tokenizer", "tokenizer.model"),
        _artifact("bos_conditioning", "bos_before_voice.npy"),
    ]
    for role in (
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
        "text_conditioner",
    ):
        result.append(_artifact(role, f"{role}.onnx", "fp32"))
        if role in {
            "flow_lm_main",
            "flow_lm_flow",
            "mimi_decoder",
            "mimi_encoder",
            "text_conditioner",
        }:
            result.append(_artifact(role, f"{role}_int8.onnx", "int8"))
    return result


def _profiles() -> dict[str, dict[str, str]]:
    return {
        "fp32": dict.fromkeys(sorted(REQUIRED_ONNX_ROLES), "fp32"),
        "int8": {
            "flow_lm_main": "int8",
            "flow_lm_flow": "int8",
            "mimi_decoder": "int8",
            "mimi_encoder": "fp32",
            "text_conditioner": "fp32",
        },
    }


def _catalog() -> dict[str, Any]:
    entry = {
        "id": "english_2026-04",
        "aliases": ["english", "en"],
        "language": "en",
        "layers": 6,
        "sample_rate": 24000,
        "bundle_schema": 2,
        "profiles": _profiles(),
        "artifacts": _artifacts(),
    }
    return {
        "schema": 1,
        "kind": "pocket-onnx-bundle-catalog",
        "source": {
            "provider": "huggingface",
            "repository": REPOSITORY,
            "requested_revision": "main",
            "revision": REVISION,
            "bundle_root": "onnx",
            "bundle_count": 1,
            "license": "cc-by-4.0",
            "snapshot_note": None,
        },
        "bundles": {entry["id"]: entry},
    }


def _upstream_bundle() -> dict[str, Any]:
    data = _catalog()["bundles"]["english_2026-04"]
    return {
        "language": data["language"],
        "sample_rate": data["sample_rate"],
        "layers": data["layers"],
        "schema_version": data["bundle_schema"],
        "aliases": data["aliases"],
        "profiles": data["profiles"],
        "artifacts": data["artifacts"],
    }


class TestConstants:
    def test_roles(self) -> None:
        assert VALID_ROLES == REQUIRED_ONNX_ROLES | STATIC_ROLES


class TestSafeNames:
    def test_valid_name(self) -> None:
        _safe_name("english_2026-04", "test")

    @pytest.mark.parametrize("value", ["", ".", "..", "test/name", "test name", "_test"])
    def test_unsafe_name_raises(self, value: str) -> None:
        with pytest.raises(CatalogError):
            _safe_name(value, "test")


def test_extract_bundle_id() -> None:
    assert _extract_bundle_id("onnx/english_2026-04/bundle.json") == "english_2026-04"
    with pytest.raises(CatalogError, match="Invalid bundle path"):
        _extract_bundle_id("invalid/path")


class TestVerifyCatalog:
    def test_valid_catalog(self) -> None:
        verify_catalog(_catalog())

    def test_rejects_legacy_kind_and_list(self) -> None:
        bad = _catalog()
        bad["kind"] = "pocket-bundle-catalog"
        bad["bundles"] = list(bad["bundles"].values())
        with pytest.raises(CatalogError, match="Unexpected catalog kind"):
            verify_catalog(bad)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [("schema", 2, "schema must be 1"), ("kind", "wrong", "Unexpected catalog kind")],
    )
    def test_invalid_top_level(self, field: str, value: Any, message: str) -> None:
        bad = _catalog()
        bad[field] = value
        with pytest.raises(CatalogError, match=message):
            verify_catalog(bad)

    def test_invalid_source_and_count(self) -> None:
        bad = _catalog()
        bad["source"]["provider"] = "wrong"
        with pytest.raises(CatalogError, match="provider must be huggingface"):
            verify_catalog(bad)
        bad = _catalog()
        bad["source"]["revision"] = "short"
        with pytest.raises(CatalogError, match="40-character SHA"):
            verify_catalog(bad)
        bad = _catalog()
        bad["source"]["bundle_count"] = 99
        with pytest.raises(CatalogError, match="Bundle count mismatch"):
            verify_catalog(bad)

    def test_rejects_map_id_and_alias_collisions(self) -> None:
        bad = _catalog()
        entry = bad["bundles"].pop("english_2026-04")
        bad["bundles"]["other"] = entry
        with pytest.raises(CatalogError, match="does not match"):
            verify_catalog(bad)
        bad = _catalog()
        bad["bundles"]["english_2026-04"]["aliases"] = ["en"]
        other = copy.deepcopy(bad["bundles"]["english_2026-04"])
        other["id"] = "other"
        other["aliases"] = ["en"]
        bad["bundles"]["other"] = other
        bad["source"]["bundle_count"] = 2
        with pytest.raises(CatalogError, match="ambiguous"):
            verify_catalog(bad)

    def test_rejects_missing_roles_profiles_paths_and_integrity(self) -> None:
        bad = _catalog()
        bad["bundles"]["english_2026-04"]["artifacts"] = [
            a for a in bad["bundles"]["english_2026-04"]["artifacts"] if a["role"] != "tokenizer"
        ]
        with pytest.raises(CatalogError, match="missing static role"):
            verify_catalog(bad)
        bad = _catalog()
        bad["bundles"]["english_2026-04"]["profiles"]["fp32"]["unknown"] = "fp32"
        with pytest.raises(CatalogError, match="unknown role"):
            verify_catalog(bad)
        bad = _catalog()
        bad["bundles"]["english_2026-04"]["artifacts"][0]["filename"] = "other.json"
        with pytest.raises(CatalogError, match="filename does not match"):
            verify_catalog(bad)
        bad = _catalog()
        bad["bundles"]["english_2026-04"]["artifacts"][0]["size"] = None
        with pytest.raises(CatalogError, match="invalid size"):
            verify_catalog(bad)
        bad = _catalog()
        bad["bundles"]["english_2026-04"]["artifacts"][0]["sha256"] = None
        with pytest.raises(CatalogError, match="invalid sha256"):
            verify_catalog(bad)


class TestLoadCatalog:
    def test_load_valid_catalog(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(_catalog()), encoding="utf-8")
        assert load_catalog(path)["schema"] == 1

    @pytest.mark.parametrize("content", ["not json", "{}"])
    def test_load_invalid_catalog(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(CatalogError):
            load_catalog(path)


def test_build_catalog_from_explicit_fixture_is_deterministic() -> None:
    upstream = _upstream_bundle()
    with (
        patch(
            "onnxvoice.catalog_tools.pocket._list_bundle_paths",
            return_value=[f"{BASE}/bundle.json"],
        ),
        patch(
            "onnxvoice.catalog_tools.pocket._fetch_bundle_json", return_value=(upstream, "b" * 64)
        ),
    ):
        result1 = build_catalog(resolved_revision=REVISION)
        result2 = build_catalog(resolved_revision=REVISION)
    assert result1 == result2
    verify_catalog(result1)


def test_build_catalog_discovers_upstream_files_and_metadata() -> None:
    upstream = _upstream_bundle()
    upstream.pop("artifacts")
    upstream.pop("profiles")
    tree = []
    for artifact in _artifacts():
        tree.append(
            {"path": artifact["path"], "size": artifact["size"], "lfs": {"oid": artifact["sha256"]}}
        )
    with (
        patch(
            "onnxvoice.catalog_tools.pocket._list_bundle_paths",
            return_value=[f"{BASE}/bundle.json"],
        ),
        patch(
            "onnxvoice.catalog_tools.pocket._fetch_bundle_json", return_value=(upstream, "b" * 64)
        ),
        patch("onnxvoice.catalog_tools.pocket._repository_tree", return_value=tree),
    ):
        result = build_catalog(resolved_revision=REVISION)
    verify_catalog(result)
    entry = result["bundles"]["english_2026-04"]
    assert len(entry["artifacts"]) == 13
    assert entry["profiles"]["int8"]["mimi_encoder"] == "fp32"


def test_build_catalog_requires_bundles() -> None:
    with (
        patch("onnxvoice.catalog_tools.pocket._list_bundle_paths", return_value=[]),
        pytest.raises(CatalogError, match="No bundles found"),
    ):
        build_catalog(resolved_revision=REVISION)
