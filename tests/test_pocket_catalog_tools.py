"""Tests for Pocket catalog builder and verifier tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from onnxvoice.catalog_tools.pocket import (
    REQUIRED_ONNX_ROLES,
    STATIC_ROLES,
    VALID_ROLES,
    CatalogError,
    _extract_bundle_id,
    _safe_name,
    build_catalog,
    load_catalog,
    verify_catalog,
)

SAMPLE_CATALOG: dict[str, Any] = {
    "schema": 1,
    "kind": "pocket-bundle-catalog",
    "source": {
        "provider": "huggingface",
        "repository": "KevinAHM/pocket-tts-onnx",
        "requested_revision": "main",
        "revision": "a" * 40,
        "bundle_count": 1,
    },
    "bundles": [
        {
            "id": "english_2026-04",
            "aliases": ["english", "en"],
            "sample_rate": 24000,
            "language": "en",
            "layers": 6,
            "bundle_schema": 2,
            "profiles": {
                "int8": {
                    "flow_lm_main": "int8",
                    "flow_lm_flow": "int8",
                    "mimi_decoder": "int8",
                    "mimi_encoder": "fp32",
                    "text_conditioner": "fp32",
                }
            },
            "source_revision": "a" * 40,
            "source_repository": "KevinAHM/pocket-tts-onnx",
            "artifacts": [
                {
                    "role": "bundle_metadata",
                    "filename": "bundle.json",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/bundle.json?download=true",
                },
                {
                    "role": "tokenizer",
                    "filename": "tokenizer.model",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/tokenizer.model?download=true",
                },
                {
                    "role": "bos_conditioning",
                    "filename": "bos_before_voice.npy",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/bos_before_voice.npy?download=true",
                },
                {
                    "role": "flow_lm_main",
                    "filename": "flow_lm_main_int8.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/flow_lm_main_int8.onnx?download=true",
                    "quality": "int8",
                },
                {
                    "role": "flow_lm_main",
                    "filename": "flow_lm_main_fp32.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/flow_lm_main_fp32.onnx?download=true",
                    "quality": "fp32",
                },
                {
                    "role": "flow_lm_flow",
                    "filename": "flow_lm_flow_int8.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/flow_lm_flow_int8.onnx?download=true",
                    "quality": "int8",
                },
                {
                    "role": "flow_lm_flow",
                    "filename": "flow_lm_flow_fp32.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/flow_lm_flow_fp32.onnx?download=true",
                    "quality": "fp32",
                },
                {
                    "role": "mimi_decoder",
                    "filename": "mimi_decoder_int8.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/mimi_decoder_int8.onnx?download=true",
                    "quality": "int8",
                },
                {
                    "role": "mimi_decoder",
                    "filename": "mimi_decoder_fp32.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/mimi_decoder_fp32.onnx?download=true",
                    "quality": "fp32",
                },
                {
                    "role": "mimi_encoder",
                    "filename": "mimi_encoder.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/mimi_encoder.onnx?download=true",
                    "quality": "fp32",
                },
                {
                    "role": "text_conditioner",
                    "filename": "text_conditioner.onnx",
                    "url": f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{'a' * 40}/onnx/english_2026-04/text_conditioner.onnx?download=true",
                    "quality": "fp32",
                },
            ],
        }
    ],
}


class TestConstants:
    """Test constant definitions."""

    def test_valid_roles(self) -> None:
        assert VALID_ROLES == REQUIRED_ONNX_ROLES | STATIC_ROLES

    def test_required_onnx_roles(self) -> None:
        expected = {
            "flow_lm_main",
            "flow_lm_flow",
            "mimi_decoder",
            "mimi_encoder",
            "text_conditioner",
        }
        assert expected == REQUIRED_ONNX_ROLES

    def test_static_roles(self) -> None:
        expected = {
            "bundle_metadata",
            "tokenizer",
            "bos_conditioning",
        }
        assert expected == STATIC_ROLES


class TestSafeName:
    """Test _safe_name validation."""

    def test_valid_name(self) -> None:
        _safe_name("test_name", "test")

    def test_empty_name_raises(self) -> None:
        with pytest.raises(CatalogError, match="must be a non-empty string"):
            _safe_name("", "test")

    def test_slash_raises(self) -> None:
        with pytest.raises(CatalogError, match="path separators are forbidden"):
            _safe_name("test/name", "test")

    def test_dot_segment_raises(self) -> None:
        with pytest.raises(CatalogError, match="dot segments are forbidden"):
            _safe_name(".", "test")

    def test_unsafe_name_raises(self) -> None:
        with pytest.raises(CatalogError, match="unsafe name"):
            _safe_name("test name", "test")


class TestExtractBundleId:
    """Test _extract_bundle_id."""

    def test_valid_path(self) -> None:
        assert _extract_bundle_id("onnx/english_2026-04/bundle.json") == "english_2026-04"

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(CatalogError, match="Invalid bundle path"):
            _extract_bundle_id("invalid/path")


class TestVerifyCatalog:
    """Test verify_catalog."""

    def test_valid_catalog(self) -> None:
        verify_catalog(SAMPLE_CATALOG)

    def test_not_dict_raises(self) -> None:
        with pytest.raises(CatalogError, match="must be an object"):
            verify_catalog([])

    def test_invalid_schema_raises(self) -> None:
        bad = {**SAMPLE_CATALOG, "schema": 2}
        with pytest.raises(CatalogError, match="schema must be 1"):
            verify_catalog(bad)

    def test_invalid_kind_raises(self) -> None:
        bad = {**SAMPLE_CATALOG, "kind": "wrong"}
        with pytest.raises(CatalogError, match="Unexpected catalog kind"):
            verify_catalog(bad)

    def test_invalid_source_provider_raises(self) -> None:
        bad = {
            **SAMPLE_CATALOG,
            "source": {**SAMPLE_CATALOG["source"], "provider": "wrong"},
        }
        with pytest.raises(CatalogError, match="provider must be huggingface"):
            verify_catalog(bad)

    def test_invalid_revision_raises(self) -> None:
        bad = {
            **SAMPLE_CATALOG,
            "source": {**SAMPLE_CATALOG["source"], "revision": "short"},
        }
        with pytest.raises(CatalogError, match="40-character SHA"):
            verify_catalog(bad)

    def test_bundle_count_mismatch_raises(self) -> None:
        bad = {
            **SAMPLE_CATALOG,
            "source": {**SAMPLE_CATALOG["source"], "bundle_count": 99},
        }
        with pytest.raises(CatalogError, match="Bundle count mismatch"):
            verify_catalog(bad)

    def test_duplicate_alias_raises(self) -> None:
        bad_bundle = {
            **SAMPLE_CATALOG["bundles"][0],
            "id": "other",
            "aliases": ["english"],  # Already used by first bundle
        }
        bad = {
            **SAMPLE_CATALOG,
            "source": {**SAMPLE_CATALOG["source"], "bundle_count": 2},
            "bundles": [*SAMPLE_CATALOG["bundles"], bad_bundle],
        }
        with pytest.raises(CatalogError, match="ambiguous"):
            verify_catalog(bad)

    def test_missing_static_role_raises(self) -> None:
        bad_bundle = {
            **SAMPLE_CATALOG["bundles"][0],
            "artifacts": [
                a for a in SAMPLE_CATALOG["bundles"][0]["artifacts"] if a["role"] != "tokenizer"
            ],
        }
        bad = {**SAMPLE_CATALOG, "bundles": [bad_bundle]}
        with pytest.raises(CatalogError, match="missing static role"):
            verify_catalog(bad)

    def test_no_quality_variant_raises(self) -> None:
        bad_bundle = {
            **SAMPLE_CATALOG["bundles"][0],
            "artifacts": [
                a for a in SAMPLE_CATALOG["bundles"][0]["artifacts"] if a["role"] != "mimi_encoder"
            ],
        }
        bad = {**SAMPLE_CATALOG, "bundles": [bad_bundle]}
        with pytest.raises(CatalogError, match="no quality variant"):
            verify_catalog(bad)

    def test_invalid_profile_reference_raises(self) -> None:
        bad_bundle = {
            **SAMPLE_CATALOG["bundles"][0],
            "profiles": {
                "int8": {
                    "flow_lm_main": "int8",
                    "flow_lm_flow": "int8",
                    "mimi_decoder": "int8",
                    "mimi_encoder": "fp32",
                    "text_conditioner": "fp32",
                    "unknown_role": "fp32",  # Invalid
                }
            },
        }
        bad = {**SAMPLE_CATALOG, "bundles": [bad_bundle]}
        with pytest.raises(CatalogError, match="unknown role"):
            verify_catalog(bad)


class TestLoadCatalog:
    """Test load_catalog."""

    def test_load_valid_catalog(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(SAMPLE_CATALOG))

        result = load_catalog(path)
        assert result["schema"] == 1
        assert len(result["bundles"]) == 1

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        path.write_text("not json")

        with pytest.raises(CatalogError, match="Unable to load catalog"):
            load_catalog(path)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"

        with pytest.raises(CatalogError, match="Unable to load catalog"):
            load_catalog(path)


class TestBuildCatalog:
    """Test build_catalog (with mocked network)."""

    @patch("onnxvoice.catalog_tools.pocket._list_bundle_paths")
    @patch("onnxvoice.catalog_tools.pocket._fetch_bundle_json")
    def test_build_deterministic(
        self,
        mock_fetch: MagicMock,
        mock_list: MagicMock,
    ) -> None:
        mock_list.return_value = ["onnx/english_2026-04/bundle.json"]
        mock_fetch.return_value = (
            {
                "language": "en",
                "sample_rate": 24000,
                "layers": 6,
                "schema_version": 2,
                "aliases": ["english", "en"],
                "profiles": {
                    "int8": {
                        "flow_lm_main": "int8",
                        "flow_lm_flow": "int8",
                        "mimi_decoder": "int8",
                        "mimi_encoder": "fp32",
                        "text_conditioner": "fp32",
                    }
                },
                "artifacts": {
                    "bundle_metadata": {"filename": "bundle.json"},
                    "tokenizer": {"filename": "tokenizer.model"},
                    "bos_conditioning": {"filename": "bos.npy"},
                    "flow_lm_main": {"filename": "flow.onnx", "quality": "int8"},
                    "flow_lm_flow": {"filename": "flow.onnx", "quality": "int8"},
                    "mimi_decoder": {"filename": "decoder.onnx", "quality": "int8"},
                    "mimi_encoder": {"filename": "encoder.onnx", "quality": "fp32"},
                    "text_conditioner": {"filename": "conditioner.onnx", "quality": "fp32"},
                },
            },
            "sha256hash",
        )

        result1 = build_catalog(resolved_revision="a" * 40)
        result2 = build_catalog(resolved_revision="a" * 40)

        # Should be deterministic
        assert result1 == result2

        assert result1["schema"] == 1
        assert result1["kind"] == "pocket-onnx-bundle-catalog"
        assert result1["source"]["revision"] == "a" * 40
        verify_catalog(result1)
        assert len(result1["bundles"]) == 1

    @patch("onnxvoice.catalog_tools.pocket._list_bundle_paths")
    def test_build_no_bundles_raises(self, mock_list: MagicMock) -> None:
        mock_list.return_value = []

        with pytest.raises(CatalogError, match="No bundles found"):
            build_catalog(resolved_revision="a" * 40)
