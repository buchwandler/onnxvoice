"""Tests for Pocket catalog parsing and profile selection."""

from __future__ import annotations

import pytest

from onnxvoice.catalog import (
    POCKET_ARTIFACT_ROLES,
    POCKET_QUALIFIED_ROLES,
    _parse_pocket,
    _select_pocket_profile,
)
from onnxvoice.errors import CatalogError

SAMPLE_CATALOG = {
    "source": {
        "repository": "KevinAHM/pocket-tts-onnx",
        "revision": "abc123def456abc123def456abc123def456abc1",
        "requested_revision": "main",
    },
    "bundles": [
        {
            "id": "english_2026-04",
            "language": "en",
            "layers": 6,
            "bundle_schema": 2,
            "sample_rate": 24000,
            "aliases": ["english", "en"],
            "max_token_per_chunk": 200,
            "model_recommended_frames_after_eos": 15,
            "remove_semicolons": True,
            "pad_with_spaces_for_short_inputs": True,
            "profiles": {
                "int8": {
                    "flow_lm_main": "int8",
                    "flow_lm_flow": "int8",
                    "mimi_decoder": "int8",
                    "mimi_encoder": "fp32",
                    "text_conditioner": "fp32",
                },
                "fp32": {
                    "flow_lm_main": "fp32",
                    "flow_lm_flow": "fp32",
                    "mimi_decoder": "fp32",
                    "mimi_encoder": "fp32",
                    "text_conditioner": "fp32",
                },
            },
            "artifacts": [
                {
                    "role": "bundle_metadata",
                    "filename": "bundle.json",
                    "url": "https://example.com/bundle.json",
                    "size": 1024,
                    "sha256": "abc123",
                },
                {
                    "role": "tokenizer",
                    "filename": "tokenizer.model",
                    "url": "https://example.com/tokenizer.model",
                    "size": 512,
                    "sha256": "def456",
                },
                {
                    "role": "bos_conditioning",
                    "filename": "bos_before_voice.npy",
                    "url": "https://example.com/bos_before_voice.npy",
                    "size": 256,
                    "sha256": "ghi789",
                },
                {
                    "role": "flow_lm_main",
                    "filename": "flow_lm_main_int8.onnx",
                    "url": "https://example.com/flow_lm_main_int8.onnx",
                    "quality": "int8",
                    "size": 1024 * 1024,
                    "sha256": "jkl012",
                },
                {
                    "role": "flow_lm_main",
                    "filename": "flow_lm_main_fp32.onnx",
                    "url": "https://example.com/flow_lm_main_fp32.onnx",
                    "quality": "fp32",
                    "size": 4 * 1024 * 1024,
                    "sha256": "mno345",
                },
                {
                    "role": "flow_lm_flow",
                    "filename": "flow_lm_flow_int8.onnx",
                    "url": "https://example.com/flow_lm_flow_int8.onnx",
                    "quality": "int8",
                    "size": 512 * 1024,
                    "sha256": "pqr678",
                },
                {
                    "role": "flow_lm_flow",
                    "filename": "flow_lm_flow_fp32.onnx",
                    "url": "https://example.com/flow_lm_flow_fp32.onnx",
                    "quality": "fp32",
                    "size": 2 * 1024 * 1024,
                    "sha256": "stu901",
                },
                {
                    "role": "mimi_decoder",
                    "filename": "mimi_decoder_int8.onnx",
                    "url": "https://example.com/mimi_decoder_int8.onnx",
                    "quality": "int8",
                    "size": 256 * 1024,
                    "sha256": "vwx234",
                },
                {
                    "role": "mimi_decoder",
                    "filename": "mimi_decoder_fp32.onnx",
                    "url": "https://example.com/mimi_decoder_fp32.onnx",
                    "quality": "fp32",
                    "size": 1024 * 1024,
                    "sha256": "yza567",
                },
                {
                    "role": "mimi_encoder",
                    "filename": "mimi_encoder.onnx",
                    "url": "https://example.com/mimi_encoder.onnx",
                    "quality": "fp32",
                    "size": 512 * 1024,
                    "sha256": "bcd890",
                },
                {
                    "role": "text_conditioner",
                    "filename": "text_conditioner.onnx",
                    "url": "https://example.com/text_conditioner.onnx",
                    "quality": "fp32",
                    "size": 128 * 1024,
                    "sha256": "efg123",
                },
            ],
        }
    ],
}


class TestPocketArtifactRoles:
    """Test that pocket artifact roles are properly defined."""

    def test_all_roles_defined(self) -> None:
        expected = {
            "bundle_metadata",
            "tokenizer",
            "bos_conditioning",
            "flow_lm_main",
            "flow_lm_flow",
            "mimi_decoder",
            "mimi_encoder",
            "text_conditioner",
        }
        assert expected == POCKET_ARTIFACT_ROLES

    def test_qualified_roles_subset(self) -> None:
        assert POCKET_QUALIFIED_ROLES.issubset(POCKET_ARTIFACT_ROLES)
        assert len(POCKET_QUALIFIED_ROLES) == 5


class TestParsePocket:
    """Test Pocket catalog parsing."""

    def test_parse_basic_catalog(self) -> None:
        items = _parse_pocket(SAMPLE_CATALOG)
        assert len(items) == 1

        item = items[0]
        assert item.system == "pocket"
        assert item.id == "english_2026-04"
        assert item.kind == "bundle"
        assert item.sample_rate == 24000
        assert item.aliases == ("english", "en")

    def test_parse_artifacts(self) -> None:
        items = _parse_pocket(SAMPLE_CATALOG)
        item = items[0]

        roles = {a.role for a in item.artifacts}
        assert roles == POCKET_ARTIFACT_ROLES

        # Check we have multiple qualities for qualified roles
        for role in POCKET_QUALIFIED_ROLES:
            qualities = {a.quality for a in item.artifacts if a.role == role}
            assert len(qualities) >= 1, f"Role {role!r} should have at least one quality variant"

    def test_parse_metadata(self) -> None:
        items = _parse_pocket(SAMPLE_CATALOG)
        item = items[0]

        assert item.metadata["language"] == "en"
        assert item.metadata["layers"] == 6
        assert item.metadata["bundle_schema"] == 2
        assert item.metadata["source_revision"] == "abc123def456abc123def456abc123def456abc1"
        assert item.metadata["source_repository"] == "KevinAHM/pocket-tts-onnx"
        assert item.metadata["requested_revision"] == "main"
        assert item.metadata["max_token_per_chunk"] == 200
        assert item.metadata["model_recommended_frames_after_eos"] == 15
        assert item.metadata["remove_semicolons"] is True
        assert item.metadata["pad_with_spaces_for_short_inputs"] is True

    def test_parse_profiles(self) -> None:
        items = _parse_pocket(SAMPLE_CATALOG)
        item = items[0]

        profiles = item.metadata["profiles"]
        assert "int8" in profiles
        assert "fp32" in profiles

        int8_profile = profiles["int8"]
        assert int8_profile["flow_lm_main"] == "int8"
        assert int8_profile["mimi_encoder"] == "fp32"

    def test_parse_missing_bundles(self) -> None:
        with pytest.raises(CatalogError, match="missing the 'bundles' list"):
            _parse_pocket({})

    def test_parse_missing_bundle_id(self) -> None:
        bad_catalog = {"bundles": [{"language": "en", "artifacts": []}]}
        with pytest.raises(CatalogError, match="missing 'id'"):
            _parse_pocket(bad_catalog)

    def test_parse_unknown_role(self) -> None:
        bad_catalog = {
            "bundles": [
                {
                    "id": "test",
                    "artifacts": [{"role": "unknown_role", "filename": "test.onnx"}],
                }
            ]
        }
        with pytest.raises(CatalogError, match="Unknown Pocket artifact role"):
            _parse_pocket(bad_catalog)

    def test_parse_multiple_bundles(self) -> None:
        catalog = {
            "source": {"repository": "test", "revision": "a" * 40},
            "bundles": [
                {
                    "id": "bundle1",
                    "sample_rate": 24000,
                    "artifacts": [
                        {"role": "bundle_metadata", "filename": "bundle.json"},
                        {"role": "tokenizer", "filename": "tokenizer.model"},
                        {"role": "bos_conditioning", "filename": "bos.npy"},
                        {"role": "flow_lm_main", "filename": "flow.onnx", "quality": "fp32"},
                        {"role": "flow_lm_flow", "filename": "flow.onnx", "quality": "fp32"},
                        {"role": "mimi_decoder", "filename": "decoder.onnx", "quality": "fp32"},
                        {"role": "mimi_encoder", "filename": "encoder.onnx", "quality": "fp32"},
                        {
                            "role": "text_conditioner",
                            "filename": "conditioner.onnx",
                            "quality": "fp32",
                        },
                    ],
                    "profiles": {
                        "fp32": {
                            "flow_lm_main": "fp32",
                            "flow_lm_flow": "fp32",
                            "mimi_decoder": "fp32",
                            "mimi_encoder": "fp32",
                            "text_conditioner": "fp32",
                        }
                    },
                },
                {
                    "id": "bundle2",
                    "sample_rate": 22050,
                    "artifacts": [
                        {"role": "bundle_metadata", "filename": "bundle.json"},
                        {"role": "tokenizer", "filename": "tokenizer.model"},
                        {"role": "bos_conditioning", "filename": "bos.npy"},
                        {"role": "flow_lm_main", "filename": "flow.onnx", "quality": "int8"},
                        {"role": "flow_lm_flow", "filename": "flow.onnx", "quality": "int8"},
                        {"role": "mimi_decoder", "filename": "decoder.onnx", "quality": "int8"},
                        {"role": "mimi_encoder", "filename": "encoder.onnx", "quality": "fp32"},
                        {
                            "role": "text_conditioner",
                            "filename": "conditioner.onnx",
                            "quality": "fp32",
                        },
                    ],
                    "profiles": {
                        "int8": {
                            "flow_lm_main": "int8",
                            "flow_lm_flow": "int8",
                            "mimi_decoder": "int8",
                            "mimi_encoder": "fp32",
                            "text_conditioner": "fp32",
                        }
                    },
                },
            ],
        }
        items = _parse_pocket(catalog)
        assert len(items) == 2
        assert items[0].id == "bundle1"
        assert items[1].id == "bundle2"


class TestSelectPocketProfile:
    """Test Pocket profile selection."""

    def _get_item(self):
        items = _parse_pocket(SAMPLE_CATALOG)
        return items[0]

    def test_select_int8_profile(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, "int8")

        assert selected.metadata["selected_quality"] == "int8"

        # Check qualified roles have correct qualities
        for artifact in selected.artifacts:
            if (
                artifact.role == "flow_lm_main"
                or artifact.role == "flow_lm_flow"
                or artifact.role == "mimi_decoder"
            ):
                assert artifact.quality == "int8"
            elif artifact.role == "mimi_encoder" or artifact.role == "text_conditioner":
                assert artifact.quality == "fp32"

    def test_select_fp32_profile(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, "fp32")

        assert selected.metadata["selected_quality"] == "fp32"

        for artifact in selected.artifacts:
            if artifact.role in POCKET_QUALIFIED_ROLES:
                assert artifact.quality == "fp32"

    def test_select_default_profile(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, None)

        # Default should be int8
        assert selected.metadata["selected_quality"] == "int8"

    def test_select_unknown_profile(self) -> None:
        item = self._get_item()
        with pytest.raises(CatalogError, match="Unknown Pocket profile"):
            _select_pocket_profile(item, "unknown")

    def test_static_artifacts_preserved(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, "int8")

        static_roles = {"bundle_metadata", "tokenizer", "bos_conditioning"}
        selected_static = {a.role for a in selected.artifacts if a.role in static_roles}
        assert selected_static == static_roles

    def test_one_artifact_per_qualified_role(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, "int8")

        for role in POCKET_QUALIFIED_ROLES:
            matching = [a for a in selected.artifacts if a.role == role]
            assert len(matching) == 1, f"Expected exactly one {role!r} artifact"

    def test_sample_rate_preserved(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, "int8")
        assert selected.sample_rate == 24000

    def test_aliases_preserved(self) -> None:
        item = self._get_item()
        selected = _select_pocket_profile(item, "int8")
        assert selected.aliases == ("english", "en")
