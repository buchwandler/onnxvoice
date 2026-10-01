"""Tests for the inventory module: language normalization, gender, filters, size, and update comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from onnxvoice.inventory import (
    InventoryFilter,
    InventoryRecord,
    _match_code,
    catalog_size_bytes,
    compare_installation_to_catalog,
    effective_distribution,
    effective_quality,
    gender_from_metadata,
    inventory_record_from_catalog,
    inventory_record_from_installation,
    language_base,
    language_codes_from_metadata,
    language_tags_match,
    logical_size_bytes,
    matches_filter,
    matches_language,
    merge_inventory,
    normalize_gender,
    normalize_language_code,
    normalize_language_tag,
    primary_language_from_metadata,
    query_inventory,
    version_label,
    voice_language_codes_from_catalog,
    voice_metadata_from_catalog,
)
from onnxvoice.types import Artifact, CatalogItem, Installation, InstalledArtifact

# ---------------------------------------------------------------------------
# Language normalization tests
# ---------------------------------------------------------------------------


class TestLanguageNormalization:
    """Test language code normalization across all system formats."""

    def test_piper_dict_language(self):
        """Piper stores language as dict with 'code'."""
        metadata = {"language": {"code": "en_US", "name": "English (US)"}}
        codes = language_codes_from_metadata(metadata)
        assert codes == ("en-US",)

    def test_pocket_string_language(self):
        """Pocket stores language as a plain string."""
        metadata = {"language": "en"}
        codes = language_codes_from_metadata(metadata)
        assert codes == ("en",)

    def test_kokoro_language_codes_list(self):
        """Kokoro stores language_codes as a list."""
        metadata = {"language_codes": ["en-us", "en-gb"]}
        codes = language_codes_from_metadata(metadata)
        assert codes == ("en-US", "en-GB")

    def test_kokoro_takes_priority_over_language_field(self):
        """When language_codes is present, it's used even if language also exists."""
        metadata = {"language": "de", "language_codes": ["en-us"]}
        codes = language_codes_from_metadata(metadata)
        assert codes == ("en-US",)

    def test_normalize_language_code_dict(self):
        assert normalize_language_code({"code": "en_US"}) == "en-US"

    def test_normalize_language_code_string(self):
        assert normalize_language_code("en") == "en"
        assert normalize_language_code("en_US") == "en-US"

    def test_normalize_language_tag_canonical_casing(self):
        assert normalize_language_tag("en_US") == "en-US"
        assert normalize_language_tag("en-us") == "en-US"
        assert normalize_language_tag("EN_us") == "en-US"
        assert normalize_language_tag("pt_BR") == "pt-BR"
        assert normalize_language_tag("zh_cn") == "zh-CN"
        assert normalize_language_tag("en") == "en"

    def test_language_base(self):
        assert language_base("en") == "en"
        assert language_base("en-US") == "en"
        assert language_base("en-GB") == "en"
        assert language_base("de-DE") == "de"
        assert language_base("pt-BR") == "pt"

    def test_language_tags_match_generic_and_specific_tags(self):
        assert language_tags_match("en", "en")
        assert language_tags_match("en", "en-US")
        assert language_tags_match("en", "en-GB")
        assert language_tags_match("en-US", "en")
        assert language_tags_match("en-US", "en-US")
        assert not language_tags_match("en-US", "en-GB")
        assert language_tags_match("en_US", "en")
        assert not language_tags_match("en_US", "en_GB")

    def test_normalize_language_code_empty(self):
        assert normalize_language_code(None) == ""
        assert normalize_language_code({}) == ""
        assert normalize_language_code("") == ""

    def test_primary_language_from_metadata(self):
        meta = {"language_codes": ["en-us", "en-gb"]}
        assert primary_language_from_metadata(meta) == "en-US"

    def test_primary_language_returns_none_when_empty(self):
        assert primary_language_from_metadata({}) is None

    def test_subtag_matching_en_matches_en_us(self):
        """--lang en should match en-US."""
        assert _match_code("en", "en-US") is True

    def test_subtag_matching_en_matches_en_gb(self):
        assert _match_code("en", "en-GB") is True

    def test_subtag_matching_en_us_matches_en_us(self):
        assert _match_code("en-US", "en-US") is True

    def test_subtag_matching_en_us_not_match_en_gb(self):
        """--lang en-US should NOT match en-GB."""
        assert _match_code("en-US", "en-GB") is False

    def test_subtag_matching_case_insensitive(self):
        assert _match_code("en-us", "EN_US") is True

    def test_matches_language_piper(self):
        meta = {"language": {"code": "en_US"}}
        assert matches_language(meta, "en") is True
        assert matches_language(meta, "en-US") is True
        assert matches_language(meta, "de") is False

    def test_matches_language_pocket(self):
        meta = {"language": "en"}
        assert matches_language(meta, "en") is True
        assert matches_language(meta, "en-US") is True

    def test_matches_language_kokoro(self):
        meta = {"language_codes": ["en-us", "en-gb"]}
        assert matches_language(meta, "en-US") is True
        assert matches_language(meta, "en") is True
        assert matches_language(meta, "de") is False

    def test_matches_language_empty_metadata(self):
        assert matches_language({}, "en") is False


# ---------------------------------------------------------------------------
# Gender normalization tests
# ---------------------------------------------------------------------------


class TestGenderNormalization:
    """Test gender normalization and metadata extraction."""

    def test_normalize_male(self):
        assert normalize_gender("male") == "male"

    def test_normalize_female(self):
        assert normalize_gender("female") == "female"

    def test_normalize_neutral(self):
        assert normalize_gender("neutral") == "neutral"

    def test_normalize_unknown(self):
        assert normalize_gender("unknown") == "unknown"

    def test_normalize_case_insensitive(self):
        assert normalize_gender("Male") == "male"
        assert normalize_gender("FEMALE") == "female"

    def test_normalize_missing_is_unknown(self):
        assert normalize_gender(None) == "unknown"
        assert normalize_gender("") == "unknown"

    def test_normalize_invalid_is_unknown(self):
        assert normalize_gender("other") == "unknown"
        assert normalize_gender(42) == "unknown"

    def test_gender_from_metadata_present(self):
        assert gender_from_metadata({"gender": "male"}) == "male"

    def test_gender_from_metadata_missing(self):
        assert gender_from_metadata({}) == "unknown"

    def test_no_inference_from_id(self):
        """Gender must NOT be inferred from IDs like 'ryan' or 'amy'."""
        meta = {"id": "ryan"}  # No gender key
        assert gender_from_metadata(meta) == "unknown"

    def test_no_inference_from_name(self):
        meta = {"name": "Amy"}  # No gender key
        assert gender_from_metadata(meta) == "unknown"

    def test_entry_with_gendered_name_but_no_gender_metadata_stays_unknown(self):
        """Even if the name looks gendered, metadata absence means 'unknown'."""
        meta = {"name": "Female Voice", "language": "en"}
        assert gender_from_metadata(meta) == "unknown"


class TestVoiceMetadataExtraction:
    def test_per_voice_details_take_precedence(self):
        item = CatalogItem(
            system="pocket",
            id="bundle",
            kind="bundle",
            artifacts=(),
            voices=("alba", "other"),
            metadata={
                "language": {"code": "en_US", "name": "English (US)"},
                "gender": "male",
                "voice_details": [
                    {
                        "id": "alba",
                        "language": "pt",
                        "locale": "pt_br",
                        "language_label": "Brazilian Portuguese",
                        "gender": "female",
                    }
                ],
            },
        )

        details = voice_metadata_from_catalog(item, "alba")

        assert details.language == "pt"
        assert details.locale == "pt-BR"
        assert details.language_label == "Brazilian Portuguese"
        assert details.gender == "female"

    def test_item_level_exact_metadata_and_multi_voice_gender(self):
        item = CatalogItem(
            system="kokoro",
            id="model",
            kind="model",
            artifacts=(),
            voices=("voice-a", "voice-b"),
            metadata={
                "language": {"code": "en_US", "name": "English (US)"},
                "language_codes": ["en-GB"],
                "gender": "female",
            },
        )

        details = voice_metadata_from_catalog(item, "voice-a")

        assert details.language == "en"
        assert details.locale == "en-US"
        assert details.language_label == "English (US)"
        assert details.gender == "unknown"

    def test_bare_region_is_not_used_as_language_label(self):
        item = CatalogItem(
            system="piper",
            id="voice",
            kind="voice",
            artifacts=(),
            metadata={
                "language": {"code": "en_US", "name": "US"},
                "language_label": "US",
            },
        )

        details = voice_metadata_from_catalog(item, "voice")

        assert details.locale == "en-US"
        assert details.language_label == "en-US"

    def test_generic_single_voice_metadata_and_unknown_fallback(self):
        item = CatalogItem(
            system="pocket",
            id="bundle",
            kind="bundle",
            artifacts=(),
            voices=("alba",),
            metadata={"language": "en", "gender": "female"},
        )
        details = voice_metadata_from_catalog(item, "alba")
        assert (details.language, details.locale, details.language_label) == ("en", "en", "en")
        assert details.gender == "female"

        unknown = voice_metadata_from_catalog(
            CatalogItem(system="test", id="item", kind="item", artifacts=()), "item"
        )
        assert (unknown.language, unknown.locale, unknown.language_label) == ("", "", "unknown")
        assert unknown.gender == "unknown"

    def test_missing_pocket_voice_details_keeps_gender_unknown(self):
        item = CatalogItem(
            system="pocket",
            id="bundle",
            kind="bundle",
            artifacts=(),
            voices=("alba",),
            metadata={"language": "en"},
        )

        details = voice_metadata_from_catalog(item, "alba")

        assert details.language == "en"
        assert details.locale == "en"
        assert details.gender == "unknown"


# ---------------------------------------------------------------------------
# Effective quality / distribution tests
# ---------------------------------------------------------------------------


class TestVoiceLanguageCapabilities:
    def test_child_language_metadata_precedes_bundle_capabilities(self):
        item = CatalogItem(
            system="generic",
            id="bundle",
            kind="bundle",
            artifacts=(),
            voices=("voice-a", "voice-b"),
            metadata={
                "language_codes": ("en", "de"),
                "voice_details": {
                    "voice-a": {"locale": "en"},
                    "voice-b": {"locale": "de"},
                },
            },
        )

        assert voice_language_codes_from_catalog(item, "voice-a") == ("en",)
        assert voice_language_codes_from_catalog(item, "voice-b") == ("de",)

    def test_bundle_capabilities_are_inherited_when_voice_metadata_is_absent(self):
        item = CatalogItem(
            system="generic",
            id="bundle",
            kind="bundle",
            artifacts=(),
            voices=("voice-a",),
            metadata={"language_codes": ("en", "de", "ja")},
        )

        assert voice_language_codes_from_catalog(item, "voice-a") == ("en", "de", "ja")

    def test_descriptive_locale_is_the_last_language_fallback(self):
        item = CatalogItem(
            system="generic",
            id="voice-a",
            kind="voice",
            artifacts=(),
            metadata={"locale": "pt_br"},
        )

        assert voice_language_codes_from_catalog(item, "voice-a") == ("pt-BR",)


class TestEffectiveSelection:
    def test_effective_quality_selected_over_quality(self):
        meta = {"selected_quality": "fp16", "quality": "fp32"}
        assert effective_quality(meta) == "fp16"

    def test_effective_quality_fallback_to_quality(self):
        meta = {"quality": "fp32"}
        assert effective_quality(meta) == "fp32"

    def test_effective_quality_none(self):
        assert effective_quality({}) is None

    def test_effective_distribution_selected_over_distribution_id(self):
        meta = {"selected_distribution": "cpu", "distribution_id": "gpu"}
        assert effective_distribution(meta) == "cpu"

    def test_effective_distribution_fallback(self):
        meta = {"distribution_id": "gpu"}
        assert effective_distribution(meta) == "gpu"

    def test_effective_distribution_none(self):
        assert effective_distribution({}) is None


# ---------------------------------------------------------------------------
# Version label tests
# ---------------------------------------------------------------------------


class TestVersionLabel:
    def test_piper_uses_source_revision(self):
        item = CatalogItem(
            system="piper",
            id="test",
            kind="voice",
            artifacts=(),
            metadata={"source_revision": "abc123def456"},
        )
        assert version_label(item) == "abc123de"  # truncated to 8

    def test_kokoro_prefers_model_version(self):
        item = CatalogItem(
            system="kokoro",
            id="test",
            kind="model",
            artifacts=(),
            metadata={"model_version": "1.0"},
        )
        assert version_label(item) == "1.0"

    def test_external_returns_local(self):
        item = CatalogItem(
            system="test",
            id="test",
            kind="model",
            artifacts=(),
            metadata={"external": True},
        )
        assert version_label(item) == "local"


# ---------------------------------------------------------------------------
# Size helper tests
# ---------------------------------------------------------------------------


class TestSizeHelpers:
    def test_logical_size_bytes(self):
        inst = Installation(
            system="test",
            id="one",
            kind="model",
            path=Path("/tmp"),
            artifacts=(
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "sha", 100),
                InstalledArtifact("config", "b.json", Path("/tmp/b.json"), "sha", 50),
            ),
        )
        assert logical_size_bytes(inst) == 150

    def test_catalog_size_bytes(self):
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(
                Artifact("model", "a.onnx", size=100),
                Artifact("config", "b.json", size=50),
            ),
        )
        assert catalog_size_bytes(item) == 150

    def test_catalog_size_bytes_unknown(self):
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(
                Artifact("model", "a.onnx", size=100),
                Artifact("config", "b.json", size=None),
            ),
        )
        assert catalog_size_bytes(item) is None


# ---------------------------------------------------------------------------
# Inventory filter tests
# ---------------------------------------------------------------------------


def _make_record(**kwargs: Any) -> InventoryRecord:
    """Helper to create InventoryRecord with defaults."""
    defaults: dict[str, Any] = {
        "system": "piper",
        "id": "test",
        "kind": "voice",
        "language_codes": ("en-US",),
        "gender": "unknown",
        "quality": None,
        "distribution": None,
        "version": None,
        "installed": False,
        "installation": None,
        "catalog_item": None,
        "size_bytes": 100,
        "status": "available",
        "update_status": "not_checked",
    }
    defaults.update(kwargs)
    return InventoryRecord(**defaults)


class TestInventoryFilter:
    def test_filter_by_system(self):
        spec = InventoryFilter(systems=("piper",))
        rec = _make_record(system="piper")
        assert matches_filter(rec, spec) is True

    def test_filter_by_system_mismatch(self):
        spec = InventoryFilter(systems=("piper",))
        rec = _make_record(system="kokoro")
        assert matches_filter(rec, spec) is False

    def test_filter_by_kind(self):
        spec = InventoryFilter(kinds=("voice",))
        rec = _make_record(kind="voice")
        assert matches_filter(rec, spec) is True

    def test_filter_by_language(self):
        spec = InventoryFilter(languages=("en-US",))
        rec = _make_record(language_codes=("en-US",))
        assert matches_filter(rec, spec) is True

    def test_specific_language_filter_matches_generic_catalog_language(self):
        spec = InventoryFilter(languages=("en-US",))
        rec = _make_record(language_codes=("en",))
        assert matches_filter(rec, spec) is True

    def test_filter_by_language_subtag(self):
        spec = InventoryFilter(languages=("en",))
        rec = _make_record(language_codes=("en-US",))
        assert matches_filter(rec, spec) is True

    def test_filter_by_language_no_match(self):
        spec = InventoryFilter(languages=("de",))
        rec = _make_record(language_codes=("en-US",))
        assert matches_filter(rec, spec) is False

    def test_filter_by_gender(self):
        spec = InventoryFilter(genders=("male",))
        rec = _make_record(gender="male")
        assert matches_filter(rec, spec) is True

    def test_filter_by_gender_mismatch(self):
        spec = InventoryFilter(genders=("male",))
        rec = _make_record(gender="female")
        assert matches_filter(rec, spec) is False

    def test_filter_by_status(self):
        spec = InventoryFilter(statuses=("installed",))
        rec = _make_record(status="installed")
        assert matches_filter(rec, spec) is True

    def test_filter_by_quality(self):
        spec = InventoryFilter(qualities=("fp16",))
        rec = _make_record(quality="fp16")
        assert matches_filter(rec, spec) is True

    def test_filter_no_criteria_matches_all(self):
        spec = InventoryFilter()
        rec = _make_record()
        assert matches_filter(rec, spec) is True


# ---------------------------------------------------------------------------
# Update comparison tests
# ---------------------------------------------------------------------------


class TestUpdateComparison:
    def _installation(self, artifacts):
        return Installation(
            system="test",
            id="one",
            kind="model",
            path=Path("/tmp"),
            artifacts=tuple(artifacts),
        )

    def _catalog_item(self, artifacts):
        return CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=tuple(artifacts),
        )

    def test_current_sha256_match(self):
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_update_available_sha256_differs(self):
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx", sha256="def456", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"
        assert "model" in result.changed_roles

    def test_current_size_match_no_sha(self):
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx", size=100),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"

    def test_update_available_size_differs_no_sha(self):
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx", size=200),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"

    def test_unknown_no_identity(self):
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx"),  # No sha256, no size
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "unknown"

    def test_update_available_artifact_added(self):
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
                Artifact("config", "b.json", sha256="xyz", size=50),
            ]
        )
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "update_available"

    def test_current_catalog_revision_changes_but_artifacts_match(self):
        """Unrelated catalog source revision changes should NOT cause false positive."""
        inst = self._installation(
            [
                InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "abc123", 100),
            ]
        )
        item = self._catalog_item(
            [
                Artifact("model", "a.onnx", sha256="abc123", size=100),
            ]
        )
        # Even if the catalog item has different metadata, artifacts match
        result = compare_installation_to_catalog(inst, item)
        assert result.status == "current"


# ---------------------------------------------------------------------------
# Inventory record creation tests
# ---------------------------------------------------------------------------


class TestInventoryRecordCreation:
    def test_record_from_installation(self):
        inst = Installation(
            system="piper",
            id="test-voice",
            kind="voice",
            path=Path("/tmp"),
            artifacts=(InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "sha", 100),),
            metadata={"language": {"code": "en_US"}, "gender": "male"},
        )
        rec = inventory_record_from_installation(inst)
        assert rec.system == "piper"
        assert rec.kind == "voice"
        assert rec.language_codes == ("en-US",)
        assert rec.gender == "male"
        assert rec.installed is True
        assert rec.size_bytes == 100

    def test_record_from_catalog(self):
        item = CatalogItem(
            system="piper",
            id="test-voice",
            kind="voice",
            artifacts=(Artifact("model", "a.onnx", size=100),),
            metadata={"language": {"code": "en_US"}},
        )
        rec = inventory_record_from_catalog(item)
        assert rec.system == "piper"
        assert rec.status == "available"
        assert rec.gender == "unknown"  # No gender in metadata

    def test_external_installation_has_local_status(self):
        inst = Installation(
            system="test",
            id="local",
            kind="external",
            path=Path("/tmp"),
            artifacts=(),
            metadata={"external": True},
        )
        rec = inventory_record_from_installation(inst)
        assert rec.status == "local"


# ---------------------------------------------------------------------------
# Merge inventory tests
# ---------------------------------------------------------------------------


class TestMergeInventory:
    def test_installed_and_catalog_merged(self):
        inst = Installation(
            system="piper",
            id="voice-a",
            kind="voice",
            path=Path("/tmp"),
            artifacts=(InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "sha", 100),),
            metadata={"language": {"code": "en_US"}},
        )
        cat_installed = CatalogItem(
            system="piper",
            id="voice-a",
            kind="voice",
            artifacts=(Artifact("model", "a.onnx", sha256="sha", size=100),),
            metadata={"language": {"code": "en_US"}},
        )
        cat_available = CatalogItem(
            system="piper",
            id="voice-b",
            kind="voice",
            artifacts=(Artifact("model", "b.onnx", size=200),),
            metadata={"language": {"code": "en_US"}},
        )
        records = merge_inventory([inst], [cat_installed, cat_available])
        assert len(records) == 2
        by_ref = {r.ref: r for r in records}
        assert by_ref["piper:voice-a"].installed is True
        assert by_ref["piper:voice-b"].installed is False

    def test_query_inventory_with_filter(self):
        inst_en = Installation(
            system="piper",
            id="en-voice",
            kind="voice",
            path=Path("/tmp"),
            artifacts=(InstalledArtifact("model", "a.onnx", Path("/tmp/a.onnx"), "sha", 100),),
            metadata={"language": {"code": "en_US"}},
        )
        inst_de = Installation(
            system="piper",
            id="de-voice",
            kind="voice",
            path=Path("/tmp"),
            artifacts=(InstalledArtifact("model", "b.onnx", Path("/tmp/b.onnx"), "sha", 200),),
            metadata={"language": {"code": "de_DE"}},
        )
        spec = InventoryFilter(languages=("en",))
        records = query_inventory([inst_en, inst_de], None, spec=spec)
        assert len(records) == 1
        assert records[0].id == "en-voice"
