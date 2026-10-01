from __future__ import annotations

import pytest

from onnxvoice.errors import AssetNotFoundError, VoiceNotFoundError
from onnxvoice.manager import OnnxVoice
from onnxvoice.types import Artifact, CatalogItem
from onnxvoice.voices import catalog_voice_ids, iter_catalog_voices, parse_voice_ref


def _piper_voice() -> CatalogItem:
    return CatalogItem(
        system="piper",
        id="en_US-amy-medium",
        kind="voice",
        artifacts=(Artifact("model", "voice.onnx"),),
        metadata={"language": {"code": "en_US", "name": "English (US)"}},
    )


def _kokoro_model() -> CatalogItem:
    return CatalogItem(
        system="kokoro",
        id="v1.0",
        kind="model",
        artifacts=(),
        voices=("af_heart",),
        metadata={
            "language_codes": ["en-US"],
            "voice_details": {"af_heart": {"locale": "en-US", "gender": "female"}},
        },
    )


def _pocket_bundle() -> CatalogItem:
    return CatalogItem(
        system="pocket",
        id="english_2026-04",
        kind="bundle",
        artifacts=(),
        voices=("alba",),
        metadata={"language": "en", "voice_details": {"alba": {"locale": "en"}}},
    )


def _supertonic_bundle() -> CatalogItem:
    return CatalogItem(
        system="supertonic",
        id="MODEL_OR_BUNDLE",
        kind="bundle",
        artifacts=(),
        voices=("F1",),
        metadata={"language_codes": ["ko"]},
    )


class _Catalog:
    def __init__(self, items: tuple[CatalogItem, ...]) -> None:
        self.items = items
        self.calls: list[tuple[str, bool, object]] = []

    def systems(self) -> tuple[str, ...]:
        return tuple(sorted({item.system for item in self.items}))

    def list(self, system: str, *, refresh: bool = False, progress=None):
        self.calls.append((system, refresh, progress))
        return [item for item in self.items if item.system == system]

    def resolve(self, ref: str, *, refresh: bool = False, progress=None):
        self.calls.append((ref, refresh, progress))
        return next(item for item in self.items if item.ref == ref)


def test_parse_semantic_voice_reference_preserves_catalog_spelling() -> None:
    asset_ref = parse_voice_ref("piper:en_US-amy-medium")
    assert (asset_ref.system, asset_ref.asset_id, asset_ref.voice_id) == (
        "piper",
        "en_US-amy-medium",
        None,
    )
    assert asset_ref.backing_ref == "piper:en_US-amy-medium"

    named_ref = parse_voice_ref("kokoro:v1.0/af_heart")
    assert (named_ref.system, named_ref.asset_id, named_ref.voice_id) == (
        "kokoro",
        "v1.0",
        "af_heart",
    )
    assert named_ref.backing_ref == "kokoro:v1.0"


@pytest.mark.parametrize(
    "ref", ["", "piper:", "piper:asset/", "piper:asset/voice/extra", "piper:asset "]
)
def test_parse_semantic_voice_reference_rejects_malformed_input(ref: str) -> None:
    with pytest.raises(ValueError):
        parse_voice_ref(ref)


def test_catalog_voice_iteration_is_system_neutral() -> None:
    single = CatalogItem(system="custom", id="voice-a", kind="voice", artifacts=())
    bundle = CatalogItem(
        system="another",
        id="bundle-a",
        kind="bundle",
        artifacts=(),
        voices=("voice-b", "voice-c"),
    )
    empty = CatalogItem(system="third", id="model", kind="model", artifacts=())

    assert catalog_voice_ids(single) == ("voice-a",)
    assert catalog_voice_ids(bundle) == ("voice-b", "voice-c")
    assert catalog_voice_ids(empty) == ()
    assert tuple(iter_catalog_voices((single, bundle, empty))) == (
        (single, "voice-a"),
        (bundle, "voice-b"),
        (bundle, "voice-c"),
    )


def test_list_voices_flattens_all_catalogs_and_filters_descriptive_language() -> None:
    items = (_supertonic_bundle(), _pocket_bundle(), _kokoro_model(), _piper_voice())
    manager = object.__new__(OnnxVoice)
    manager.catalog = _Catalog(items)

    records = manager.list_voices()
    assert [
        (record.ref, record.system, record.asset_id, record.voice_id) for record in records
    ] == [
        ("kokoro:v1.0/af_heart", "kokoro", "v1.0", "af_heart"),
        ("piper:en_US-amy-medium", "piper", "en_US-amy-medium", "en_US-amy-medium"),
        ("pocket:english_2026-04/alba", "pocket", "english_2026-04", "alba"),
        ("supertonic:MODEL_OR_BUNDLE/F1", "supertonic", "MODEL_OR_BUNDLE", "F1"),
    ]
    assert records[0].backing_ref == "kokoro:v1.0"
    assert records[0].locale == "en-US"
    assert records[0].gender == "female"

    english = manager.list_voices(language="en-US")
    assert {record.ref for record in english} == {
        "kokoro:v1.0/af_heart",
        "piper:en_US-amy-medium",
        "pocket:english_2026-04/alba",
    }


def test_multilingual_bundle_capabilities_filter_list_and_resolve() -> None:
    languages = ("en", "ko", "ja", "de")
    item = CatalogItem(
        system="supertonic",
        id="supertonic-3",
        kind="bundle",
        artifacts=(),
        voices=("F1", "M1"),
        metadata={"language_codes": languages},
    )
    manager = object.__new__(OnnxVoice)
    manager.catalog = _Catalog((item,))

    for language in ("de", "ja"):
        assert {record.voice_id for record in manager.list_voices(language=language)} == {
            "F1",
            "M1",
        }

    assert manager.list_voices(language="fr") == []
    record = manager.resolve_voice("supertonic:supertonic-3/F1")
    assert record.languages == languages


def test_per_voice_language_capabilities_restrict_generic_bundle_filtering() -> None:
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
    manager = object.__new__(OnnxVoice)
    manager.catalog = _Catalog((item,))

    records = {record.voice_id: record for record in manager.list_voices("generic")}
    assert records["voice-a"].languages == ("en",)
    assert records["voice-b"].languages == ("de",)
    assert [record.voice_id for record in manager.list_voices("generic", language="de")] == [
        "voice-b"
    ]
    assert manager.resolve_voice("generic:bundle/voice-a").languages == ("en",)
    assert manager.resolve_voice("generic:bundle/voice-b").languages == ("de",)


def test_list_voices_passes_catalog_options_and_uses_requested_system() -> None:
    manager = object.__new__(OnnxVoice)
    manager.catalog = _Catalog((_piper_voice(), _supertonic_bundle()))
    progress = object()

    records = manager.list_voices("SUPERtonic", refresh=True, progress=progress)

    assert [record.ref for record in records] == ["supertonic:MODEL_OR_BUNDLE/F1"]
    assert manager.catalog.calls == [("supertonic", True, progress)]


def test_resolve_voice_supports_asset_voices_and_named_child_voices() -> None:
    items = (_piper_voice(), _kokoro_model(), _pocket_bundle(), _supertonic_bundle())
    manager = object.__new__(OnnxVoice)
    manager.catalog = _Catalog(items)
    progress = object()

    piper = manager.resolve_voice("piper:en_US-amy-medium/en_US-amy-medium")
    kokoro = manager.resolve_voice("kokoro:v1.0/af_heart", refresh=True, progress=progress)
    pocket = manager.resolve_voice("pocket:english_2026-04/alba")
    supertonic = manager.resolve_voice("supertonic:MODEL_OR_BUNDLE/F1")

    assert piper.ref == "piper:en_US-amy-medium"
    assert (piper.system, piper.asset_id, piper.voice_id) == (
        "piper",
        "en_US-amy-medium",
        "en_US-amy-medium",
    )
    assert kokoro.ref == "kokoro:v1.0/af_heart"
    assert kokoro.backing_ref == "kokoro:v1.0"
    assert kokoro.gender == "female"
    assert pocket.ref == "pocket:english_2026-04/alba"
    assert supertonic.ref == "supertonic:MODEL_OR_BUNDLE/F1"
    assert manager.catalog.calls[1] == ("kokoro:v1.0", True, progress)


def test_resolve_voice_requires_catalog_declared_child_voice() -> None:
    manager = object.__new__(OnnxVoice)
    manager.catalog = _Catalog((_kokoro_model(), _piper_voice()))

    with pytest.raises(VoiceNotFoundError, match="required"):
        manager.resolve_voice("kokoro:v1.0")
    with pytest.raises(VoiceNotFoundError, match="not declared"):
        manager.resolve_voice("kokoro:v1.0/not-a-voice")
    with pytest.raises(VoiceNotFoundError, match="not declared"):
        manager.resolve_voice("piper:en_US-amy-medium/not-a-voice")


def test_resolve_voice_propagates_missing_catalog_asset() -> None:
    class MissingCatalog(_Catalog):
        def resolve(self, ref: str, *, refresh: bool = False, progress=None):
            raise AssetNotFoundError(f"Unknown catalog item: {ref}")

    manager = object.__new__(OnnxVoice)
    manager.catalog = MissingCatalog(())

    with pytest.raises(AssetNotFoundError, match="Unknown catalog item"):
        manager.resolve_voice("piper:missing")
