from __future__ import annotations

import inspect

import pytest

from onnxvoice.catalog import parse_ref
from onnxvoice.catalog_tools.voice_selectors import (
    check_registry_and_catalog,
    missing_registry_voices,
    unassigned_catalog_voices,
)
from onnxvoice.cli import _render_voice_records, build_parser
from onnxvoice.errors import (
    VoiceSelectorError,
    VoiceSelectorNotFoundError,
    VoiceSelectorRegistryError,
    VoiceSelectorRetiredError,
)
from onnxvoice.manager import OnnxVoice
from onnxvoice.types import Artifact, CatalogItem, VoiceRecord
from onnxvoice.voice_selectors import (
    VoiceSelectorRegistry,
    catalog_voice_keys,
    format_voice_selector,
    is_voice_selector,
    iter_voice_identities,
    load_voice_selector_registry,
    parse_voice_selector,
    resolve_voice_selector,
)


def _entry(
    language: str, code: str, slot: int, system: str, asset: str, voice: str, state="active"
):
    return {
        "language": language,
        "engine_code": code,
        "slot": slot,
        "system": system,
        "asset_id": asset,
        "voice_id": voice,
        "state": state,
    }


def _registry(*entries):
    return VoiceSelectorRegistry.from_data(
        {"schema": 1, "engine_codes": {"kokoro": "ko", "piper": "pi"}, "entries": list(entries)}
    )


def _piper_voice(voice_id: str, language: str = "de_DE") -> CatalogItem:
    return CatalogItem(
        system="piper",
        id=voice_id,
        kind="voice",
        artifacts=(Artifact("model", "voice.onnx"),),
        metadata={"language": {"code": language}, "gender": None},
    )


def _kokoro_model(model_id: str, voices: tuple[str, ...]) -> CatalogItem:
    return CatalogItem(
        system="kokoro",
        id=model_id,
        kind="model",
        artifacts=(),
        voices=voices,
        metadata={"language_codes": ["de"]},
    )


EXPECTED_EN_US_KOKORO = [
    ("en_us-ko-1", "v1.0", "af_alloy"),
    ("en_us-ko-2", "v1.0", "af_aoede"),
    ("en_us-ko-3", "v1.0", "af_bella"),
    ("en_us-ko-4", "v1.0", "af_heart"),
    ("en_us-ko-5", "v1.0", "af_jessica"),
    ("en_us-ko-6", "v1.0", "af_kore"),
    ("en_us-ko-7", "v1.0", "af_nicole"),
    ("en_us-ko-8", "v1.0", "af_nova"),
    ("en_us-ko-9", "v1.0", "af_river"),
    ("en_us-ko-10", "v1.0", "af_sarah"),
    ("en_us-ko-11", "v1.0", "af_sky"),
    ("en_us-ko-12", "v1.0", "am_adam"),
    ("en_us-ko-13", "v1.0", "am_echo"),
    ("en_us-ko-14", "v1.0", "am_eric"),
    ("en_us-ko-15", "v1.0", "am_fenrir"),
    ("en_us-ko-16", "v1.0", "am_liam"),
    ("en_us-ko-17", "v1.0", "am_michael"),
    ("en_us-ko-18", "v1.0", "am_onyx"),
    ("en_us-ko-19", "v1.0", "am_puck"),
    ("en_us-ko-20", "v1.0", "am_santa"),
    ("en_us-ko-21", "v1.0", "af_ameliaearhart"),
    ("en_us-ko-22", "v1.0", "af_libritts5338"),
    ("en_us-ko-23", "v1.0", "am_libritts1272"),
    ("en_us-ko-24", "v1.0", "am_libritts6241"),
    ("en_us-ko-25", "v1.0", "am_vincentprice"),
    ("en_us-ko-26", "v1.1-zh", "af_maple"),
    ("en_us-ko-27", "v1.1-zh", "af_sol"),
]


def test_canonical_selector_grammar():
    assert format_voice_selector("de", "ko", 1) == "de-ko-1"
    assert format_voice_selector("en_us", "pi", 12) == "en_us-pi-12"
    assert parse_voice_selector("en_us-ko-3").slot == 3
    assert parse_voice_selector("en-us-ko-3").selector == "en_us-ko-3"
    assert is_voice_selector("de-ko-1")
    for value in ("de-ko-0", "de-ko-01", "-ko-1", "de-ko", "de-xx-1-extra"):
        assert not is_voice_selector(value)
        with pytest.raises(VoiceSelectorError):
            parse_voice_selector(value)


def test_packaged_registry_contains_all_en_us_kokoro_assignments():
    registry = load_voice_selector_registry()
    identities = registry.iter_identities(language="en_us", system="kokoro")
    assert len(identities) == 27
    assert [
        (identity.selector, identity.asset_id, identity.voice_id) for identity in identities
    ] == EXPECTED_EN_US_KOKORO
    for selector, asset_id, voice_id in EXPECTED_EN_US_KOKORO:
        identity = resolve_voice_selector(selector, registry=registry)
        assert identity.system == "kokoro"
        assert identity.asset_id == asset_id
        assert identity.voice_id == voice_id
        assert (
            registry.selector_for_voice(system="kokoro", asset_id=asset_id, voice_id=voice_id)
            == identity
        )
    assert resolve_voice_selector("en-us-ko-4", registry=registry).selector == "en_us-ko-4"
    assert resolve_voice_selector("en_us-ko-4", registry=registry).asset_id == "v1.0"
    assert resolve_voice_selector("en_us-ko-4", registry=registry).voice_id == "af_heart"
    assert resolve_voice_selector("en_us-ko-26", registry=registry).canonical_key == (
        "kokoro",
        "v1.1-zh",
        "af_maple",
    )
    assert resolve_voice_selector("en_us-ko-27", registry=registry).canonical_key == (
        "kokoro",
        "v1.1-zh",
        "af_sol",
    )


def test_engine_codes_and_packaged_baseline():
    assert resolve_voice_selector("de-ko-1").system == "kokoro"
    assert resolve_voice_selector("de-ko-1").canonical_key == ("kokoro", "de-anna", "df_anna")
    assert resolve_voice_selector("de-pi-1").canonical_key == (
        "piper",
        "de_DE-eva_k-x_low",
        "de_DE-eva_k-x_low",
    )
    assert {identity.engine_code for identity in iter_voice_identities()} == {"ko", "pi"}
    assert load_voice_selector_registry().identities == iter_voice_identities()


def test_packaged_german_selectors_remain_unchanged():
    expected = [
        ("de-ko-1", "de-anna", "df_anna"),
        ("de-ko-2", "de-crane", "default"),
        ("de-ko-3", "de-thorsten", "thorsten"),
        ("de-ko-4", "v1.2-de-martin", "martin"),
    ]
    assert [
        (identity.selector, identity.asset_id, identity.voice_id)
        for identity in iter_voice_identities(language="de", system="kokoro")
    ] == expected


def test_registry_rejects_duplicate_slots_identity_and_mismatch():
    with pytest.raises(VoiceSelectorRegistryError):
        _registry(
            _entry("de", "pi", 1, "piper", "voice-a", "voice-a"),
            _entry("de", "pi", 1, "piper", "voice-b", "voice-b"),
        )
    with pytest.raises(VoiceSelectorRegistryError):
        _registry(
            _entry("de", "pi", 1, "piper", "voice-a", "voice-a"),
            _entry("de", "pi", 2, "piper", "voice-a", "voice-a"),
        )
    with pytest.raises(VoiceSelectorRegistryError):
        _registry(_entry("de", "ko", 1, "piper", "voice-a", "voice-a"))


def test_insertion_reorder_and_filter_do_not_renumber():
    registry = _registry(
        _entry("de", "pi", 1, "piper", "karlsson", "karlsson"),
        _entry("de", "pi", 2, "piper", "kerstin", "kerstin"),
    )
    assert (
        registry.selector_for_voice(
            system="piper", asset_id="karlsson", voice_id="karlsson"
        ).selector
        == "de-pi-1"
    )
    assert (
        registry.selector_for_voice(system="piper", asset_id="kerstin", voice_id="kerstin").selector
        == "de-pi-2"
    )
    assert registry.selector_for_voice(system="piper", asset_id="eva", voice_id="eva") is None
    assert tuple(i.selector for i in registry.iter_identities(language="de")) == (
        "de-pi-1",
        "de-pi-2",
    )
    assert tuple(i.selector for i in registry.iter_identities(system="piper")) == (
        "de-pi-1",
        "de-pi-2",
    )


def test_tombstones_are_not_reused():
    registry = _registry(
        _entry("de", "pi", 1, "piper", "old", "old", "retired"),
        _entry("de", "pi", 2, "piper", "new", "new"),
    )
    with pytest.raises(VoiceSelectorRetiredError):
        registry.resolve("de-pi-1")
    assert registry.resolve("de-pi-1", include_retired=True).state == "retired"
    assert registry.selector_for_voice(system="piper", asset_id="old", voice_id="old") is None
    assert (
        registry.selector_for_voice(
            system="piper", asset_id="old", voice_id="old", include_retired=True
        ).selector
        == "de-pi-1"
    )


def test_kokoro_identity_includes_backing_model():
    registry = _registry(
        _entry("de", "ko", 1, "kokoro", "model-a", "default"),
        _entry("de", "ko", 2, "kokoro", "model-b", "default"),
    )
    a = registry.selector_for_voice(system="kokoro", asset_id="model-a", voice_id="default")
    b = registry.selector_for_voice(system="kokoro", asset_id="model-b", voice_id="default")
    assert a.canonical_key != b.canonical_key
    assert a.selector == "de-ko-1"
    assert b.selector == "de-ko-2"


def test_asset_reference_boundary_is_unchanged():
    with pytest.raises(ValueError):
        parse_ref("de-ko-1")
    assert resolve_voice_selector("de-ko-1").backing_ref == "kokoro:de-anna"


def test_catalog_projection_reports_assigned_and_unassigned_voices():
    piper = _piper_voice("de_DE-eva_k-x_low")
    kokoro = _kokoro_model("de-anna", ("df_anna", "new_voice"))
    missing = _piper_voice("not-in-registry")
    items = [piper, kokoro, missing]
    unassigned = unassigned_catalog_voices(items)
    assert {(item.system, asset, voice) for item, asset, voice in unassigned} == {
        ("kokoro", "de-anna", "new_voice"),
        ("piper", "not-in-registry", "not-in-registry"),
    }
    records = object.__new__(OnnxVoice)
    records.catalog = type("Catalog", (), {"list": lambda self, system, **_: items})()
    projected = records.list_voices(system="kokoro", language="de", include_unassigned=True)
    assert any(record.selector == "de-ko-1" and record.available for record in projected)
    assert any(record.selector is None and record.voice_id == "new_voice" for record in projected)
    assert all(record.gender == "unknown" for record in projected)


def test_pocket_voice_states_are_projected_without_implicit_selector() -> None:
    pocket = CatalogItem(
        system="pocket",
        id="english_2026-04",
        kind="bundle",
        artifacts=(),
        voices=("alba",),
    )
    assert catalog_voice_keys([pocket]) == ((pocket, "english_2026-04", "alba"),)

def test_current_en_us_kokoro_catalog_has_no_unassigned_voices():
    catalog = [
        _kokoro_model(
            "v1.0",
            tuple(
                voice_id for _, asset_id, voice_id in EXPECTED_EN_US_KOKORO if asset_id == "v1.0"
            ),
        ),
        _kokoro_model(
            "v1.1-zh",
            tuple(
                voice_id for _, asset_id, voice_id in EXPECTED_EN_US_KOKORO if asset_id == "v1.1-zh"
            ),
        ),
    ]
    assert unassigned_catalog_voices(catalog) == ()


def test_catalog_drift_check_reports_missing_and_unassigned():
    issues = check_registry_and_catalog([_piper_voice("de_DE-eva_k-x_low")])
    assert any("missing from catalog" in issue for issue in issues)
    assert not any("unassigned catalog voice" in issue for issue in issues)
    assert len(missing_registry_voices([_piper_voice("de_DE-eva_k-x_low")])) == 40


def test_cli_parser_and_json_projection():
    args = build_parser().parse_args(["voices", "list", "--system", "piper", "--format", "json"])
    assert args.command == "voices"
    assert args.voices_command == "list"
    assert args.system == "piper"
    assert args.format == "json"
    assert build_parser().parse_args(["voices", "show", "de-ko-1"]).selector == "de-ko-1"


def test_voice_records_render_identity_fields(capsys):
    identity = resolve_voice_selector("de-ko-1")
    _render_voice_records(
        [VoiceRecord(identity, True, _kokoro_model("de-anna", ("df_anna",)), ("de",), "unknown")],
        "json",
    )
    output = capsys.readouterr().out
    assert '"selector": "de-ko-1"' in output
    assert '"backing_ref": "kokoro:de-anna"' in output


def test_selector_module_has_no_onnx_runtime_dependency():
    assert (
        "onnxruntime"
        not in inspect.getsource(__import__("onnxvoice.voice_selectors", fromlist=["*"])).lower()
    )
    assert (
        "onnxruntime"
        not in inspect.getsource(
            __import__("onnxvoice.catalog_tools.voice_selectors", fromlist=["*"])
        ).lower()
    )


def test_unknown_selector_has_specific_error():
    with pytest.raises(VoiceSelectorNotFoundError):
        resolve_voice_selector("de-ko-999")
