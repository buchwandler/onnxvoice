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


def test_catalog_drift_check_reports_missing_and_unassigned():
    issues = check_registry_and_catalog([_piper_voice("de_DE-eva_k-x_low")])
    assert any("missing from catalog" in issue for issue in issues)
    assert not any("unassigned catalog voice" in issue for issue in issues)
    assert len(missing_registry_voices([_piper_voice("de_DE-eva_k-x_low")])) == 13


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
