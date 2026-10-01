from __future__ import annotations

import json

from onnxvoice.cli import build_parser, main
from onnxvoice.types import CatalogItem, VoiceMetadata, VoiceRecord


class _Manager:
    def __init__(self, record: VoiceRecord) -> None:
        self.record = record
        self.list_args = None
        self.resolve_args = None

    def list_voices(self, **kwargs):
        self.list_args = kwargs
        return [self.record]

    def resolve_voice(self, ref: str, **kwargs):
        self.resolve_args = (ref, kwargs)
        return self.record


def _record() -> VoiceRecord:
    item = CatalogItem(
        system="kokoro",
        id="v1.0",
        kind="model",
        artifacts=(),
        voices=("af_heart",),
    )
    return VoiceRecord(
        catalog_item=item,
        voice_id="af_heart",
        metadata=VoiceMetadata("en", "en-US", "American English", "female"),
        languages=("en-US",),
    )


_SUPERTONIC_LANGUAGES = (
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
    "na",
)


def _supertonic_record() -> VoiceRecord:
    item = CatalogItem(
        system="supertonic",
        id="supertonic-3",
        kind="bundle",
        artifacts=(),
        voices=("F1",),
        metadata={"language_codes": _SUPERTONIC_LANGUAGES},
    )
    return VoiceRecord(
        catalog_item=item,
        voice_id="F1",
        metadata=VoiceMetadata("en", "en", "en", "unknown"),
        languages=_SUPERTONIC_LANGUAGES,
    )


def test_voices_list_parser_accepts_supertonic() -> None:
    args = build_parser().parse_args(["voices", "list", "--system", "supertonic"])

    assert args.system == "supertonic"


def test_voices_parser_exposes_only_catalog_filters_and_semantic_ref() -> None:
    parser = build_parser()
    commands = next(action for action in parser._actions if action.dest == "command").choices
    voices = commands["voices"]
    subcommands = next(
        action for action in voices._actions if action.dest == "voices_command"
    ).choices
    list_parser = subcommands["list"]
    show_parser = subcommands["show"]

    list_options = {option for action in list_parser._actions for option in action.option_strings}
    show_options = {option for action in show_parser._actions for option in action.option_strings}
    assert list_options - {"-h", "--help"} == {
        "--system",
        "--lang",
        "--language",
        "--refresh",
        "--format",
    }
    assert show_options - {"-h", "--help"} == {"--refresh", "--format"}
    assert next(action for action in show_parser._actions if action.dest == "ref").dest == "ref"


def test_voices_list_renders_semantic_refs_as_json(monkeypatch, capsys) -> None:
    manager = _Manager(_record())
    monkeypatch.setattr("onnxvoice.cli._manager", lambda _args: manager)

    assert (
        main(["voices", "list", "--system", "kokoro", "--lang", "en-US", "--format", "json"]) == 0
    )

    assert manager.list_args == {"system": "kokoro", "language": "en-US", "refresh": False}
    payload = json.loads(capsys.readouterr().out)["items"][0]
    assert payload == {
        "ref": "kokoro:v1.0/af_heart",
        "system": "kokoro",
        "asset_id": "v1.0",
        "voice_id": "af_heart",
        "backing_ref": "kokoro:v1.0",
        "languages": ["en-US"],
        "language": "en",
        "locale": "en-US",
        "language_label": "American English",
        "gender": "female",
    }


def test_supertonic_list_and_show_json_keep_canonical_identity(monkeypatch, capsys) -> None:
    manager = _Manager(_supertonic_record())
    monkeypatch.setattr("onnxvoice.cli._manager", lambda _args: manager)

    assert (
        main(["voices", "list", "--system", "supertonic", "--lang", "de", "--format", "json"]) == 0
    )
    assert manager.list_args == {"system": "supertonic", "language": "de", "refresh": False}
    listed = json.loads(capsys.readouterr().out)["items"][0]
    expected = {
        "ref": "supertonic:supertonic-3/F1",
        "system": "supertonic",
        "asset_id": "supertonic-3",
        "voice_id": "F1",
        "backing_ref": "supertonic:supertonic-3",
        "languages": list(_SUPERTONIC_LANGUAGES),
        "language": "en",
        "locale": "en",
        "language_label": "en",
        "gender": "unknown",
    }
    assert listed == expected

    assert main(["voices", "show", "supertonic:supertonic-3/F1", "--format", "json"]) == 0
    assert manager.resolve_args == ("supertonic:supertonic-3/F1", {"refresh": False})
    assert json.loads(capsys.readouterr().out) == expected


def test_voices_list_table_uses_semantic_columns(monkeypatch, capsys) -> None:
    manager = _Manager(_record())
    monkeypatch.setattr("onnxvoice.cli._manager", lambda _args: manager)

    assert main(["voices", "list"]) == 0

    output = capsys.readouterr().out
    header = output.splitlines()[0]
    assert all(
        column in header for column in ("REF", "SYSTEM", "ASSET", "VOICE", "LOCALE", "GENDER")
    )
    assert "kokoro:v1.0/af_heart" in output


def test_voices_show_resolves_semantic_ref(monkeypatch, capsys) -> None:
    manager = _Manager(_record())
    monkeypatch.setattr("onnxvoice.cli._manager", lambda _args: manager)

    assert main(["voices", "show", "kokoro:v1.0/af_heart", "--refresh", "--format", "json"]) == 0

    assert manager.resolve_args == ("kokoro:v1.0/af_heart", {"refresh": True})
    assert json.loads(capsys.readouterr().out)["ref"] == "kokoro:v1.0/af_heart"
