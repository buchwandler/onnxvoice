from __future__ import annotations

import json
from pathlib import Path

from onnxvoice.manager import OnnxVoice

_FIXTURE = Path(__file__).parent / "fixtures" / "inflect_models.json"


def _catalog_file(tmp_path: Path, *, gender: str = "male") -> Path:
    catalog = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    catalog["models"]["nano-v2"]["voices"]["default"]["gender"] = gender
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def test_inflect_voice_inventory_lists_default_voice_per_model(tmp_path: Path) -> None:
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"inflect": str(_catalog_file(tmp_path))},
    )

    assert [item.id for item in manager.list("inflect")] == ["nano-v2", "micro-v2"]
    records = manager.list_voices("inflect")
    assert [record.ref for record in records] == [
        "inflect:micro-v2/default",
        "inflect:nano-v2/default",
    ]


def test_inflect_voice_language_filter(tmp_path: Path) -> None:
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"inflect": str(_catalog_file(tmp_path))},
    )

    records = manager.list_voices("inflect", language="en-US")
    assert len(records) == 2
    assert all(record.locale == "en-US" for record in records)
    assert manager.list_voices("inflect", language="fr") == []
    assert manager.list("inflect", language="en")[0].id == "nano-v2"


def test_inflect_child_ref_resolves_default_voice(tmp_path: Path) -> None:
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"inflect": str(_catalog_file(tmp_path))},
    )

    resolved = manager.resolve_voice("inflect:nano/default")
    assert resolved.ref == "inflect:nano-v2/default"
    assert resolved.asset_id == "nano-v2"
    assert resolved.voice_id == "default"


def test_inflect_voice_metadata_preserves_gender_and_language(tmp_path: Path) -> None:
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"inflect": str(_catalog_file(tmp_path, gender="female"))},
    )

    record = manager.resolve_voice("inflect:nano-v2/default")
    assert record.gender == "female"
    assert record.language == "en"
    assert record.locale == "en-US"
    assert record.languages == ("en-US", "en")
    details = record.catalog_item.metadata["voice_details"]["default"]
    assert details["synthetic"] is True
    assert details["gender"] == "female"
