from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_inflect
from onnxvoice.errors import AssetNotFoundError, CatalogError

_FIXTURE = Path(__file__).parent / "fixtures" / "inflect_models.json"


def _catalog() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_default_sources_contains_inflect() -> None:
    assert DEFAULT_SOURCES["inflect"].endswith(
        "inflect-onnx-bundles/main/catalog/models.json"
    )


def test_parse_inflect_nano_and_micro() -> None:
    items = _parse_inflect(_catalog())

    assert [item.id for item in items] == ["nano-v2", "micro-v2"]
    for item in items:
        assert (item.system, item.kind, item.sample_rate) == ("inflect", "model", 24000)
        assert item.voices == ("default",)
        assert item.default_voice == "default"
        assert [artifact.role for artifact in item.artifacts] == ["duration", "decode"]
        assert [artifact.filename for artifact in item.artifacts] == [
            "duration.onnx",
            "decode.onnx",
        ]
        assert item.artifact("duration").metadata["path"] == "onnx/duration.onnx"
        assert item.artifact("decode").metadata["path"] == "onnx/decode.onnx"


def test_inflect_aliases_resolve_to_canonical_ids(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_catalog()), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"inflect": str(path)})

    assert client.resolve("inflect:nano").id == "nano-v2"
    assert client.resolve("inflect:Inflect-Nano-v2-ONNX").id == "nano-v2"
    assert client.resolve("inflect:micro").id == "micro-v2"
    assert client.resolve("inflect:Inflect-Micro-v2-ONNX").id == "micro-v2"
    with pytest.raises(AssetNotFoundError):
        client.resolve("inflect:missing")


def test_inflect_catalog_preserves_provenance() -> None:
    item = _parse_inflect(_catalog())[0]

    assert item.metadata["source_repository"] == "owensong/Inflect-Nano-v2"
    assert item.metadata["source_revision"] == "67fc9249e0bb8ab4b196d25cb09b6dd43f5e4f13"
    assert item.metadata["license"] == "Apache-2.0"
    assert item.metadata["upstream"]["repository"] == "owensong/Inflect-Nano-v2-ONNX"
    assert item.metadata["runtime"] == {
        "profile": "inflect-v2-split-v1",
        "precision": "fp32",
        "onnx_opset": 17,
        "layout": "split",
    }


def test_inflect_catalog_preserves_controls() -> None:
    item = _parse_inflect(_catalog())[0]

    assert item.metadata["controls"] == {
        "speed": {"default": 1.0, "minimum": 0.5, "maximum": 2.0},
        "variation": {"default": 0.667, "minimum": 0.0, "maximum": 1.0},
        "seed_default": 0,
    }
    assert item.metadata["source_model_sha256"] == (
        "bfca468489c9069361d6b87b295a17eef611af8ff09854ce0b80305d9122f5b3"
    )


def test_inflect_catalog_marks_split_runtime_layout() -> None:
    assert _parse_inflect(_catalog())[0].metadata["runtime"]["layout"] == "split"


def test_inflect_catalog_has_duration_and_decode_roles() -> None:
    item = _parse_inflect(_catalog())[0]
    assert {artifact.role for artifact in item.artifacts} == {"duration", "decode"}
    assert item.artifact("duration").metadata["source"]["path"] == "onnx/duration.onnx"
    assert item.artifact("decode").metadata["source"]["path"] == "onnx/decode.onnx"


def test_inflect_voice_details_are_catalog_derived() -> None:
    item = _parse_inflect(_catalog())[0]
    assert item.metadata["voice_details"]["default"] == {
        "id": "default",
        "name": "Default",
        "language": "en",
        "locale": "en-US",
        "gender": "male",
        "synthetic": True,
    }
    assert item.metadata["language_codes"] == ("en-US",)



def test_inflect_catalog_preserves_unknown_voice_gender() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["voices"]["default"]["gender"] = "unknown"
    item = _parse_inflect(catalog)[0]
    assert item.metadata["voice_details"]["default"]["gender"] == "unknown"


@pytest.mark.parametrize("field", ["unexpected", "source"])
def test_inflect_catalog_rejects_unknown_top_level_fields(field: str) -> None:
    catalog = _catalog()
    catalog[field] = {}
    with pytest.raises(CatalogError, match="exactly schema, kind, and models"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_unknown_model_fields() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["surprise"] = True
    with pytest.raises(CatalogError, match="fields mismatch.*unexpected: surprise"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_duplicate_alias() -> None:
    catalog = _catalog()
    catalog["models"]["micro-v2"]["aliases"] = ["NANO"]
    with pytest.raises(CatalogError, match="duplicate or conflicting alias"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_unpinned_huggingface_url() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["artifacts"][0]["url"] = (
        "https://huggingface.co/owensong/Inflect-Nano-v2-ONNX/resolve/main/onnx/duration.onnx"
    )
    with pytest.raises(CatalogError, match="not pinned to upstream repository/revision"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_bad_sha256() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["artifacts"][0]["sha256"] = "A" * 64
    with pytest.raises(CatalogError, match="lowercase 64-character hex"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_wrong_runtime_profile() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["runtime"]["profile"] = "inflect-v3"
    with pytest.raises(CatalogError, match="unsupported runtime profile"):
        _parse_inflect(catalog)


@pytest.mark.parametrize(
    ("control", "field", "value"),
    [
        ("speed", "default", True),
        ("speed", "minimum", 2.1),
        ("speed", "maximum", float("inf")),
        ("variation", "default", -0.1),
        ("variation", "minimum", 0.8),
        ("variation", "maximum", 1.1),
    ],
)
def test_inflect_catalog_rejects_invalid_control_bounds(
    control: str, field: str, value: Any
) -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["controls"][control][field] = value
    with pytest.raises(CatalogError, match="control"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_boolean_seed_default() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["controls"]["seed_default"] = True
    with pytest.raises(CatalogError, match="seed_default must be an integer"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_invalid_fixed_voice_contract() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["voices"]["default"]["gender"] = "robot"
    with pytest.raises(CatalogError, match="voice gender"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_non_24000_sample_rate() -> None:
    catalog = _catalog()
    catalog["models"]["nano-v2"]["sample_rate"] = 22050
    with pytest.raises(CatalogError, match="sample_rate must be 24000"):
        _parse_inflect(catalog)


def test_inflect_catalog_rejects_cross_model_casefold_id_collision() -> None:
    catalog = _catalog()
    catalog["models"]["micro-v2"]["aliases"] = ["NANO-V2"]
    with pytest.raises(CatalogError, match="duplicate or conflicting alias"):
        _parse_inflect(catalog)
