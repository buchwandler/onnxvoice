from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_kitten
from onnxvoice.errors import AssetNotFoundError, CatalogError

_REPOSITORY = "KittenML/test-model"
_REVISION = "a" * 40
_MODEL_ID = "nano-0.8-int8"


def _url(filename: str) -> str:
    return f"https://huggingface.co/{_REPOSITORY}/resolve/{_REVISION}/{filename}"


def _model() -> dict[str, Any]:
    return {
        "id": _MODEL_ID,
        "aliases": ["nano-int8"],
        "name": "Kitten TTS Nano",
        "version": "0.8",
        "language": "en-US",
        "sample_rate": 24000,
        "quality": "int8",
        "runtime": {"profile": "ONNX2"},
        "upstream": {
            "provider": "huggingface",
            "repository": _REPOSITORY,
            "revision": _REVISION,
            "license": "apache-2.0",
        },
        "artifacts": [
            {
                "role": "model",
                "format": "onnx",
                "filename": "model.onnx",
                "url": _url("model.onnx"),
                "size": 123,
                "sha256": "b" * 64,
            },
            {
                "role": "voices",
                "format": "npz",
                "filename": "voices.npz",
                "url": _url("voices.npz"),
                "size": 45,
                "sha256": "c" * 64,
            },
        ],
        "voice_aliases": {"Bella": "expr-voice-2-f", "Jasper": "expr-voice-2-m"},
        "speed_priors": {"expr-voice-2-f": 0.8},
        "metadata": {"parameters_millions": 15},
    }


def _catalog() -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "kitten-onnx-model-catalog",
        "models": {_MODEL_ID: _model()},
    }


class TestParseKitten:
    def test_parses_strict_model_contract_and_provenance(self) -> None:
        item = _parse_kitten(_catalog())[0]

        assert (item.system, item.id, item.kind) == ("kitten", _MODEL_ID, "model")
        assert item.aliases == ("nano-int8",)
        assert item.sample_rate == 24000
        assert item.voices == ("Bella", "Jasper")
        assert item.default_voice is None
        assert [
            (artifact.role, artifact.format, artifact.quality) for artifact in item.artifacts
        ] == [
            ("model", "onnx", "int8"),
            ("voices", "npz", None),
        ]
        assert item.metadata["voice_aliases"] == {
            "Bella": "expr-voice-2-f",
            "Jasper": "expr-voice-2-m",
        }
        assert item.metadata["speed_priors"] == {"expr-voice-2-f": 0.8}
        assert item.metadata["language"] == "en-US"
        assert item.metadata["runtime"] == {"profile": "ONNX2"}
        assert item.metadata["upstream"]["repository"] == _REPOSITORY
        assert item.metadata["source_repository"] == _REPOSITORY
        assert item.metadata["source_revision"] == _REVISION
        assert item.artifact("model").metadata["source"]["path"] == "model.onnx"
        assert item.metadata["license"] == "apache-2.0"
        assert item.metadata["parameters_millions"] == 15

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("schema", 2, "schema must be 1"),
            ("schema", True, "schema must be 1"),
            ("kind", "other", "Unexpected Kitten catalog kind"),
            ("source", {}, "exactly schema, kind, and models"),
            ("models", {}, "non-empty mapping"),
            ("models", [], "non-empty mapping"),
        ],
    )
    def test_rejects_invalid_catalog_envelope(
        self, field: str, value: object, message: str
    ) -> None:
        catalog = _catalog()
        catalog[field] = value
        with pytest.raises(CatalogError, match=message):
            _parse_kitten(catalog)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda model: model.update(id="another-id"), "does not match id"),
            (lambda model: model.update(aliases=["../escape"]), "alias is unsafe"),
            (lambda model: model.update(aliases=[_MODEL_ID]), "conflicting alias"),
            (lambda model: model.update(aliases=["nano-int8", "nano-int8"]), "conflicting alias"),
            (
                lambda model: model.update(sample_rate=True),
                "sample_rate must be a positive integer",
            ),
            (lambda model: model.update(sample_rate=0), "sample_rate must be a positive integer"),
            (lambda model: model.update(quality="int4"), "quality must be"),
            (lambda model: model.update(quality=[]), "quality must be"),
            (lambda model: model.update(runtime={"profile": "ONNX3"}), "profile must be"),
            (lambda model: model.update(runtime={"profile": []}), "profile must be"),
            (
                lambda model: model.update(runtime={"profile": "ONNX2", "extra": True}),
                "fields mismatch",
            ),
            (lambda model: model.update(language=" "), "language must be a non-empty string"),
            (lambda model: model.update(speed_priors={"unknown-style": 0.8}), "unknown voice"),
            (
                lambda model: model.update(speed_priors={"expr-voice-2-f": 0}),
                "positive finite number",
            ),
            (
                lambda model: model.update(speed_priors={"expr-voice-2-f": float("inf")}),
                "positive finite number",
            ),
            (
                lambda model: model.update(speed_priors={"expr-voice-2-f": 1 << 5000}),
                "positive finite number",
            ),
            (
                lambda model: model.update(speed_priors={"expr-voice-2-f": True}),
                "positive finite number",
            ),
            (
                lambda model: model.update(voice_aliases={"../Bella": "expr-voice-2-f"}),
                "public voice alias is unsafe",
            ),
            (
                lambda model: model.update(voice_aliases={"Bella": "../style"}),
                "internal voice ID is unsafe",
            ),
            (lambda model: model.update(extra=True), "unexpected: extra"),
        ],
    )
    def test_rejects_invalid_model_fields(self, mutate, message: str) -> None:
        catalog = _catalog()
        mutate(catalog["models"][_MODEL_ID])
        with pytest.raises(CatalogError, match=message):
            _parse_kitten(catalog)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda artifacts: artifacts.pop(), "exactly model and voices"),
            (lambda artifacts: artifacts.append(copy.deepcopy(artifacts[0])), "duplicate 'model'"),
            (
                lambda artifacts: artifacts.append(
                    {**copy.deepcopy(artifacts[1]), "role": "extra"}
                ),
                "unknown Kitten artifact role",
            ),
            (lambda artifacts: artifacts[0].update(format="npz"), "format must be 'onnx'"),
            (lambda artifacts: artifacts[0].update(filename="../model.onnx"), "filename is unsafe"),
            (lambda artifacts: artifacts[0].update(size=False), "size must be a positive integer"),
            (lambda artifacts: artifacts[0].update(size=0), "size must be a positive integer"),
            (lambda artifacts: artifacts[0].update(sha256="A" * 64), "lowercase 64-character hex"),
            (
                lambda artifacts: artifacts[0].update(
                    url=artifacts[0]["url"].replace(_REVISION, "main")
                ),
                "not pinned",
            ),
            (
                lambda artifacts: artifacts[0].update(
                    url=artifacts[0]["url"].replace(_REPOSITORY, "wrong/repo")
                ),
                "not pinned",
            ),
            (lambda artifacts: artifacts[0].update(extra=True), "unexpected: extra"),
        ],
    )
    def test_rejects_invalid_artifact_contract(self, mutate, message: str) -> None:
        catalog = _catalog()
        mutate(catalog["models"][_MODEL_ID]["artifacts"])
        with pytest.raises(CatalogError, match=message):
            _parse_kitten(catalog)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("provider", "local", "provider must be huggingface"),
            ("repository", "owner/name/extra", "owner/name form"),
            ("revision", "A" * 40, "lowercase 40-character SHA"),
            ("license", "", "license must be a non-empty string"),
        ],
    )
    def test_rejects_invalid_upstream_provenance(
        self, field: str, value: str, message: str
    ) -> None:
        catalog = _catalog()
        catalog["models"][_MODEL_ID]["upstream"][field] = value
        with pytest.raises(CatalogError, match=message):
            _parse_kitten(catalog)

    def test_rejects_unsafe_canonical_model_id(self) -> None:
        catalog = _catalog()
        catalog["models"]["../escape"] = catalog["models"].pop(_MODEL_ID)

        with pytest.raises(CatalogError, match="model ID is unsafe"):
            _parse_kitten(catalog)

    def test_rejects_alias_collisions_across_models(self) -> None:
        catalog = _catalog()
        second = copy.deepcopy(catalog["models"][_MODEL_ID])
        second["id"] = "micro-0.8"
        second["aliases"] = ["nano-int8"]
        catalog["models"]["micro-0.8"] = second

        with pytest.raises(CatalogError, match="conflicting alias"):
            _parse_kitten(catalog)


def test_catalog_client_uses_default_source_environment_override_and_alias_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "kitten.json"
    path.write_text(json.dumps(_catalog()), encoding="utf-8")
    monkeypatch.setenv("ONNXVOICE_KITTEN_CATALOG", str(path))
    client = CatalogClient(cache_dir=tmp_path / "cache")

    assert DEFAULT_SOURCES["kitten"].endswith("kitten-onnx-bundles/main/catalog/models.json")
    assert client.sources is not None
    assert client.sources["kitten"] == str(path)
    item = client.resolve("kitten:nano-int8")
    assert item.id == _MODEL_ID
    assert item.artifact("model").quality == "int8"
    assert item.artifact("voices").quality is None

    with pytest.raises(AssetNotFoundError):
        client.resolve("kitten:missing")


def test_sibling_kitten_bundle_catalog_matches_runtime_parser() -> None:
    catalog_path = Path(__file__).resolve().parents[2] / "kitten-onnx-bundles/catalog/models.json"
    if not catalog_path.is_file():
        pytest.skip("sibling kitten-onnx-bundles checkout is unavailable")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = _parse_kitten(catalog)
    assert {item.id for item in items} == {
        "nano-0.8-int8",
        "nano-0.8-fp32",
        "micro-0.8",
        "mini-0.8",
    }
    assert [item.metadata["quality"] for item in items] == ["int8", "fp32", None, None]
    assert items[0].voices[0] == "Bella"
    assert items[0].artifact("model").quality == "int8"
    assert items[0].artifact("voices").quality is None
    assert (
        items[0].metadata["source_revision"]
        == catalog["models"][items[0].id]["upstream"]["revision"]
    )
