from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_supertonic
from onnxvoice.errors import AssetNotFoundError, CatalogError
from onnxvoice.manager import OnnxVoice
from onnxvoice.systems.supertonic import SupertonicAdapter

REPOSITORY = "supertone-oss-archive/supertonic-3"
REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
MODEL_COMPONENTS = (
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
)
STYLE_COMPONENTS = tuple([f"F{i}" for i in range(1, 6)] + [f"M{i}" for i in range(1, 6)])


def _url(path: str, revision: str = REVISION) -> str:
    return f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{path}?download=true"


def _artifact(role: str, component: str | None, path: str, index: int) -> dict:
    return {
        "role": role,
        "component": component,
        "filename": path.rsplit("/", 1)[-1],
        "path": path,
        "url": _url(path),
        "size": index + 1,
        "sha256": f"{index:064x}",
        "format": path.rsplit(".", 1)[-1],
    }


def _catalog() -> dict:
    artifacts = [
        _artifact("config", None, "onnx/tts.json", 1),
        _artifact("unicode_indexer", None, "onnx/unicode_indexer.json", 2),
    ]
    artifacts.extend(
        _artifact("model", component, f"onnx/{component}.onnx", index + 3)
        for index, component in enumerate(MODEL_COMPONENTS)
    )
    artifacts.extend(
        _artifact("voice_style", component, f"voice_styles/{component}.json", index + 7)
        for index, component in enumerate(STYLE_COMPONENTS)
    )
    return {
        "schema": 1,
        "kind": "supertonic-onnx-bundle-catalog",
        "source": {
            "provider": "huggingface",
            "repository": REPOSITORY,
            "requested_revision": "main",
            "revision": REVISION,
            "bundle_count": 1,
        },
        "bundles": {
            "supertonic-3": {
                "aliases": ["supertonic", "st3"],
                "artifacts": artifacts,
                "sample_rate": 44100,
                "languages": ["en", "ko", "na"],
                "default_language": "en",
                "voices": [
                    {"name": component, "artifact_component": component}
                    for component in STYLE_COMPONENTS
                ],
                "default_voice": "F1",
                "runtime": {
                    "layout": "supertonic-3-v1",
                    "latent_dim": 24,
                    "chunk_size": 512,
                },
                "metadata": {"upstream": "fixture"},
            }
        },
    }


class TestParseSupertonic:
    def test_parses_bundle_artifacts_aliases_languages_voices_and_metadata(self) -> None:
        item = _parse_supertonic(_catalog())[0]

        assert (item.system, item.id, item.kind) == ("supertonic", "supertonic-3", "bundle")
        assert item.aliases == ("supertonic", "st3")
        assert item.sample_rate == 44100
        assert item.voices == STYLE_COMPONENTS
        assert item.default_voice == "F1"
        assert item.metadata["language_codes"] == ("en", "ko", "na")
        assert item.metadata["runtime"]["layout"] == "supertonic-3-v1"
        assert item.metadata["source_revision"] == REVISION
        assert item.metadata["source_repository"] == REPOSITORY
        assert item.metadata["requested_revision"] == "main"
        assert item.metadata["upstream"] == "fixture"
        assert len(item.artifacts) == 16
        assert {a.component for a in item.artifacts if a.role == "model"} == set(MODEL_COMPONENTS)
        assert item.artifact("model", component="vocoder").metadata["path"] == "onnx/vocoder.onnx"
        assert item.artifact("config").metadata["source"] == {
            "provider": "huggingface",
            "repository": REPOSITORY,
            "revision": REVISION,
            "path": "onnx/tts.json",
            "gated": False,
        }

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("schema", 2, "schema must be 1"),
            ("kind", "other", "Unexpected Supertonic catalog kind"),
            ("bundles", [], "bundles must be a mapping"),
        ],
    )
    def test_rejects_invalid_envelope(self, field: str, value: object, message: str) -> None:
        catalog = _catalog()
        catalog[field] = value
        with pytest.raises(CatalogError, match=message):
            _parse_supertonic(catalog)

    def test_rejects_wrong_bundle_count(self) -> None:
        catalog = _catalog()
        catalog["source"]["bundle_count"] = 2
        with pytest.raises(CatalogError, match="bundle_count"):
            _parse_supertonic(catalog)

    def test_rejects_duplicate_role_component(self) -> None:
        catalog = _catalog()
        artifacts = catalog["bundles"]["supertonic-3"]["artifacts"]
        artifacts.append(copy.deepcopy(artifacts[2]))
        with pytest.raises(CatalogError, match=r"duplicate \(role, component\)"):
            _parse_supertonic(catalog)

    def test_rejects_missing_model_component(self) -> None:
        catalog = _catalog()
        entry = catalog["bundles"]["supertonic-3"]
        entry["artifacts"] = [
            artifact
            for artifact in entry["artifacts"]
            if not (artifact["role"] == "model" and artifact["component"] == "vocoder")
        ]
        with pytest.raises(CatalogError, match="missing model component"):
            _parse_supertonic(catalog)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda a: a.update(path="../escape.onnx"), "unsafe artifact path"),
            (lambda a: a.update(sha256="A" * 64), "sha256 must be lowercase hex"),
            (lambda a: a.update(size=0), "size must be a positive integer"),
            (lambda a: a.update(url=a["url"].replace(REVISION, "main")), "not pinned to source"),
            (lambda a: a.update(url=a["url"].replace(REVISION, "b" * 40)), "not pinned to source"),
        ],
    )
    def test_rejects_invalid_artifact_integrity_or_source(self, mutate, message: str) -> None:
        catalog = _catalog()
        mutate(catalog["bundles"]["supertonic-3"]["artifacts"][0])
        with pytest.raises(CatalogError, match=message):
            _parse_supertonic(catalog)

    @pytest.mark.parametrize(
        ("source_field", "value", "message"),
        [
            ("repository", "owner/name/extra", "owner/name"),
            ("revision", "A" * 40, "lowercase 40-character SHA"),
            ("provider", "local", "source provider must be huggingface"),
        ],
    )
    def test_rejects_invalid_source(self, source_field: str, value: str, message: str) -> None:
        catalog = _catalog()
        catalog["source"][source_field] = value
        with pytest.raises(CatalogError, match=message):
            _parse_supertonic(catalog)

    def test_rejects_unsafe_ids_components_and_alias_collisions(self) -> None:
        catalog = _catalog()
        entry = catalog["bundles"].pop("supertonic-3")
        catalog["bundles"]["../bundle"] = entry
        with pytest.raises(CatalogError, match="bundle ID is unsafe"):
            _parse_supertonic(catalog)

        catalog = _catalog()
        entry = catalog["bundles"]["supertonic-3"]
        entry["artifacts"][2]["component"] = "../vocoder"
        with pytest.raises(CatalogError, match="component is unsafe"):
            _parse_supertonic(catalog)

        catalog = _catalog()
        catalog["bundles"]["supertonic-3"]["aliases"] = ["supertonic-3"]
        with pytest.raises(CatalogError, match="conflicting alias"):
            _parse_supertonic(catalog)

    def test_rejects_style_references_without_artifact_and_bad_runtime_defaults(self) -> None:
        catalog = _catalog()
        entry = catalog["bundles"]["supertonic-3"]
        entry["voices"][0]["artifact_component"] = "missing-style"
        with pytest.raises(CatalogError, match="missing style artifact"):
            _parse_supertonic(catalog)

        catalog = _catalog()
        catalog["bundles"]["supertonic-3"]["default_voice"] = "missing"
        with pytest.raises(CatalogError, match="default_voice"):
            _parse_supertonic(catalog)

        catalog = _catalog()
        catalog["bundles"]["supertonic-3"]["default_language"] = "fr"
        with pytest.raises(CatalogError, match="default_language"):
            _parse_supertonic(catalog)

        catalog = _catalog()
        catalog["bundles"]["supertonic-3"]["runtime"]["layout"] = "other"
        with pytest.raises(CatalogError, match="runtime layout"):
            _parse_supertonic(catalog)

    def test_rejects_duplicate_styles_and_voices(self) -> None:
        catalog = _catalog()
        artifacts = catalog["bundles"]["supertonic-3"]["artifacts"]
        artifacts.append(copy.deepcopy(artifacts[-1]))
        with pytest.raises(CatalogError, match=r"duplicate \(role, component\)"):
            _parse_supertonic(catalog)

        catalog = _catalog()
        voices = catalog["bundles"]["supertonic-3"]["voices"]
        voices.append(copy.deepcopy(voices[0]))
        with pytest.raises(CatalogError, match="duplicate voice"):
            _parse_supertonic(catalog)


def test_catalog_client_registers_and_resolves_supertonic_without_quality_filtering(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "supertonic.json"
    path.write_text(json.dumps(_catalog()), encoding="utf-8")
    monkeypatch.setenv("ONNXVOICE_SUPERTONIC_CATALOG", str(path))
    client = CatalogClient(cache_dir=tmp_path / "cache")

    assert "supertonic" in client.systems()
    assert DEFAULT_SOURCES["supertonic"].endswith(
        "supertonic-onnx-bundles/main/catalog/bundles.json"
    )
    item = client.resolve("supertonic:st3")
    assert len([artifact for artifact in item.artifacts if artifact.role == "model"]) == 4
    assert len(item.artifacts) == 16
    with pytest.raises(AssetNotFoundError, match="Supertonic has no quality profiles"):
        client.resolve("supertonic:supertonic-3", quality="fp32")


def _installable_catalog() -> tuple[dict, dict[str, bytes], tuple[str, ...]]:
    catalog = copy.deepcopy(_catalog())
    languages = (
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
    bundle = catalog["bundles"]["supertonic-3"]
    bundle["languages"] = list(languages)
    payloads = {}
    config = {
        "ae": {"sample_rate": 44100, "base_chunk_size": 512},
        "ttl": {"chunk_compress_factor": 6, "latent_dim": 24},
    }
    for artifact in bundle["artifacts"]:
        if artifact["role"] == "config":
            payload = json.dumps(config).encode()
        elif artifact["role"] == "unicode_indexer":
            payload = b"{}"
        elif artifact["role"] == "model":
            payload = f"fake-onnx:{artifact['component']}".encode()
        else:
            payload = json.dumps({"voice": artifact["component"]}).encode()
        payloads[artifact["path"]] = payload
        artifact["size"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    return catalog, payloads, languages


def test_install_open_offline_reopen_and_inventory_use_standard_bundle_flow(
    tmp_path, monkeypatch
) -> None:
    import onnxvoice.store as store_module

    catalog, payloads, languages = _installable_catalog()
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    downloads = []

    def download(source, *, local_dir: Path, offline: bool) -> Path:
        assert not offline
        downloads.append(source.path)
        target = local_dir / Path(source.path).name
        target.write_bytes(payloads[source.path])
        return target

    monkeypatch.setattr(store_module, "download_huggingface_file", download)
    manager = OnnxVoice(cache_dir=cache_dir, catalog_sources={"supertonic": str(catalog_path)})
    installation = manager.install("supertonic:supertonic-3")
    manager.store.verify(installation)

    assert len(downloads) == 16
    assert set(downloads) == set(payloads)
    assert len(installation.artifacts) == 16
    for artifact in installation.artifacts:
        expected = payloads[artifact.metadata["path"]]
        assert artifact.path.read_bytes() == expected
        assert hashlib.sha256(expected).hexdigest() == artifact.sha256

    inventory = manager.inventory(system="supertonic")
    assert len(inventory) == 1
    record = inventory[0]
    assert record.language_codes == languages
    assert record.catalog_item is not None
    assert record.catalog_item.voices == STYLE_COMPONENTS
    assert {
        artifact.component
        for artifact in record.installation.artifacts
        if artifact.role == "voice_style"
    } == set(STYLE_COMPONENTS)
    voice_records = manager.list_voices(system="supertonic")
    assert {voice.ref for voice in voice_records} == {
        f"supertonic:supertonic-3/{voice_id}" for voice_id in STYLE_COMPONENTS
    }

    monkeypatch.setattr(
        store_module,
        "download_huggingface_file",
        lambda *_args, **_kwargs: pytest.fail("offline reopen attempted a download"),
    )
    offline_manager = OnnxVoice(
        cache_dir=cache_dir,
        catalog_sources={"supertonic": str(catalog_path)},
        offline=True,
    )
    reopened = offline_manager.open(installation)
    assert isinstance(reopened, SupertonicAdapter)
    assert reopened._settings() == (44100, 512, 6, 24)
