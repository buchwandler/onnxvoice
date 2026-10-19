from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from _kitten2_fixture import CURRENT_VOICES, create_bundle

import onnxvoice.cli as cli_module
from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_kitten2
from onnxvoice.catalog_tools import kitten2 as kitten2_catalog_tools
from onnxvoice.errors import CatalogError

_REVISION = "a" * 40
_REPOSITORY = "example/kitten2-bundle"


def _catalog(root: Path, *, revision: str = _REVISION) -> dict[str, Any]:
    bundle_path = create_bundle(root)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    artifacts = [
        {
            "role": "bundle",
            "path": "bundle.json",
            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        }
    ]
    artifacts.extend(
        {"role": "asset", "path": path, "sha256": digest}
        for path, digest in sorted(bundle["files"].items())
    )
    return {
        "schema": 1,
        "kind": "kitten2-onnx-bundle-catalog",
        "bundles": {
            "kitten-tts-v2": {
                "id": "kitten-tts-v2",
                "aliases": ["v2", "default"],
                "name": "KittenTTS 2 ONNX",
                "layout": "kitten2-onnx-bundle-v1",
                "sample_rate": 24000,
                "default_voice": "Bruno",
                "voices": ["Bruno"],
                "source": {
                    "provider": "huggingface",
                    "repository": _REPOSITORY,
                    "revision": revision,
                    "license": "Apache-2.0",
                },
                "runtime": {"bundle_format": "kitten2-onnx-bundle-v1"},
                "metadata": {"default_preset": "stable"},
                "artifacts": artifacts,
            }
        },
    }


def test_native_parser_preserves_pinned_nested_artifacts_and_provenance(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    items = _parse_kitten2(catalog)
    assert len(items) == 1
    item = items[0]
    assert (item.system, item.id, item.kind) == ("kitten2", "kitten-tts-v2", "bundle")
    assert item.aliases == ("v2", "default")
    assert item.voices == ("Bruno",)
    assert item.default_voice == "Bruno"
    assert item.sample_rate == 24000
    assert item.metadata["runtime"] == {
        "layout": "kitten2-onnx-bundle-v1",
        "sample_rate": 24000,
    }
    artifact = next(
        artifact
        for artifact in item.artifacts
        if artifact.filename == "decoder/mel2wav/mel2wav.onnx.data"
    )
    assert artifact.role == "asset"
    expected_sha = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))["files"][
        "decoder/mel2wav/mel2wav.onnx.data"
    ]
    assert artifact.sha256 == expected_sha
    assert artifact.url == (
        f"https://huggingface.co/{_REPOSITORY}/resolve/{_REVISION}/"
        "decoder/mel2wav/mel2wav.onnx.data?download=true"
    )


def test_catalog_client_registers_native_source_and_rejects_mutable_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog(tmp_path / "bundle", revision="main")), encoding="utf-8")
    assert DEFAULT_SOURCES["kitten2"].endswith("kitten2-onnx-bundles/main/catalog/bundles.json")
    client = CatalogClient(
        cache_dir=tmp_path / "cache",
        sources={"kitten2": str(path)},
        offline=True,
    )
    with pytest.raises(CatalogError, match="commit SHA"):
        client.list("kitten2")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda catalog: catalog.update(schema=2), "schema must be 1"),
        (
            lambda catalog: catalog["bundles"]["kitten-tts-v2"]["sample_rate"].__class__,
            "sample_rate must be 24000",
        ),
        (
            lambda catalog: catalog["bundles"]["kitten-tts-v2"]["source"].update(revision="main"),
            "commit SHA",
        ),
        (
            lambda catalog: catalog["bundles"]["kitten-tts-v2"]["artifacts"][1].update(
                path="../outside"
            ),
            "Unsafe Kitten2 artifact path",
        ),
        (
            lambda catalog: catalog["bundles"]["kitten-tts-v2"]["artifacts"][1].update(
                path=".cache/huggingface/model.bin"
            ),
            "outside the Kitten2 runtime namespaces",
        ),
        (
            lambda catalog: catalog["bundles"]["kitten-tts-v2"]["artifacts"][1].update(
                sha256="bad"
            ),
            "sha256 must be lowercase",
        ),
    ],
)
def test_catalog_rejects_malformed_records(tmp_path: Path, change: Any, message: str) -> None:
    catalog = _catalog(tmp_path)
    if message == "sample_rate must be 24000":
        catalog["bundles"]["kitten-tts-v2"]["sample_rate"] = 16000
    else:
        change(catalog)
    with pytest.raises(CatalogError, match=message):
        _parse_kitten2(catalog)


def test_revision_resolution_and_catalog_build_pin_manifest_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = create_bundle(tmp_path / "upstream", voice_names=CURRENT_VOICES)
    bundle_bytes = bundle_path.read_bytes()
    requests: list[str] = []

    class Response:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    def fake_urlopen(request: Any, timeout: int) -> Response:
        requests.append(request.full_url)
        if "/api/models/" in request.full_url:
            return Response(json.dumps({"sha": _REVISION}).encode("utf-8"))
        assert request.full_url.endswith(f"/resolve/{_REVISION}/bundle.json?download=true")
        return Response(bundle_bytes)

    monkeypatch.setattr(kitten2_catalog_tools.urllib.request, "urlopen", fake_urlopen)
    catalog = kitten2_catalog_tools.build_catalog(
        repository=_REPOSITORY, revision="main", license="Apache-2.0"
    )
    source = kitten2_catalog_tools.build_source(catalog)
    kitten2_catalog_tools.verify_catalog(catalog, source)
    record = catalog["bundles"]["kitten-tts-v2"]
    assert record["source"]["revision"] == _REVISION
    assert len(requests) == 2
    assert len(record["artifacts"]) == len(json.loads(bundle_bytes)["files"]) + 1
    upstream_manifest = json.loads(bundle_bytes)
    assert record["voices"] == list(CURRENT_VOICES)
    assert {artifact["path"] for artifact in record["artifacts"]} == {
        "bundle.json",
        *upstream_manifest["files"],
    }
    assert kitten2_catalog_tools.resolve_revision(_REPOSITORY, _REVISION) == _REVISION


def test_builder_rejects_polluted_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = create_bundle(tmp_path / "upstream")
    manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest["files"][".cache/huggingface/model.bin"] = "0" * 64
    bundle_bytes = json.dumps(manifest).encode("utf-8")

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return bundle_bytes

    monkeypatch.setattr(
        kitten2_catalog_tools.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    with pytest.raises(CatalogError, match="outside the runtime namespaces"):
        kitten2_catalog_tools.build_catalog(
            repository=_REPOSITORY, revision=_REVISION, license="Apache-2.0"
        )


def test_builder_rejects_missing_voice_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = create_bundle(tmp_path / "upstream")
    manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest["files"].pop("voices/Bruno.meta.json")
    bundle_bytes = json.dumps(manifest).encode("utf-8")

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return bundle_bytes

    monkeypatch.setattr(
        kitten2_catalog_tools.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    with pytest.raises(CatalogError, match="voice metadata is not declared"):
        kitten2_catalog_tools.build_catalog(
            repository=_REPOSITORY, revision=_REVISION, license="Apache-2.0"
        )


def test_catalog_verify_is_offline_and_rejects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _catalog(tmp_path / "bundle")
    source = kitten2_catalog_tools.build_source(catalog)
    catalog_path = tmp_path / "catalog.json"
    source_path = tmp_path / "source.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")

    def fail_network(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("offline catalog verification attempted network access")

    monkeypatch.setattr(kitten2_catalog_tools.urllib.request, "urlopen", fail_network)
    assert kitten2_catalog_tools.load_catalog(catalog_path, source_path) == catalog
    source["repositories"][0]["revision"] = "b" * 40
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(CatalogError, match="does not match"):
        kitten2_catalog_tools.load_catalog(catalog_path, source_path)


def test_cli_build_and_verify_kitten2_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _catalog(tmp_path / "bundle")
    monkeypatch.setattr(kitten2_catalog_tools, "build_catalog", lambda **_kwargs: catalog)
    catalog_path = tmp_path / "catalog.json"
    source_path = tmp_path / "source.json"
    assert (
        cli_module.main(
            [
                "catalog",
                "kitten2",
                "build",
                "--repository",
                _REPOSITORY,
                "--revision",
                "main",
                "--output",
                str(catalog_path),
                "--source-output",
                str(source_path),
            ]
        )
        == 0
    )
    assert (
        cli_module.main(
            [
                "catalog",
                "kitten2",
                "verify",
                "--catalog",
                str(catalog_path),
                "--source",
                str(source_path),
            ]
        )
        == 0
    )
    assert json.loads(catalog_path.read_text(encoding="utf-8")) == catalog
