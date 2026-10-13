from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_sopro
from onnxvoice.errors import AssetNotFoundError, CatalogError, IntegrityError
from onnxvoice.manager import OnnxVoice

REPOSITORY = "samuel-vitorino/sopro-v2-turbo-onnx"
COMMIT = "a" * 40
REVISION = "7a645ef2aaecfea9"
PROFILES = {
    "wasm-uint8": {"provider": "wasm", "dtype": "float32"},
    "webgpu-fp16": {"provider": "webgpu", "dtype": "float16"},
}


def _url(path: str) -> str:
    return f"https://huggingface.co/{REPOSITORY}/resolve/{COMMIT}/{path}?download=true"


def _artifact(
    role: str,
    component: str | None,
    profile: str | None,
    path: str,
    payload: bytes | None,
) -> dict[str, Any]:
    return {
        "role": role,
        "component": component,
        "profile": profile,
        "path": path,
        "filename": Path(path).name,
        "url": _url(path),
        "sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
        "bytes": len(payload) if payload is not None else None,
        "provider": PROFILES[profile]["provider"] if role == "graph" and profile else None,
        "dtype": PROFILES[profile]["dtype"] if role == "graph" and profile else None,
    }


def _catalog() -> dict[str, Any]:
    artifacts = [
        _artifact("manifest", None, None, "manifest.json", None),
        _artifact("tokenizer", None, None, "tokenizer.json", None),
    ]
    profiles: dict[str, dict[str, Any]] = {}
    for profile, info in PROFILES.items():
        reference_path = "common/reference.onnx"
        core_path = f"{profile}/semantic_core.onnx"
        shard_path = f"{profile}/weights-semantic.bin"
        profiles[profile] = {
            **info,
            "graphs": {"reference": reference_path, "semantic_core": core_path},
            "shards": {"semantic": shard_path},
        }
        for component, path in profiles[profile]["graphs"].items():
            payload = f"graph:{profile}:{component}".encode()
            if component == "reference":
                payload = b"shared-reference-graph"
            artifacts.append(_artifact("graph", component, profile, path, payload))
        artifacts.append(
            _artifact("shard", "semantic", profile, shard_path, f"shard:{profile}".encode())
        )
    return {
        "schema": 1,
        "kind": "sopro-onnx-bundle-catalog",
        "status": "pinned",
        "bundles": {
            "v2-turbo": {
                "id": "v2-turbo",
                "ref": "sopro:v2-turbo",
                "aliases": ["turbo"],
                "model": REPOSITORY,
                "revision": REVISION,
                "source_commit": COMMIT,
                "sample_rate": 24000,
                "languages": ["en", "pt", "fr", "de"],
                "profiles": profiles,
                "artifacts": artifacts,
            }
        },
    }


def _payloads(catalog: dict[str, Any]) -> dict[str, bytes]:
    payloads = {}
    for artifact in catalog["bundles"]["v2-turbo"]["artifacts"]:
        path = artifact["path"]
        if path == "manifest.json":
            payload = b'{"format":1}'
        elif path == "tokenizer.json":
            payload = b'{"model":{}}'
        elif path == "common/reference.onnx":
            payload = b"shared-reference-graph"
        elif artifact["role"] == "graph":
            payload = f"graph:{artifact['profile']}:{artifact['component']}".encode()
        else:
            payload = f"shard:{artifact['profile']}".encode()
        payloads[path] = payload
        if artifact["role"] in {"graph", "shard"}:
            artifact["bytes"] = len(payload)
            artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    return payloads


def test_parse_pinned_catalog_and_preserve_profile_semantics() -> None:
    item = _parse_sopro(_catalog())[0]

    assert (item.system, item.id, item.ref, item.kind) == (
        "sopro",
        "v2-turbo",
        "sopro:v2-turbo",
        "bundle",
    )
    assert item.aliases == ("turbo",)
    assert item.sample_rate == 24000
    assert item.metadata["language_codes"] == ("en", "pt", "fr", "de")
    assert item.metadata["source_commit"] == COMMIT
    assert item.metadata["manifest_revision"] == REVISION
    assert set(item.metadata["profiles"]) == set(PROFILES)
    assert item.artifact("graph", component="reference", quality="wasm-uint8").filename == (
        "common/reference.onnx"
    )
    assert (
        item.artifact("shard", component="semantic", quality="wasm-uint8").metadata["path"]
        == "wasm-uint8/weights-semantic.bin"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", True, "schema must be 1"),
        ("schema", 2, "schema must be 1"),
        ("kind", "other", "Unexpected Sopro catalog kind"),
        ("status", "bootstrap", "status must be 'pinned'"),
        ("bundles", [], "non-empty mapping"),
    ],
)
def test_reject_invalid_catalog_envelope(field: str, value: Any, message: str) -> None:
    catalog = _catalog()
    catalog[field] = value
    with pytest.raises(CatalogError, match=message):
        _parse_sopro(catalog)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda entry: entry.update(ref="sopro:other"), "canonical ref"),
        (lambda entry: entry.update(source_commit="main"), "40-character SHA"),
        (lambda entry: entry.update(model="../model"), "owner/name repository identifier"),
        (lambda entry: entry.update(sample_rate=True), "positive integer"),
        (lambda entry: entry.update(languages=[]), "non-empty sequence"),
        (lambda entry: entry.update(aliases=["../escape"]), "unsafe"),
        (lambda entry: entry.update(aliases=["v2-turbo"]), "conflicting alias"),
        (
            lambda entry: entry["profiles"]["wasm-uint8"]["graphs"].update(reference="../bad.onnx"),
            "Unsafe",
        ),
        (
            lambda entry: entry["artifacts"][2].update(url="https://example.com/model.onnx"),
            "not pinned",
        ),
        (lambda entry: entry["artifacts"][2].update(sha256="A" * 64), "lowercase"),
        (lambda entry: entry["artifacts"][2].update(bytes=0), "positive integer"),
        (
            lambda entry: entry["artifacts"].append(copy.deepcopy(entry["artifacts"][2])),
            "duplicate artifact",
        ),
    ],
)
def test_reject_invalid_bundle_and_artifact_metadata(mutate, message: str) -> None:
    catalog = _catalog()
    mutate(catalog["bundles"]["v2-turbo"])
    with pytest.raises(CatalogError, match=message):
        _parse_sopro(catalog)


def test_reject_missing_tokenizer_and_conflicting_duplicate_path() -> None:
    catalog = _catalog()
    catalog["bundles"]["v2-turbo"]["artifacts"] = [
        artifact
        for artifact in catalog["bundles"]["v2-turbo"]["artifacts"]
        if artifact["role"] != "tokenizer"
    ]
    with pytest.raises(CatalogError, match="missing required static artifacts"):
        _parse_sopro(catalog)

    catalog = _catalog()
    catalog["bundles"]["v2-turbo"]["artifacts"][5]["sha256"] = "b" * 64
    with pytest.raises(CatalogError, match="conflicting duplicate artifact path"):
        _parse_sopro(catalog)


def test_catalog_client_resolves_alias_and_requires_explicit_profile(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"sopro": str(catalog_path)})

    assert DEFAULT_SOURCES["sopro"].endswith("sopro-onnx-bundles/main/catalog/bundles.json")
    assert "sopro" in client.systems()
    with pytest.raises(AssetNotFoundError, match="profile must be selected"):
        client.resolve("sopro:v2-turbo")
    with pytest.raises(AssetNotFoundError, match="Unknown Sopro profile"):
        client.resolve("sopro:v2-turbo", quality="unknown")

    item = client.resolve("sopro:turbo", quality="wasm-uint8")
    assert item.ref == "sopro:v2-turbo"
    assert item.metadata["selected_quality"] == "wasm-uint8"
    assert item.metadata["runtime"]["profile"] == "wasm-uint8"
    assert {artifact.quality for artifact in item.artifacts if artifact.role == "graph"} == {
        "wasm-uint8"
    }
    assert {artifact.role for artifact in item.artifacts} == {
        "manifest",
        "tokenizer",
        "graph",
        "shard",
    }


def test_install_preserves_nested_paths_and_offline_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnxvoice.store as store_module

    catalog = _catalog()
    payloads = _payloads(catalog)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    downloads: list[str] = []

    def download(_self, _source, destination, *, artifact: str, **_kwargs) -> None:
        downloads.append(artifact)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[artifact])

    monkeypatch.setattr(store_module.AssetStore, "_download", download)
    manager = OnnxVoice(cache_dir=cache_dir, catalog_sources={"sopro": str(catalog_path)})
    installation = manager.install("sopro:v2-turbo", quality="wasm-uint8")
    manager.store.verify(installation)

    assert len(installation.artifacts) == 5
    assert set(downloads) == {artifact.filename for artifact in installation.artifacts}
    assert {"common/reference.onnx", "wasm-uint8/semantic_core.onnx"} <= {
        artifact.filename for artifact in installation.artifacts
    }
    for artifact in installation.artifacts:
        assert artifact.path.relative_to(installation.path) == Path(artifact.filename)
        assert artifact.path.read_bytes() == payloads[artifact.filename]

    def no_download(*_args, **_kwargs) -> None:
        pytest.fail("offline reopen attempted a download")

    monkeypatch.setattr(store_module.AssetStore, "_download", no_download)
    offline_manager = OnnxVoice(
        cache_dir=cache_dir,
        catalog_sources={"sopro": str(catalog_path)},
        offline=True,
    )
    reopened = offline_manager.install("sopro:turbo", quality="wasm-uint8")
    offline_manager.store.verify(reopened)
    assert reopened.path == installation.path

    broken = reopened.artifacts[0]
    broken.path.write_bytes(b"corrupt")
    with pytest.raises(IntegrityError):
        offline_manager.store.verify(reopened)
