from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_chatterbox
from onnxvoice.catalog_tools import chatterbox
from onnxvoice.errors import AssetNotFoundError, CatalogError

REVISION = "a" * 40
DIGEST = "b" * 64
COMPONENTS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")


def seed() -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "chatterbox-onnx-bundle-seed",
        "bundles": {
            "nano-v2": {
                "aliases": ["nano"],
                "language": "en",
                "sample_rate": 24000,
                "default_profile": "cpu",
                "source": {
                    "provider": "huggingface",
                    "repository": "owner/nano",
                    "requested_revision": REVISION,
                },
                "frontend": {
                    "tokenizer_role": "tokenizer",
                    "tokenizer_path": "tokenizer.json",
                    "tokenizer_contract": "nano-v2",
                },
                "runtime": {
                    "start_speech_token": 6561,
                    "stop_speech_token": 6562,
                    "silence_token": 4299,
                    "trailing_silence_tokens": 3,
                    "default_max_new_tokens": 1024,
                    "default_repetition_penalty": 1.2,
                },
                "profiles": {
                    "fp32": {
                        component: {"path": f"onnx/{component}.onnx", "quality": "fp32"}
                        for component in COMPONENTS
                    },
                    "cpu": {
                        "speech_encoder": {"path": "onnx/speech_encoder.onnx", "quality": "fp32"},
                        "embed_tokens": {"path": "onnx/embed_tokens.onnx", "quality": "fp32"},
                        "language_model": {
                            "path": "onnx/language_model_int8.onnx",
                            "quality": "int8_mixed",
                        },
                        "conditional_decoder": {
                            "path": "onnx/conditional_decoder_int8.onnx",
                            "quality": "int8",
                        },
                    },
                },
            }
        },
    }


def mock_tree(monkeypatch: pytest.MonkeyPatch, source_seed: dict[str, Any]) -> None:
    paths = {"tokenizer.json"}
    for bundle in source_seed["bundles"].values():
        for profile in bundle["profiles"].values():
            for declaration in profile.values():
                path = declaration["path"]
                paths.add(path)
                if path.endswith("language_model.onnx"):
                    paths.add(path + "_data")
                if path.endswith("language_model_int8.onnx"):
                    paths.add(path + ".data")
    tree = {
        path: {"path": path, "type": "file", "size": 4, "lfs": {"oid": DIGEST}} for path in paths
    }
    monkeypatch.setattr(chatterbox, "_repository_tree", lambda repository, revision: tree)


def test_default_source_and_alias_profile_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_seed = seed()
    mock_tree(monkeypatch, raw_seed)
    catalog = chatterbox.build_catalog(raw_seed)
    assert DEFAULT_SOURCES["chatterbox"].endswith(
        "chatterbox-onnx-bundles/main/catalog/bundles.json"
    )
    items = _parse_chatterbox(catalog)
    assert [item.ref for item in items] == ["chatterbox:nano-v2"]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path, sources={"chatterbox": str(catalog_path)})

    cpu = client.resolve("chatterbox:nano", quality="cpu")
    assert cpu.selected_quality == "cpu"
    assert cpu.metadata["frontend"]["tokenizer_contract"] == "nano-v2"
    assert {a.component: a.quality for a in cpu.artifacts if a.role == "graph"} == {
        "speech_encoder": "fp32",
        "embed_tokens": "fp32",
        "language_model": "int8_mixed",
        "conditional_decoder": "int8",
    }
    assert {a.component for a in cpu.artifacts if a.role == "external_data"} == {"language_model"}
    assert all(
        a.quality != "fp32" or a.component in {"speech_encoder", "embed_tokens"}
        for a in cpu.artifacts
        if a.role == "graph"
    )
    fp32 = client.resolve("chatterbox:nano-v2", quality="fp32")
    assert {a.quality for a in fp32.artifacts if a.role == "graph"} == {"fp32"}
    with pytest.raises(AssetNotFoundError):
        client.resolve("chatterbox:nano", quality="q4f16")


def test_builder_creates_explicit_pinned_artifacts_and_source_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_seed = seed()
    mock_tree(monkeypatch, raw_seed)
    catalog = chatterbox.build_catalog(raw_seed)
    source = chatterbox.build_source(catalog)
    chatterbox.verify_catalog(catalog, source)
    assert source["bundle_count"] == 1
    assert source["artifact_count"] == len(catalog["bundles"]["nano-v2"]["artifacts"])
    for artifact in catalog["bundles"]["nano-v2"]["artifacts"]:
        assert f"/resolve/{REVISION}/" in artifact["url"]
        assert len(artifact["sha256"]) == 64 and artifact["size"] > 0
    assert chatterbox.canonical_json(catalog) == chatterbox.canonical_json(
        json.loads(chatterbox.canonical_json(catalog))
    )


def test_builder_rejects_duplicate_alias_and_unsafe_paths() -> None:
    duplicate = seed()
    duplicate["bundles"]["nano-v2"]["aliases"] = ["turbo"]
    duplicate["bundles"]["turbo"] = {"aliases": ["turbo"]}
    with pytest.raises(chatterbox.CatalogError, match="Duplicate"):
        chatterbox._validate_seed(duplicate)

    unsafe = seed()
    unsafe["bundles"]["nano-v2"]["profiles"]["fp32"]["speech_encoder"]["path"] = "../escape.onnx"
    with pytest.raises(chatterbox.CatalogError, match="unsafe"):
        chatterbox._validate_seed(unsafe)


def test_parser_rejects_non_pinned_urls_bad_integrity_and_bad_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_seed = seed()
    mock_tree(monkeypatch, raw_seed)
    catalog = chatterbox.build_catalog(raw_seed)
    malformed = json.loads(json.dumps(catalog))
    malformed["bundles"]["nano-v2"]["source"]["revision"] = "main"
    with pytest.raises(CatalogError, match="revision"):
        _parse_chatterbox(malformed)

    malformed = json.loads(json.dumps(catalog))
    malformed["bundles"]["nano-v2"]["artifacts"][0]["url"] = (
        "https://huggingface.co/owner/nano/resolve/main/tokenizer.json"
    )
    with pytest.raises(CatalogError, match="pinned"):
        _parse_chatterbox(malformed)

    malformed = json.loads(json.dumps(catalog))
    malformed["bundles"]["nano-v2"]["artifacts"][0]["sha256"] = "bad"
    with pytest.raises(CatalogError, match="sha256"):
        _parse_chatterbox(malformed)

    malformed = json.loads(json.dumps(catalog))
    malformed["bundles"]["nano-v2"]["artifacts"][0]["size"] = 0
    with pytest.raises(CatalogError, match="size"):
        _parse_chatterbox(malformed)


def test_parser_rejects_alias_collisions_and_mismatched_external_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_seed = seed()
    mock_tree(monkeypatch, raw_seed)
    catalog = chatterbox.build_catalog(raw_seed)
    malformed = json.loads(json.dumps(catalog))
    malformed["bundles"]["nano-v2"]["aliases"] = ["nano", "nano"]
    with pytest.raises(CatalogError, match="alias"):
        _parse_chatterbox(malformed)

    malformed = json.loads(json.dumps(catalog))
    for artifact in malformed["bundles"]["nano-v2"]["artifacts"]:
        if artifact["role"] == "external_data":
            artifact["source_path"] = "onnx/unrelated.onnx.data"
            artifact["filename"] = "unrelated.onnx.data"
            artifact["url"] = (
                f"https://huggingface.co/owner/nano/resolve/{REVISION}/onnx/unrelated.onnx.data?download=true"
            )
            break
    with pytest.raises(CatalogError, match="match"):
        _parse_chatterbox(malformed)


def test_parser_rejects_unselected_profile_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_seed = seed()
    mock_tree(monkeypatch, raw_seed)
    catalog = chatterbox.build_catalog(raw_seed)
    extra = json.loads(json.dumps(catalog))
    bundle = extra["bundles"]["nano-v2"]
    graph = next(item for item in bundle["artifacts"] if item["role"] == "graph")
    unused = json.loads(json.dumps(graph))
    unused["quality"] = "unsupported"
    unused["source_path"] = "onnx/language_model_unsupported.onnx"
    unused["filename"] = "language_model_unsupported.onnx"
    unused["url"] = (
        f"https://huggingface.co/owner/nano/resolve/{REVISION}/onnx/language_model_unsupported.onnx?download=true"
    )
    bundle["artifacts"].append(unused)
    with pytest.raises(CatalogError, match="unselected"):
        _parse_chatterbox(extra)
