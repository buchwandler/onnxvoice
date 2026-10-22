from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.catalog import DEFAULT_SOURCES, CatalogClient, _parse_qwen3
from onnxvoice.catalog_tools import qwen3
from onnxvoice.errors import CatalogError

BUNDLE_TYPES = {"base-0.6b": "base", "voice-design-1.7b": "voice_design"}
GENERATION = ("talker_prefill", "talker_decode", "code_predictor", "vocoder")
ENCODERS = ("speaker_encoder", "tokenizer_encoder")
TOKENIZERS = {
    "tokenizer_json": "tokenizer.json",
    "vocab": "vocab.json",
    "merges": "merges.txt",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens": "special_tokens_map.json",
    "added_tokens": "added_tokens.json",
}
EMBEDDINGS = (
    "text_embedding",
    "text_projection_fc1_weight",
    "text_projection_fc1_bias",
    "text_projection_fc2_weight",
    "text_projection_fc2_bias",
    "talker_codec_embedding",
    *(f"cp_codec_embedding_{i}" for i in range(15)),
)


def _url(repository: str, revision: str, path: str) -> str:
    return qwen3.huggingface_resolve_url(repository, revision, path)


def _artifact(
    bundle_id: str,
    role: str,
    component: str | None,
    quality: str | None,
    path: str,
    index: int,
) -> dict[str, Any]:
    repository = "fixture/" + bundle_id
    revision = "a" * 40 if bundle_id == "base-0.6b" else "b" * 40
    return {
        "role": role,
        "component": component,
        "quality": quality,
        "path": path,
        "url": _url(repository, revision, path),
        "size": index + 1,
        "sha256": hashlib.sha256(f"{bundle_id}:{path}".encode()).hexdigest(),
    }


def _catalog() -> dict[str, Any]:
    bundles = {}
    index = 0
    for bundle_id, model_type in BUNDLE_TYPES.items():
        repository = "fixture/" + bundle_id
        revision = "a" * 40 if bundle_id == "base-0.6b" else "b" * 40
        aliases = ["base", "clone"] if model_type == "base" else ["design", "voicedesign"]
        models = list(GENERATION) + (list(ENCODERS) if model_type == "base" else [])
        artifacts: list[dict[str, Any]] = []

        def add(
            role: str,
            component: str | None,
            quality: str | None,
            path: str,
            *,
            _bundle_id: str = bundle_id,
            _artifacts: list[dict[str, Any]] = artifacts,
        ) -> None:
            nonlocal index
            _artifacts.append(_artifact(_bundle_id, role, component, quality, path, index))
            index += 1

        add("config", None, None, "config.json")
        for component, filename in TOKENIZERS.items():
            add("tokenizer", component, None, f"tokenizer/{filename}")
        for component in EMBEDDINGS:
            add("embedding", component, None, f"embeddings/{component}.npy")
        for component in ENCODERS if model_type == "base" else ():
            add("model", component, None, f"{component}.onnx")
            add("model_data", component, None, f"{component}.onnx.data")
        for quality in ("fp32", "int4"):
            for component in GENERATION:
                add("model", component, quality, f"{quality}/{component}.onnx")
                add("model_data", component, quality, f"{quality}/{component}.onnx.data")
        bundles[bundle_id] = {
            "id": bundle_id,
            "aliases": aliases,
            "layout": "qwen3-wavekat-v1",
            "model_type": model_type,
            "sample_rate": 24000,
            "source": {
                "provider": "huggingface",
                "repository": repository,
                "revision": revision,
                "license": "apache-2.0",
            },
            "profiles": {quality: {"components": sorted(models)} for quality in ("fp32", "int4")},
            "artifacts": artifacts,
        }
    return {
        "schema": 1,
        "kind": "qwen3-onnx-bundle-catalog",
        "source": {"provider": "huggingface", "bundle_count": 2},
        "bundles": bundles,
    }


def _tree_for(seed: dict[str, Any], bundle_id: str) -> dict[str, dict[str, Any]]:
    entry = seed["bundles"][bundle_id]
    paths = {"config.json"}
    paths.update(f"tokenizer/{name}" for name in TOKENIZERS.values())
    paths.update(f"embeddings/{name}.npy" for name in EMBEDDINGS)
    if entry["model_type"] == "base":
        for component in ENCODERS:
            paths.add(f"{component}.onnx")
            paths.add(f"{component}.onnx.data")
    for quality in ("fp32", "int4"):
        paths.update(f"{quality}/{name}.onnx" for name in GENERATION)
        paths.update(f"{quality}/{name}.onnx.data" for name in GENERATION)
    return {
        path: {
            "path": path,
            "type": "file",
            "size": len(path) + 1,
            "lfs": {"oid": hashlib.sha256(path.encode()).hexdigest()},
        }
        for path in paths
    }


class TestParseQwen3:
    def test_parses_both_bundles_and_component_inventory(self) -> None:
        items = _parse_qwen3(_catalog())
        assert [(item.system, item.id, item.kind) for item in items] == [
            ("qwen3", "base-0.6b", "bundle"),
            ("qwen3", "voice-design-1.7b", "bundle"),
        ]
        base, design = items
        assert base.aliases == ("base", "clone")
        assert base.metadata["model_type"] == "base"
        assert base.metadata["runtime"]["layout"] == "qwen3-wavekat-v1"
        assert {a.component for a in base.artifacts if a.role == "embedding"} == set(EMBEDDINGS)
        assert {a.component for a in design.artifacts if a.role == "model"} == set(GENERATION)
        assert (
            base.artifact("model", component="talker_prefill", quality="int4").metadata["path"]
            == "int4/talker_prefill.onnx"
        )

    def test_default_catalog_and_quality_selection(self, tmp_path: Path) -> None:
        assert DEFAULT_SOURCES["qwen3"].endswith("qwen3-onnx-bundles/main/catalog/bundles.json")
        source = tmp_path / "catalog.json"
        source.write_text(json.dumps(_catalog()), encoding="utf-8")
        client = CatalogClient(
            cache_dir=tmp_path / "cache", sources={"qwen3": str(source)}, offline=True
        )
        item = client.resolve("qwen3:design", quality="int4")
        assert item.id == "voice-design-1.7b"
        assert item.metadata["selected_quality"] == "int4"
        assert {a.quality for a in item.artifacts if a.role in {"model", "model_data"}} == {"int4"}
        assert client.resolve("qwen3:base", quality="fp32").metadata["model_type"] == "base"

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda c: c["source"].update(bundle_count=1), "bundle_count"),
            (
                lambda c: c["bundles"]["base-0.6b"]["source"].update(revision="main"),
                "40-character SHA",
            ),
            (
                lambda c: c["bundles"]["base-0.6b"].update(layout="other"),
                "unsupported Qwen3 layout",
            ),
            (lambda c: c["bundles"]["base-0.6b"].update(model_type="custom_voice"), "model_type"),
            (
                lambda c: c["bundles"]["base-0.6b"]["profiles"]["int4"].update(components=[]),
                "missing required",
            ),
            (lambda c: c["bundles"]["base-0.6b"]["artifacts"].pop(), "external-data dependencies"),
            (
                lambda c: c["bundles"]["base-0.6b"]["artifacts"][0].update(path="../escape"),
                "Unsafe",
            ),
            (
                lambda c: c["bundles"]["base-0.6b"]["artifacts"][0].update(size=0),
                "positive integer",
            ),
            (lambda c: c["bundles"]["base-0.6b"]["artifacts"][0].update(sha256=None), "sha256"),
            (
                lambda c: c["bundles"]["voice-design-1.7b"]["aliases"].append("base"),
                "duplicate or ambiguous",
            ),
            (lambda c: c["bundles"]["voice-design-1.7b"]["artifacts"].pop(0), "config artifact"),
        ],
    )
    def test_rejects_invalid_normalized_catalog(self, mutate, message: str) -> None:
        damaged = copy.deepcopy(_catalog())
        mutate(damaged)
        with pytest.raises(CatalogError, match=message):
            _parse_qwen3(damaged)

    def test_rejects_mutable_revision_unsafe_path_and_missing_external_data(
        self, monkeypatch
    ) -> None:
        seed = json.loads(qwen3.DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
        bad = copy.deepcopy(seed)
        bad["bundles"]["base-0.6b"]["profiles"]["int4"]["include"][0] = "../config.json"
        with pytest.raises(qwen3.CatalogError, match="unsafe include"):
            qwen3._seed_bundles(bad)
        bad = copy.deepcopy(seed)
        bad["bundles"]["base-0.6b"]["profiles"]["fp32"]["include"].remove("fp32/vocoder.onnx.data")
        trees = {bundle_id: _tree_for(bad, bundle_id) for bundle_id in bad["bundles"]}
        monkeypatch.setattr(
            qwen3,
            "_repository_tree",
            lambda repo, rev: trees["base-0.6b" if "0.6B" in repo else "voice-design-1.7b"],
        )
        with pytest.raises(qwen3.CatalogError, match="external-data dependencies"):
            qwen3.build_catalog(bad)


class TestBuildQwen3Catalog:
    def test_builds_normalized_catalog_and_verifies_offline(self, monkeypatch) -> None:
        seed = json.loads(qwen3.DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
        trees = {bundle_id: _tree_for(seed, bundle_id) for bundle_id in seed["bundles"]}
        monkeypatch.setattr(
            qwen3,
            "_repository_tree",
            lambda repo, revision: trees["base-0.6b" if "0.6B" in repo else "voice-design-1.7b"],
        )
        first = qwen3.build_catalog(seed)
        second = qwen3.build_catalog(seed)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert first["source"]["bundle_count"] == 2
        assert first["bundles"]["base-0.6b"]["profiles"]["int4"]["components"] == sorted(
            (*GENERATION, *ENCODERS)
        )
        source = qwen3.build_source(first)
        monkeypatch.setattr(qwen3, "_repository_tree", lambda *args: pytest.fail("network"))
        monkeypatch.setattr(qwen3, "_read_url", lambda *args: pytest.fail("network"))
        qwen3.verify_catalog(first, source)
        with pytest.raises(qwen3.CatalogError, match="does not match"):
            qwen3.verify_catalog(first, {**source, "sources": []})

    def test_rejects_empty_seed_include(self, monkeypatch) -> None:
        seed = json.loads(qwen3.DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
        seed["bundles"]["base-0.6b"]["profiles"]["fp32"]["include"][0] = "absent/*.onnx"
        monkeypatch.setattr(
            qwen3, "_repository_tree", lambda repo, rev: _tree_for(seed, "base-0.6b")
        )
        with pytest.raises(qwen3.CatalogError, match="matched no files"):
            qwen3.build_catalog(seed)

    def test_load_catalog_requires_valid_json_and_offline_integrity(self, tmp_path: Path) -> None:
        catalog = _catalog()
        source = qwen3.build_source(catalog)
        cat_path = tmp_path / "bundles.json"
        src_path = tmp_path / "source.json"
        cat_path.write_text(json.dumps(catalog), encoding="utf-8")
        src_path.write_text(json.dumps(source), encoding="utf-8")
        assert qwen3.load_catalog(cat_path, src_path) == catalog
        source["sources"][0]["revision"] = "c" * 40
        src_path.write_text(json.dumps(source), encoding="utf-8")
        with pytest.raises(qwen3.CatalogError, match="does not match"):
            qwen3.load_catalog(cat_path, src_path)
