from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import onnxvoice.manager as manager_module
from onnxvoice.catalog import CatalogClient
from onnxvoice.catalog_tools import qwen3
from onnxvoice.errors import RuntimeContractError
from onnxvoice.manager import OnnxVoice
from onnxvoice.systems import Qwen3Adapter, registered_systems

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
    *(f"cp_codec_embedding_{index}" for index in range(15)),
)
BUNDLE_TYPES = {"base-0.6b": "base", "voice-design-1.7b": "voice_design"}


def _config_data(model_type: str) -> dict[str, Any]:
    hidden = 8 if model_type == "voice_design" else 4
    return {
        "model_id": f"fixture/{model_type}",
        "tts_model_type": model_type,
        "sample_rate": 24000,
        "im_start_token_id": 0,
        "im_end_token_id": 1,
        "tts_pad_token_id": 2,
        "tts_bos_token_id": 3,
        "tts_eos_token_id": 4,
        "codec_eos_token_id": 5,
        "codec_think_id": 6,
        "codec_nothink_id": 7,
        "codec_think_bos_id": 8,
        "codec_think_eos_id": 9,
        "codec_pad_id": 10,
        "codec_bos_id": 11,
        "talker_hidden_size": hidden,
        "talker_num_layers": 2,
        "talker_num_attention_heads": 2,
        "talker_num_kv_heads": 1,
        "talker_head_dim": 4,
        "talker_vocab_size": 32,
        "talker_text_hidden_size": 3,
        "talker_num_code_groups": 16,
        "cp_hidden_size": 4,
        "cp_num_layers": 2,
        "cp_num_attention_heads": 2,
        "cp_num_kv_heads": 1,
        "cp_head_dim": 4,
        "cp_vocab_size": 8,
        "cp_num_code_groups": 16,
        "codec_language_id": {"english": 12},
    }


def _embedding_array(component: str, model_type: str) -> Any:

    config = _config_data(model_type)
    hidden = config["talker_hidden_size"]
    shapes = {
        "text_embedding": (16, config["talker_text_hidden_size"]),
        "text_projection_fc1_weight": (5, config["talker_text_hidden_size"]),
        "text_projection_fc1_bias": (5,),
        "text_projection_fc2_weight": (hidden, 5),
        "text_projection_fc2_bias": (hidden,),
        "talker_codec_embedding": (config["talker_vocab_size"], hidden),
    }
    if component.startswith("cp_codec_embedding_"):
        shape = (config["cp_vocab_size"], hidden)
    else:
        shape = shapes[component]
    return np.zeros(shape, dtype=np.float32)


def _artifact_bytes(path: str, model_type: str) -> bytes:

    if path == "config.json":
        return json.dumps(_config_data(model_type), sort_keys=True).encode()
    if path.startswith("embeddings/") and path.endswith(".npy"):
        component = Path(path).stem
        buffer = io.BytesIO()
        np.save(buffer, _embedding_array(component, model_type), allow_pickle=False)
        return buffer.getvalue()
    return f"fixture:{path}".encode()


def _catalog() -> dict[str, Any]:
    bundles: dict[str, Any] = {}
    for bundle_id, model_type in BUNDLE_TYPES.items():
        repository = f"fixture/{bundle_id}"
        revision = "a" * 40 if model_type == "base" else "b" * 40
        artifacts: list[dict[str, Any]] = []

        def add(
            role: str,
            component: str | None,
            quality: str | None,
            path: str,
            *,
            _repository: str = repository,
            _revision: str = revision,
            _artifacts: list[dict[str, Any]] = artifacts,
            _model_type: str = model_type,
        ) -> None:
            content = _artifact_bytes(path, _model_type)
            _artifacts.append(
                {
                    "role": role,
                    "component": component,
                    "quality": quality,
                    "path": path,
                    "url": qwen3.huggingface_resolve_url(_repository, _revision, path),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )

        add("config", None, None, "config.json")
        for component, filename in TOKENIZERS.items():
            add("tokenizer", component, None, f"tokenizer/{filename}")
        for component in EMBEDDINGS:
            add("embedding", component, None, f"embeddings/{component}.npy")
        components = list(GENERATION)
        if model_type == "base":
            components.extend(ENCODERS)
            for component in ENCODERS:
                add("model", component, None, f"{component}.onnx")
                add("model_data", component, None, f"{component}.onnx.data")
        for quality in ("fp32", "int4"):
            for component in GENERATION:
                add("model", component, quality, f"{quality}/{component}.onnx")
                add("model_data", component, quality, f"{quality}/{component}.onnx.data")
        artifacts.sort(
            key=lambda item: (
                item["role"],
                item["component"] or "",
                item["quality"] or "",
                item["path"],
            )
        )
        bundles[bundle_id] = {
            "id": bundle_id,
            "aliases": ["base", "clone"] if model_type == "base" else ["design", "voicedesign"],
            "layout": "qwen3-wavekat-v1",
            "model_type": model_type,
            "sample_rate": 24000,
            "source": {
                "provider": "huggingface",
                "repository": repository,
                "revision": revision,
                "license": "apache-2.0",
            },
            "profiles": {
                quality: {"components": sorted(components)} for quality in ("fp32", "int4")
            },
            "artifacts": artifacts,
        }
    return {
        "schema": 1,
        "kind": "qwen3-onnx-bundle-catalog",
        "source": {"provider": "huggingface", "bundle_count": 2},
        "bundles": bundles,
    }


def _local_files(root: Path, model_type: str) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def add(key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "config":
            content = _artifact_bytes("config.json", model_type)
        elif key.startswith("embedding:"):
            component = key.split(":", 1)[1]
            content = _artifact_bytes(f"embeddings/{component}.npy", model_type)
        else:
            content = f"fixture:{key}".encode()
        path.write_bytes(content)
        files[key] = path

    add("config", root / "config.json")
    for component, filename in TOKENIZERS.items():
        add(f"tokenizer:{component}", root / "tokenizer" / filename)
    for component in EMBEDDINGS:
        add(f"embedding:{component}", root / "embeddings" / f"{component}.npy")
    components = list(GENERATION)
    if model_type == "base":
        components.extend(ENCODERS)
    for component in components:
        prefix = Path("shared") if component in ENCODERS else Path("int4")
        graph = root / prefix / f"{component}.onnx"
        add(f"model:{component}", graph)
        add(f"model_data:{component}", Path(str(graph) + ".data"))
    return files


def _write_catalog(tmp_path: Path) -> Path:
    catalog = _catalog()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("model_type", ["voice_design", "base"])
def test_qwen3_open_local_detects_type_and_never_uses_store_or_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    assert "qwen3" in registered_systems()
    files = _local_files(tmp_path, model_type)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Qwen3 open_local must not access catalog or store")

    monkeypatch.setattr(manager_module.AssetStore, "__init__", forbidden)
    monkeypatch.setattr(CatalogClient, "__init__", forbidden)
    adapter = OnnxVoice.open_local(
        system="qwen3",
        artifacts=files,
        runtime={"layout": "qwen3-wavekat-v1", "model_type": model_type, "quality": "int4"},
        sample_rate=24000,
        providers="cpu",
    )
    try:
        assert isinstance(adapter, Qwen3Adapter)
        assert adapter.installation.metadata["managed"] is False
        assert adapter.installation.metadata["runtime"]["model_type"] == model_type
        assert adapter.installation.metadata["runtime"]["quality"] == "int4"
        assert adapter.installation.sample_rate == 24000
        assert {artifact.path for artifact in adapter.installation.artifacts} == {
            path.resolve() for path in files.values()
        }
        for component in (*GENERATION, *(ENCODERS if model_type == "base" else ())):
            graph = adapter.installation.artifact("model", component=component)
            data = adapter.installation.artifact("model_data", component=component)
            assert data.path == Path(str(graph.path) + ".data")
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("runtime", "sample_rate", "error"),
    [
        ({"layout": "other", "quality": "int4"}, 24000, "layout"),
        ({"layout": "qwen3-wavekat-v1", "quality": "fp16"}, 24000, "quality"),
        (
            {"layout": "qwen3-wavekat-v1", "quality": "int4", "model_type": "base"},
            24000,
            "model_type",
        ),
        ({"layout": "qwen3-wavekat-v1", "quality": "int4"}, 16000, "sample_rate"),
    ],
)
def test_qwen3_open_local_rejects_runtime_mismatches(
    tmp_path: Path,
    runtime: dict[str, Any],
    sample_rate: int,
    error: str,
) -> None:
    files = _local_files(tmp_path, "voice_design")
    with pytest.raises(RuntimeContractError, match=error):
        OnnxVoice.open_local(system="qwen3", files=files, runtime=runtime, sample_rate=sample_rate)


def test_qwen3_open_local_rejects_incomplete_inventory_and_nonadjacent_data(tmp_path: Path) -> None:
    files = _local_files(tmp_path / "missing", "voice_design")
    del files["embedding:text_embedding"]
    with pytest.raises(RuntimeContractError, match="Incomplete Qwen3 local artifact inventory"):
        OnnxVoice.open_local(
            system="qwen3",
            files=files,
            runtime={"layout": "qwen3-wavekat-v1", "model_type": "voice_design", "quality": "int4"},
        )

    files = _local_files(tmp_path / "adjacency", "voice_design")
    detached = tmp_path / "detached" / "talker_prefill.onnx.data"
    detached.parent.mkdir(parents=True)
    detached.write_bytes(b"external weights")
    files["model_data:talker_prefill"] = detached
    with pytest.raises(RuntimeContractError, match="must be adjacent"):
        OnnxVoice.open_local(
            system="qwen3",
            files=files,
            runtime={"layout": "qwen3-wavekat-v1", "model_type": "voice_design", "quality": "int4"},
        )


def test_qwen3_managed_install_preserves_profile_paths_and_open_is_local_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = _write_catalog(tmp_path)
    for bundle_id, model_type in BUNDLE_TYPES.items():
        for quality in ("fp32", "int4"):
            manager = OnnxVoice(
                cache_dir=tmp_path / f"cache-{bundle_id}-{quality}",
                catalog_sources={"qwen3": str(catalog_path)},
            )

            def write_fixture(
                _source: str,
                destination: Path,
                *,
                artifact: str,
                _model_type: str = model_type,
                **_kwargs: Any,
            ) -> None:
                destination.write_bytes(_artifact_bytes(artifact, _model_type))

            monkeypatch.setattr(manager.store, "_download", write_fixture)
            installation = manager.install(f"qwen3:{bundle_id}", quality=quality)
            manager.store.verify(installation)
            assert installation.id == bundle_id
            assert installation.metadata["runtime"] == {
                "layout": "qwen3-wavekat-v1",
                "model_type": model_type,
                "quality": quality,
                "sample_rate": 24000,
            }
            expected_qualities = {quality} | ({None} if model_type == "base" else set())
            assert {
                artifact.quality for artifact in installation.artifacts if artifact.role == "model"
            } == expected_qualities
            for artifact in installation.artifacts:
                relative = artifact.metadata["path"]
                assert artifact.filename == relative
                assert artifact.path == installation.path / relative
            components = set(GENERATION) | (set(ENCODERS) if model_type == "base" else set())
            for component in components:
                graph = installation.artifact(
                    "model", component=component, quality=None if component in ENCODERS else quality
                )
                data = installation.artifact(
                    "model_data",
                    component=component,
                    quality=None if component in ENCODERS else quality,
                )
                assert data.path == Path(str(graph.path) + ".data")

            def no_catalog(*_args: Any, **_kwargs: Any) -> Any:
                pytest.fail("Opening an Installation must not load a catalog")

            with monkeypatch.context() as scoped:
                scoped.setattr(CatalogClient, "load_raw", no_catalog)
                scoped.setattr(urllib.request, "urlopen", no_catalog)
                scoped.setattr(manager.store, "_download", no_catalog)
                adapter = manager.open(installation, provider="cpu")
                try:
                    assert isinstance(adapter, Qwen3Adapter)
                    assert adapter.quality == quality
                    assert adapter.model_type == model_type
                finally:
                    adapter.close()
