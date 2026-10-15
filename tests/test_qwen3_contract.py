from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import onnxvoice.systems.qwen3 as qwen3_module
from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.qwen3 import Qwen3Adapter
from onnxvoice.systems.qwen3_contract import (
    CODE_GROUPS,
    REQUIRED_EMBEDDINGS,
    Qwen3Bundle,
    Qwen3Config,
    Qwen3EmbeddingStore,
    validate_graph_abi,
)
from onnxvoice.types import (
    Installation,
    InstalledArtifact,
    RuntimeDiagnostic,
    SessionDiagnostic,
    TensorSpec,
)


def _config(model_type: str = "base") -> dict[str, Any]:
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
        "talker_num_code_groups": CODE_GROUPS,
        "cp_hidden_size": 4,
        "cp_num_layers": 2,
        "cp_num_attention_heads": 2,
        "cp_num_kv_heads": 1,
        "cp_head_dim": 4,
        "cp_vocab_size": 8,
        "cp_num_code_groups": CODE_GROUPS,
        "codec_language_id": {"english": 12},
        "max_position_embeddings": 512,
    }


def _embedding_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    hidden = config["talker_hidden_size"]
    arrays = {
        "text_embedding": np.zeros((16, config["talker_text_hidden_size"]), np.float32),
        "text_projection_fc1_weight": np.zeros((5, config["talker_text_hidden_size"]), np.float32),
        "text_projection_fc1_bias": np.zeros((5,), np.float32),
        "text_projection_fc2_weight": np.zeros((hidden, 5), np.float32),
        "text_projection_fc2_bias": np.zeros((hidden,), np.float32),
        "talker_codec_embedding": np.zeros((config["talker_vocab_size"], hidden), np.float32),
    }
    for index in range(CODE_GROUPS - 1):
        arrays[f"cp_codec_embedding_{index}"] = np.zeros(
            (config["cp_vocab_size"], hidden), np.float32
        )
    return arrays


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)


def _tensor(name: str, ort_type: str, *shape: Any) -> TensorSpec:
    return TensorSpec(name=name, ort_type=ort_type, shape=tuple(shape))


def _graph_specs(
    component: str, config: Qwen3Config
) -> tuple[tuple[TensorSpec, ...], tuple[TensorSpec, ...]]:
    f32, i64 = "tensor(float)", "tensor(int64)"
    hidden = config.talker_hidden_size
    if component == "talker_prefill":
        inputs = (
            _tensor("inputs_embeds", f32, 1, "sequence", hidden),
            _tensor("attention_mask", i64, 1, "sequence"),
            _tensor("position_ids", i64, 3, 1, "sequence"),
        )
        outputs = [
            _tensor("logits", f32, 1, "sequence", config.talker_vocab_size),
            _tensor("hidden_states", f32, 1, "sequence", hidden),
        ]
        for index in range(config.talker_num_layers * 2):
            outputs.append(
                _tensor(
                    f"present_key_values.{index // 2}.{'key' if index % 2 == 0 else 'value'}",
                    f32,
                    1,
                    config.talker_num_kv_heads,
                    "sequence",
                    config.talker_head_dim,
                )
            )
        return inputs, tuple(outputs)
    if component == "talker_decode":
        cache = (
            config.talker_num_layers,
            1,
            config.talker_num_kv_heads,
            "sequence",
            config.talker_head_dim,
        )
        inputs = (
            _tensor("inputs_embeds", f32, 1, 1, hidden),
            _tensor("attention_mask", i64, 1, "sequence"),
            _tensor("position_ids", i64, 3, 1, 1),
            _tensor("past_keys", f32, *cache),
            _tensor("past_values", f32, *cache),
        )
        outputs = (
            _tensor("logits", f32, 1, 1, config.talker_vocab_size),
            _tensor("hidden_states", f32, 1, 1, hidden),
            _tensor("present_keys", f32, *cache),
            _tensor("present_values", f32, *cache),
        )
        return inputs, outputs
    if component == "code_predictor":
        cache = (config.cp_num_layers, 1, config.cp_num_kv_heads, "sequence", config.cp_head_dim)
        inputs = (
            _tensor("inputs_embeds", f32, 1, "sequence", hidden),
            _tensor("generation_steps", i64, 1),
            _tensor("past_keys", f32, *cache),
            _tensor("past_values", f32, *cache),
        )
        outputs = (
            _tensor("logits", f32, 1, "sequence", config.cp_vocab_size),
            _tensor("present_keys", f32, *cache),
            _tensor("present_values", f32, *cache),
        )
        return inputs, outputs
    if component == "vocoder":
        return (_tensor("codes", i64, 1, CODE_GROUPS, "frames"),), (
            _tensor("waveform", f32, 1, "samples"),
        )
    if component == "speaker_encoder":
        return (_tensor("mels", f32, 1, "frames", 128),), (
            _tensor("speaker_embedding", f32, 1, hidden),
        )
    if component == "tokenizer_encoder":
        return (_tensor("waveform", f32, 1, 240_000),), (
            _tensor("codes", i64, 1, CODE_GROUPS, "frames"),
        )
    raise AssertionError(component)


def _write_installation(root: Path, model_type: str = "voice_design") -> Installation:
    config_raw = _config(model_type)
    artifacts: list[InstalledArtifact] = []

    def add(
        role: str,
        relative: str,
        *,
        component: str | None = None,
        quality: str | None = None,
        content: bytes,
    ) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(
            InstalledArtifact(
                role=role,
                filename=relative,
                path=path,
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
                component=component,
                quality=quality,
                metadata={"path": relative},
            )
        )

    add("config", "config.json", content=json.dumps(config_raw).encode())
    for tokenizer_component, filename in {
        "tokenizer_json": "tokenizer.json",
        "vocab": "vocab.json",
        "merges": "merges.txt",
        "tokenizer_config": "tokenizer_config.json",
        "special_tokens": "special_tokens_map.json",
        "added_tokens": "added_tokens.json",
    }.items():
        add("tokenizer", f"tokenizer/{filename}", component=tokenizer_component, content=b"fixture")
    for component, array in _embedding_arrays(config_raw).items():
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        add(
            "embedding",
            f"embeddings/{component}.npy",
            component=component,
            content=buffer.getvalue(),
        )

    components = ["talker_prefill", "talker_decode", "code_predictor", "vocoder"]
    if model_type == "base":
        components.extend(("speaker_encoder", "tokenizer_encoder"))
    for component in components:
        quality = None if component in {"speaker_encoder", "tokenizer_encoder"} else "int4"
        graph = f"{component}.onnx" if quality is None else f"int4/{component}.onnx"
        add("model", graph, component=component, quality=quality, content=b"graph")
        add("model_data", graph + ".data", component=component, quality=quality, content=b"weights")
    return Installation(
        system="qwen3",
        id="voice-design-1.7b" if model_type == "voice_design" else "base-0.6b",
        kind="model",
        path=root,
        artifacts=tuple(artifacts),
        sample_rate=24000,
        metadata={
            "runtime": {
                "layout": "qwen3-wavekat-v1",
                "model_type": model_type,
                "quality": "int4",
                "sample_rate": 24000,
            },
            "selected_quality": "int4",
        },
    )


def test_qwen3_config_uses_installed_dimensions_and_token_ids() -> None:
    config = Qwen3Config.from_mapping(_config("voice_design"))
    assert config.model_type == "voice_design"
    assert config.talker_hidden_size == 8
    assert config.talker_num_layers == 2
    assert config.talker_num_code_groups == 16
    assert config.codec_eos_token_id == 5
    assert config.codec_language_id["english"] == 12
    assert config.max_position_embeddings == 512
    assert isinstance(config.codec_language_id, MappingProxyType)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tts_model_type", [], "tts_model_type"),
        ("sample_rate", True, "sample_rate"),
        ("talker_num_layers", 0, "talker_num_layers"),
        ("talker_num_code_groups", 15, "code groups"),
        ("codec_language_id", {"english": True}, "language ID"),
    ],
)
def test_qwen3_config_rejects_invalid_abi_values(field: str, value: Any, message: str) -> None:
    config = _config()
    config[field] = value
    with pytest.raises(RuntimeContractError, match=message):
        Qwen3Config.from_mapping(config)


def test_qwen3_embedding_store_checks_inventory_and_dimensions(tmp_path: Path) -> None:
    config_mapping = _config()
    config = Qwen3Config.from_mapping(config_mapping)
    arrays = _embedding_arrays(config_mapping)
    paths: dict[str, Path] = {}
    for component, array in arrays.items():
        path = tmp_path / f"{component}.npy"
        _save_array(path, array)
        paths[component] = path
    store = Qwen3EmbeddingStore.load(paths, config)
    assert set(store.arrays) == REQUIRED_EMBEDDINGS
    assert not store["text_embedding"].flags.writeable
    optional_paths = dict(paths)
    bias_path = tmp_path / "small_to_mtp_projection_bias.npy"
    weight_path = tmp_path / "small_to_mtp_projection_weight.npy"
    _save_array(bias_path, np.zeros((4,), np.float32))
    _save_array(weight_path, np.zeros((4, 4), np.float32))
    optional_paths["small_to_mtp_projection_bias"] = bias_path
    optional_paths["small_to_mtp_projection_weight"] = weight_path
    assert (
        "small_to_mtp_projection_weight" in Qwen3EmbeddingStore.load(optional_paths, config).arrays
    )
    with pytest.raises(RuntimeContractError, match="as a pair"):
        Qwen3EmbeddingStore.load({**paths, "small_to_mtp_projection_bias": bias_path}, config)

    with pytest.raises(RuntimeContractError, match="missing"):
        Qwen3EmbeddingStore.load(
            {key: value for key, value in paths.items() if key != "text_embedding"}, config
        )

    bad_paths = dict(paths)
    bad_path = tmp_path / "bad.npy"
    _save_array(bad_path, np.zeros((3, 5), np.float32))
    bad_paths["text_projection_fc1_weight"] = bad_path
    with pytest.raises(RuntimeContractError, match="first text projection weight"):
        Qwen3EmbeddingStore.load(bad_paths, config)


def test_qwen3_bundle_load_checks_config_type_and_quality(tmp_path: Path) -> None:
    raw = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    paths: dict[str, Path] = {}
    for component, array in _embedding_arrays(raw).items():
        path = tmp_path / "embeddings" / f"{component}.npy"
        _save_array(path, array)
        paths[component] = path
    components = (
        "talker_prefill",
        "talker_decode",
        "code_predictor",
        "vocoder",
        "speaker_encoder",
        "tokenizer_encoder",
    )
    model_paths = {component: tmp_path / f"{component}.onnx" for component in components}
    data_paths = {component: tmp_path / f"{component}.onnx.data" for component in components}
    bundle = Qwen3Bundle.load(
        config_path=config_path,
        embedding_paths=paths,
        model_paths=model_paths,
        external_data_paths=data_paths,
        expected_model_type="base",
        quality="int4",
    )
    assert bundle.config.model_type == "base"
    assert bundle.model_paths["talker_prefill"].name == "talker_prefill.onnx"
    with pytest.raises(RuntimeContractError, match="model type"):
        Qwen3Bundle.load(
            config_path=config_path,
            embedding_paths=paths,
            model_paths={},
            external_data_paths={},
            expected_model_type="voice_design",
            quality="int4",
        )


@pytest.mark.parametrize(
    "component",
    [
        "talker_prefill",
        "talker_decode",
        "code_predictor",
        "vocoder",
        "speaker_encoder",
        "tokenizer_encoder",
    ],
)
def test_qwen3_graph_contract_accepts_wavekat_specs(component: str) -> None:
    config = Qwen3Config.from_mapping(_config())
    inputs, outputs = _graph_specs(component, config)
    validate_graph_abi(component, config, inputs, outputs)


@pytest.mark.parametrize("component", ("talker_decode", "code_predictor"))
def test_cache_graphs_require_present_cache_output_names(component: str) -> None:
    config = Qwen3Config.from_mapping(_config())
    inputs, outputs = _graph_specs(component, config)
    expected_names = (
        ("logits", "hidden_states", "present_keys", "present_values")
        if component == "talker_decode"
        else ("logits", "present_keys", "present_values")
    )

    assert tuple(spec.name for spec in outputs) == expected_names
    invalid_outputs = tuple(
        _tensor(
            {"present_keys": "past_keys", "present_values": "past_values"}.get(
                spec.name, spec.name
            ),
            spec.ort_type,
            *spec.shape,
        )
        for spec in outputs
    )
    with pytest.raises(RuntimeContractError, match="names mismatch"):
        validate_graph_abi(component, config, inputs, invalid_outputs)


def test_qwen3_graph_contract_rejects_names_dtypes_ranks_and_cache_layout() -> None:
    config = Qwen3Config.from_mapping(_config())
    inputs, outputs = _graph_specs("talker_decode", config)
    bad_inputs = tuple(
        _tensor(spec.name, "tensor(float16)", *spec.shape) if spec.name == "past_keys" else spec
        for spec in inputs
    )
    with pytest.raises(RuntimeContractError, match="must have type"):
        validate_graph_abi("talker_decode", config, bad_inputs, outputs)

    bad_cache = tuple(
        _tensor(spec.name, spec.ort_type, 3, *spec.shape[1:]) if spec.name == "past_keys" else spec
        for spec in inputs
    )
    with pytest.raises(RuntimeContractError, match="dimension 0"):
        validate_graph_abi("talker_decode", config, bad_cache, outputs)

    with pytest.raises(RuntimeContractError, match="names mismatch"):
        validate_graph_abi("talker_decode", config, inputs[:-1], outputs)

    prefill_inputs, prefill_outputs = _graph_specs("talker_prefill", config)
    with pytest.raises(RuntimeContractError, match="KV tensors"):
        validate_graph_abi("talker_prefill", config, prefill_inputs, prefill_outputs[:-1])


def test_qwen3_graph_contract_rejects_invalid_audio_abi() -> None:
    config = Qwen3Config.from_mapping(_config())
    _inputs, outputs = _graph_specs("vocoder", config)
    bad_codes = (_tensor("codes", "tensor(int32)", 1, CODE_GROUPS, "frames"),)
    with pytest.raises(RuntimeContractError, match="must have type"):
        validate_graph_abi("vocoder", config, bad_codes, outputs)

    tokenizer_inputs, _tokenizer_outputs = _graph_specs("tokenizer_encoder", config)
    wrong_codes = (_tensor("codes", "tensor(int64)", 1, 15, "frames"),)
    with pytest.raises(RuntimeContractError, match="dimension 1"):
        validate_graph_abi("tokenizer_encoder", config, tokenizer_inputs, wrong_codes)


class _FakeSession:
    def __init__(
        self,
        model: Path,
        *,
        component: str,
        providers: Any,
        provider_options: Any = None,
        session_options: Any = None,
        strict_providers: bool = False,
        active_override: tuple[str, ...] | None = None,
        resolved_override: tuple[str, ...] | None = None,
    ) -> None:
        del provider_options, session_options, strict_providers
        self.model = model
        self.component = component
        config_path = next(
            parent / "config.json" for parent in model.parents if (parent / "config.json").is_file()
        )
        config = Qwen3Config.from_path(config_path)
        self.input_specs, self.output_specs = _graph_specs(component, config)
        self.resolved_providers = resolved_override or (
            ("CPUExecutionProvider",) if providers == "cpu" else ("CUDAExecutionProvider",)
        )
        self.active_providers = active_override or self.resolved_providers
        self.closed = 0

    def diagnostics(self) -> SessionDiagnostic:
        return SessionDiagnostic(
            component=self.component,
            model_path=self.model,
            providers_requested=self.resolved_providers,
            providers_active=self.active_providers,
            inputs=tuple(spec.name for spec in self.input_specs),
            outputs=tuple(spec.name for spec in self.output_specs),
        )

    def close(self) -> None:
        self.closed += 1


def test_qwen3_sessions_are_lazy_validated_diagnosed_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[_FakeSession] = []

    def factory(model: Path, **kwargs: Any) -> _FakeSession:
        session = _FakeSession(model, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(qwen3_module, "OnnxSession", factory)
    adapter = Qwen3Adapter(_write_installation(tmp_path), providers="cpu")
    assert created == []
    before = adapter.diagnostics()
    assert before.sessions == ()
    assert created == []

    adapter._ensure_sessions()
    assert len(created) == 4
    diagnostic: RuntimeDiagnostic = adapter.diagnostics()
    assert {session.component for session in diagnostic.sessions} == {
        "talker_prefill",
        "talker_decode",
        "code_predictor",
        "vocoder",
    }
    assert diagnostic.properties["model_type"] == "voice_design"
    assert diagnostic.properties["quality"] == "int4"
    assert diagnostic.properties["talker_hidden_size"] == 8
    assert diagnostic.properties["providers_active"] == ("CPUExecutionProvider",)
    assert "reference_audio" not in diagnostic.properties

    adapter.close()
    assert all(session.closed == 1 for session in created)
    assert adapter._sessions == {}
    adapter.close()
    assert all(session.closed == 1 for session in created)


def test_qwen3_base_sessions_include_both_encoders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []

    def factory(model: Path, **kwargs: Any) -> _FakeSession:
        session = _FakeSession(model, **kwargs)
        created.append(session.component)
        return session

    monkeypatch.setattr(qwen3_module, "OnnxSession", factory)
    adapter = Qwen3Adapter(_write_installation(tmp_path, "base"), providers="cpu")
    adapter._ensure_sessions()
    assert created == [
        "speaker_encoder",
        "tokenizer_encoder",
        "talker_prefill",
        "talker_decode",
        "code_predictor",
        "vocoder",
    ]
    adapter.close()


def test_qwen3_rejects_provider_fallback_and_partial_provider_pipelines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[_FakeSession] = []

    def fallback_factory(model: Path, **kwargs: Any) -> _FakeSession:
        session = _FakeSession(
            model,
            **kwargs,
            active_override=("CPUExecutionProvider",),
            resolved_override=("CUDAExecutionProvider",),
        )
        created.append(session)
        return session

    monkeypatch.setattr(qwen3_module, "OnnxSession", fallback_factory)
    adapter = Qwen3Adapter(_write_installation(tmp_path / "fallback"), providers="cuda")
    with pytest.raises(RuntimeContractError, match="did not activate any requested provider"):
        adapter._ensure_sessions()
    assert adapter._sessions == {}
    assert created[0].closed == 1

    created = []
    count = 0

    def inconsistent_factory(model: Path, **kwargs: Any) -> _FakeSession:
        nonlocal count
        count += 1
        provider = "CPUExecutionProvider" if count == 1 else "CUDAExecutionProvider"
        session = _FakeSession(
            model,
            **kwargs,
            active_override=(provider,),
            resolved_override=(provider,),
        )
        created.append(session)
        return session

    monkeypatch.setattr(qwen3_module, "OnnxSession", inconsistent_factory)
    adapter = Qwen3Adapter(_write_installation(tmp_path / "mixed"), providers="cpu")
    with pytest.raises(RuntimeContractError, match="inconsistent provider policies"):
        adapter._ensure_sessions()
    assert adapter._sessions == {}
    assert len(created) == 2
    assert all(session.closed == 1 for session in created)
