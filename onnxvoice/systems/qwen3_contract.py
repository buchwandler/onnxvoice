from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..types import TensorSpec

LAYOUT = "qwen3-wavekat-v1"
SAMPLE_RATE = 24_000
CODE_GROUPS = 16

REQUIRED_EMBEDDINGS = frozenset(
    {
        "text_embedding",
        "text_projection_fc1_weight",
        "text_projection_fc1_bias",
        "text_projection_fc2_weight",
        "text_projection_fc2_bias",
        "talker_codec_embedding",
        *(f"cp_codec_embedding_{index}" for index in range(CODE_GROUPS - 1)),
    }
)
OPTIONAL_EMBEDDINGS = frozenset({"small_to_mtp_projection_bias", "small_to_mtp_projection_weight"})
_TOKEN_IDS = (
    "im_start_token_id",
    "im_end_token_id",
    "tts_pad_token_id",
    "tts_bos_token_id",
    "tts_eos_token_id",
    "codec_eos_token_id",
    "codec_think_id",
    "codec_nothink_id",
    "codec_think_bos_id",
    "codec_think_eos_id",
    "codec_pad_id",
    "codec_bos_id",
)
_DIMENSIONS = (
    "talker_hidden_size",
    "talker_num_layers",
    "talker_num_attention_heads",
    "talker_num_kv_heads",
    "talker_head_dim",
    "talker_vocab_size",
    "talker_text_hidden_size",
    "talker_num_code_groups",
    "cp_hidden_size",
    "cp_num_layers",
    "cp_num_attention_heads",
    "cp_num_kv_heads",
    "cp_head_dim",
    "cp_vocab_size",
    "cp_num_code_groups",
)


def _integer(config: Mapping[str, Any], name: str, *, positive: bool = True) -> int:
    value = config.get(name)
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise RuntimeContractError(f"Qwen3 config field {name!r} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Validated WaveKat config values used by the runtime ABI."""

    model_type: str
    model_id: str
    sample_rate: int
    talker_hidden_size: int
    talker_num_layers: int
    talker_num_attention_heads: int
    talker_num_kv_heads: int
    talker_head_dim: int
    talker_vocab_size: int
    talker_text_hidden_size: int
    talker_num_code_groups: int
    cp_hidden_size: int
    cp_num_layers: int
    cp_num_attention_heads: int
    cp_num_kv_heads: int
    cp_head_dim: int
    cp_vocab_size: int
    cp_num_code_groups: int
    im_start_token_id: int
    im_end_token_id: int
    tts_pad_token_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    codec_eos_token_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_pad_id: int
    codec_bos_id: int
    codec_language_id: Mapping[str, int]
    max_position_embeddings: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Qwen3Config:
        config_path = Path(path)
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(f"Could not read Qwen3 config {config_path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise RuntimeContractError("Qwen3 config must contain a JSON object")
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Qwen3Config:
        model_type = value.get("tts_model_type")
        if not isinstance(model_type, str) or model_type not in {"base", "voice_design"}:
            raise RuntimeContractError(
                "Qwen3 config tts_model_type must be 'base' or 'voice_design'"
            )
        sample_rate = _integer(value, "sample_rate")
        if sample_rate != SAMPLE_RATE:
            raise RuntimeContractError(f"Qwen3 WaveKat config requires sample_rate={SAMPLE_RATE}")
        dimensions = {name: _integer(value, name) for name in _DIMENSIONS}
        if dimensions["talker_num_code_groups"] != CODE_GROUPS:
            raise RuntimeContractError(f"WaveKat ABI requires {CODE_GROUPS} talker code groups")
        if dimensions["cp_num_code_groups"] != CODE_GROUPS:
            raise RuntimeContractError(f"WaveKat ABI requires {CODE_GROUPS} predictor code groups")
        if dimensions["talker_num_kv_heads"] > dimensions["talker_num_attention_heads"]:
            raise RuntimeContractError(
                "Qwen3 talker KV-head count exceeds its attention-head count"
            )
        if dimensions["cp_num_kv_heads"] > dimensions["cp_num_attention_heads"]:
            raise RuntimeContractError(
                "Qwen3 predictor KV-head count exceeds its attention-head count"
            )

        token_ids = {name: _integer(value, name, positive=False) for name in _TOKEN_IDS}
        codec_token_names = tuple(name for name in _TOKEN_IDS if name.startswith("codec_"))
        if any(token_ids[name] >= dimensions["talker_vocab_size"] for name in codec_token_names):
            raise RuntimeContractError("Qwen3 codec token ID is outside the talker vocabulary")
        language_ids_raw = value.get("codec_language_id")
        if not isinstance(language_ids_raw, Mapping):
            raise RuntimeContractError("Qwen3 config codec_language_id must be an object")
        language_ids: dict[str, int] = {}
        for language, language_id in language_ids_raw.items():
            if not isinstance(language, str) or not language:
                raise RuntimeContractError("Qwen3 codec language names must be non-empty strings")
            if isinstance(language_id, bool) or not isinstance(language_id, int) or language_id < 0:
                raise RuntimeContractError(
                    f"Qwen3 codec language ID for {language!r} must be non-negative"
                )
            if language_id >= dimensions["talker_vocab_size"]:
                raise RuntimeContractError(
                    f"Qwen3 codec language ID for {language!r} is outside the vocabulary"
                )
            language_ids[language] = language_id

        model_id = value.get("model_id", "")
        if not isinstance(model_id, str):
            raise RuntimeContractError("Qwen3 config model_id must be a string")
        max_positions = value.get("max_position_embeddings")
        if max_positions is not None and (
            isinstance(max_positions, bool)
            or not isinstance(max_positions, int)
            or max_positions <= 0
        ):
            raise RuntimeContractError("Qwen3 max_position_embeddings must be a positive integer")

        return cls(
            model_type=model_type,
            model_id=model_id,
            sample_rate=sample_rate,
            **dimensions,
            **token_ids,
            codec_language_id=MappingProxyType(language_ids),
            max_position_embeddings=max_positions,
        )

    @property
    def codec_token_ids(self) -> tuple[int, ...]:
        return (
            self.codec_eos_token_id,
            self.codec_think_id,
            self.codec_nothink_id,
            self.codec_think_bos_id,
            self.codec_think_eos_id,
            self.codec_pad_id,
            self.codec_bos_id,
            *self.codec_language_id.values(),
        )

    @property
    def text_token_ids(self) -> tuple[int, ...]:
        return (
            self.im_start_token_id,
            self.im_end_token_id,
            self.tts_pad_token_id,
            self.tts_bos_token_id,
            self.tts_eos_token_id,
        )


@dataclass(frozen=True, slots=True)
class Qwen3EmbeddingStore:
    """Read-only NumPy arrays loaded and shape-checked once per adapter."""

    arrays: Mapping[str, np.ndarray]

    def __getitem__(self, component: str) -> np.ndarray:
        return self.arrays[component]

    @classmethod
    def load(
        cls,
        paths: Mapping[str, str | Path],
        config: Qwen3Config,
    ) -> Qwen3EmbeddingStore:
        if any(not isinstance(component, str) or not component for component in paths):
            raise RuntimeContractError("Qwen3 embedding names must be non-empty strings")
        actual = set(paths)
        missing = REQUIRED_EMBEDDINGS - actual
        extra = actual - REQUIRED_EMBEDDINGS - OPTIONAL_EMBEDDINGS
        if missing or extra:
            raise RuntimeContractError(
                f"Qwen3 embedding inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        optional_present = actual & OPTIONAL_EMBEDDINGS
        if optional_present and optional_present != OPTIONAL_EMBEDDINGS:
            raise RuntimeContractError(
                "Qwen3 optional small-to-MTP embeddings must be supplied as a pair"
            )

        arrays: dict[str, np.ndarray] = {}
        for component, path in paths.items():
            try:
                array = np.load(Path(path), mmap_mode="r", allow_pickle=False)
            except (OSError, TypeError, ValueError, EOFError) as exc:
                raise RuntimeContractError(
                    f"Could not load Qwen3 embedding {component!r}: {exc}"
                ) from exc
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.floating)
            ):
                raise RuntimeContractError(
                    f"Qwen3 embedding {component!r} must be a floating NumPy array"
                )
            arrays[component] = array

        text = arrays["text_embedding"]
        fc1_weight = arrays["text_projection_fc1_weight"]
        fc1_bias = arrays["text_projection_fc1_bias"]
        fc2_weight = arrays["text_projection_fc2_weight"]
        fc2_bias = arrays["text_projection_fc2_bias"]
        codec = arrays["talker_codec_embedding"]
        if text.ndim != 2 or text.shape[1] != config.talker_text_hidden_size:
            raise RuntimeContractError("Qwen3 text_embedding dimensions disagree with config")
        if fc1_weight.ndim != 2 or fc1_weight.shape[1] != text.shape[1]:
            raise RuntimeContractError("Qwen3 first text projection weight has an invalid shape")
        if fc1_bias.shape != (fc1_weight.shape[0],):
            raise RuntimeContractError("Qwen3 first text projection bias has an invalid shape")
        if fc2_weight.ndim != 2 or fc2_weight.shape != (
            config.talker_hidden_size,
            fc1_weight.shape[0],
        ):
            raise RuntimeContractError("Qwen3 second text projection weight has an invalid shape")
        if fc2_bias.shape != (config.talker_hidden_size,):
            raise RuntimeContractError("Qwen3 second text projection bias has an invalid shape")
        if text.shape[0] <= max(config.text_token_ids):
            raise RuntimeContractError(
                "Qwen3 text_embedding does not contain all configured text token IDs"
            )
        if codec.shape != (config.talker_vocab_size, config.talker_hidden_size):
            raise RuntimeContractError(
                "Qwen3 talker_codec_embedding dimensions disagree with config"
            )
        if max(config.codec_token_ids, default=0) >= codec.shape[0]:
            raise RuntimeContractError(
                "Qwen3 talker codec embedding does not contain all configured codec IDs"
            )
        for index in range(config.talker_num_code_groups - 1):
            embedding = arrays[f"cp_codec_embedding_{index}"]
            if embedding.shape != (config.cp_vocab_size, config.talker_hidden_size):
                raise RuntimeContractError(
                    f"Qwen3 predictor embedding {index} dimensions disagree with config"
                )
        for component in optional_present:
            if arrays[component].ndim == 0:
                raise RuntimeContractError(
                    f"Qwen3 optional embedding {component!r} must not be scalar"
                )
        return cls(MappingProxyType(arrays))


@dataclass(frozen=True, slots=True)
class Qwen3Bundle:
    """Validated config, embeddings, and graph paths for an installed WaveKat bundle."""

    config: Qwen3Config
    embeddings: Qwen3EmbeddingStore
    model_paths: Mapping[str, Path]
    external_data_paths: Mapping[str, Path]

    @classmethod
    def load(
        cls,
        *,
        config_path: str | Path,
        embedding_paths: Mapping[str, str | Path],
        model_paths: Mapping[str, str | Path],
        external_data_paths: Mapping[str, str | Path],
        expected_model_type: str,
        quality: str,
    ) -> Qwen3Bundle:
        config = Qwen3Config.from_path(config_path)
        if config.model_type != expected_model_type:
            raise RuntimeContractError("Qwen3 config model type conflicts with runtime metadata")
        if not isinstance(quality, str) or quality not in {"fp32", "int4"}:
            raise RuntimeContractError("Qwen3 bundle quality must be 'fp32' or 'int4'")
        expected_graphs = {"talker_prefill", "talker_decode", "code_predictor", "vocoder"}
        if config.model_type == "base":
            expected_graphs.update({"speaker_encoder", "tokenizer_encoder"})
        if set(model_paths) != expected_graphs or set(external_data_paths) != expected_graphs:
            raise RuntimeContractError(
                "Qwen3 bundle graph/data inventory does not match its model type"
            )
        for component, source in model_paths.items():
            graph = Path(source)
            data = Path(external_data_paths[component])
            if graph.name != f"{component}.onnx" or data.name != graph.name + ".data":
                raise RuntimeContractError(
                    f"Qwen3 bundle graph/data names disagree for {component!r}"
                )
            if data.resolve() != Path(str(graph.resolve()) + ".data"):
                raise RuntimeContractError(f"Qwen3 bundle data must be adjacent to {component!r}")
        embeddings = Qwen3EmbeddingStore.load(embedding_paths, config)
        return cls(
            config=config,
            embeddings=embeddings,
            model_paths=MappingProxyType({key: Path(path) for key, path in model_paths.items()}),
            external_data_paths=MappingProxyType(
                {key: Path(path) for key, path in external_data_paths.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class _TensorContract:
    dtype: str
    rank: int
    shape: tuple[int | None, ...]


def _validate_tensor(spec: TensorSpec, contract: _TensorContract, *, component: str) -> None:
    if spec.ort_type != contract.dtype:
        raise RuntimeContractError(
            f"Qwen3 {component} tensor {spec.name!r} must have type {contract.dtype}, got {spec.ort_type}"
        )
    if len(spec.shape) != contract.rank:
        raise RuntimeContractError(
            f"Qwen3 {component} tensor {spec.name!r} must have rank {contract.rank}, got {len(spec.shape)}"
        )
    for axis, expected in enumerate(contract.shape):
        actual = spec.shape[axis]
        if expected is not None and isinstance(actual, int) and actual != expected:
            raise RuntimeContractError(
                f"Qwen3 {component} tensor {spec.name!r} dimension {axis} must be {expected}, got {actual}"
            )


def _spec_map(
    specs: Sequence[TensorSpec], *, component: str, direction: str
) -> dict[str, TensorSpec]:
    result: dict[str, TensorSpec] = {}
    for spec in specs:
        if spec.name in result:
            raise RuntimeContractError(
                f"Qwen3 {component} graph has duplicate {direction} {spec.name!r}"
            )
        result[spec.name] = spec
    return result


def _validate_named_specs(
    specs: Sequence[TensorSpec],
    expected: Mapping[str, _TensorContract],
    *,
    component: str,
    direction: str,
) -> dict[str, TensorSpec]:
    actual = _spec_map(specs, component=component, direction=direction)
    if actual.keys() != expected.keys():
        raise RuntimeContractError(
            f"Qwen3 {component} {direction} names mismatch; expected={sorted(expected)}, got={sorted(actual)}"
        )
    if direction == "output" and tuple(spec.name for spec in specs) != tuple(expected):
        raise RuntimeContractError(f"Qwen3 {component} output order does not match the runtime ABI")
    for name, contract in expected.items():
        _validate_tensor(actual[name], contract, component=component)
    return actual


def validate_graph_abi(
    component: str,
    config: Qwen3Config,
    input_specs: Sequence[TensorSpec],
    output_specs: Sequence[TensorSpec],
) -> None:
    """Validate one WaveKat graph's names, ranks, dtypes, and config-derived dimensions."""
    f32 = "tensor(float)"
    i64 = "tensor(int64)"
    seq = None
    if component == "talker_prefill":
        inputs = {
            "inputs_embeds": _TensorContract(f32, 3, (1, seq, config.talker_hidden_size)),
            "attention_mask": _TensorContract(i64, 2, (1, seq)),
            "position_ids": _TensorContract(i64, 3, (3, 1, seq)),
        }
        _validate_named_specs(input_specs, inputs, component=component, direction="input")
        if len(output_specs) != 2 + 2 * config.talker_num_layers:
            raise RuntimeContractError(
                f"Qwen3 talker_prefill must return logits, hidden states, and {2 * config.talker_num_layers} KV tensors"
            )
        _spec_map(output_specs, component=component, direction="output")
        if [spec.name for spec in output_specs[:2]] != ["logits", "hidden_states"]:
            raise RuntimeContractError(
                "Qwen3 talker_prefill outputs must begin with logits and hidden_states"
            )
        _validate_tensor(
            output_specs[0],
            _TensorContract(f32, 3, (1, seq, config.talker_vocab_size)),
            component=component,
        )
        _validate_tensor(
            output_specs[1],
            _TensorContract(f32, 3, (1, seq, config.talker_hidden_size)),
            component=component,
        )
        kv = _TensorContract(f32, 4, (1, config.talker_num_kv_heads, seq, config.talker_head_dim))
        for spec in output_specs[2:]:
            _validate_tensor(spec, kv, component=component)
        return

    if component == "talker_decode":
        cache = (
            config.talker_num_layers,
            1,
            config.talker_num_kv_heads,
            seq,
            config.talker_head_dim,
        )
        inputs = {
            "inputs_embeds": _TensorContract(f32, 3, (1, 1, config.talker_hidden_size)),
            "attention_mask": _TensorContract(i64, 2, (1, seq)),
            "position_ids": _TensorContract(i64, 3, (3, 1, 1)),
            "past_keys": _TensorContract(f32, 5, cache),
            "past_values": _TensorContract(f32, 5, cache),
        }
        _validate_named_specs(input_specs, inputs, component=component, direction="input")
        outputs = {
            "logits": _TensorContract(f32, 3, (1, 1, config.talker_vocab_size)),
            "hidden_states": _TensorContract(f32, 3, (1, 1, config.talker_hidden_size)),
            "present_keys": _TensorContract(f32, 5, cache),
            "present_values": _TensorContract(f32, 5, cache),
        }
        _validate_named_specs(output_specs, outputs, component=component, direction="output")
        return

    if component == "code_predictor":
        cache = (config.cp_num_layers, 1, config.cp_num_kv_heads, seq, config.cp_head_dim)
        inputs = {
            "inputs_embeds": _TensorContract(f32, 3, (1, seq, config.talker_hidden_size)),
            "generation_steps": _TensorContract(i64, 1, (1,)),
            "past_keys": _TensorContract(f32, 5, cache),
            "past_values": _TensorContract(f32, 5, cache),
        }
        _validate_named_specs(input_specs, inputs, component=component, direction="input")
        outputs = {
            "logits": _TensorContract(f32, 3, (1, seq, config.cp_vocab_size)),
            "present_keys": _TensorContract(f32, 5, cache),
            "present_values": _TensorContract(f32, 5, cache),
        }
        _validate_named_specs(output_specs, outputs, component=component, direction="output")
        return

    if component == "vocoder":
        _validate_named_specs(
            input_specs,
            {"codes": _TensorContract(i64, 3, (1, CODE_GROUPS, seq))},
            component=component,
            direction="input",
        )
        if len(output_specs) != 1:
            raise RuntimeContractError("Qwen3 vocoder must expose exactly one waveform output")
        output = output_specs[0]
        if output.ort_type != f32 or len(output.shape) not in {1, 2, 3}:
            raise RuntimeContractError(
                "Qwen3 vocoder waveform output must be a rank-1/2/3 float tensor"
            )
        return

    if component == "speaker_encoder":
        _validate_named_specs(
            input_specs,
            {"mels": _TensorContract(f32, 3, (1, seq, 128))},
            component=component,
            direction="input",
        )
        _validate_named_specs(
            output_specs,
            {"speaker_embedding": _TensorContract(f32, 2, (1, config.talker_hidden_size))},
            component=component,
            direction="output",
        )
        return

    if component == "tokenizer_encoder":
        _validate_named_specs(
            input_specs,
            {"waveform": _TensorContract(f32, 2, (1, 240_000))},
            component=component,
            direction="input",
        )
        if len(output_specs) != 1:
            raise RuntimeContractError(
                "Qwen3 tokenizer_encoder must expose exactly one codec output"
            )
        _validate_tensor(
            output_specs[0], _TensorContract(i64, 3, (1, CODE_GROUPS, seq)), component=component
        )
        return

    raise RuntimeContractError(f"Unknown Qwen3 graph component {component!r}")
