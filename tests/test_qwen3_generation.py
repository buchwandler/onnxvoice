from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.qwen3 import Qwen3Adapter
from onnxvoice.systems.qwen3_audio import Qwen3VoiceState
from onnxvoice.systems.qwen3_contract import CODE_GROUPS, Qwen3Config, Qwen3EmbeddingStore
from onnxvoice.systems.qwen3_generation import (
    build_voice_clone_prefill,
    build_voice_design_prefill,
    generate_codes,
    vocode_codes,
    voice_design_result,
)
from onnxvoice.systems.qwen3_sampling import (
    Qwen3GenerationConfig,
    mask_talker_logits,
    sample_logits,
)


def _config() -> Qwen3Config:
    return Qwen3Config.from_mapping(
        {
            "tts_model_type": "voice_design",
            "model_id": "fixture/voice-design",
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
            "codec_language_id": {"english": 12},
            "talker_hidden_size": 4,
            "talker_num_layers": 2,
            "talker_num_attention_heads": 2,
            "talker_num_kv_heads": 1,
            "talker_head_dim": 4,
            "talker_vocab_size": 1100,
            "talker_text_hidden_size": 3,
            "talker_num_code_groups": CODE_GROUPS,
            "cp_hidden_size": 4,
            "cp_num_layers": 2,
            "cp_num_attention_heads": 2,
            "cp_num_kv_heads": 1,
            "cp_head_dim": 4,
            "cp_vocab_size": 8,
            "cp_num_code_groups": CODE_GROUPS,
        }
    )


def _embeddings() -> Qwen3EmbeddingStore:
    config = _config()
    arrays: dict[str, np.ndarray] = {
        "text_embedding": np.zeros((64, config.talker_text_hidden_size), np.float32),
        "text_projection_fc1_weight": np.zeros((5, config.talker_text_hidden_size), np.float32),
        "text_projection_fc1_bias": np.zeros((5,), np.float32),
        "text_projection_fc2_weight": np.zeros((config.talker_hidden_size, 5), np.float32),
        "text_projection_fc2_bias": np.zeros((config.talker_hidden_size,), np.float32),
        "talker_codec_embedding": np.repeat(
            np.arange(config.talker_vocab_size, dtype=np.float32)[:, None],
            config.talker_hidden_size,
            axis=1,
        ),
    }
    for index in range(CODE_GROUPS - 1):
        arrays[f"cp_codec_embedding_{index}"] = np.repeat(
            np.arange(config.cp_vocab_size, dtype=np.float32)[:, None],
            config.talker_hidden_size,
            axis=1,
        )
    return Qwen3EmbeddingStore(MappingProxyType(arrays))


_TEXT_IDS = [0, 20, 21, 30, 31, 1, 22, 0, 23, 22]
_INSTRUCTION_IDS = [0, 24, 25, 26, 1, 22]


class _FakeTalkerPrefill:
    def __init__(self, config: Qwen3Config) -> None:
        self.config = config
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(inputs)
        length = inputs["inputs_embeds"].shape[1]
        logits = np.full((1, length, self.config.talker_vocab_size), -20, np.float32)
        logits[0, -1, 20] = 5
        logits[0, -1, self.config.codec_eos_token_id] = 9
        logits[0, -1, -1] = 100
        hidden = np.ones((1, length, self.config.talker_hidden_size), np.float32)
        key = np.ones(
            (1, self.config.talker_num_kv_heads, length, self.config.talker_head_dim), np.float32
        )
        value = np.full_like(key, 2)
        outputs: list[np.ndarray] = [logits, hidden]
        for _ in range(self.config.talker_num_layers):
            outputs.extend((key.copy(), value.copy()))
        return outputs


class _FakeTalkerDecode:
    def __init__(self, config: Qwen3Config) -> None:
        self.config = config
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(inputs)
        config = self.config
        length = inputs["attention_mask"].shape[1]
        logits = np.full((1, 1, config.talker_vocab_size), -20, np.float32)
        logits[0, 0, 20] = 5
        logits[0, 0, config.codec_eos_token_id] = 9
        logits[0, 0, -1] = 100
        hidden = np.full((1, 1, config.talker_hidden_size), 2, np.float32)
        extra_key = np.full(
            (config.talker_num_layers, 1, config.talker_num_kv_heads, 1, config.talker_head_dim),
            3,
            np.float32,
        )
        extra_value = np.full_like(extra_key, 4)
        keys = np.concatenate([inputs["past_keys"], extra_key], axis=3)
        values = np.concatenate([inputs["past_values"], extra_value], axis=3)
        assert keys.shape[3] == length
        return [logits, hidden, keys, values]


class _FakeCodePredictor:
    def __init__(self, config: Qwen3Config) -> None:
        self.config = config
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(inputs)
        config = self.config
        input_length = inputs["inputs_embeds"].shape[1]
        past_length = inputs["past_keys"].shape[3]
        logits = np.full((1, input_length, config.cp_vocab_size), -10, np.float32)
        logits[0, -1, 3] = 10
        extra_shape = (
            config.cp_num_layers,
            1,
            config.cp_num_kv_heads,
            input_length,
            config.cp_head_dim,
        )
        extra_key = np.ones(extra_shape, np.float32)
        extra_value = np.full(extra_shape, 2, np.float32)
        keys = np.concatenate([inputs["past_keys"], extra_key], axis=3)
        values = np.concatenate([inputs["past_values"], extra_value], axis=3)
        assert keys.shape[3] == past_length + input_length
        return [logits, keys, values]


class _FakeVocoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(inputs)
        return [np.ones((1, 1, 480), np.float32)]


def _sessions(
    config: Qwen3Config,
) -> tuple[_FakeTalkerPrefill, _FakeTalkerDecode, _FakeCodePredictor]:
    return _FakeTalkerPrefill(config), _FakeTalkerDecode(config), _FakeCodePredictor(config)


def test_voice_design_prefill_uses_wrapped_ids_and_wavekat_embedding_order() -> None:
    config = _config()
    embeddings = _embeddings()
    prefill = build_voice_design_prefill(
        config, embeddings, _TEXT_IDS, _INSTRUCTION_IDS, language="english"
    )
    assert prefill.shape == (1, 18, config.talker_hidden_size)
    np.testing.assert_array_equal(prefill[0, 9], np.full((4,), 6, np.float32))
    np.testing.assert_array_equal(prefill[0, 10], np.full((4,), 8, np.float32))
    np.testing.assert_array_equal(prefill[0, 11], np.full((4,), 12, np.float32))
    np.testing.assert_array_equal(prefill[0, 12], np.full((4,), 9, np.float32))
    np.testing.assert_array_equal(prefill[0, 13], np.full((4,), 10, np.float32))
    np.testing.assert_array_equal(prefill[0, 14], np.full((4,), 10, np.float32))
    np.testing.assert_array_equal(prefill[0, 15], np.full((4,), 10, np.float32))
    np.testing.assert_array_equal(prefill[0, 16], np.full((4,), 10, np.float32))
    np.testing.assert_array_equal(prefill[0, 17], np.full((4,), 11, np.float32))

    unknown = build_voice_design_prefill(
        config, embeddings, _TEXT_IDS, _INSTRUCTION_IDS, language="unknown"
    )
    assert unknown.shape[1] == prefill.shape[1] - 1
    np.testing.assert_array_equal(unknown[0, 9], np.full((4,), 7, np.float32))


def test_voice_design_prefill_rejects_unwrapped_or_out_of_range_ids() -> None:
    config = _config()
    embeddings = _embeddings()
    with pytest.raises(RuntimeContractError, match="wrapped chat format"):
        build_voice_design_prefill(
            config, embeddings, [30, *_TEXT_IDS[1:]], _INSTRUCTION_IDS, language="english"
        )
    with pytest.raises(RuntimeContractError, match="outside the installed text embedding"):
        build_voice_design_prefill(
            config,
            embeddings,
            [0, 20, 21, 99, 31, 1, 22, 0, 23, 22],
            _INSTRUCTION_IDS,
            language="english",
        )


def test_wavekat_sampler_is_seeded_and_masks_reserved_tokens_eos_and_repetition() -> None:
    logits = np.arange(1100, dtype=np.float32)
    masked = mask_talker_logits(
        logits,
        eos_token_id=5,
        generated_group0=[],
        frame_index=0,
        repetition_penalty=1.05,
    )
    assert np.isneginf(masked[1099])
    assert np.isneginf(masked[5])
    assert np.isfinite(masked[4])

    eos_allowed = mask_talker_logits(
        logits, eos_token_id=5, generated_group0=[], frame_index=2, repetition_penalty=1.05
    )
    assert eos_allowed[5] == logits[5]
    penalized = mask_talker_logits(
        logits, eos_token_id=5, generated_group0=[4], frame_index=2, repetition_penalty=2.0
    )
    assert penalized[4] == logits[4] / 2.0

    sample_config = Qwen3GenerationConfig(temperature=0.8, top_k=5, seed=42)
    first = [sample_logits(logits, sample_config, np.random.default_rng(42)) for _ in range(1)]
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    sample_a = [sample_logits(logits, sample_config, rng_a) for _ in range(20)]
    sample_b = [sample_logits(logits, sample_config, rng_b) for _ in range(20)]
    assert first == sample_a[:1]
    assert sample_a == sample_b
    assert set(sample_a) <= set(range(1095, 1100))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"top_k": -1},
        {"repetition_penalty": 0},
        {"max_new_tokens": 0},
        {"seed": -1},
    ],
)
def test_qwen3_generation_config_rejects_invalid_sampling_controls(kwargs: dict[str, Any]) -> None:
    with pytest.raises(RuntimeContractError):
        Qwen3GenerationConfig(**kwargs)


def test_generation_runs_one_prefill_fifteen_predictor_steps_per_frame_and_propagates_kv() -> None:
    config = _config()
    embeddings = _embeddings()
    prefill_session, decode_session, predictor_session = _sessions(config)
    generated = generate_codes(
        config=config,
        embeddings=embeddings,
        prefill_session=prefill_session,
        decode_session=decode_session,
        predictor_session=predictor_session,
        inputs_embeds=np.zeros((1, 5, config.talker_hidden_size), np.float32),
        trailing_hidden=np.zeros((1, config.talker_hidden_size), np.float32),
        generation=Qwen3GenerationConfig(top_k=1, max_new_tokens=4, seed=7),
    )
    assert generated.eos_observed
    assert generated.codes.shape == (2, CODE_GROUPS)
    assert np.all(generated.codes[:, 0] == 20)
    assert np.all(generated.codes[:, 1:] == 3)
    assert len(prefill_session.calls) == 1
    assert len(decode_session.calls) == 2
    assert len(predictor_session.calls) == 2 * (CODE_GROUPS - 1)
    assert [int(call["generation_steps"][0]) for call in predictor_session.calls] == list(
        range(15)
    ) * 2
    expected_cache_lengths = [0, *range(2, 16)] * 2
    assert [
        call["past_keys"].shape[3] for call in predictor_session.calls
    ] == expected_cache_lengths
    assert [
        call["past_values"].shape[3] for call in predictor_session.calls
    ] == expected_cache_lengths
    np.testing.assert_array_equal(
        decode_session.calls[0]["past_keys"][0, 0, 0, :5, 0], np.ones((5,), np.float32)
    )
    assert decode_session.calls[1]["past_keys"].shape[3] == 6


def test_voice_design_result_returns_finite_float32_24khz_audio_and_sixteen_code_frames() -> None:
    config = _config()
    embeddings = _embeddings()
    prefill_session, decode_session, predictor_session = _sessions(config)
    vocoder = _FakeVocoder()
    result = voice_design_result(
        config=config,
        embeddings=embeddings,
        sessions={
            "talker_prefill": prefill_session,
            "talker_decode": decode_session,
            "code_predictor": predictor_session,
            "vocoder": vocoder,
        },
        token_ids=_TEXT_IDS,
        instruction_token_ids=_INSTRUCTION_IDS,
        language="english",
        generation=Qwen3GenerationConfig(top_k=1, max_new_tokens=2, seed=10),
        model_id="fixture/voice-design",
        quality="int4",
        providers_active=("CPUExecutionProvider",),
    )
    assert result.sample_rate == 24_000
    assert result.audio.dtype == np.float32
    assert result.audio.shape == (480,)
    assert np.isfinite(result.audio).all()
    assert result.outputs["codes"].shape == (2, CODE_GROUPS)
    assert result.metadata["frames_generated"] == 2
    assert result.metadata["providers_active"] == ("CPUExecutionProvider",)
    assert len(vocoder.calls) == 1
    assert vocoder.calls[0]["codes"].shape == (1, CODE_GROUPS, 2)


def test_vocoder_rejects_empty_nonfinite_or_malformed_output() -> None:
    codes = np.zeros((1, CODE_GROUPS), np.int64)

    class BadVocoder:
        def __init__(self, output: np.ndarray) -> None:
            self.output = output

        def run(self, _inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            return [self.output]

    with pytest.raises(RuntimeContractError, match="non-finite"):
        vocode_codes(BadVocoder(np.asarray([np.nan], np.float32)), codes)
    with pytest.raises(RuntimeContractError, match="empty"):
        vocode_codes(BadVocoder(np.empty((1, 0), np.float32)), codes)

    class NoOutput:
        def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            del self, inputs
            return []

    with pytest.raises(RuntimeContractError, match="no outputs"):
        vocode_codes(NoOutput(), codes)
    with pytest.raises(RuntimeContractError, match=r"\[frames, 16\]"):
        vocode_codes(BadVocoder(np.ones((1,), np.float32)), np.zeros((2, 15), np.int64))


def test_qwen3_adapter_infer_uses_frontend_tokens_and_returns_runtime_result() -> None:
    config = _config()
    prefill_session, decode_session, predictor_session = _sessions(config)
    vocoder = _FakeVocoder()
    adapter = cast(Any, Qwen3Adapter.__new__(Qwen3Adapter))
    adapter.model_type = "voice_design"
    adapter.quality = "int4"
    adapter.config = config
    adapter.embeddings = _embeddings()
    adapter.installation = SimpleNamespace(id="fixture/voice-design")
    adapter._sessions = {
        "talker_prefill": prefill_session,
        "talker_decode": decode_session,
        "code_predictor": predictor_session,
        "vocoder": vocoder,
    }
    adapter._ensure_sessions = lambda: None
    result = adapter.infer(
        _TEXT_IDS,
        instruction_token_ids=_INSTRUCTION_IDS,
        language="english",
        top_k=1,
        max_new_tokens=1,
        seed=11,
    )
    assert result.sample_rate == 24_000
    assert result.metadata["layout"] == "qwen3-wavekat-v1"
    assert result.metadata["seed"] == 11
    assert result.audio.dtype == np.float32


def test_base_prefill_prepends_reference_codes_and_trims_vocoded_reference_region() -> None:
    config = replace(_config(), model_type="base")
    embeddings = _embeddings()
    voice = Qwen3VoiceState(
        speaker_embedding=np.full((1, config.talker_hidden_size), 0.5, np.float32),
        reference_codes=np.ones((1, CODE_GROUPS, 2), np.int64),
        reference_token_ids=tuple(_INSTRUCTION_IDS),
        sample_rate=24_000,
        reference_samples=1921,
        model_id=config.model_id,
    )
    prefill = build_voice_clone_prefill(config, embeddings, _TEXT_IDS, voice, language="english")
    assert prefill.shape == (1, 16, config.talker_hidden_size)
    np.testing.assert_array_equal(prefill[0, 3:7, 0], np.asarray([6, 8, 12, 9], np.float32))
    np.testing.assert_array_equal(prefill[0, 7], np.full((4,), 0.5, np.float32))
    np.testing.assert_array_equal(prefill[0, 8:13], np.full((5, 4), 10, np.float32))
    np.testing.assert_array_equal(prefill[0, 13], np.full((4,), 11, np.float32))
    np.testing.assert_array_equal(prefill[0, 14:16], np.full((2, 4), 16, np.float32))

    prefill_session, decode_session, predictor_session = _sessions(config)
    vocoder = _FakeVocoder()
    adapter = cast(Any, Qwen3Adapter.__new__(Qwen3Adapter))
    adapter.model_type = "base"
    adapter.quality = "int4"
    adapter.config = config
    adapter.embeddings = embeddings
    adapter.installation = SimpleNamespace(id=config.model_id)
    adapter._sessions = {
        "talker_prefill": prefill_session,
        "talker_decode": decode_session,
        "code_predictor": predictor_session,
        "vocoder": vocoder,
    }
    adapter._ensure_sessions = lambda: None
    result = adapter.infer(
        _TEXT_IDS,
        voice_state=voice,
        language="english",
        top_k=1,
        max_new_tokens=1,
        seed=17,
    )
    assert result.sample_rate == 24_000
    assert result.audio.dtype == np.float32
    assert result.audio.shape == (160,)
    assert np.isfinite(result.audio).all()
    assert result.outputs["codes"].shape == (1, CODE_GROUPS)
    assert result.metadata["reference_frames"] == 2
    assert vocoder.calls[0]["codes"].shape == (1, CODE_GROUPS, 3)
    np.testing.assert_array_equal(
        vocoder.calls[0]["codes"][0, :, :2], np.ones((CODE_GROUPS, 2), np.int64)
    )
    np.testing.assert_array_equal(vocoder.calls[0]["codes"][0, :, 2], np.asarray([20, *([3] * 15)]))
