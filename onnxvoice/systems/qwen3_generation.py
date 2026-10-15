from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..types import InferenceResult
from .qwen3_audio import (
    QWEN3_CODEC_SAMPLES_PER_FRAME,
    Qwen3VoiceState,
    validate_reference_token_ids,
)
from .qwen3_contract import CODE_GROUPS, SAMPLE_RATE, Qwen3Config, Qwen3EmbeddingStore
from .qwen3_sampling import Qwen3GenerationConfig, mask_talker_logits, sample_logits

# The supported tokenizer/exporter ABI passes these complete, chat-wrapped token sequences.
VOICE_TEXT_PREFIX_TOKENS = 3
VOICE_TEXT_SUFFIX_TOKENS = 5
RESERVED_TALKER_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class GeneratedCodes:
    """Autoregressively generated WaveKat codes in [frames, 16] layout."""

    codes: np.ndarray
    eos_observed: bool


def _token_ids(value: Sequence[int], *, name: str, rows: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise RuntimeContractError(f"Qwen3 {name} must be a one-dimensional integer sequence")
    try:
        tokens = np.asarray(list(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(
            f"Qwen3 {name} must be a one-dimensional integer sequence"
        ) from exc
    if tokens.ndim != 1 or tokens.size == 0 or not np.issubdtype(tokens.dtype, np.integer):
        raise RuntimeContractError(
            f"Qwen3 {name} must be a non-empty one-dimensional integer sequence"
        )
    tokens = tokens.astype(np.int64, copy=False)
    if np.any(tokens < 0) or np.any(tokens >= rows):
        raise RuntimeContractError(
            f"Qwen3 {name} contains an ID outside the installed text embedding"
        )
    return tokens


def project_text_tokens(ids: np.ndarray, embeddings: Qwen3EmbeddingStore) -> np.ndarray:
    """Apply WaveKat's embedding -> linear -> SiLU -> linear text projection."""
    values = np.asarray(ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise RuntimeContractError(
            "Qwen3 text projection requires one-dimensional integer token IDs"
        )
    arrays = embeddings.arrays
    try:
        text = arrays["text_embedding"][values]
        hidden = text @ arrays["text_projection_fc1_weight"].T
        hidden += arrays["text_projection_fc1_bias"]
        hidden *= 1.0 / (1.0 + np.exp(-hidden))
        projected = hidden @ arrays["text_projection_fc2_weight"].T
        projected += arrays["text_projection_fc2_bias"]
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeContractError(f"Qwen3 text embedding projection failed: {exc}") from exc
    result = np.asarray(projected, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise RuntimeContractError("Qwen3 text embedding projection returned invalid values")
    return result


def _codec_prefix_ids(config: Qwen3Config, language: str) -> tuple[int, ...]:
    language_id = config.codec_language_id.get(language)
    if language_id is None:
        return (config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id)
    return (
        config.codec_think_id,
        config.codec_think_bos_id,
        language_id,
        config.codec_think_eos_id,
    )


def build_voice_design_prefill(
    config: Qwen3Config,
    embeddings: Qwen3EmbeddingStore,
    token_ids: Sequence[int],
    instruction_token_ids: Sequence[int],
    *,
    language: str,
) -> np.ndarray:
    """Build WaveKat VoiceDesign embeddings from frontend-tokenized chat sequences."""
    if config.model_type != "voice_design":
        raise RuntimeContractError("VoiceDesign prefill requires a voice_design config")
    rows = embeddings["text_embedding"].shape[0]
    text = _token_ids(token_ids, name="text token IDs", rows=rows)
    instruction = _token_ids(instruction_token_ids, name="instruction token IDs", rows=rows)
    if text.size <= VOICE_TEXT_PREFIX_TOKENS + VOICE_TEXT_SUFFIX_TOKENS:
        raise RuntimeContractError("Qwen3 VoiceDesign text IDs must contain wrapped non-empty text")
    if (
        text[0] != config.im_start_token_id
        or text[-VOICE_TEXT_SUFFIX_TOKENS] != config.im_end_token_id
    ):
        raise RuntimeContractError(
            "Qwen3 VoiceDesign text IDs must use the supported wrapped chat format"
        )
    if text[-3] != config.im_start_token_id:
        raise RuntimeContractError("Qwen3 VoiceDesign text IDs must end with the assistant header")
    if (
        instruction.size < 2
        or instruction[0] != config.im_start_token_id
        or instruction[-2] != config.im_end_token_id
    ):
        raise RuntimeContractError(
            "Qwen3 instruction IDs must use the supported wrapped chat format"
        )

    projected_instruction = project_text_tokens(instruction, embeddings)
    projected_header = project_text_tokens(text[:VOICE_TEXT_PREFIX_TOKENS], embeddings)
    projected_content = project_text_tokens(
        text[VOICE_TEXT_PREFIX_TOKENS:-VOICE_TEXT_SUFFIX_TOKENS], embeddings
    )
    codec = embeddings["talker_codec_embedding"]
    projected_pad = project_text_tokens(np.asarray([config.tts_pad_token_id]), embeddings)[0]
    projected_bos = project_text_tokens(np.asarray([config.tts_bos_token_id]), embeddings)[0]
    projected_eos = project_text_tokens(np.asarray([config.tts_eos_token_id]), embeddings)[0]
    codec_pad = codec[config.codec_pad_id]
    codec_bos = codec[config.codec_bos_id]

    prefix_ids = _codec_prefix_ids(config, language)
    prefix = [(projected_pad + codec[token_id])[None, :] for token_id in prefix_ids]
    segments = [
        projected_instruction,
        projected_header,
        *prefix,
        (projected_bos + codec_pad)[None, :],
        projected_content + codec_pad[None, :],
        (projected_eos + codec_pad)[None, :],
        (projected_pad + codec_bos)[None, :],
    ]
    prefill = np.concatenate(segments, axis=0)[None, :, :].astype(np.float32, copy=False)
    if prefill.shape[2] != config.talker_hidden_size or not np.isfinite(prefill).all():
        raise RuntimeContractError("Qwen3 VoiceDesign prefill has invalid dimensions or values")
    return np.ascontiguousarray(prefill)


def build_voice_clone_prefill(
    config: Qwen3Config,
    embeddings: Qwen3EmbeddingStore,
    token_ids: Sequence[int],
    voice: Qwen3VoiceState,
    *,
    language: str,
) -> np.ndarray:
    """Interleave reference transcript, target text, speaker state, and reference codes."""
    if config.model_type != "base":
        raise RuntimeContractError("Base cloning prefill requires a base config")
    rows = embeddings["text_embedding"].shape[0]
    target = _token_ids(token_ids, name="text token IDs", rows=rows)
    if target.size <= VOICE_TEXT_PREFIX_TOKENS + VOICE_TEXT_SUFFIX_TOKENS:
        raise RuntimeContractError("Qwen3 Base target IDs must contain wrapped non-empty text")
    if (
        target[0] != config.im_start_token_id
        or target[-VOICE_TEXT_SUFFIX_TOKENS] != config.im_end_token_id
    ):
        raise RuntimeContractError(
            "Qwen3 Base target IDs must use the supported wrapped chat format"
        )
    if target[-3] != config.im_start_token_id:
        raise RuntimeContractError("Qwen3 Base target IDs must end with the assistant header")
    reference_ids = validate_reference_token_ids(
        voice.reference_token_ids,
        text_embedding_rows=rows,
        im_start_token_id=config.im_start_token_id,
        im_end_token_id=config.im_end_token_id,
    )
    reference_content = np.asarray(reference_ids[VOICE_TEXT_PREFIX_TOKENS:-2], dtype=np.int64)
    target_content = target[VOICE_TEXT_PREFIX_TOKENS:-VOICE_TEXT_SUFFIX_TOKENS]
    combined_text = np.concatenate((reference_content, target_content))
    projected_header = project_text_tokens(target[:VOICE_TEXT_PREFIX_TOKENS], embeddings)
    projected_content = project_text_tokens(combined_text, embeddings)
    projected_pad = project_text_tokens(np.asarray([config.tts_pad_token_id]), embeddings)[0]
    projected_bos = project_text_tokens(np.asarray([config.tts_bos_token_id]), embeddings)[0]
    projected_eos = project_text_tokens(np.asarray([config.tts_eos_token_id]), embeddings)[0]

    codec = embeddings["talker_codec_embedding"]
    predictor_embeddings = tuple(
        embeddings.arrays[f"cp_codec_embedding_{index}"] for index in range(CODE_GROUPS - 1)
    )
    speaker = np.asarray(voice.speaker_embedding, dtype=np.float32)
    if speaker.shape != (1, config.talker_hidden_size) or not np.isfinite(speaker).all():
        raise RuntimeContractError("Qwen3 Base voice speaker embedding disagrees with config")
    codes = np.asarray(voice.reference_codes)
    reference_frames = math.ceil(voice.reference_samples / QWEN3_CODEC_SAMPLES_PER_FRAME)
    if codes.shape != (1, CODE_GROUPS, reference_frames):
        raise RuntimeContractError(
            "Qwen3 Base reference codes disagree with reference audio length"
        )
    if np.any(codes[0, 0] >= config.talker_vocab_size) or np.any(
        codes[0, 1:] >= config.cp_vocab_size
    ):
        raise RuntimeContractError("Qwen3 Base reference code ID exceeds its embedding vocabulary")

    codec_pad = codec[config.codec_pad_id]
    codec_bos = codec[config.codec_bos_id]
    frame_embeddings = np.zeros((reference_frames, config.talker_hidden_size), dtype=np.float32)
    frame_embeddings += codec[codes[0, 0]]
    for group_index, predictor_embedding in enumerate(predictor_embeddings):
        frame_embeddings += predictor_embedding[codes[0, group_index + 1]]
    reference_codec_embeddings = np.concatenate((codec_bos[None, :], frame_embeddings), axis=0)
    text_with_end = np.concatenate((projected_content, projected_eos[None, :]), axis=0)
    prefix = [
        (projected_pad + codec[token_id])[None, :]
        for token_id in _codec_prefix_ids(config, language)
    ]
    segments = [
        projected_header,
        *prefix,
        (projected_pad + speaker[0])[None, :],
        (projected_bos + codec_pad)[None, :],
        text_with_end + codec_pad[None, :],
        reference_codec_embeddings + projected_pad[None, :],
    ]
    prefill = np.concatenate(segments, axis=0)[None, :, :].astype(np.float32, copy=False)
    if prefill.shape[2] != config.talker_hidden_size or not np.isfinite(prefill).all():
        raise RuntimeContractError("Qwen3 Base prefill has invalid dimensions or values")
    return np.ascontiguousarray(prefill)


def _run(session: Any, inputs: dict[str, np.ndarray], *, component: str) -> list[np.ndarray]:
    try:
        outputs = session.run(inputs)
    except Exception as exc:
        raise RuntimeContractError(f"Qwen3 {component} inference failed: {exc}") from exc
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or not outputs:
        raise RuntimeContractError(f"Qwen3 {component} returned no outputs")
    try:
        return [np.asarray(output) for output in outputs]
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(f"Qwen3 {component} returned invalid outputs") from exc


def _finite_array(
    value: np.ndarray, *, shape: tuple[int, ...], component: str, name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise RuntimeContractError(
            f"Qwen3 {component} {name} must have finite shape {shape}, got {array.shape}"
        )
    return array


def _initial_talker_cache(
    outputs: Sequence[np.ndarray], config: Qwen3Config, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    expected_count = 2 + 2 * config.talker_num_layers
    if len(outputs) != expected_count:
        raise RuntimeContractError(
            f"Qwen3 talker_prefill returned {len(outputs)} outputs; expected {expected_count}"
        )
    key_shape = (1, config.talker_num_kv_heads, sequence_length, config.talker_head_dim)
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for layer in range(config.talker_num_layers):
        keys.append(
            _finite_array(
                outputs[2 + layer * 2],
                shape=key_shape,
                component="talker_prefill",
                name=f"key {layer}",
            )
        )
        values.append(
            _finite_array(
                outputs[3 + layer * 2],
                shape=key_shape,
                component="talker_prefill",
                name=f"value {layer}",
            )
        )
    return np.stack(keys), np.stack(values)


def _validate_predictor_cache(
    value: np.ndarray,
    *,
    config: Qwen3Config,
    sequence_length: int,
    name: str,
) -> np.ndarray:
    shape = (
        config.cp_num_layers,
        1,
        config.cp_num_kv_heads,
        sequence_length,
        config.cp_head_dim,
    )
    return _finite_array(value, shape=shape, component="code_predictor", name=name)


def _validate_talker_cache(
    value: np.ndarray,
    *,
    config: Qwen3Config,
    sequence_length: int,
    name: str,
) -> np.ndarray:
    shape = (
        config.talker_num_layers,
        1,
        config.talker_num_kv_heads,
        sequence_length,
        config.talker_head_dim,
    )
    return _finite_array(value, shape=shape, component="talker_decode", name=name)


def generate_codes(
    *,
    config: Qwen3Config,
    embeddings: Qwen3EmbeddingStore,
    prefill_session: Any,
    decode_session: Any,
    predictor_session: Any,
    inputs_embeds: np.ndarray,
    trailing_hidden: np.ndarray,
    generation: Qwen3GenerationConfig,
) -> GeneratedCodes:
    """Run the shared talker/code-predictor loop once from a prepared prefill."""
    prefill = np.asarray(inputs_embeds, dtype=np.float32)
    if prefill.ndim != 3 or prefill.shape[0] != 1 or prefill.shape[1] == 0:
        raise RuntimeContractError(
            "Qwen3 talker prefill embeddings must have shape [1, sequence, hidden]"
        )
    if prefill.shape[2] != config.talker_hidden_size or not np.isfinite(prefill).all():
        raise RuntimeContractError(
            "Qwen3 talker prefill embeddings disagree with config or are non-finite"
        )
    trailing = np.asarray(trailing_hidden, dtype=np.float32)
    if trailing.shape == (config.talker_hidden_size,):
        trailing = trailing[None, :]
    if trailing.shape != (1, config.talker_hidden_size) or not np.isfinite(trailing).all():
        raise RuntimeContractError("Qwen3 trailing hidden state must have finite shape [1, hidden]")

    sequence_length = prefill.shape[1]
    positions = np.arange(sequence_length, dtype=np.int64).reshape(1, 1, sequence_length)
    positions = np.repeat(positions, 3, axis=0)
    prefill_outputs = _run(
        prefill_session,
        {
            "inputs_embeds": np.ascontiguousarray(prefill),
            "attention_mask": np.ones((1, sequence_length), dtype=np.int64),
            "position_ids": positions,
        },
        component="talker_prefill",
    )
    vocab_size = config.talker_vocab_size
    logits = _finite_array(
        prefill_outputs[0],
        shape=(1, sequence_length, vocab_size),
        component="talker_prefill",
        name="logits",
    )
    hidden = _finite_array(
        prefill_outputs[1],
        shape=(1, sequence_length, config.talker_hidden_size),
        component="talker_prefill",
        name="hidden_states",
    )
    past_keys, past_values = _initial_talker_cache(prefill_outputs, config, sequence_length)

    codec_embedding = embeddings["talker_codec_embedding"]
    predictor_embeddings = tuple(
        embeddings.arrays[f"cp_codec_embedding_{index}"] for index in range(CODE_GROUPS - 1)
    )
    rng = np.random.default_rng(generation.seed)
    group0_history: list[int] = []
    frames: list[list[int]] = []
    current_position = sequence_length
    eos_observed = False
    cp_cache_shape = (
        config.cp_num_layers,
        1,
        config.cp_num_kv_heads,
        0,
        config.cp_head_dim,
    )

    for frame_index in range(generation.max_new_tokens):
        talker_scores = mask_talker_logits(
            logits[0, -1],
            eos_token_id=config.codec_eos_token_id,
            generated_group0=group0_history,
            frame_index=frame_index,
            repetition_penalty=generation.repetition_penalty,
            reserved_tokens=RESERVED_TALKER_TOKENS,
        )
        first_code = sample_logits(talker_scores, generation, rng)
        if first_code == config.codec_eos_token_id:
            eos_observed = True
            break
        group0_history.append(first_code)
        frame = [first_code]
        try:
            predictor_input = np.concatenate(
                [hidden[0, -1:, :], codec_embedding[first_code][None, :]], axis=0
            )[None, :, :].astype(np.float32)
        except IndexError as exc:
            raise RuntimeContractError(
                "Qwen3 sampled talker code is outside its embedding table"
            ) from exc
        cp_keys = np.zeros(cp_cache_shape, dtype=np.float32)
        cp_values = np.zeros(cp_cache_shape, dtype=np.float32)
        predictor_position = predictor_input.shape[1]

        for group_index in range(CODE_GROUPS - 1):
            predictor_outputs = _run(
                predictor_session,
                {
                    "inputs_embeds": predictor_input,
                    "generation_steps": np.asarray([group_index], dtype=np.int64),
                    "past_keys": cp_keys,
                    "past_values": cp_values,
                },
                component="code_predictor",
            )
            if len(predictor_outputs) != 3:
                raise RuntimeContractError("Qwen3 code_predictor must return logits and KV caches")
            cp_logits = predictor_outputs[0]
            if (
                cp_logits.ndim != 3
                or cp_logits.shape[0] != 1
                or cp_logits.shape[1] != predictor_input.shape[1]
            ):
                raise RuntimeContractError("Qwen3 code_predictor logits have an invalid shape")
            if cp_logits.shape[2] != config.cp_vocab_size or not np.isfinite(cp_logits).all():
                raise RuntimeContractError(
                    "Qwen3 code_predictor logits disagree with config or are non-finite"
                )
            cp_keys = _validate_predictor_cache(
                predictor_outputs[1],
                config=config,
                sequence_length=predictor_position,
                name="past_keys",
            )
            cp_values = _validate_predictor_cache(
                predictor_outputs[2],
                config=config,
                sequence_length=predictor_position,
                name="past_values",
            )
            code = sample_logits(cp_logits[0, -1], generation, rng)
            frame.append(code)
            try:
                predictor_input = predictor_embeddings[group_index][code][None, None, :].astype(
                    np.float32
                )
            except IndexError as exc:
                raise RuntimeContractError(
                    "Qwen3 sampled predictor code is outside its embedding table"
                ) from exc
            if not np.isfinite(predictor_input).all():
                raise RuntimeContractError("Qwen3 predictor embedding contains non-finite values")
            predictor_position += 1

        if len(frame) != CODE_GROUPS:
            raise RuntimeContractError(
                f"Qwen3 code predictor produced {len(frame)} codes, expected {CODE_GROUPS}"
            )
        frames.append(frame)
        next_embedding = codec_embedding[first_code].astype(np.float32, copy=True)
        for group_index, code in enumerate(frame[1:]):
            next_embedding += predictor_embeddings[group_index][code]
        next_embedding += trailing[0]
        next_embedding = next_embedding.reshape(1, 1, config.talker_hidden_size)
        next_position = current_position + 1
        decode_outputs = _run(
            decode_session,
            {
                "inputs_embeds": next_embedding,
                "attention_mask": np.ones((1, next_position), dtype=np.int64),
                "position_ids": np.full((3, 1, 1), current_position, dtype=np.int64),
                "past_keys": past_keys,
                "past_values": past_values,
            },
            component="talker_decode",
        )
        if len(decode_outputs) != 4:
            raise RuntimeContractError(
                "Qwen3 talker_decode must return logits, hidden states, and KV caches"
            )
        logits = _finite_array(
            decode_outputs[0],
            shape=(1, 1, vocab_size),
            component="talker_decode",
            name="logits",
        )
        hidden = _finite_array(
            decode_outputs[1],
            shape=(1, 1, config.talker_hidden_size),
            component="talker_decode",
            name="hidden_states",
        )
        past_keys = _validate_talker_cache(
            decode_outputs[2], config=config, sequence_length=next_position, name="past_keys"
        )
        past_values = _validate_talker_cache(
            decode_outputs[3], config=config, sequence_length=next_position, name="past_values"
        )
        current_position = next_position

    if not frames:
        raise RuntimeContractError("Qwen3 talker generated no codec frames")
    return GeneratedCodes(np.asarray(frames, dtype=np.int64), eos_observed)


def vocode_codes(vocoder_session: Any, codes: np.ndarray) -> np.ndarray:
    """Vocode [frames, 16] codes and normalize output to finite mono float32."""
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[1] != CODE_GROUPS or values.shape[0] == 0:
        raise RuntimeContractError("Qwen3 vocoder codes must have non-empty [frames, 16] shape")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise RuntimeContractError("Qwen3 vocoder codes must contain non-negative integers")
    output = _run(
        vocoder_session,
        {"codes": np.ascontiguousarray(values.T[None, :, :], dtype=np.int64)},
        component="vocoder",
    )
    if len(output) != 1:
        raise RuntimeContractError("Qwen3 vocoder must return exactly one waveform")
    try:
        audio = np.asarray(output[0], dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError("Qwen3 vocoder output must be a float waveform") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeContractError("Qwen3 vocoder returned empty or non-finite audio")
    return np.ascontiguousarray(audio)


def voice_design_result(
    *,
    config: Qwen3Config,
    embeddings: Qwen3EmbeddingStore,
    sessions: Mapping[str, Any],
    token_ids: Sequence[int],
    instruction_token_ids: Sequence[int],
    language: str,
    generation: Qwen3GenerationConfig,
    model_id: str,
    quality: str,
    providers_active: Sequence[str] = (),
) -> InferenceResult:
    """Run one fully-tokenized VoiceDesign request through WaveKat's graph pipeline."""
    prefill = build_voice_design_prefill(
        config, embeddings, token_ids, instruction_token_ids, language=language
    )
    trailing_hidden = project_text_tokens(np.asarray([config.tts_pad_token_id]), embeddings)
    codes = generate_codes(
        config=config,
        embeddings=embeddings,
        prefill_session=sessions["talker_prefill"],
        decode_session=sessions["talker_decode"],
        predictor_session=sessions["code_predictor"],
        inputs_embeds=prefill,
        trailing_hidden=trailing_hidden,
        generation=generation,
    )
    audio = vocode_codes(sessions["vocoder"], codes.codes)
    return InferenceResult(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        outputs={"codes": codes.codes},
        metadata=MappingProxyType(
            {
                "model_type": config.model_type,
                "model_id": model_id,
                "layout": "qwen3-wavekat-v1",
                "quality": quality,
                "frames_generated": int(codes.codes.shape[0]),
                "eos_observed": codes.eos_observed,
                "seed": generation.seed,
                "temperature": generation.temperature,
                "top_k": generation.top_k,
                "repetition_penalty": generation.repetition_penalty,
                "sample_rate": SAMPLE_RATE,
                "providers_active": tuple(providers_active),
            }
        ),
    )


def voice_clone_result(
    *,
    config: Qwen3Config,
    embeddings: Qwen3EmbeddingStore,
    sessions: Mapping[str, Any],
    token_ids: Sequence[int],
    voice: Qwen3VoiceState,
    language: str,
    generation: Qwen3GenerationConfig,
    model_id: str,
    quality: str,
    providers_active: Sequence[str] = (),
) -> InferenceResult:
    """Generate Base speech, prepend reference codes for vocoding, and trim that region."""
    if voice.model_id != model_id or voice.layout != "qwen3-wavekat-v1":
        raise RuntimeContractError(
            "Qwen3 voice state is incompatible with the installed Base bundle"
        )
    prefill = build_voice_clone_prefill(config, embeddings, token_ids, voice, language=language)
    trailing_hidden = project_text_tokens(np.asarray([config.tts_pad_token_id]), embeddings)
    generated = generate_codes(
        config=config,
        embeddings=embeddings,
        prefill_session=sessions["talker_prefill"],
        decode_session=sessions["talker_decode"],
        predictor_session=sessions["code_predictor"],
        inputs_embeds=prefill,
        trailing_hidden=trailing_hidden,
        generation=generation,
    )
    reference_codes = np.asarray(voice.reference_codes[0].T, dtype=np.int64)
    all_codes = np.concatenate((reference_codes, generated.codes), axis=0)
    waveform = vocode_codes(sessions["vocoder"], all_codes)
    reference_frames = reference_codes.shape[0]
    trim_samples = int(reference_frames / all_codes.shape[0] * waveform.size)
    audio = np.ascontiguousarray(waveform[trim_samples:], dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeContractError(
            "Qwen3 Base reference-region trimming returned empty or non-finite audio"
        )
    return InferenceResult(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        outputs={"codes": generated.codes},
        metadata=MappingProxyType(
            {
                "model_type": config.model_type,
                "model_id": model_id,
                "layout": "qwen3-wavekat-v1",
                "quality": quality,
                "frames_generated": int(generated.codes.shape[0]),
                "reference_frames": int(reference_frames),
                "eos_observed": generated.eos_observed,
                "seed": generation.seed,
                "temperature": generation.temperature,
                "top_k": generation.top_k,
                "repetition_penalty": generation.repetition_penalty,
                "sample_rate": SAMPLE_RATE,
                "providers_active": tuple(providers_active),
            }
        ),
    )
