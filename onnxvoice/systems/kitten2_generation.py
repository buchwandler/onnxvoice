from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import OnnxSession
from .kitten2_sampling import SamplingSettings, is_eos_token, sample
from .kitten2_tokens import TokenMap


@dataclass(frozen=True)
class GenerationResult:
    raw_ids: list[int]
    s3_ids: list[int]


class Kitten2LanguageModel:
    """Drive the exported Qwen3 LM with explicit embeddings and KV cache.

    This class handles speaker projection and LM token generation. It does not yet
    reproduce the S3 waveform decoder.
    """

    def __init__(
        self,
        lm_session: OnnxSession,
        speaker_session: OnnxSession,
        metadata: Mapping[str, Any],
        embeddings_path: str | Path,
    ) -> None:
        self.session = lm_session
        self.speaker_session = speaker_session
        self.meta = dict(metadata)
        runtime_profile = self.meta.get("runtime_profile")
        io_dtype = self.meta.get("io_dtype")
        if runtime_profile is None or io_dtype is None:
            raise ValueError("LM metadata must declare runtime_profile and io_dtype")
        if io_dtype == "bfloat16":
            raise NotImplementedError(
                "bfloat16 LM I/O requires ONNX Runtime I/O binding; NumPy session.run is unsupported"
            )
        if io_dtype not in {"float32", "float16"}:
            raise ValueError(f"Unsupported LM I/O dtype: {io_dtype!r}")
        self.input_mode = self.meta.get("input_mode", "inputs_embeds")
        if self.input_mode == "inputs_embeds":
            self.embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
            self.embedding_storage_dtype = self.meta.get(
                "embedding_storage_dtype", str(self.embeddings.dtype)
            )
            if self.embedding_storage_dtype == "bfloat16-u16":
                if self.embeddings.dtype != np.uint16:
                    raise ValueError("bfloat16-u16 embedding storage must use uint16")
            elif str(self.embeddings.dtype) != self.embedding_storage_dtype:
                raise ValueError("Embedding array dtype differs from LM metadata")
        elif self.input_mode == "input_ids":
            if self.meta.get("embedding_file") is not None:
                raise ValueError("input_ids LM metadata must not reference a separate embedding table")
            self.embeddings = None
            self.embedding_storage_dtype = None
        else:
            raise ValueError(f"Unsupported LM input mode: {self.input_mode!r}")
        runtime_memory_profile = self.meta.get("runtime_memory_profile", "default")
        if runtime_memory_profile not in {"default", "low-memory-exact-ternary"}:
            raise ValueError(f"Unsupported runtime memory profile: {runtime_memory_profile!r}")
        self.layers = int(self.meta["num_hidden_layers"])
        self.kv_heads = int(self.meta["num_key_value_heads"])
        self.head_dim = int(self.meta["head_dim"])
        graph_inputs = {item.name: item.ort_type for item in self.session.input_specs}
        expected_type = {"float32": "tensor(float)", "float16": "tensor(float16)"}[io_dtype]
        if self.input_mode == "inputs_embeds":
            graph_input_type = graph_inputs.get("inputs_embeds")
            if graph_input_type != expected_type:
                raise ValueError(
                    f"LM metadata io_dtype {io_dtype!r} does not match inputs_embeds type "
                    f"{graph_input_type!r}"
                )
        else:
            if graph_inputs.get("input_ids") != "tensor(int64)":
                raise ValueError("input_ids LM graph must accept int64 token IDs")
            if graph_inputs.get("speaker_hidden") != expected_type:
                raise ValueError("LM metadata io_dtype does not match speaker_hidden graph input")
        self.model_numpy_dtype = {"float32": np.float32, "float16": np.float16}[io_dtype]

    def _empty_cache(self) -> dict[str, np.ndarray]:
        shape = (1, self.kv_heads, 0, self.head_dim)
        return {
            name: np.zeros(shape, dtype=self.model_numpy_dtype)
            for i in range(self.layers)
            for name in (f"past_key_{i}", f"past_value_{i}")
        }

    def _project_speaker(self, speaker_embedding: np.ndarray) -> np.ndarray:
        speaker_embedding = np.asarray(speaker_embedding, dtype=np.float32)
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None, :]
        return self.speaker_session.run({"speaker_embedding": speaker_embedding})[0]
    def _embed(
        self,
        token_ids: np.ndarray,
        *,
        speaker_embedding: np.ndarray | None = None,
        prefill: bool,
    ) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if self.embedding_storage_dtype == "bfloat16-u16":
            bits = np.asarray(self.embeddings[token_ids], dtype=np.uint32) << 16
            values = bits.view(np.float32).astype(self.model_numpy_dtype)
        else:
            values = np.asarray(self.embeddings[token_ids], dtype=self.model_numpy_dtype)
        if prefill and speaker_embedding is not None:
            speaker_hidden = self._project_speaker(speaker_embedding)
            values = values.copy()
            values[0, 0, :] = speaker_hidden[0].astype(values.dtype, copy=False)
        return values

    def step(
        self,
        token_ids: np.ndarray,
        *,
        past: dict[str, np.ndarray] | None = None,
        speaker_embedding: np.ndarray | None = None,
        position: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run prefill or one cached decode chunk.

        ``token_ids`` must have shape ``[1, S]``. If ``past`` is omitted this is
        treated as prefill and an empty cache is supplied.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("token_ids must have shape [1, S]")

        is_prefill = past is None
        past = self._empty_cache() if past is None else past
        past_len = next(iter(past.values())).shape[2]
        start = past_len if position is None else int(position)

        attention_mask = np.ones((1, past_len + token_ids.shape[1]), dtype=np.int64)
        position_ids = np.arange(
            start,
            start + token_ids.shape[1],
            dtype=np.int64,
        )[None, :]
        inputs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past,
        }
        if self.input_mode == "inputs_embeds":
            inputs["inputs_embeds"] = self._embed(
                token_ids,
                speaker_embedding=speaker_embedding,
                prefill=is_prefill,
            )
        else:
            speaker_hidden = np.zeros(
                (1, int(self.meta["hidden_size"])), dtype=self.model_numpy_dtype
            )
            if is_prefill and speaker_embedding is not None:
                speaker_hidden = self._project_speaker(speaker_embedding).astype(
                    self.model_numpy_dtype, copy=False
                )
            inputs["input_ids"] = token_ids
            inputs["speaker_hidden"] = speaker_hidden
        outputs = self.session.run(inputs)
        logits = outputs[0]
        new_past: dict[str, np.ndarray] = {}
        for i in range(self.layers):
            new_past[f"past_key_{i}"] = outputs[1 + 2 * i]
            new_past[f"past_value_{i}"] = outputs[2 + 2 * i]
        return logits, new_past

    def generate(
        self,
        prompt_ids: list[int],
        *,
        token_map: TokenMap,
        max_new_tokens: int,
        speaker_embedding: np.ndarray | None = None,
        settings: SamplingSettings | None = None,
        seed: int | None = None,
        stop_ids: set[int] | None = None,
        logits_callback: Callable[[int, np.ndarray], None] | None = None,
    ) -> GenerationResult:
        """Generate raw LM IDs and their in-range, zero-based S3 counterparts."""
        if not prompt_ids:
            raise ValueError("prompt_ids may not be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return GenerationResult(raw_ids=[], s3_ids=[])

        settings = settings or SamplingSettings()
        rng = np.random.default_rng(seed)
        history = [int(token) for token in prompt_ids]
        prompt_length = len(history)
        logits, past = self.step(
            np.asarray([history], dtype=np.int64),
            speaker_embedding=speaker_embedding,
        )
        raw_ids: list[int] = []

        for step_index in range(max_new_tokens):
            if logits_callback is not None:
                logits_callback(step_index, np.asarray(logits[0]).copy())
            token = sample(
                logits[0],
                history,
                token_map,
                settings,
                float(rng.random()),
                prompt_length=prompt_length,
            )
            raw_ids.append(token)
            should_stop = is_eos_token(token, token_map) if stop_ids is None else token in stop_ids
            if should_stop:
                break
            history.append(token)
            logits, past = self.step(
                np.asarray([[token]], dtype=np.int64),
                past=past,
            )

        s3_ids = [
            token - token_map.audio_id_base
            for token in raw_ids
            if token_map.audio_id_base <= token < token_map.audio_id_end
        ]
        return GenerationResult(raw_ids=raw_ids, s3_ids=s3_ids)


