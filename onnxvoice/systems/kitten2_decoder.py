from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..runtime import OnnxSession

S3GEN_SILENCE_TOKEN = 4299
S3GEN_SILENCE_TAIL_TOKENS = 3
S3GEN_MEL_FRAMES_PER_TOKEN = 2
S3GEN_SAMPLE_RATE = 24000
S3GEN_SAMPLES_PER_MEL_FRAME = 480
S3GEN_SPEECH_VOCAB_SIZE = 6561


def _cpu_session_options(profile: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if profile == "default":
        return options
    if profile != "low-memory-exact-ternary":
        raise ValueError(f"Unsupported CPU runtime memory profile: {profile!r}")
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def clean_s3_tokens(
    speech_tokens: list[int] | np.ndarray, vocab_size: int = S3GEN_SPEECH_VOCAB_SIZE
) -> np.ndarray:
    """Keep only valid zero-based codec IDs accepted by the flow graph."""
    tokens = np.asarray(speech_tokens, dtype=np.int64)
    if tokens.ndim == 2 and tokens.shape[0] == 1:
        tokens = tokens[0]
    elif tokens.ndim != 1:
        raise ValueError("speech_tokens must have shape [T] or [1, T]")
    return tokens[(tokens >= 0) & (tokens < vocab_size)]


def append_s3_silence_tail(
    speech_tokens: list[int] | np.ndarray,
    token_id: int = S3GEN_SILENCE_TOKEN,
    count: int = S3GEN_SILENCE_TAIL_TOKENS,
) -> np.ndarray:
    """Append the silence tokens required by upstream S3 lookahead."""
    tokens = np.asarray(speech_tokens, dtype=np.int64)
    if tokens.ndim == 1:
        tail_shape: tuple[int, ...] = (count,)
    elif tokens.ndim == 2 and tokens.shape[0] == 1:
        tail_shape = (1, count)
    else:
        raise ValueError("speech_tokens must have shape [T] or [1, T]")
    tail = np.full(tail_shape, token_id, dtype=np.int64)
    return np.concatenate((tokens, tail), axis=-1)

class Kitten2Decoder:
    """Run the deterministic S3 flow and HiFT ONNX stages for zero-based S3 IDs."""

    def __init__(
        self,
        flow_session: OnnxSession,
        mel2wav_session: OnnxSession,
        flow_meta: Mapping[str, Any],
        mel2wav_meta: Mapping[str, Any],
    ) -> None:
        self.flow_session = flow_session
        self.mel2wav_session = mel2wav_session
        self.flow_meta = dict(flow_meta)
        self.mel2wav_meta = dict(mel2wav_meta)
        if self.flow_meta.get("format") != "kitten2-onnx-flow-v1":
            raise ValueError("Unsupported S3 flow graph metadata")
        if self.mel2wav_meta.get("format") != "kitten2-onnx-mel2wav-v1":
            raise ValueError("Unsupported mel2wav graph metadata")
        tail = self.flow_meta["silence_tail"]
        self.token_mel_ratio = int(self.flow_meta["token_mel_ratio"])
        self.speech_vocab_size = int(self.flow_meta["speech_vocab_size"])
        self.silence_token_id = int(tail["token_id"])
        self.silence_tail_count = int(tail["count"])
        self.samples_per_mel_frame = int(self.mel2wav_meta["samples_per_mel_frame"])
        self.sample_rate = int(self.mel2wav_meta["sample_rate"])
        self.trim_fade_samples = int(self.mel2wav_meta["trim_fade_samples"])
        self.flow_channels = int(self.flow_meta["input_shapes"]["flow_noise"][1])
        self.vocoder_harmonics = int(self.mel2wav_meta["input_shapes"]["phase_noise"][1])

    def decode(
        self,
        speech_tokens: list[int] | np.ndarray,
        conditioning: Mapping[str, np.ndarray],
        *,
        seed: int | None = None,
        flow_noise: np.ndarray | None = None,
        phase_noise: np.ndarray | None = None,
        sine_noise: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Decode S3 IDs and apply the upstream silence lookahead and trim fade."""
        tokens = clean_s3_tokens(speech_tokens, self.speech_vocab_size)
        if tokens.size == 0:
            return None
        tokens = append_s3_silence_tail(
            tokens, self.silence_token_id, self.silence_tail_count
        )[None, :]

        prompt_token = np.asarray(conditioning["conditioning_prompt_token"], dtype=np.int64)
        prompt_token_len = np.asarray(conditioning["conditioning_prompt_token_len"], dtype=np.int64)
        prompt_feat = np.asarray(conditioning["conditioning_prompt_feat"], dtype=np.float32)
        embedding = np.asarray(conditioning["conditioning_embedding"], dtype=np.float32)
        if prompt_token.shape[0] != 1 or prompt_token_len.shape != (1,):
            raise ValueError("conditioning prompt tokens must have batch size one")
        if int(prompt_token_len[0]) != prompt_token.shape[1]:
            raise ValueError("conditioning_prompt_token_len does not match its token array")

        rng = np.random.default_rng(seed)
        flow_shape = (
            1,
            self.flow_channels,
            self.token_mel_ratio * (int(prompt_token_len[0]) + tokens.shape[1]),
        )
        if flow_noise is None:
            flow_noise = rng.standard_normal(flow_shape, dtype=np.float32)
        else:
            flow_noise = np.asarray(flow_noise, dtype=np.float32)
        if flow_noise.shape != flow_shape:
            raise ValueError(f"flow_noise must have shape {flow_shape}, found {flow_noise.shape}")

        flow_inputs = {
            "speech_tokens": tokens,
            "speech_token_lengths": np.asarray([tokens.shape[1]], dtype=np.int64),
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
            "flow_noise": flow_noise,
        }
        mel = self.flow_session.session.run(["mel"], flow_inputs)[0]

        phase_shape = (1, self.vocoder_harmonics, 1)
        if phase_noise is None:
            phase_noise = rng.uniform(-np.pi, np.pi, size=phase_shape).astype(np.float32)
            phase_noise[:, 0, :] = 0
        else:
            phase_noise = np.asarray(phase_noise, dtype=np.float32)
        sine_shape = (
            1,
            self.vocoder_harmonics,
            mel.shape[2] * self.samples_per_mel_frame,
        )
        if sine_noise is None:
            sine_noise = rng.standard_normal(sine_shape, dtype=np.float32)
        else:
            sine_noise = np.asarray(sine_noise, dtype=np.float32)
        if phase_noise.shape != phase_shape:
            raise ValueError(f"phase_noise must have shape {phase_shape}, found {phase_noise.shape}")
        if sine_noise.shape != sine_shape:
            raise ValueError(f"sine_noise must have shape {sine_shape}, found {sine_noise.shape}")

        waveform = self.mel2wav_session.session.run(
            ["waveform"],
            {"mel": mel, "phase_noise": phase_noise, "sine_noise": sine_noise},
        )[0]
        n_trim = self.trim_fade_samples
        fade = np.zeros(2 * n_trim, dtype=np.float32)
        fade[n_trim:] = (np.cos(np.linspace(np.pi, 0, n_trim, dtype=np.float32)) + 1) / 2
        waveform[:, : fade.shape[0]] *= fade
        return waveform
