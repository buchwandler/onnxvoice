from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..errors import RuntimeContractError


@dataclass(frozen=True, slots=True)
class Qwen3GenerationConfig:
    """WaveKat sampling controls; the seed is local to one inference request."""

    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise RuntimeContractError("Qwen3 temperature must be a finite number in (0, 5]")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or not 0.0 < temperature <= 5.0:
            raise RuntimeContractError("Qwen3 temperature must be a finite number in (0, 5]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral) or self.top_k < 0:
            raise RuntimeContractError("Qwen3 top_k must be a non-negative integer")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, Real)
            or not math.isfinite(float(self.repetition_penalty))
            or self.repetition_penalty <= 0
        ):
            raise RuntimeContractError("Qwen3 repetition_penalty must be a finite positive number")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, Integral)
            or not 1 <= self.max_new_tokens <= 8192
        ):
            raise RuntimeContractError(
                "Qwen3 max_new_tokens must be an integer from 1 through 8192"
            )
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or not 0 <= self.seed <= 0xFFFFFFFF
        ):
            raise RuntimeContractError("Qwen3 seed must be an integer from 0 through 4294967295")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        object.__setattr__(self, "max_new_tokens", int(self.max_new_tokens))
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))


def sample_logits(
    logits: np.ndarray,
    config: Qwen3GenerationConfig,
    rng: np.random.Generator,
) -> int:
    """Apply WaveKat temperature/top-k semantics and sample one finite logit vector."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or np.isnan(scores).any() or np.isposinf(scores).any():
        raise RuntimeContractError(
            "Qwen3 sampling logits must be a non-empty vector without NaN/+inf"
        )
    scores = scores.copy()
    scores /= config.temperature
    if config.top_k and config.top_k < scores.size:
        indices = np.argpartition(scores, -config.top_k)[-config.top_k :]
        filtered = np.full_like(scores, -np.inf)
        filtered[indices] = scores[indices]
        scores = filtered
    maximum = float(np.max(scores))
    if not math.isfinite(maximum):
        raise RuntimeContractError("Qwen3 sampling has no finite candidate logits")
    probabilities = np.exp(scores - maximum)
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeContractError("Qwen3 sampling probabilities are invalid")
    probabilities /= total
    return int(rng.choice(scores.size, p=probabilities))


def mask_talker_logits(
    logits: np.ndarray,
    *,
    eos_token_id: int,
    generated_group0: list[int],
    frame_index: int,
    repetition_penalty: float,
    reserved_tokens: int = 1024,
) -> np.ndarray:
    """Apply WaveKat reserved-token, early-EOS, and group-0 repetition masks."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or np.isnan(scores).any() or np.isposinf(scores).any():
        raise RuntimeContractError(
            "Qwen3 talker logits must be a non-empty vector without NaN/+inf"
        )
    if isinstance(eos_token_id, bool) or not 0 <= eos_token_id < scores.size:
        raise RuntimeContractError("Qwen3 codec EOS ID is outside the talker logits vocabulary")
    if reserved_tokens < 0:
        raise RuntimeContractError("Qwen3 reserved token count must be non-negative")
    result = scores.copy()
    result[max(0, result.size - reserved_tokens) :] = -np.inf
    result[eos_token_id] = scores[eos_token_id]
    if frame_index < 2:
        result[eos_token_id] = -np.inf
    if repetition_penalty != 1.0 and generated_group0:
        seen = np.asarray(generated_group0, dtype=np.int64)
        if np.any(seen < 0) or np.any(seen >= result.size):
            raise RuntimeContractError("Qwen3 repetition history contains an invalid token ID")
        previous = result[seen]
        result[seen] = np.where(
            previous > 0,
            previous / repetition_penalty,
            previous * repetition_penalty,
        )
    return result
