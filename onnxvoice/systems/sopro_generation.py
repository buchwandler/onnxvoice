from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SoproGenerationDefaults:
    """Validated immutable generation settings exposed by a Sopro adapter."""

    temperature: float
    top_p: float
    top_k: int
    steps: int
    max_seconds: float
    min_seconds: float
    ref_seconds: float
    max_segment_chars: int
    style_tokens: int
    prompt_tokens: int
    stream_chunk_frames: int
