"""Prompt assembly for reference-conditioned KittenTTS 2 generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from .kitten2_decoder_reference import load_decoder_reference
from .kitten2_expression import inspect_expression_markup
from .kitten2_tokenizer import Kitten2Tokenizer
from .kitten2_tokens import TokenMap


def build_reference_prefix(
    token_map: TokenMap,
    reference_text_ids: Sequence[int],
    reference_audio_tokens: Sequence[int],
) -> list[int]:
    """Build the in-context reference transcript and 0-based S3 token prefix."""
    audio_ids = [token_map.audio_id_base + token for token in reference_audio_tokens]
    if token_map.uses_reference_role_tokens:
        return [
            token_map.start_id,
            token_map.reference_text_start_id,
            *reference_text_ids,
            token_map.reference_text_end_id,
            token_map.reference_speech_start_id,
            *audio_ids,
            token_map.reference_speech_end_id,
        ]
    return [
        token_map.start_id,
        token_map.text_start_id,
        *reference_text_ids,
        token_map.speech_start_id,
        *audio_ids,
        token_map.speech_end_id,
    ]


def build_generation_prompt(
    token_map: TokenMap,
    target_text_ids: Sequence[int],
    reference_prefix: Sequence[int] | None = None,
    emotion_ids: Sequence[int] | None = None,
) -> list[int]:
    """Build the target segment, optionally preceded by a reference prefix."""
    head = list(reference_prefix) if reference_prefix else [token_map.start_id]
    final_marker = [token_map.final_seg_id] if token_map.final_seg_id else []
    return [
        *head,
        *(emotion_ids or []),
        token_map.text_start_id,
        *target_text_ids,
        *final_marker,
        token_map.speech_start_id,
    ]


@dataclass(frozen=True, slots=True)
class PreparedTarget:
    text: str
    token_ids: tuple[int, ...]
    has_expression: bool


@dataclass(frozen=True)
class PreparedVoice:
    speaker_embedding: np.ndarray
    reference_prefix: list[int]
    conditioning: dict[str, np.ndarray]


class Kitten2Frontend:
    def __init__(self, root: str | Path, manifest: dict[str, Any], token_map: TokenMap) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.token_map = token_map
        self._tokenizer: Kitten2Tokenizer | None = None
        self._emotion_control_ids: tuple[int, ...] | None = None

    def _load_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        tokenizer_path = self.root / self.manifest["tokenizer"]
        self._tokenizer = Kitten2Tokenizer(tokenizer_path / "tokenizer.json")

    def prepare_target(self, text: str) -> PreparedTarget:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")
        expression = inspect_expression_markup(text)
        self._load_tokenizer()
        tokenizer = self._tokenizer
        assert tokenizer is not None
        return PreparedTarget(
            text=text,
            token_ids=tuple(tokenizer.encode(text)),
            has_expression=expression.has_expression,
        )

    @property
    def emotion_control_ids(self) -> tuple[int, ...]:
        if self._emotion_control_ids is None:
            emotion_control = self.manifest["generation"].get("emotion_control")
            if emotion_control is None:
                self._emotion_control_ids = ()
            else:
                self._load_tokenizer()
                tokenizer = self._tokenizer
                assert tokenizer is not None
                self._emotion_control_ids = tuple(tokenizer.encode(emotion_control))
        return self._emotion_control_ids

    def prepare_voice(self, voice: str) -> PreparedVoice:
        voices = self.manifest["voices"]
        if voice not in voices:
            raise ValueError(
                f"Unknown voice {voice!r}; available voices: {', '.join(sorted(voices))}"
            )
        self._load_tokenizer()
        tokenizer = self._tokenizer
        assert tokenizer is not None
        reference = load_decoder_reference(self.root / voices[voice])
        transcript = reference.metadata.get("reference_transcript")
        if not isinstance(transcript, str):
            raise RuntimeContractError(f"Voice {voice!r} is missing its reference transcript")
        reference_text_ids = tokenizer.encode(transcript)
        reference_prefix = build_reference_prefix(
            self.token_map,
            reference_text_ids,
            reference.arrays["reference_tokens"].tolist(),
        )
        conditioning = {
            key: reference.arrays[key]
            for key in (
                "conditioning_prompt_token",
                "conditioning_prompt_token_len",
                "conditioning_prompt_feat",
                "conditioning_embedding",
            )
        }
        return PreparedVoice(
            speaker_embedding=np.asarray(reference.arrays["speaker_embedding"], dtype=np.float32),
            reference_prefix=reference_prefix,
            conditioning=conditioning,
        )
