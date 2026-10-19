"""Recognition of model-specific KittenTTS 2 expression markup."""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMOTIONS = (
    "angry",
    "contemplative",
    "excited",
    "joyful",
    "mundane",
    "nervous",
    "sad",
    "stern",
    "surprised",
    "tender",
)
_VOCAL_EVENTS = (
    "gasp",
    "giggle",
    "growl",
    "gulp",
    "laugh",
    "pause",
    "scoff",
    "sigh",
    "sob",
    "um",
)
_EMOTION_PATTERN = r"\[(?P<emotion>" + "|".join(_EMOTIONS) + r")\]"
_EVENT_PATTERN = r"<(?P<event>" + "|".join(_VOCAL_EVENTS) + r")>"
_EMOTION_TAG = re.compile(_EMOTION_PATTERN)
_VOCAL_EVENT = re.compile(_EVENT_PATTERN)
_LEADING_EMOTION = re.compile(r"^\s*(?P<tag>" + _EMOTION_PATTERN + r")")
_EMPHASIS_SPAN = re.compile(r"\(\(\(([^()\n]{1,80})\)\)\)")


def expression_markup_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return source ranges occupied by supported Kitten2 markup."""
    spans = [match.span() for match in _EMOTION_TAG.finditer(text)]
    spans.extend(match.span() for match in _VOCAL_EVENT.finditer(text))
    spans.extend(match.span() for match in _EMPHASIS_SPAN.finditer(text))
    return tuple(sorted(spans))


@dataclass(frozen=True, slots=True)
class ExpressionInfo:
    has_expression: bool
    emotion: str | None
    event_count: int
    emphasis_count: int


def inspect_expression_markup(text: str) -> ExpressionInfo:
    """Report supported Kitten2 markup without changing the supplied text."""
    leading_emotion = _LEADING_EMOTION.match(text)
    events = tuple(_VOCAL_EVENT.finditer(text))
    emphasis_count = sum(1 for _ in _EMPHASIS_SPAN.finditer(text))
    return ExpressionInfo(
        has_expression=bool(leading_emotion or events or emphasis_count),
        emotion=leading_emotion.group("emotion") if leading_emotion else None,
        event_count=len(events),
        emphasis_count=emphasis_count,
    )
