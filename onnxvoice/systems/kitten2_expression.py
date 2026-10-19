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
_EXPRESSION_TAG = re.compile(
    r"\[(?P<emotion>" + "|".join(_EMOTIONS) + r")\]"
    r"|<(?P<event>" + "|".join(_VOCAL_EVENTS) + r")>",
    re.IGNORECASE,
)
_EMPHASIS_SPAN = re.compile(r"\(\(\(([^()\n]{1,80})\)\)\)")


def expression_markup_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return source ranges occupied by recognized Kitten2 markup."""
    spans = [match.span() for match in _EXPRESSION_TAG.finditer(text)]
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
    matches = tuple(_EXPRESSION_TAG.finditer(text))
    emotions = [match.group("emotion").lower() for match in matches if match.group("emotion")]
    event_count = sum(match.group("event") is not None for match in matches)
    emphasis_count = sum(1 for _ in _EMPHASIS_SPAN.finditer(text))
    return ExpressionInfo(
        has_expression=bool(emotions or event_count or emphasis_count),
        emotion=emotions[0] if emotions else None,
        event_count=event_count,
        emphasis_count=emphasis_count,
    )
