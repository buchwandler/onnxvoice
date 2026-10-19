from __future__ import annotations

import pytest

from onnxvoice.systems.kitten2_expression import inspect_expression_markup


@pytest.mark.parametrize(
    "emotion",
    [
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
    ],
)
def test_recognizes_supported_emotions(emotion: str) -> None:
    text = f"[{emotion}] Hello."

    result = inspect_expression_markup(text)

    assert result.has_expression
    assert result.emotion == emotion
    assert result.event_count == 0
    assert result.emphasis_count == 0
    assert text == f"[{emotion}] Hello."


def test_leading_emotion_allows_whitespace_and_uses_only_the_first_tag() -> None:
    text = " \t\n[joyful] hello [sad]"
    result = inspect_expression_markup(text)

    assert result.has_expression
    assert result.emotion == "joyful"
    assert text == " \t\n[joyful] hello [sad]"


@pytest.mark.parametrize(
    "event",
    ["gasp", "giggle", "growl", "gulp", "laugh", "pause", "scoff", "sigh", "sob", "um"],
)
def test_recognizes_supported_vocal_events(event: str) -> None:
    result = inspect_expression_markup(f"Hello <{event}>.")

    assert result.has_expression
    assert result.emotion is None
    assert result.event_count == 1
    assert result.emphasis_count == 0


@pytest.mark.parametrize("text", ["[JOYFUL] hello", "<LAUGH>"])
def test_uppercase_markup_does_not_activate_expression(text: str) -> None:
    result = inspect_expression_markup(text)

    assert not result.has_expression
    assert result.emotion is None
    assert result.event_count == 0


def test_nonleading_emotion_does_not_declare_request_emotion() -> None:
    result = inspect_expression_markup("hello [sad] world")

    assert not result.has_expression
    assert result.emotion is None


def test_recognizes_emphasis_spans_up_to_eighty_non_newline_characters() -> None:
    text = f"(((short phrase))) ((({'x' * 80})))"

    result = inspect_expression_markup(text)

    assert result.has_expression
    assert result.emphasis_count == 2


@pytest.mark.parametrize(
    "text",
    [
        "(((" + "x" * 81 + ")))",
        "((()))",
        "(((nested (phrase)))",
        "(((unclosed)",
        "(((first\nsecond)))",
        "((word))",
        "[whisper] hello",
        "hello <door>",
        "((hello))",
        "[angry-ish] hello",
    ],
)
def test_unknown_or_unsupported_markup_does_not_activate_expression(text: str) -> None:
    result = inspect_expression_markup(text)

    assert not result.has_expression
    assert result.emotion is None
    assert result.event_count == 0
    assert result.emphasis_count == 0


def test_adjacent_markup_and_punctuation_are_recognized_without_rewriting() -> None:
    text = "[angry]hello(((really))),okay<laugh><sigh>"

    result = inspect_expression_markup(text)

    assert result.has_expression
    assert result.emotion == "angry"
    assert result.event_count == 2
    assert result.emphasis_count == 1
    assert text == "[angry]hello(((really))),okay<laugh><sigh>"
