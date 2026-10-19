from __future__ import annotations

from onnxvoice.systems.kitten2_chunking import split_kitten2_text
from onnxvoice.systems.kitten2_expression import expression_markup_spans


def test_chunking_preserves_exact_text_at_whitespace_and_punctuation_boundaries() -> None:
    text = "First sentence. " + "A long phrase, with punctuation; " * 8 + "Last sentence!  "

    chunks = split_kitten2_text(text, chunk_chars=45, chunk_min_chars=0)

    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(len(chunk) <= 45 for chunk in chunks)


def test_chunking_does_not_split_supported_expression_markup() -> None:
    text = "a" * 16 + " [angry] " + "b" * 16 + " (((one, two))) " + "c" * 16
    spans = expression_markup_spans(text)

    chunks = split_kitten2_text(text, chunk_chars=18, chunk_min_chars=0)

    offsets = []
    position = 0
    for chunk in chunks[:-1]:
        position += len(chunk)
        offsets.append(position)
    assert all(not start < boundary < end for boundary in offsets for start, end in spans)
    assert any("[angry]" in chunk for chunk in chunks)
    assert any("(((one, two)))" in chunk for chunk in chunks)
    assert "".join(chunks) == text


def test_chunking_hard_splits_unbroken_text_without_changing_characters() -> None:
    text = "x" * 53

    chunks = split_kitten2_text(text, chunk_chars=20)

    assert [len(chunk) for chunk in chunks] == [20, 20, 13]
    assert "".join(chunks) == text


def test_chunk_min_merges_short_neighboring_chunks_without_inserting_text() -> None:
    text = ". " + "x" * 23

    chunks = split_kitten2_text(text, chunk_chars=12, chunk_min_chars=5)

    assert [len(chunk) for chunk in chunks] == [14, 11]
    assert "".join(chunks) == text


def test_disabled_chunking_returns_original_text_as_one_chunk() -> None:
    text = "  Keep punctuation, spacing, and [unknown] markup.  "

    assert split_kitten2_text(text, chunk_chars=0, chunk_min_chars=1) == [text]
