"""KittenTTS 2 text chunk boundaries that preserve the input verbatim."""

from __future__ import annotations

import unicodedata

from .kitten2_expression import expression_markup_spans


def split_kitten2_text(text: str, chunk_chars: int, chunk_min_chars: int = 0) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]

    protected_spans = expression_markup_spans(text)
    boundaries = [
        position
        for position, char in enumerate(text, start=1)
        if (char.isspace() or unicodedata.category(char).startswith("P"))
        and not any(start < position < end for start, end in protected_spans)
    ]
    chunks: list[str] = []
    start = 0
    while len(text) - start > chunk_chars:
        target = start + chunk_chars
        candidates = [position for position in boundaries if start < position <= target]
        if candidates:
            boundary = candidates[-1]
        else:
            boundary = target
            for span_start, span_end in protected_spans:
                if span_start < boundary < span_end:
                    boundary = span_start if span_start > start else span_end
                    break
        chunks.append(text[start:boundary])
        start = boundary

    chunks.append(text[start:])
    if chunk_min_chars > 0:
        chunks = _merge_short_chunks(chunks, chunk_chars, chunk_min_chars)
    return chunks


def _merge_short_chunks(chunks: list[str], chunk_chars: int, chunk_min_chars: int) -> list[str]:
    limit = int(chunk_chars * 1.4)
    merged = list(chunks)
    while len(merged) > 1:
        for index, chunk in enumerate(merged):
            if len(chunk) >= chunk_min_chars:
                continue
            previous_fits = index > 0 and len(merged[index - 1]) + len(chunk) <= limit
            next_fits = index + 1 < len(merged) and len(chunk) + len(merged[index + 1]) <= limit
            if previous_fits and (
                not next_fits or len(merged[index - 1]) <= len(merged[index + 1])
            ):
                merged[index - 1] += chunk
                merged.pop(index)
                break
            if next_fits:
                merged[index] += merged[index + 1]
                merged.pop(index + 1)
                break
        else:
            return merged
    return merged
