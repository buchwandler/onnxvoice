"""Standard-library table renderer for onnxvoice CLI.

Supports stable column ordering, ANSI suppression on redirect,
human-readable byte formatting (IEC units), deterministic output,
TTY-aware truncation, and plain/tsv/json format modes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Byte formatting (IEC units)
# ---------------------------------------------------------------------------

_IEC_UNITS = [
    (1 << 40, "TiB"),
    (1 << 30, "GiB"),
    (1 << 20, "MiB"),
    (1 << 10, "KiB"),
    (1, "B"),
]


def format_bytes(n: int | None) -> str:
    """Format a byte count in human-readable IEC units.

    Examples:
        0 -> "0 B"
        812 -> "812 B"
        66270000 -> "63.2 MiB"
        1438814042 -> "1.34 GiB"
    """
    if n is None:
        return "-"
    if n == 0:
        return "0 B"
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    for threshold, unit in _IEC_UNITS:
        if abs_n >= threshold:
            value = abs_n / threshold
            if value >= 100:
                return f"{sign}{value:.0f} {unit}"
            elif value >= 10:
                return f"{sign}{value:.1f} {unit}"
            else:
                return f"{sign}{value:.2f} {unit}"
    return f"{sign}{abs_n} B"


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _is_tty(stream: TextIO) -> bool:
    """Check if a stream is a TTY."""
    return hasattr(stream, "isatty") and stream.isatty()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and text[j] not in "mABCDEFGHJKSTfn":
                j += 1
            i = j + 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def _visible_len(text: str) -> int:
    """Visible length of text after stripping ANSI codes."""
    return len(_strip_ansi(text))


def _truncate(text: str, max_width: int) -> str:
    """Truncate text to max_width visible characters."""
    if _visible_len(text) <= max_width:
        return text
    # Build up character by character
    result: list[str] = []
    visible = 0
    in_escape = False
    for ch in text:
        if ch == "\x1b":
            in_escape = True
            result.append(ch)
            continue
        if in_escape:
            result.append(ch)
            if ch == "m":
                in_escape = False
            continue
        if visible >= max_width - 1:
            break
        result.append(ch)
        visible += 1
    return "".join(result) + "~"


def render_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    file: TextIO | None = None,
    max_col_width: int = 40,
) -> None:
    """Render a table with stable column ordering.

    - ANSI colors stripped when output is redirected.
    - Columns aligned to content width (up to max_col_width).
    - No output if columns/rows are empty.
    """
    if file is None:
        file = sys.stdout

    is_tty = _is_tty(file)

    # Strip ANSI if not a TTY
    if not is_tty:
        clean_columns = [_strip_ansi(c) for c in columns]
        clean_rows = [[_strip_ansi(cell) for cell in row] for row in rows]
    else:
        clean_columns = list(columns)
        clean_rows = [list(row) for row in rows]

    if not clean_columns:
        return

    # Calculate column widths
    col_widths = [len(c) for c in clean_columns]
    for row in clean_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], _visible_len(cell))

    # Apply max width
    col_widths = [min(w, max_col_width) for w in col_widths]

    # Header
    header_parts = []
    for i, col in enumerate(clean_columns):
        w = col_widths[i] if i < len(col_widths) else 0
        header_parts.append(col.ljust(w))
    file.write("  ".join(header_parts) + "\n")

    # Separator
    sep_parts = ["-" * w for w in col_widths]
    file.write("  ".join(sep_parts) + "\n")

    # Rows
    for row in clean_rows:
        parts = []
        for i in range(len(clean_columns)):
            cell = row[i] if i < len(row) else ""
            w = col_widths[i] if i < len(col_widths) else 0
            if _visible_len(cell) > w and w > 0:
                cell = _truncate(cell, w)
            parts.append(cell.ljust(w))
        file.write("  ".join(parts) + "\n")


def render_plain(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    file: TextIO | None = None,
) -> None:
    """Render one record per line, tab-separated (TSV)."""
    if file is None:
        file = sys.stdout

    for row in rows:
        file.write("\t".join(str(cell) for cell in row) + "\n")


def render_tsv(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    file: TextIO | None = None,
) -> None:
    """Render as TSV with header row."""
    if file is None:
        file = sys.stdout

    file.write("\t".join(columns) + "\n")
    for row in rows:
        file.write("\t".join(str(cell) for cell in row) + "\n")


def render_json(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    file: TextIO | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Render as JSON array of objects.

    Each row becomes an object keyed by column names.
    Extra keys are merged into the root object.
    """
    if file is None:
        file = sys.stdout

    items = []
    for row in rows:
        obj: dict[str, Any] = {}
        for i, col in enumerate(columns):
            obj[col] = row[i] if i < len(row) else None
        items.append(obj)

    output: dict[str, Any] = {"items": items}
    if extra:
        output.update(extra)

    json.dump(output, file, ensure_ascii=False, indent=2, sort_keys=True)
    file.write("\n")


def render_inventory(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    format: str = "table",
    file: TextIO | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Dispatch to the appropriate renderer based on format string.

    Supported formats: table, plain, tsv, json
    """
    if format == "json":
        render_json(columns, rows, file=file, extra=extra)
    elif format == "plain":
        render_plain(columns, rows, file=file)
    elif format == "tsv":
        render_tsv(columns, rows, file=file)
    else:
        render_table(columns, rows, file=file)
