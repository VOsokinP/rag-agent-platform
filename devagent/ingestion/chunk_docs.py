"""Heading-based chunking of Markdown documentation.

Each section under a heading becomes one chunk, carrying its full heading path
as the symbol so a citation reads `Dependencies > First Steps` rather than a
bare filename. Sections longer than `max_chars` are split on paragraph
boundaries, with every piece keeping the same heading path.
"""

import re

from devagent.ingestion.chunker import DOC, Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def chunk_markdown(source: str, file_path: str, max_chars: int = 4000) -> list[Chunk]:
    """Split Markdown into one chunk per heading section."""
    if not source.strip():
        return []

    chunks: list[Chunk] = []
    for symbol, start_line, body in _sections(source):
        for text, offset in _split_oversized(body, max_chars):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    file_path=file_path,
                    symbol=symbol,
                    kind=DOC,
                    start_line=start_line + offset,
                    end_line=start_line + offset + text.count("\n"),
                    text=text,
                )
            )
    return chunks


def _sections(source: str) -> list[tuple[str | None, int, str]]:
    """Yield (heading path, 1-based start line, section text) for each section."""
    lines = source.splitlines()
    sections: list[tuple[str | None, int, str]] = []
    path: list[str] = []
    current: list[str] = []
    current_symbol: str | None = None
    current_start = 1

    def flush() -> None:
        text = "\n".join(current).strip()
        if text:
            sections.append((current_symbol, current_start, text))

    for number, line in enumerate(lines, start=1):
        match = _HEADING.match(line)
        if match is None:
            current.append(line)
            continue

        flush()
        level, title = len(match.group(1)), match.group(2)
        # Trim the path to this heading's depth, then push this heading on.
        path = path[: level - 1]
        path.append(title)
        current_symbol = " > ".join(path)
        current_start = number
        current = [line]

    flush()
    return sections


def _split_oversized(text: str, max_chars: int) -> list[tuple[str, int]]:
    """Split text on paragraph boundaries into pieces of at most max_chars.

    Returns (piece, line offset from the section start) pairs.
    """
    if len(text) <= max_chars:
        return [(text, 0)]

    pieces: list[tuple[str, int]] = []
    buffer: list[str] = []
    buffer_len = 0
    offset = 0
    consumed_lines = 0

    for paragraph in text.split("\n\n"):
        addition = len(paragraph) + 2
        if buffer and buffer_len + addition > max_chars:
            joined = "\n\n".join(buffer)
            pieces.append((joined, offset))
            consumed_lines += joined.count("\n") + 2
            offset = consumed_lines
            buffer, buffer_len = [], 0
        buffer.append(paragraph)
        buffer_len += addition

    if buffer:
        pieces.append(("\n\n".join(buffer), offset))

    return pieces
