"""Heading-based chunking of Markdown documentation.

Each section under a heading becomes one chunk, carrying its full heading path
as the symbol so a citation reads `Dependencies > First Steps` rather than a
bare filename. Fenced code blocks are tracked so that a `#` comment inside a
code example is never mistaken for a heading — FastAPI's docs are largely
fenced Python, where that mistake would fabricate sections that don't exist.

Sections longer than `max_chars` are split on paragraph boundaries, then line
boundaries, and finally on character count. Line numbers for a chunk produced by
that last resort are approximate, since the chunk is then a fragment of a single
source line.
"""

import re

from devagent.ingestion.chunker import DOC, Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


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

    in_fence = False
    fence_char = ""
    fence_length = 0

    for number, line in enumerate(lines, start=1):
        fence_match = _FENCE.match(line)
        if fence_match is not None:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence, fence_char, fence_length = True, marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                in_fence = False
            current.append(line)
            continue

        if in_fence:
            current.append(line)
            continue

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


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph, preferring line boundaries.

    A single line longer than `max_chars` is cut on character count as a last
    resort: an oversized chunk is rejected or silently truncated by the
    embeddings API, which is a worse failure than an awkward split.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces: list[str] = []
    buffer: list[str] = []
    length = 0

    def flush_buffer() -> None:
        nonlocal buffer, length
        if buffer:
            pieces.append("\n".join(buffer))
            buffer, length = [], 0

    for line in paragraph.split("\n"):
        if len(line) > max_chars:
            flush_buffer()
            pieces.extend(
                line[start : start + max_chars] for start in range(0, len(line), max_chars)
            )
            continue
        if buffer and length + len(line) + 1 > max_chars:
            flush_buffer()
        buffer.append(line)
        length += len(line) + 1

    flush_buffer()
    return pieces


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

    oversized_expanded = [
        piece
        for paragraph in text.split("\n\n")
        for piece in _split_paragraph(paragraph, max_chars)
    ]
    for paragraph in oversized_expanded:
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
