"""Heading-based chunking of Markdown documentation.

Each section under a heading becomes one chunk, carrying its full heading path
as the symbol so a citation reads `Dependencies > First Steps` rather than a
bare filename. Fenced code blocks are tracked so that a `#` comment inside a
code example is never mistaken for a heading — FastAPI's docs are largely
fenced Python, where that mistake would fabricate sections that don't exist.

Sections longer than `max_chars` are split on paragraph boundaries, then line
boundaries, and finally on character count. Line numbers for a chunk produced by
that last resort are approximate, since the chunk is then a fragment of a single
source line; they are clamped to the section's own line span so an approximation
can never point into a different section or past the end of the file.
"""

import re

from devagent.ingestion.chunker import DOC, Chunk, pack

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def chunk_markdown(source: str, file_path: str, max_chars: int = 4000) -> list[Chunk]:
    """Split Markdown into one chunk per heading section."""
    if not source.strip():
        return []

    chunks: list[Chunk] = []
    taken: set[tuple[int, int]] = set()
    for symbol, start_line, last_line, body in _sections(source):
        for text, offset in _split_oversized(body, max_chars):
            if not text.strip():
                continue
            # Clamp to the section's own line span. The offsets for an oversized
            # section are approximate — hard-splitting one very long line yields
            # several pieces that all live on that single source line — so without
            # a clamp the offsets run past the section, land on another section's
            # lines, and two chunks end up with the same (start_line, end_line).
            # That pair is the uq_chunk_location key, and Postgres aborts an
            # ON CONFLICT statement that names the same key twice.
            start = min(start_line + offset, last_line)
            end = min(start + text.count("\n"), last_line)
            location = _free_range(taken, start, end, start_line, last_line)
            if location is None:
                # The section has no distinct range left to give. Dropping the
                # overflow of an already-truncated long line beats emitting a
                # duplicate key that would abort the whole ingest.
                continue
            taken.add(location)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    symbol=symbol,
                    kind=DOC,
                    start_line=location[0],
                    end_line=location[1],
                    text=text,
                )
            )
    return chunks


def _free_range(
    taken: set[tuple[int, int]],
    start: int,
    end: int,
    section_start: int,
    section_end: int,
) -> tuple[int, int] | None:
    """Find an unused (start_line, end_line) pair inside the section.

    `(repo, file_path, start_line, end_line)` is `uq_chunk_location`, so two
    chunks of one file may not share a pair. Ordinary chunks never collide and
    keep their exact range — the first candidate tried is the one passed in.
    Collisions only arise between hard-split fragments of a single line longer
    than `max_chars`, whose line numbers the module docstring already describes
    as approximate. For those, the end is widened first (the content really does
    begin at `start`), then the start is walked back toward the section heading.
    Returns None when the section has no free pair left.
    """
    if (start, end) not in taken:
        return (start, end)
    for candidate_end in range(end + 1, section_end + 1):
        if (start, candidate_end) not in taken:
            return (start, candidate_end)
    for candidate_start in range(start - 1, section_start - 1, -1):
        for candidate_end in range(section_end, candidate_start - 1, -1):
            if (candidate_start, candidate_end) not in taken:
                return (candidate_start, candidate_end)
    return None


def _sections(source: str) -> list[tuple[str | None, int, int, str]]:
    """Yield (heading path, start line, last line, text) for each section.

    Both line numbers are 1-based and refer to the source. The last line lets a
    caller clamp an oversized section's split pieces to the lines the section
    actually occupies.
    """
    lines = source.splitlines()
    sections: list[tuple[str | None, int, int, str]] = []
    path: list[str] = []
    current: list[str] = []
    current_symbol: str | None = None
    current_start = 1

    def flush() -> None:
        text = "\n".join(current).strip()
        if text:
            last = current_start + len(current) - 1
            sections.append((current_symbol, current_start, last, text))

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
    run: list[str] = []

    def flush_run() -> None:
        """Emit whatever has accumulated since the last hard-cut line."""
        for start, stop in pack(run, max_chars, separator_len=1):
            pieces.append("\n".join(run[start:stop]))
        run.clear()

    for line in paragraph.split("\n"):
        if len(line) > max_chars:
            flush_run()
            pieces.extend(
                line[start : start + max_chars]
                for start in range(0, len(line), max_chars)
            )
            continue
        run.append(line)

    flush_run()
    return pieces


def _split_oversized(text: str, max_chars: int) -> list[tuple[str, int]]:
    """Split text on paragraph boundaries into pieces of at most max_chars.

    Returns (piece, line offset from the section start) pairs.
    """
    if len(text) <= max_chars:
        return [(text, 0)]

    paragraphs = [
        piece
        for paragraph in text.split("\n\n")
        for piece in _split_paragraph(paragraph, max_chars)
    ]

    pieces: list[tuple[str, int]] = []
    consumed_lines = 0
    offset = 0
    for start, stop in pack(paragraphs, max_chars, separator_len=2):
        joined = "\n\n".join(paragraphs[start:stop])
        pieces.append((joined, offset))
        # A paragraph break is two lines of source, so a piece consumes its
        # own newlines plus the blank line that followed it.
        consumed_lines += joined.count("\n") + 2
        offset = consumed_lines

    return pieces
