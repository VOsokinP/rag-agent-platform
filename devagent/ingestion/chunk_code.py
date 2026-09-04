"""AST-based chunking of Python source.

One chunk per top-level function, class, and method, plus a single `<module>`
chunk holding the file's module-level code (imports, constants). Chunking on
definition boundaries rather than fixed-size windows means a function is never
split mid-body, and citations name a real symbol instead of a line range.

Chunks are capped at `max_chars` because the embeddings API rejects the whole
request when any single input exceeds its token limit — one oversized chunk
would otherwise silently discard its entire batch. A class that exceeds the cap
becomes a header chunk (its methods are already chunked separately); a function
that exceeds it is split on line boundaries.
"""

import ast
import logging

from devagent.ingestion.chunker import CODE, Chunk, pack

logger = logging.getLogger(__name__)

_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def chunk_python_source(
    source: str, file_path: str, max_chars: int = 8000
) -> list[Chunk]:
    """Split Python source into chunks. Returns [] if the source cannot be parsed.

    No returned chunk's text exceeds `max_chars`. The default leaves headroom
    under `text-embedding-3-small`'s 8,192-token input limit at the ~4 chars per
    token that Python source averages.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("Skipping %s: could not parse (%s)", file_path, exc)
        return []

    lines = source.splitlines()
    chunks: list[Chunk] = []
    definition_line_numbers: set[int] = set()

    for node in tree.body:
        if not isinstance(node, _DEFINITION_NODES):
            continue

        definition_line_numbers.update(range(_start_line(node), _end_line(node) + 1))
        chunk = _chunk_from_node(node, node.name, lines, file_path)
        if chunk is not None:
            if isinstance(node, ast.ClassDef) and len(chunk.text) > max_chars:
                header = _class_header_chunk(node, lines, file_path)
                if header is not None:
                    chunks.extend(_split_oversized_chunk(header, max_chars))
            else:
                chunks.extend(_split_oversized_chunk(chunk, max_chars))

        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = _chunk_from_node(
                        member, f"{node.name}.{member.name}", lines, file_path
                    )
                    if method_chunk is not None:
                        chunks.extend(_split_oversized_chunk(method_chunk, max_chars))

    module_chunk = _module_chunk(lines, definition_line_numbers, file_path)
    if module_chunk is not None:
        chunks.extend(_split_oversized_chunk(module_chunk, max_chars))

    return chunks


def _class_header_chunk(
    node: ast.ClassDef, lines: list[str], file_path: str
) -> Chunk | None:
    """A chunk covering a class's decorators, signature, and docstring only.

    Used when the full class body exceeds the embedding input limit. The methods
    are already separate chunks, so re-splitting the body would embed them twice;
    the header still gives retrieval something that describes the class itself.
    """
    start = _start_line(node)
    end = node.body[0].lineno - 1 if node.body else node.lineno
    docstring = ast.get_docstring(node)
    if docstring is not None and node.body:
        end = getattr(node.body[0], "end_lineno", node.body[0].lineno)
    text = "\n".join(lines[start - 1 : end]).strip("\n")
    if not text.strip():
        return None
    return Chunk(
        file_path=file_path,
        symbol=node.name,
        kind=CODE,
        start_line=start,
        end_line=end,
        text=text,
    )


def _split_oversized_chunk(chunk: Chunk, max_chars: int) -> list[Chunk]:
    """Split a chunk that would exceed the embedding input limit.

    Splits on line boundaries so every piece stays syntactically readable, and
    derives each piece's line range from its offset within the original, so the
    pieces never share a (start_line, end_line) pair — that pair is part of the
    database's uniqueness key.
    """
    if len(chunk.text) <= max_chars:
        return [chunk]

    lines = chunk.text.split("\n")
    return [
        Chunk(
            file_path=chunk.file_path,
            symbol=chunk.symbol,
            kind=chunk.kind,
            start_line=chunk.start_line + start,
            end_line=chunk.start_line + stop - 1,
            text="\n".join(lines[start:stop]),
        )
        for start, stop in pack(lines, max_chars, separator_len=1)
    ]


def _start_line(node: ast.stmt) -> int:
    """The first line of a definition, counting its decorators.

    `node.lineno` points at the `def`/`class` keyword, so a decorated definition
    would otherwise lose its decorators to the module chunk — and on this corpus
    the decorator is often the most identifying line in the whole definition.
    """
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return node.lineno


def _end_line(node: ast.stmt) -> int:
    """The last line of a node, falling back to its start line."""
    return getattr(node, "end_lineno", None) or node.lineno


def _chunk_from_node(
    node: ast.stmt, symbol: str, lines: list[str], file_path: str
) -> Chunk | None:
    """Build a chunk for one definition, or None if it has no source text."""
    start, end = _start_line(node), _end_line(node)
    text = "\n".join(lines[start - 1 : end]).strip("\n")
    if not text.strip():
        return None
    return Chunk(
        file_path=file_path,
        symbol=symbol,
        kind=CODE,
        start_line=start,
        end_line=end,
        text=text,
    )


def _module_chunk(
    lines: list[str], definition_line_numbers: set[int], file_path: str
) -> Chunk | None:
    """Collect the lines that aren't part of any definition into one chunk.

    Returns None when nothing but whitespace is left over, which is the normal
    case for a file that is entirely class and function definitions.
    """
    kept = [
        (number, line)
        for number, line in enumerate(lines, start=1)
        if number not in definition_line_numbers and line.strip()
    ]
    if not kept:
        return None

    text = "\n".join(line for _, line in kept).strip()
    if not text:
        return None

    return Chunk(
        file_path=file_path,
        symbol="<module>",
        kind=CODE,
        start_line=kept[0][0],
        end_line=kept[-1][0],
        text=text,
    )
