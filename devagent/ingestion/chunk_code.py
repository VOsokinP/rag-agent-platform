"""AST-based chunking of Python source.

One chunk per top-level function, class, and method, plus a single `<module>`
chunk holding the file's module-level code (imports, constants). Chunking on
definition boundaries rather than fixed-size windows means a function is never
split mid-body, and citations name a real symbol instead of a line range.
"""

import ast
import logging

from devagent.ingestion.chunker import CODE, Chunk

logger = logging.getLogger(__name__)

_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def chunk_python_source(source: str, file_path: str) -> list[Chunk]:
    """Split Python source into chunks. Returns [] if the source cannot be parsed."""
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

        definition_line_numbers.update(range(node.lineno, _end_line(node) + 1))
        chunks.append(_chunk_from_node(node, node.name, source, file_path))

        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        _chunk_from_node(
                            member, f"{node.name}.{member.name}", source, file_path
                        )
                    )

    module_chunk = _module_chunk(lines, definition_line_numbers, file_path)
    if module_chunk is not None:
        chunks.append(module_chunk)

    return chunks


def _end_line(node: ast.AST) -> int:
    """The last line of a node, falling back to its start line."""
    return getattr(node, "end_lineno", None) or node.lineno


def _chunk_from_node(node: ast.AST, symbol: str, source: str, file_path: str) -> Chunk:
    text = ast.get_source_segment(source, node) or ""
    return Chunk(
        file_path=file_path,
        symbol=symbol,
        kind=CODE,
        start_line=node.lineno,
        end_line=_end_line(node),
        text=text,
    )


def _module_chunk(
    lines: list[str], definition_line_numbers: set[int], file_path: str
) -> Chunk | None:
    """Collect the lines that aren't part of any definition into one chunk.

    Returns None when there is nothing but whitespace left over, which is the
    normal case for a file that is entirely class and function definitions.
    """
    kept = [
        line
        for number, line in enumerate(lines, start=1)
        if number not in definition_line_numbers
    ]
    text = "\n".join(kept).strip()
    if not text:
        return None

    return Chunk(
        file_path=file_path,
        symbol="<module>",
        kind=CODE,
        start_line=1,
        end_line=len(lines),
        text=text,
    )
