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

        definition_line_numbers.update(range(_start_line(node), _end_line(node) + 1))
        chunk = _chunk_from_node(node, node.name, lines, file_path)
        if chunk is not None:
            chunks.append(chunk)

        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = _chunk_from_node(
                        member, f"{node.name}.{member.name}", lines, file_path
                    )
                    if method_chunk is not None:
                        chunks.append(method_chunk)

    module_chunk = _module_chunk(lines, definition_line_numbers, file_path)
    if module_chunk is not None:
        chunks.append(module_chunk)

    return chunks


def _start_line(node: ast.AST) -> int:
    """The first line of a definition, counting its decorators.

    `node.lineno` points at the `def`/`class` keyword, so a decorated definition
    would otherwise lose its decorators to the module chunk — and on this corpus
    the decorator is often the most identifying line in the whole definition.
    """
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return node.lineno


def _end_line(node: ast.AST) -> int:
    """The last line of a node, falling back to its start line."""
    return getattr(node, "end_lineno", None) or node.lineno


def _chunk_from_node(
    node: ast.AST, symbol: str, lines: list[str], file_path: str
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
