"""The Chunk record shared by every chunker."""

from dataclasses import dataclass

CODE = "code"
DOC = "doc"


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of a repository.

    `symbol` is the human-meaningful name used in citations: a qualified
    function or class name for code (`APIRoute.get_route_handler`), a heading
    path for docs (`Dependencies > First Steps`), or `<module>` for the
    module-level code of a Python file.
    """

    file_path: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    text: str
