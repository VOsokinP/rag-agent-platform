"""The Chunk record and the greedy packer shared by every chunker."""

from collections.abc import Iterator, Sequence
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


def pack(
    items: Sequence[str], max_chars: int, separator_len: int
) -> Iterator[tuple[int, int]]:
    """Yield `(start, stop)` index ranges of consecutive items that fit together.

    Greedy: a group grows until the next item would take it past `max_chars`,
    counting `separator_len` characters for the join between items. An item that
    is itself over the limit is yielded alone rather than dropped — splitting it
    is the caller's business, since only the caller knows what a valid piece is.

    Every chunker needs this, and each used to write the separator arithmetic
    itself. One wrong `+ 1` produces chunks a little over the embedding input
    limit, which the API rejects for the whole batch, so the arithmetic lives
    here once.
    """
    start = 0
    length = 0
    for index, item in enumerate(items):
        addition = len(item) + separator_len
        if index > start and length + addition > max_chars:
            yield start, index
            start, length = index, 0
        length += addition
    if start < len(items):
        yield start, len(items)
