"""The agent's tools.

Every tool returns text and never raises. A tool that raises unwinds the graph
and turns a situation the model could have recovered from -- a wrong path, a
missing file -- into a 500 with no answer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devagent.retrieval.vector_search import RetrievedChunk
from devagent.retrieval.vector_search import search as vector_search

MAX_SPAN = 400


@dataclass
class ToolContext:
    """Everything the tools need, passed explicitly rather than imported.

    Holding these on a context object is what lets the unit tests exercise each
    tool with fakes and no database, network, or Docker.
    """

    workspace_root: Path
    provider: Any
    session: Any
    runner: Any
    repo: str | None = None
    citations: list[RetrievedChunk] = field(default_factory=list)

    def resolve(self, path: str) -> Path:
        """Resolve `path` inside the workspace, or raise ValueError."""
        candidate = (self.workspace_root / path).resolve()
        if not candidate.is_relative_to(self.workspace_root.resolve()):
            raise ValueError("outside the workspace")
        return candidate

    def cite(self, chunks: list[RetrievedChunk]) -> None:
        """Record retrieved chunks, keeping one entry per span.

        Two searches over similar ground return overlapping chunks, and without
        this the same lines appear several times in the response's sources.
        """
        by_span = {(c.file_path, c.start_line, c.end_line): c for c in self.citations}
        for chunk in chunks:
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            existing = by_span.get(key)
            if existing is None or chunk.score > existing.score:
                by_span[key] = chunk
        self.citations[:] = list(by_span.values())


def search_code(ctx: ToolContext, query: str, k: int = 8) -> str:
    """Find code and docs chunks relevant to `query`."""
    chunks = vector_search(query, ctx.provider, ctx.session, repo=ctx.repo, k=k)
    if not chunks:
        return "No matching chunks were found."
    # Citations come from what retrieval actually returned, never from model
    # output -- the Milestone 1 rule, preserved.
    ctx.cite(chunks)
    return "\n\n".join(
        f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
        f"({chunk.symbol or '-'}, score {chunk.score:+.2f})\n{chunk.text}"
        for chunk in chunks
    )


def read_file(ctx: ToolContext, path: str, start: int = 1, end: int = 200) -> str:
    """Read lines `start`..`end` of a file in the workspace."""
    try:
        resolved = ctx.resolve(path)
    except ValueError:
        return f"Refused: {path} is outside the workspace."
    if not resolved.is_file():
        return f"Not found: {path}"

    end = min(end, start + MAX_SPAN)
    body = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    span = body[max(start - 1, 0) : end]
    numbered = "\n".join(f"{n:>6}  {line}" for n, line in enumerate(span, start=start))
    return f"{path} lines {start}-{start + len(span) - 1}:\n{numbered}"
