"""The agent's tools.

Every tool returns text and never raises. A tool that raises unwinds the graph
and turns a situation the model could have recovered from -- a wrong path, a
missing file -- into a 500 with no answer.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devagent.retrieval.vector_search import RetrievedChunk
from devagent.retrieval.vector_search import search as vector_search

MAX_SPAN = 400


def _confine(root: Path, path: str) -> Path:
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("outside the workspace")
    return candidate


@dataclass
class ToolContext:
    """Everything the tools need, passed explicitly rather than imported.

    Holding these on a context object is what lets the unit tests exercise each
    tool with fakes and no database, network, or Docker.
    """

    workspace_root: Path
    # The ingested checkout, kept alongside the throwaway copy: blame is about
    # committed history, which a user's patch never touches, and unshallowing a
    # directory that is about to be deleted would pay a fetch per request.
    source_repo: Path
    provider: Any
    session: Any
    runner: Any
    repo: str | None = None
    # How many chunks a search returns when the model does not say. Comes
    # from the request, so a caller can widen retrieval without new code.
    k: int = 8
    citations: list[RetrievedChunk] = field(default_factory=list)

    def resolve(self, path: str) -> Path:
        """Resolve `path` inside the workspace, or raise ValueError."""
        return _confine(self.workspace_root, path)

    def resolve_source(self, path: str) -> Path:
        """Resolve `path` inside the ingested checkout, or raise ValueError."""
        return _confine(self.source_repo, path)

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


def search_code(ctx: ToolContext, query: str, k: int | None = None) -> str:
    """Find code and docs chunks relevant to `query`."""
    chunks = vector_search(
        query, ctx.provider, ctx.session, repo=ctx.repo, k=k or ctx.k
    )
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


def run_tests(ctx: ToolContext, target: str) -> str:
    """Run a test file from the workspace in the sandbox."""
    result = ctx.runner.run(ctx.workspace_root, target)
    if not result.ok:
        # Distinguished deliberately: "the tests did not run" is an error to
        # surface, not a finding to report. Collapsing the two lets a broken
        # runner masquerade as a clean bill of health.
        return f"The tests could not run: {result.error}"
    verdict = "no failures" if result.failed == 0 else f"{result.failed} failed"
    return (
        f"{target}: {verdict}, {result.passed} passed "
        f"in {result.duration:.1f}s (exit {result.exit_code})\n{result.output_tail}"
    )


def git_blame(ctx: ToolContext, path: str, start: int, end: int) -> str:
    """Show who last changed lines `start`..`end` of a file."""
    try:
        ctx.resolve_source(path)
    except ValueError:
        return f"Refused: {path} is outside the workspace."

    if _is_shallow(ctx.source_repo):
        # The ingest clone is --depth 1, so blame would credit every line to a
        # single squashed commit. Unshallowing is slow, but it runs against the
        # checkout that outlives the request, so it happens once.
        completed = subprocess.run(
            ["git", "fetch", "--unshallow"],
            cwd=ctx.source_repo, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            return f"Could not fetch history for blame: {completed.stderr.strip()}"

    completed = subprocess.run(
        ["git", "blame", "-L", f"{start},{end}", "--", path],
        cwd=ctx.source_repo, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        return f"blame failed: {completed.stderr.strip()}"
    return completed.stdout


def _is_shallow(repo_dir: Path) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo_dir, capture_output=True, text=True,
    )
    return completed.stdout.strip() == "true"
