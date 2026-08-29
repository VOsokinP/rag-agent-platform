"""Walk a checked-out repository, chunk it, embed it, and persist it."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

from devagent.db.models import Chunk as ChunkRow
from devagent.ingestion.chunk_code import chunk_python_source
from devagent.ingestion.chunk_docs import chunk_markdown
from devagent.ingestion.chunker import Chunk
from devagent.ingestion.clone import iter_source_files
from devagent.llm.provider import Provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """What one ingestion run did.

    `files_skipped` lists files that had content but produced no chunks —
    unreadable or unparseable. A file that is empty (or whitespace-only) is
    not counted here: legitimately empty files (e.g. `__init__.py`) would
    otherwise drown out genuine failures.
    """

    files_processed: int = 0
    chunks_written: int = 0
    files_skipped: list[str] = field(default_factory=list)
    batches_failed: int = 0


def chunk_file(path: Path, relative_path: str) -> list[Chunk]:
    """Chunk one file based on its suffix. Returns [] if it can't be handled."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping %s: could not read (%s)", relative_path, exc)
        return []

    if path.suffix == ".py":
        return chunk_python_source(source, relative_path)
    if path.suffix == ".md":
        return chunk_markdown(source, relative_path)
    return []


def _has_content(path: Path) -> bool:
    """True if the file holds anything but whitespace."""
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError):
        # Unreadable is exactly the case worth reporting.
        return True


def build_rows(
    chunks: list[Chunk], vectors: list[list[float]], repo: str
) -> list[dict[str, Any]]:
    """Pair chunks with their embeddings into insertable row dicts."""
    return [
        {
            "repo": repo,
            "file_path": chunk.file_path,
            "symbol": chunk.symbol,
            "kind": chunk.kind,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "text": chunk.text,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def _upsert(session: Any, rows: list[dict[str, Any]]) -> None:
    """Insert rows, updating in place when the chunk location already exists.

    `session.merge()` would not work here: it matches on primary key, which is
    None for new rows, so it always inserts and a re-ingest would violate
    `uq_chunk_location` instead of refreshing the row.
    """
    statement = pg_insert(ChunkRow).values(rows)
    statement = statement.on_conflict_do_update(
        constraint="uq_chunk_location",
        set_={
            "symbol": statement.excluded.symbol,
            "kind": statement.excluded.kind,
            "text": statement.excluded.text,
            "embedding": statement.excluded.embedding,
        },
    )
    session.execute(statement)


def _delete_existing(session: Any, repo: str, file_paths: list[str]) -> None:
    """Drop the stored chunks for these files so a re-ingest can't leave orphans.

    Upserting alone is not enough: the conflict key includes the line range, so
    an edit that shifts a definition writes a new row and silently strands the
    old one, which would keep surfacing in retrieval describing code that no
    longer exists.
    """
    if not file_paths:
        return
    session.execute(
        delete(ChunkRow).where(
            ChunkRow.repo == repo, ChunkRow.file_path.in_(file_paths)
        )
    )


def ingest(
    repo: str,
    repo_dir: Path,
    provider: Provider,
    session: Any,
    batch_size: int = 100,
) -> IngestResult:
    """Chunk, embed, and upsert every included file in an existing checkout.

    A file with content that yields no chunks is reported in `files_skipped`;
    an embedding batch that fails is counted and skipped. Neither aborts the
    run — losing a whole ingest to one unparseable file or one flaky API call
    is the failure mode this guards against. Every processed file's existing
    rows are deleted before new ones are written, so a re-ingest can't leave
    stale rows describing code that no longer exists.
    """
    all_chunks: list[Chunk] = []
    skipped: list[str] = []
    processed_paths: list[str] = []
    files_processed = 0

    for absolute, relative in iter_source_files(repo_dir):
        files_processed += 1
        processed_paths.append(relative)
        chunks = chunk_file(absolute, relative)
        if not chunks:
            if _has_content(absolute):
                skipped.append(relative)
            continue
        all_chunks.extend(chunks)

    _delete_existing(session, repo, processed_paths)

    written = 0
    batches_failed = 0

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        try:
            vectors = provider.embed([chunk.text for chunk in batch])
        except Exception as exc:  # noqa: BLE001 - one bad batch must not fail the run
            logger.warning(
                "Embedding batch at offset %d failed (%s); skipping it", start, exc
            )
            batches_failed += 1
            continue

        rows = build_rows(batch, vectors, repo)
        _upsert(session, rows)
        written += len(rows)

    return IngestResult(
        files_processed=files_processed,
        chunks_written=written,
        files_skipped=skipped,
        batches_failed=batches_failed,
    )
