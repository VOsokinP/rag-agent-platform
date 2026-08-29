"""The DevAgent HTTP API."""

import logging
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from devagent.answer import EmptyCorpusError, answer_question
from devagent.api.schemas import (
    CitationOut,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from devagent.config import get_settings
from devagent.db.session import session_scope
from devagent.ingestion.clone import ensure_repo
from devagent.ingestion.pipeline import ingest
from devagent.llm.provider import Provider, get_provider
from devagent.retrieval.vector_search import search

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DevAgent", version="0.1.0")


def _repo_name_from_url(url: str) -> str:
    """Derive an `owner/name` label from a clone URL.

    Falls back to the last path segment when the URL has no owner segment, so a
    malformed URL still produces a usable label instead of raising mid-ingest.
    """
    trimmed = url.rstrip("/").removesuffix(".git")
    parts = [part for part in trimmed.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else url


def get_provider_dep() -> Provider:
    """Overridable in tests."""
    return get_provider()


def get_session_dep() -> Iterator[Any]:
    """Overridable in tests."""
    with session_scope() as session:
        yield session


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    request: IngestRequest,
    provider: Provider = Depends(get_provider_dep),
    session: Any = Depends(get_session_dep),
) -> IngestResponse:
    """Clone the target repo if needed, then chunk, embed, and store it."""
    settings = get_settings()
    repo_url = settings.repo_url
    repo_name = request.repo or _repo_name_from_url(repo_url)

    try:
        repo_dir = ensure_repo(repo_url, settings.repo_dir)
    except RuntimeError as exc:
        # ensure_repo raises RuntimeError for the two failures an operator can
        # actually act on: a partial checkout from an interrupted clone, and a
        # clone failure carrying git's own stderr. Losing that text to a generic
        # 500 would waste the work Task 6 did to make it actionable.
        #
        # Note: the message can echo the clone URL. That is fine for a
        # single-operator local tool, but if DevAgent ever grew auth or accepted
        # arbitrary URLs with embedded credentials, this would need redaction.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result = ingest(repo_name, repo_dir, provider, session)

    # Commit inside the request path, not in the dependency's exit code. An
    # exception raised while unwinding a `yield` dependency cannot change a
    # response that has already been produced, so a failed commit would return
    # 200 with a chunks_written count for rows that were rolled back.
    try:
        session.commit()
    except Exception as exc:  # noqa: BLE001 - any commit failure must not report success
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion completed but the transaction failed to commit: {exc}",
        ) from exc

    return IngestResponse(
        repo=repo_name,
        files_processed=result.files_processed,
        chunks_written=result.chunks_written,
        files_skipped=result.files_skipped,
        batches_failed=result.batches_failed,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    provider: Provider = Depends(get_provider_dep),
    session: Any = Depends(get_session_dep),
) -> QueryResponse:
    """Answer a question about the ingested repository, with citations."""
    chunks = search(
        request.question, provider, session, repo=request.repo, k=request.k
    )
    try:
        result = answer_question(request.question, chunks, provider)
    except EmptyCorpusError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        citations=[CitationOut(**vars(citation)) for citation in result.citations],
    )
