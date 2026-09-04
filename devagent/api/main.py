"""The DevAgent HTTP API."""

import logging
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from devagent.agent.graph import run_agent
from devagent.agent.tools import ToolContext
from devagent.answer import EmptyCorpusError, answer_question
from devagent.api.schemas import (
    AgentRequest,
    AgentResponse,
    CitationOut,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StepOut,
    UsageOut,
)
from devagent.config import get_settings
from devagent.db.session import session_scope
from devagent.ingestion.clone import ensure_repo
from devagent.ingestion.pipeline import ingest
from devagent.llm.chat import chat_model
from devagent.llm.provider import Provider, get_provider
from devagent.retrieval.vector_search import search
from devagent.sandbox.runner import DockerRunner
from devagent.sandbox.workspace import PatchError, workspace

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
    result = ingest(
        repo_name, repo_dir, provider, session, settings.include_globs
    )

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


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(
    request: AgentRequest,
    provider: Provider = Depends(get_provider_dep),
    session: Any = Depends(get_session_dep),
) -> AgentResponse:
    """Answer a question with tools, optionally against a patched checkout."""
    settings = get_settings()
    try:
        with workspace(settings.repo_dir, request.patch) as root:
            context = ToolContext(
                workspace_root=root,
                # Blame reads committed history, which the patch never touches
                # and which the copy does not carry.
                source_repo=settings.repo_dir,
                provider=provider,
                session=session,
                runner=DockerRunner(image=settings.runner_image),
                repo=_repo_name_from_url(settings.repo_url),
                k=request.k,
            )
            result = run_agent(request.question, context, chat_model())
    except PatchError as exc:
        # The patch is the user's input, so this is a 400 -- and it is caught
        # before any model call, so a bad diff costs nothing.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AgentResponse(
        answer=result.answer,
        steps=[StepOut(**vars(step)) for step in result.steps],
        citations=[CitationOut(**vars(c)) for c in result.citations],
        usage=UsageOut(**vars(result.usage)),
        budget_exhausted=result.budget_exhausted,
    )
