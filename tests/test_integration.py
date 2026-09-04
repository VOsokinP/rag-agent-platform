"""End-to-end tests against the real repository, a real database, and the real API.

Deselected by default. Run with:
    pytest -m integration -v
Requires: `docker compose up -d postgres`, a valid OPENAI_API_KEY, and network access.
"""

import pytest

from devagent.answer import answer_question
from devagent.config import get_settings
from devagent.db.session import init_db, session_scope
from devagent.ingestion.clone import ensure_repo, iter_source_files
from devagent.ingestion.pipeline import ingest
from devagent.llm.provider import get_provider
from devagent.retrieval.vector_search import search

pytestmark = pytest.mark.integration

REPO = "fastapi/fastapi"


@pytest.fixture(scope="module")
def ingested():
    settings = get_settings()
    init_db()
    repo_dir = ensure_repo(settings.repo_url, settings.repo_dir)
    with session_scope() as session:
        result = ingest(
            REPO, repo_dir, get_provider(), session, settings.include_globs
        )
    return result


def test_clone_produces_source_files():
    settings = get_settings()
    repo_dir = ensure_repo(settings.repo_url, settings.repo_dir)
    files = list(iter_source_files(repo_dir, settings.include_globs))
    assert len(files) > 100


def test_ingest_writes_a_substantial_number_of_chunks(ingested):
    assert ingested.chunks_written > 1000
    assert ingested.batches_failed == 0


def test_retrieval_finds_the_depends_definition(ingested):
    """FastAPI defines `Depends` twice: the frozen dataclass in `params.py` and
    the function wrapper in `param_functions.py`. Either is the definition this
    question is asking about, and which one ranks higher shifts with the docs
    that compete for the same query, so pinning the assertion to one file path
    makes the test fail on upstream drift rather than on a retrieval regression.
    """
    with session_scope() as session:
        results = search(
            "what does Depends do?", get_provider(), session, repo=REPO, k=8
        )
    assert results
    assert any(
        r.symbol == "Depends"
        and r.file_path in {"fastapi/params.py", "fastapi/param_functions.py"}
        for r in results
    )


def test_answer_is_grounded_and_cited(ingested):
    with session_scope() as session:
        chunks = search(
            "what does Depends do?", get_provider(), session, repo=REPO, k=8
        )
        answer = answer_question("what does Depends do?", chunks, get_provider())
    assert answer.answer.strip()
    assert answer.citations
    assert all(c.file_path.endswith((".py", ".md")) for c in answer.citations)
