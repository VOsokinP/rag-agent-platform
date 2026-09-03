"""Regression: the vector index must work when it is created before any data.

An IVFFlat index built on an empty table has no centroids, so rows inserted
afterwards land in lists no scan ever probes. The result is an index that
returns zero rows against a populated table -- `/query` reports an empty corpus
while the data sits in `chunks`. `init_db()` creates the schema before any
ingest, so this build order is the documented setup path, not an edge case.

Needs a live Postgres with pgvector. No API key, no network, no spend.
"""

import pytest
from sqlalchemy import MetaData, text
from sqlalchemy.orm import Session

from devagent.db.models import Chunk
from devagent.db.session import get_engine
from devagent.retrieval.vector_search import search
from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration

SCHEMA = "devagent_index_regression"
REPO = "test/repo"


@pytest.fixture
def empty_schema_session():
    """Yield a session over a throwaway schema whose index was built empty.

    A dedicated schema, rather than the real `chunks` table, is what makes the
    test meaningful: the bug only reproduces when the index is created with no
    rows present, which a table that has already been ingested cannot show.
    """
    connection = get_engine().connect()
    connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    connection.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    # `vector` is installed in public; keep it reachable for the column type.
    connection.execute(text(f"SET search_path TO {SCHEMA}, public"))
    # Copy the table definition into an explicitly schema-qualified MetaData so
    # creation cannot be skipped by the existing public.chunks.
    Chunk.__table__.to_metadata(MetaData(schema=SCHEMA)).create(bind=connection)
    connection.commit()

    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        connection.rollback()
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        connection.commit()
        connection.close()


def _store_chunks(session, provider, count):
    texts = [f"chunk number {i}" for i in range(count)]
    for index, (chunk_text, vector) in enumerate(zip(texts, provider.embed(texts))):
        session.add(
            Chunk(
                repo=REPO,
                file_path=f"module_{index}.py",
                symbol=f"symbol_{index}",
                kind="code",
                start_line=1,
                end_line=2,
                text=chunk_text,
                embedding=vector,
            )
        )
    session.commit()


def test_index_created_before_ingest_still_retrieves(empty_schema_session):
    provider = FakeProvider()
    _store_chunks(empty_schema_session, provider, count=50)

    # Force the planner onto the index. On a table this small it would otherwise
    # choose a sequential scan, which returns correct rows no matter what the
    # index holds and would let the bug through unnoticed.
    empty_schema_session.execute(text("SET enable_seqscan = off"))

    # `repo=None` mirrors the API default. It matters: a `repo` filter pushes
    # the planner back onto a sequential scan even with seqscan discouraged,
    # which is exactly how an earlier version of this test fooled itself.
    results = search("chunk number 7", provider, empty_schema_session, repo=None, k=8)

    assert len(results) == 8


def test_index_created_before_ingest_ranks_the_exact_match_first(
    empty_schema_session,
):
    """Retrieval must be correct, not merely non-empty."""
    provider = FakeProvider()
    _store_chunks(empty_schema_session, provider, count=50)
    empty_schema_session.execute(text("SET enable_seqscan = off"))

    results = search("chunk number 7", provider, empty_schema_session, repo=None, k=8)

    assert results[0].text == "chunk number 7"
