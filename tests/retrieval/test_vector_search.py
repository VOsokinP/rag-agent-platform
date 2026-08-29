from devagent.db.models import Chunk as ChunkRow
from devagent.retrieval.vector_search import RetrievedChunk, search
from tests.fakes import FakeProvider


class StubResult:
    """Stands in for the rows SQLAlchemy would return from the similarity query."""

    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class StubSession:
    """Captures the statement it was given and returns canned rows."""

    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def execute(self, statement):
        self.executed.append(statement)
        return StubResult(self.rows)


def make_row(symbol: str, distance: float):
    row = ChunkRow(
        repo="fastapi/fastapi",
        file_path="fastapi/routing.py",
        symbol=symbol,
        kind="code",
        start_line=1,
        end_line=5,
        text=f"source of {symbol}",
        embedding=[0.0] * 8,
    )
    return (row, distance)


def test_returns_retrieved_chunks():
    session = StubSession([make_row("f", 0.1), make_row("g", 0.4)])
    results = search("what is f", FakeProvider(dimensions=8), session)
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert [r.symbol for r in results] == ["f", "g"]


def test_converts_cosine_distance_to_similarity_score():
    session = StubSession([make_row("f", 0.25)])
    results = search("q", FakeProvider(dimensions=8), session)
    assert results[0].score == 0.75


def test_copies_citation_fields_from_the_row():
    session = StubSession([make_row("f", 0.1)])
    result = search("q", FakeProvider(dimensions=8), session)[0]
    assert result.file_path == "fastapi/routing.py"
    assert result.kind == "code"
    assert result.start_line == 1
    assert result.end_line == 5
    assert result.text == "source of f"


def test_embeds_the_question_once():
    provider = FakeProvider(dimensions=8)
    search("what is f", provider, StubSession([]))
    assert provider.embed_calls == [["what is f"]]


def test_empty_corpus_returns_empty_list():
    assert search("q", FakeProvider(dimensions=8), StubSession([])) == []


def _sql(statement) -> str:
    from sqlalchemy.dialects import postgresql

    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def test_limit_is_applied_to_the_statement():
    session = StubSession([])
    search("q", FakeProvider(dimensions=8), session, k=3)
    assert "LIMIT 3" in _sql(session.executed[0])


def test_default_k_is_eight():
    session = StubSession([])
    search("q", FakeProvider(dimensions=8), session)
    assert "LIMIT 8" in _sql(session.executed[0])


def test_repo_filter_adds_a_where_clause():
    session = StubSession([])
    search("q", FakeProvider(dimensions=8), session, repo="fastapi/fastapi")
    assert "WHERE" in _sql(session.executed[0])


def test_no_repo_filter_means_no_where_clause():
    session = StubSession([])
    search("q", FakeProvider(dimensions=8), session)
    assert "WHERE" not in _sql(session.executed[0])


def test_results_are_ordered_by_distance():
    """Without ORDER BY, the stub still returns rows in order and every other
    test passes — only real Postgres would reveal the loss."""
    session = StubSession([])
    search("q", FakeProvider(dimensions=8), session)
    sql = _sql(session.executed[0])
    assert "ORDER BY" in sql
    assert "<=>" in sql, "ordering must be by the cosine distance operator"
