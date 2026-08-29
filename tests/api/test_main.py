import pytest
from fastapi.testclient import TestClient

from devagent.api.main import app, _repo_name_from_url, get_provider_dep, get_session_dep
from devagent.db.models import Chunk as ChunkRow
from tests.fakes import FakeProvider


class StubResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class StubSession:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.merged = []

    def execute(self, statement):
        return StubResult(self.rows)

    def merge(self, obj):
        self.merged.append(obj)
        return obj

    def flush(self):
        pass


def make_row(symbol="Depends", distance=0.2):
    row = ChunkRow(
        repo="fastapi/fastapi",
        file_path="fastapi/param_functions.py",
        symbol=symbol,
        kind="code",
        start_line=10,
        end_line=20,
        text=f"def {symbol}(): ...",
        embedding=[0.0] * 8,
    )
    return (row, distance)


@pytest.fixture
def client():
    provider = FakeProvider(dimensions=8, answer="Depends declares a dependency.")
    session = StubSession([make_row()])
    app.dependency_overrides[get_provider_dep] = lambda: provider
    app.dependency_overrides[get_session_dep] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_an_answer(client):
    response = client.post("/query", json={"question": "what does Depends do?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Depends declares a dependency."


def test_query_returns_citations(client):
    response = client.post("/query", json={"question": "what does Depends do?"})
    citations = response.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["file_path"] == "fastapi/param_functions.py"
    assert citations[0]["symbol"] == "Depends"
    assert citations[0]["start_line"] == 10
    assert citations[0]["end_line"] == 20
    assert citations[0]["score"] == pytest.approx(0.8)


def test_query_rejects_an_empty_question(client):
    assert client.post("/query", json={"question": "   "}).status_code == 422


def test_query_rejects_a_missing_question(client):
    assert client.post("/query", json={}).status_code == 422


def test_query_on_an_empty_corpus_returns_409():
    provider = FakeProvider(dimensions=8)
    app.dependency_overrides[get_provider_dep] = lambda: provider
    app.dependency_overrides[get_session_dep] = lambda: StubSession([])
    try:
        response = TestClient(app).post("/query", json={"question": "anything"})
        assert response.status_code == 409
        assert "ingest" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_query_accepts_a_custom_k(client):
    response = client.post("/query", json={"question": "q", "k": 3})
    assert response.status_code == 200


def test_query_rejects_an_out_of_range_k(client):
    assert client.post("/query", json={"question": "q", "k": 0}).status_code == 422
    assert client.post("/query", json={"question": "q", "k": 101}).status_code == 422


def test_ingest_maps_a_clone_failure_to_502(monkeypatch, client):
    def boom(repo_url, repo_dir):
        raise RuntimeError("git clone of https://example.invalid/x failed: fatal: nope")

    monkeypatch.setattr("devagent.api.main.ensure_repo", boom)
    response = client.post("/ingest", json={})
    assert response.status_code == 502
    assert "fatal: nope" in response.json()["detail"], (
        "git's own error text must survive to the caller"
    )


def test_ingest_maps_a_partial_checkout_to_502(monkeypatch, client):
    def boom(repo_url, repo_dir):
        raise RuntimeError(
            "data/repos/fastapi exists but is not a git checkout (no .git). "
            "It may be a partial clone from an interrupted run. Remove it and try again."
        )

    monkeypatch.setattr("devagent.api.main.ensure_repo", boom)
    response = client.post("/ingest", json={})
    assert response.status_code == 502
    assert "not a git checkout" in response.json()["detail"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/fastapi/fastapi", "fastapi/fastapi"),
        ("https://github.com/fastapi/fastapi.git", "fastapi/fastapi"),
        ("https://github.com/fastapi/fastapi/", "fastapi/fastapi"),
        ("git@github.com:fastapi/fastapi.git", "git@github.com:fastapi/fastapi"),
        ("fastapi", "fastapi"),
    ],
)
def test_repo_name_from_url(url, expected):
    assert _repo_name_from_url(url) == expected
