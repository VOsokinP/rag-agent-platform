import pytest
from fastapi.testclient import TestClient

from devagent.api.main import (
    _repo_name_from_url,
    app,
    get_provider_dep,
    get_session_dep,
)
from devagent.db.models import Chunk as ChunkRow
from tests.fakes import FakeProvider


class StubResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class StubSession:
    """The subset of Session the endpoints actually use: execute and commit."""

    def __init__(self, rows=(), commit_error: Exception | None = None):
        self.rows = list(rows)
        self.commit_error = commit_error
        self.commits = 0

    def execute(self, statement):
        return StubResult(self.rows)

    def commit(self):
        self.commits += 1
        if self.commit_error is not None:
            raise self.commit_error


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


def _mini_repo(tmp_path):
    """A checkout-shaped directory holding one ingestible Python file."""
    package = tmp_path / "fastapi"
    package.mkdir()
    (package / "x.py").write_text(
        'def hello(name):\n    """Greet."""\n    return name\n',
        encoding="utf-8",
    )
    return tmp_path


def test_ingest_writes_chunks_and_reports_them(monkeypatch, client, tmp_path):
    repo_dir = _mini_repo(tmp_path)
    monkeypatch.setattr(
        "devagent.api.main.ensure_repo", lambda repo_url, target: repo_dir
    )
    response = client.post("/ingest", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["repo"] == "fastapi/fastapi", "label defaults to the configured URL"
    assert body["files_processed"] > 0
    assert body["chunks_written"] > 0
    assert body["files_skipped"] == []
    assert body["batches_failed"] == 0


def test_ingest_repo_overrides_only_the_label(monkeypatch, client, tmp_path):
    repo_dir = _mini_repo(tmp_path)
    monkeypatch.setattr(
        "devagent.api.main.ensure_repo", lambda repo_url, target: repo_dir
    )
    response = client.post("/ingest", json={"repo": "custom/name"})
    assert response.status_code == 200
    assert response.json()["repo"] == "custom/name"


def test_ingest_ignores_an_unknown_field(monkeypatch, client, tmp_path):
    """repo_url was removed: the checkout directory comes from settings, so a
    per-request URL would ingest one repository under another's name. Pydantic's
    default config ignores unknown fields, so sending it is a no-op rather than
    a silent mislabel."""
    repo_dir = _mini_repo(tmp_path)
    monkeypatch.setattr(
        "devagent.api.main.ensure_repo", lambda repo_url, target: repo_dir
    )
    response = client.post(
        "/ingest", json={"repo_url": "https://github.com/pallets/flask"}
    )
    assert response.status_code == 200
    assert response.json()["repo"] == "fastapi/fastapi"


def test_ingest_reports_500_when_the_commit_fails(monkeypatch, tmp_path):
    """A failed commit must not be reported as a successful ingest."""
    repo_dir = _mini_repo(tmp_path)
    monkeypatch.setattr(
        "devagent.api.main.ensure_repo", lambda repo_url, target: repo_dir
    )
    session = StubSession(commit_error=RuntimeError("deadlock detected"))
    app.dependency_overrides[get_provider_dep] = lambda: FakeProvider(dimensions=8)
    app.dependency_overrides[get_session_dep] = lambda: session
    try:
        response = TestClient(app).post("/ingest", json={})
        assert response.status_code == 500
        assert "deadlock detected" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_ingest_uses_the_configured_include_globs(monkeypatch, client, tmp_path):
    """REPO_URL is only a real knob if file selection follows the configuration."""
    package = tmp_path / "src"
    package.mkdir()
    (package / "y.py").write_text("def hi():\n    return 1\n", encoding="utf-8")
    monkeypatch.setenv("INCLUDE_CODE_GLOB", "src/**/*.py")
    monkeypatch.setattr(
        "devagent.api.main.ensure_repo", lambda repo_url, target: tmp_path
    )
    response = client.post("/ingest", json={})
    assert response.status_code == 200, response.text
    assert response.json()["files_processed"] == 1
