from pathlib import Path

import pytest

from devagent.ingestion.chunker import Chunk
from devagent.ingestion.pipeline import IngestResult, build_rows, chunk_file, ingest
from tests.fakes import FakeProvider


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "fastapi").mkdir()
    (tmp_path / "fastapi" / "routing.py").write_text(
        'def route():\n    """Route it."""\n    return 1\n', encoding="utf-8"
    )
    (tmp_path / "fastapi" / "broken.py").write_text("def nope(:\n", encoding="utf-8")
    docs = tmp_path / "docs" / "en" / "docs"
    docs.mkdir(parents=True)
    (docs / "index.md").write_text("# Title\n\nBody text.\n", encoding="utf-8")
    return tmp_path


class RecordingSession:
    """Captures what the pipeline would persist, without a database."""

    def __init__(self) -> None:
        self.statements: list[object] = []

    def execute(self, statement: object) -> None:
        self.statements.append(statement)

    def flush(self) -> None:
        pass


def test_chunk_file_dispatches_python(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("def f():\n    return 1\n", encoding="utf-8")
    chunks = chunk_file(path, "a.py")
    assert [c.symbol for c in chunks] == ["f"]
    assert chunks[0].kind == "code"


def test_chunk_file_dispatches_markdown(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# Heading\n\nBody.\n", encoding="utf-8")
    chunks = chunk_file(path, "a.md")
    assert [c.symbol for c in chunks] == ["Heading"]
    assert chunks[0].kind == "doc"


def test_chunk_file_returns_empty_for_unknown_suffix(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("plain", encoding="utf-8")
    assert chunk_file(path, "a.txt") == []


def test_ingest_writes_chunks(mini_repo):
    session = RecordingSession()
    result = ingest("fastapi/fastapi", mini_repo, FakeProvider(dimensions=8), session)
    assert isinstance(result, IngestResult)
    assert result.chunks_written > 0
    # One delete + at least one upsert.
    assert len(session.statements) >= 2


def test_ingest_deletes_existing_rows_before_writing(mini_repo):
    """A re-ingest must clear a file's old chunks, not just upsert over some."""
    session = RecordingSession()
    ingest("fastapi/fastapi", mini_repo, FakeProvider(dimensions=8), session)
    compiled = [str(statement) for statement in session.statements]
    assert any(text.startswith("DELETE") for text in compiled), compiled


def test_ingest_reports_skipped_unparseable_files(mini_repo):
    result = ingest(
        "fastapi/fastapi", mini_repo, FakeProvider(dimensions=8), RecordingSession()
    )
    assert "fastapi/broken.py" in result.files_skipped


def test_ingest_embeds_every_chunk(mini_repo):
    provider = FakeProvider(dimensions=8)
    result = ingest("fastapi/fastapi", mini_repo, provider, RecordingSession())
    embedded = sum(len(call) for call in provider.embed_calls)
    assert embedded == result.chunks_written


def test_ingest_respects_batch_size(mini_repo):
    provider = FakeProvider(dimensions=8)
    ingest(
        "fastapi/fastapi", mini_repo, provider, RecordingSession(), batch_size=1
    )
    assert all(len(call) == 1 for call in provider.embed_calls)


def test_build_rows_sets_repo_on_every_row():
    chunks = [
        Chunk(
            file_path="a.py",
            symbol="f",
            kind="code",
            start_line=1,
            end_line=2,
            text="def f(): ...",
        )
    ]
    rows = build_rows(chunks, [[0.0] * 8], "fastapi/fastapi")
    assert [row["repo"] for row in rows] == ["fastapi/fastapi"]


def test_build_rows_pairs_each_chunk_with_its_own_vector():
    chunks = [
        Chunk(
            file_path="a.py",
            symbol=name,
            kind="code",
            start_line=index + 1,
            end_line=index + 1,
            text=name,
        )
        for index, name in enumerate(["f", "g"])
    ]
    rows = build_rows(chunks, [[1.0] * 8, [2.0] * 8], "r")
    assert rows[0]["symbol"] == "f" and rows[0]["embedding"] == [1.0] * 8
    assert rows[1]["symbol"] == "g" and rows[1]["embedding"] == [2.0] * 8


def test_build_rows_rejects_a_length_mismatch():
    """zip(strict=True) must catch a vector/chunk misalignment rather than
    silently truncating — a misaligned embedding is a silent correctness bug."""
    chunks = [
        Chunk(
            file_path="a.py",
            symbol="f",
            kind="code",
            start_line=1,
            end_line=1,
            text="x",
        )
    ]
    with pytest.raises(ValueError):
        build_rows(chunks, [], "r")


def test_ingest_survives_a_failing_embedding_batch(mini_repo):
    """One bad batch must not cost the whole run — later batches still land."""

    class FailsFirstBatchProvider(FakeProvider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("API is down")
            return super().embed(texts)

    provider = FailsFirstBatchProvider(dimensions=8)
    session = RecordingSession()
    result = ingest("fastapi/fastapi", mini_repo, provider, session, batch_size=1)

    assert provider.calls > 1, "fixture must produce more than one batch"
    assert result.batches_failed == 1
    assert result.chunks_written == provider.calls - 1
    assert result.chunks_written > 0, "a later batch must still be persisted"


def test_ingest_reports_zero_written_when_every_batch_fails(mini_repo):
    class FailingProvider(FakeProvider):
        def embed(self, texts):
            raise RuntimeError("API is down")

    result = ingest(
        "fastapi/fastapi", mini_repo, FailingProvider(dimensions=8), RecordingSession()
    )
    assert result.batches_failed > 0
    assert result.chunks_written == 0


def test_ingest_counts_files_processed(mini_repo):
    result = ingest(
        "fastapi/fastapi", mini_repo, FakeProvider(dimensions=8), RecordingSession()
    )
    # routing.py, broken.py, index.md
    assert result.files_processed == 3
