"""The agent's tools, each exercised on its own with no model involved."""

import subprocess
from dataclasses import replace

import pytest

from devagent.agent import tools as toolset
from devagent.agent.tools import ToolContext, git_blame, read_file, run_tests, search_code
from devagent.sandbox.runner import TestResult
from devagent.retrieval.vector_search import RetrievedChunk
from tests.fakes import FakeProvider


class StubSession:
    """Stands in for a Session. Retrieval is faked, so nothing is queried."""


CHUNK = RetrievedChunk(
    file_path="fastapi/params.py",
    symbol="Depends",
    kind="code",
    start_line=745,
    end_line=749,
    text="class Depends:\n    dependency = None\n",
    score=0.44,
)


def make_ctx(tmp_path, source_repo=None):
    return ToolContext(
        workspace_root=tmp_path,
        source_repo=source_repo or tmp_path,
        provider=FakeProvider(),
        session=StubSession(),
        runner=None,
        repo="fastapi/fastapi",
    )


class FakeRunner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, workspace_root, target):
        self.calls.append(target)
        return self.result


@pytest.fixture
def retrieval_returning(monkeypatch):
    """Replace vector retrieval with a scripted sequence of results."""

    def install(*batches):
        calls = iter(batches)

        def fake_search(question, provider, session, repo=None, k=8):
            return list(next(calls))

        monkeypatch.setattr(toolset, "vector_search", fake_search)

    return install


def test_search_code_reports_location_and_score(tmp_path, retrieval_returning):
    retrieval_returning([CHUNK])
    out = search_code(make_ctx(tmp_path), "what is Depends")
    assert "fastapi/params.py" in out
    assert "745" in out
    assert "Depends" in out


def test_search_code_says_so_when_nothing_matches(tmp_path, retrieval_returning):
    retrieval_returning([])
    out = search_code(make_ctx(tmp_path), "nothing")
    assert "no matching" in out.lower()


def test_search_code_records_what_retrieval_returned(tmp_path, retrieval_returning):
    """Citations come from retrieval, never from model output -- the M1 rule."""
    retrieval_returning([CHUNK])
    ctx = make_ctx(tmp_path)
    search_code(ctx, "what is Depends")
    assert [c.file_path for c in ctx.citations] == ["fastapi/params.py"]


def test_a_chunk_found_twice_is_cited_once_at_its_best_score(tmp_path, retrieval_returning):
    """Two searches over similar ground otherwise repeat the same chunk."""
    other = replace(CHUNK, file_path="fastapi/routing.py", symbol="APIRoute", score=0.31)
    retrieval_returning([CHUNK], [replace(CHUNK, score=0.61), other])

    ctx = make_ctx(tmp_path)
    search_code(ctx, "what is Depends")
    search_code(ctx, "how is Depends resolved")

    assert len(ctx.citations) == 2
    depends = next(c for c in ctx.citations if c.file_path == "fastapi/params.py")
    assert depends.score == 0.61


def test_read_file_returns_the_requested_span(tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    out = read_file(make_ctx(tmp_path), "a.py", start=2, end=3)
    assert "two" in out and "three" in out
    assert "four" not in out


def test_read_file_refuses_to_escape_the_workspace(tmp_path):
    """A security boundary, so it gets a test rather than an assumption."""
    out = read_file(make_ctx(tmp_path), "../secrets.txt")
    assert "outside the workspace" in out


def test_read_file_refuses_an_absolute_path(tmp_path):
    out = read_file(make_ctx(tmp_path), "/etc/passwd")
    assert "outside the workspace" in out


def test_read_file_reports_a_missing_file_without_raising(tmp_path):
    out = read_file(make_ctx(tmp_path), "nope.py")
    assert "not found" in out.lower()


TARGET = "tests/test_params_repr.py"


def test_run_tests_reports_failures_as_a_finding(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.runner = FakeRunner(TestResult(True, 1, 39, 3, 4.0, "3 failed, 39 passed", None))
    out = run_tests(ctx, TARGET)
    assert "3 failed" in out
    assert "39 passed" in out


def test_run_tests_reports_a_green_run(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.runner = FakeRunner(TestResult(True, 0, 42, 0, 3.0, "42 passed", None))
    out = run_tests(ctx, TARGET)
    assert "42 passed" in out
    assert "0 failed" in out or "no failures" in out.lower()


def test_run_tests_distinguishes_a_broken_runner_from_failing_tests(tmp_path):
    """A timeout must never read as 'the tests passed'."""
    ctx = make_ctx(tmp_path)
    ctx.runner = FakeRunner(TestResult.failure("test run exceeded 300s and was killed"))
    out = run_tests(ctx, TARGET)
    assert "could not run" in out.lower()
    assert "exceeded" in out


def test_run_tests_runs_against_the_workspace_copy(tmp_path):
    """The patch lives in the copy; running the source would answer the wrong question."""
    ctx = make_ctx(tmp_path, source_repo=tmp_path / "source")
    ctx.runner = FakeRunner(TestResult(True, 0, 1, 0, 0.1, "1 passed", None))
    run_tests(ctx, TARGET)
    assert ctx.runner.calls == [TARGET]


def test_git_blame_refuses_to_escape_the_workspace(tmp_path):
    out = git_blame(make_ctx(tmp_path), "../../etc/passwd", 1, 5)
    assert "outside the workspace" in out


def test_git_blame_reads_the_source_checkout_not_the_workspace_copy(tmp_path):
    """Blame is about committed history, which a user patch never touches.

    Blaming the copy would mean unshallowing a directory that is about to be
    deleted, on every request that asks who wrote a line.
    """
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "a@example.com")
    _git(source, "config", "user.name", "Committed Author")
    (source / "a.py").write_text("original\n", encoding="utf-8")
    _git(source, "add", "a.py")
    _git(source, "commit", "-qm", "add a.py")

    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "a.py").write_text("patched\n", encoding="utf-8")

    ctx = ToolContext(
        workspace_root=copy,
        source_repo=source,
        provider=FakeProvider(),
        session=StubSession(),
        runner=None,
    )
    out = git_blame(ctx, "a.py", 1, 1)
    assert "Committed Author" in out
    assert "original" in out
    assert "patched" not in out


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
