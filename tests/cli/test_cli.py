"""Tests for the `devagent` command-line client.

Every test drives `main()` through an httpx MockTransport, so the CLI's real
request-building and output formatting run without a server, a database, a key,
or any network access.
"""

import httpx
import pytest

from devagent import cli


def transport_returning(payload, status_code=200, capture=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler)


def transport_raising(exc):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


ANSWER_PAYLOAD = {
    "answer": "Depends() declares a dependency.",
    "citations": [
        {
            "file_path": "fastapi/param_functions.py",
            "symbol": "Depends",
            "start_line": 10,
            "end_line": 40,
            "score": 0.61,
        }
    ],
}

INGEST_PAYLOAD = {
    "repo": "fastapi/fastapi",
    "files_processed": 900,
    "chunks_written": 2666,
    "files_skipped": [],
    "batches_failed": 0,
}


def test_query_prints_the_answer(capsys):
    code = cli.main(
        ["query", "what does Depends do?"],
        transport=transport_returning(ANSWER_PAYLOAD),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Depends() declares a dependency." in out


def test_query_prints_citations_with_location_and_score(capsys):
    cli.main(["query", "q"], transport=transport_returning(ANSWER_PAYLOAD))
    out = capsys.readouterr().out
    assert "fastapi/param_functions.py" in out
    assert "10-40" in out
    assert "0.61" in out


def test_query_sends_question_k_and_repo():
    captured = []
    cli.main(
        ["query", "  what does Depends do?  ", "-k", "3", "--repo", "fastapi/fastapi"],
        transport=transport_returning(ANSWER_PAYLOAD, capture=captured),
    )
    request = captured[0]
    assert request.url.path == "/query"
    body = request.read().decode()
    assert '"k":3' in body.replace(" ", "")
    assert "what does Depends do?" in body
    assert "fastapi/fastapi" in body


def test_query_omits_repo_when_not_given():
    """A repo filter changes retrieval; the CLI must not invent one."""
    captured = []
    cli.main(["query", "q"], transport=transport_returning(ANSWER_PAYLOAD, capture=captured))
    body = captured[0].read().decode().replace(" ", "")
    assert '"repo":null' in body or '"repo"' not in body


def test_query_on_empty_corpus_explains_how_to_fix_it(capsys):
    transport = transport_returning(
        {"detail": "No chunks retrieved. Has the repository been ingested yet?"},
        status_code=409,
    )
    code = cli.main(["query", "q"], transport=transport)
    captured = capsys.readouterr()
    assert code != 0
    assert "devagent ingest" in captured.out + captured.err


def test_ingest_reports_the_counts(capsys):
    code = cli.main(["ingest"], transport=transport_returning(INGEST_PAYLOAD))
    out = capsys.readouterr().out
    assert code == 0
    assert "2666" in out.replace(",", "")
    assert "900" in out


def test_ingest_warns_and_fails_when_batches_failed(capsys):
    """A failed batch leaves that file's older chunks in place, so the corpus is
    silently stale. That must not look like a clean run."""
    payload = dict(INGEST_PAYLOAD, batches_failed=2)
    code = cli.main(["ingest"], transport=transport_returning(payload))
    captured = capsys.readouterr()
    assert code != 0
    assert "stale" in (captured.out + captured.err).lower()


def test_ingest_lists_skipped_files(capsys):
    payload = dict(INGEST_PAYLOAD, files_skipped=["docs/broken.md"])
    cli.main(["ingest"], transport=transport_returning(payload))
    assert "docs/broken.md" in capsys.readouterr().out


def test_ingest_passes_the_repo_label():
    captured = []
    cli.main(
        ["ingest", "--repo", "my/label"],
        transport=transport_returning(INGEST_PAYLOAD, capture=captured),
    )
    assert "my/label" in captured[0].read().decode()


def test_health_reports_ok(capsys):
    code = cli.main(["health"], transport=transport_returning({"status": "ok"}))
    assert code == 0
    assert "ok" in capsys.readouterr().out


def test_connection_refused_names_the_command_that_starts_the_server(capsys):
    transport = transport_raising(httpx.ConnectError("refused"))
    code = cli.main(["query", "q"], transport=transport)
    message = capsys.readouterr().err
    assert code != 0
    assert "uvicorn devagent.api.main:app" in message


def test_server_error_is_reported_without_a_traceback(capsys):
    transport = transport_returning({"detail": "boom"}, status_code=500)
    code = cli.main(["query", "q"], transport=transport)
    captured = capsys.readouterr()
    assert code != 0
    assert "boom" in captured.out + captured.err


def test_url_flag_overrides_the_default_host():
    captured = []
    cli.main(
        ["--url", "http://example.test:9000", "health"],
        transport=transport_returning({"status": "ok"}, capture=captured),
    )
    assert str(captured[0].url) == "http://example.test:9000/health"


def test_url_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("DEVAGENT_URL", "http://from-env:9100")
    captured = []
    cli.main(["health"], transport=transport_returning({"status": "ok"}, capture=captured))
    assert str(captured[0].url) == "http://from-env:9100/health"


def test_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("DEVAGENT_URL", raising=False)
    captured = []
    cli.main(["health"], transport=transport_returning({"status": "ok"}, capture=captured))
    assert str(captured[0].url) == "http://localhost:8000/health"


def test_ingest_has_no_read_timeout():
    """Ingest embeds the whole corpus and runs for minutes. httpx's 5s default
    would abort the client while the server keeps working."""
    captured = []
    cli.main(["ingest"], transport=transport_returning(INGEST_PAYLOAD, capture=captured))
    assert captured[0].extensions["timeout"]["read"] is None


def test_query_keeps_a_read_timeout():
    captured = []
    cli.main(["query", "q"], transport=transport_returning(ANSWER_PAYLOAD, capture=captured))
    assert captured[0].extensions["timeout"]["read"] is not None


def test_blank_question_is_rejected_before_any_request():
    captured = []
    code = cli.main(
        ["query", "   "], transport=transport_returning(ANSWER_PAYLOAD, capture=captured)
    )
    assert code != 0
    assert captured == []


def test_long_symbols_are_truncated_so_the_table_stays_readable(capsys):
    """Markdown chunks carry the full heading path as their symbol, which can
    run to hundreds of characters and wrap every row into unreadability."""
    long_symbol = "Dependencies > First Steps > " + "Declare the dependency " * 10
    payload = {
        "answer": "a",
        "citations": [
            {
                "file_path": "docs/index.md",
                "symbol": long_symbol,
                "start_line": 1,
                "end_line": 2,
                "score": 0.5,
            }
        ],
    }
    cli.main(["query", "q"], transport=transport_returning(payload))
    out = capsys.readouterr().out
    assert long_symbol not in out
    assert "Dependencies > First Steps" in out
    assert max(len(line) for line in out.splitlines()) < 120


def test_truncation_marker_is_ascii(capsys):
    """The Windows console defaults to cp1252, where a Unicode ellipsis is a
    replacement box at best and a UnicodeEncodeError at worst."""
    payload = {
        "answer": "a",
        "citations": [
            {
                "file_path": "docs/index.md",
                "symbol": "heading " * 40,
                "start_line": 1,
                "end_line": 2,
                "score": 0.5,
            }
        ],
    }
    cli.main(["query", "q"], transport=transport_returning(payload))
    out = capsys.readouterr().out
    assert "..." in out
    out.encode("cp1252")  # raises if any character is unencodable
