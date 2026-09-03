"""The golden set is written by hand, so every rule here guards a hand-made
mistake that would otherwise be silent."""

import pytest

from devagent.eval.dataset import GoldenSetError, load_golden

VALID = """\
- question: what does Depends do?
  expect_files: [fastapi/params.py, fastapi/param_functions.py]
  expect_symbols: [Depends]
  kind: conceptual

- question: APIRoute.get_route_handler
  expect_files: [fastapi/routing.py]
  kind: identifier
"""


def write(tmp_path, text):
    path = tmp_path / "golden.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_entries_in_file_order(tmp_path):
    questions = load_golden(write(tmp_path, VALID))
    assert [q.question for q in questions] == [
        "what does Depends do?",
        "APIRoute.get_route_handler",
    ]
    assert questions[0].expect_files == ("fastapi/params.py", "fastapi/param_functions.py")
    assert questions[0].kind == "conceptual"


def test_expect_symbols_defaults_to_empty(tmp_path):
    """Symbols are recorded but never scored, so they are optional."""
    questions = load_golden(write(tmp_path, VALID))
    assert questions[1].expect_symbols == ()


def test_rejects_an_unknown_kind(tmp_path):
    text = "- question: q\n  expect_files: [a.py]\n  kind: conceptaul\n"
    with pytest.raises(GoldenSetError, match="conceptaul"):
        load_golden(write(tmp_path, text))


def test_rejects_a_blank_question(tmp_path):
    text = "- question: '   '\n  expect_files: [a.py]\n  kind: identifier\n"
    with pytest.raises(GoldenSetError, match="question"):
        load_golden(write(tmp_path, text))


def test_rejects_an_entry_with_no_expected_files(tmp_path):
    """Nothing to hit means the question can only ever score zero."""
    text = "- question: q\n  expect_files: []\n  kind: identifier\n"
    with pytest.raises(GoldenSetError, match="expect_files"):
        load_golden(write(tmp_path, text))


def test_rejects_a_duplicate_question(tmp_path):
    text = VALID + "\n- question: what does Depends do?\n  expect_files: [a.py]\n  kind: identifier\n"
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden(write(tmp_path, text))


def test_rejects_a_path_that_is_not_repo_relative(tmp_path):
    """expect_files is compared against RetrievedChunk.file_path, which is always
    a repo-relative posix path, so an absolute one could never match."""
    text = "- question: q\n  expect_files: ['/fastapi/params.py']\n  kind: identifier\n"
    with pytest.raises(GoldenSetError, match="repo-relative"):
        load_golden(write(tmp_path, text))


def test_rejects_a_top_level_mapping(tmp_path):
    text = "questions:\n  - question: q\n"
    with pytest.raises(GoldenSetError, match="list"):
        load_golden(write(tmp_path, text))


def test_reports_a_missing_file_by_path(tmp_path):
    with pytest.raises(GoldenSetError, match="golden.yaml"):
        load_golden(tmp_path / "golden.yaml")
