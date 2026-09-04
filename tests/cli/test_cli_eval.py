"""Output formatting for `devagent eval`, with a hand-built Report.

The wiring -- database, provider, gate -- is covered by the integration test in
tests/eval/test_regression_gate.py. What is worth unit-testing here is the
table, because a number printed without its sample size is how a thin
population gets misread.
"""

from devagent.cli import _corpus_commit, _corpus_empty, format_report
from devagent.eval.runner import QuestionResult, Report, score


def build_report():
    results = [
        QuestionResult("q1", "identifier", 1, ("a.py",), "code"),
        QuestionResult("q2", "identifier", None, ("z.py",), None),
        QuestionResult("q3", "conceptual", 2, ("z.py", "b.py"), "docs"),
    ]
    by_kind = {
        kind: score([r for r in results if r.kind == kind])
        for kind in ("identifier", "conceptual")
    }
    return Report(
        embedding_model="text-embedding-3-small",
        k=10,
        overall=score(results),
        by_kind=by_kind,
        results=tuple(results),
    )


def test_the_table_reports_every_kind_and_the_overall_row():
    text = format_report(build_report())
    assert "overall" in text
    assert "identifier" in text
    assert "conceptual" in text


def test_every_row_carries_its_sample_size():
    """A recall figure without n beside it invites reading noise as a result."""
    text = format_report(build_report())
    for line in text.splitlines():
        if line.startswith(("overall", "identifier", "conceptual")):
            assert "n=" in line, line


def test_the_embedding_model_is_printed():
    assert "text-embedding-3-small" in format_report(build_report())


def test_filtering_by_kind_drops_the_other_populations():
    text = format_report(build_report(), kind="identifier")
    assert "identifier" in text
    assert "conceptual" not in text


def test_the_table_reports_which_half_of_the_corpus_answered():
    """Hybrid search moves hits from docs to code; an aggregate recall number
    hides that entirely."""
    assert "hits: 1 code, 1 docs, 1 missed" in format_report(build_report())


def test_the_hit_split_follows_the_kind_filter():
    """A split describing populations that are not on screen is worse than none."""
    text = format_report(build_report(), kind="identifier")
    assert "hits: 1 code, 0 docs, 1 missed" in text


def test_a_kind_filter_notes_the_gate_is_on_overall_recall():
    """With --kind identifier the operator sees one population and a verdict
    about overall recall@5, which is not on screen -- that has to be said."""
    text = format_report(build_report(), kind="identifier")
    assert "the gate is on overall recall@5" in text


def test_no_kind_filter_prints_no_such_note():
    text = format_report(build_report())
    assert "the gate is on overall recall@5" not in text


def test_corpus_empty_is_true_when_every_result_retrieved_nothing():
    results = (
        QuestionResult("q1", "identifier", None, (), None),
        QuestionResult("q2", "conceptual", None, (), None),
    )
    by_kind = {
        kind: score([r for r in results if r.kind == kind])
        for kind in ("identifier", "conceptual")
    }
    report = Report(
        embedding_model="m",
        k=10,
        overall=score(results),
        by_kind=by_kind,
        results=results,
    )
    assert _corpus_empty(report)


def test_corpus_empty_is_false_when_anything_was_retrieved():
    assert not _corpus_empty(build_report())


def test_corpus_empty_is_false_with_no_questions_at_all():
    """n=0 is a different, already-handled state -- an empty golden set, not
    an empty corpus. Nothing here should call that a retrieval failure."""
    report = Report(
        embedding_model="m", k=10, overall=score([]), by_kind={}, results=()
    )
    assert not _corpus_empty(report)


def test_corpus_commit_is_none_for_a_directory_that_is_not_a_git_repo(tmp_path):
    """Best-effort: a missing or non-git checkout must not raise, only record
    an absent commit."""
    assert _corpus_commit(tmp_path) is None
