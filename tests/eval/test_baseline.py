"""Serialising a Report and gating a new one against it. No I/O beyond tmp_path."""

import json

import pytest

from devagent.eval.baseline import (
    TOLERANCE,
    check,
    load_baseline,
    to_baseline,
)
from devagent.eval.runner import QuestionResult, Report, score


def report_with(ranks_by_kind, embedding_model="text-embedding-3-small"):
    results = []
    for kind, ranks in ranks_by_kind.items():
        for index, rank in enumerate(ranks):
            results.append(
                QuestionResult(
                    question=f"{kind}-{index}", kind=kind, rank=rank,
                    retrieved_files=("a.py",), hit_source=None,
                )
            )
    by_kind = {
        kind: score([r for r in results if r.kind == kind]) for kind in ranks_by_kind
    }
    return Report(
        embedding_model=embedding_model, k=10, overall=score(results),
        by_kind=by_kind, results=tuple(results),
    )


def test_the_baseline_records_per_question_ranks_not_only_aggregates():
    """The paired comparison Milestone 2 needs is impossible without these."""
    report = report_with({"identifier": [1, None]})

    payload = to_baseline(report, recorded="2026-09-03")

    assert payload["ranks"] == {"identifier-0": 1, "identifier-1": None}


def test_the_baseline_records_the_embedding_model():
    report = report_with({"identifier": [1]}, embedding_model="text-embedding-3-small")
    assert to_baseline(report, recorded="2026-09-03")["embedding_model"] == (
        "text-embedding-3-small"
    )


def test_the_baseline_round_trips_through_json(tmp_path):
    report = report_with({"identifier": [1, 3], "conceptual": [None]})
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(to_baseline(report, recorded="2026-09-03")), encoding="utf-8")

    assert load_baseline(path)["overall"]["recall"]["5"] == pytest.approx(2 / 3)


def test_an_identical_report_passes_the_gate():
    report = report_with({"identifier": [1, 3]})
    assert check(report, to_baseline(report, recorded="2026-09-03")) == []


def test_a_drop_within_tolerance_passes():
    baseline = to_baseline(report_with({"identifier": [1] * 100}), recorded="2026-09-03")
    # 99 of 100 within k=5: recall@5 = 0.99, a 0.01 drop, inside TOLERANCE.
    worse = report_with({"identifier": [1] * 99 + [None]})
    assert check(worse, baseline) == []


def test_a_drop_beyond_tolerance_fails_and_says_by_how_much():
    baseline = to_baseline(report_with({"identifier": [1] * 100}), recorded="2026-09-03")
    # 90 of 100: recall@5 = 0.90, a 0.10 drop, well beyond TOLERANCE.
    worse = report_with({"identifier": [1] * 90 + [None] * 10})

    failures = check(worse, baseline)

    assert len(failures) == 1
    assert "recall@5" in failures[0]
    assert "0.90" in failures[0] and "1.00" in failures[0]


def test_a_changed_embedding_model_is_reported_separately_from_a_regression():
    """A red gate has to say which of the two kinds of red it is."""
    baseline = to_baseline(report_with({"identifier": [1]}), recorded="2026-09-03")
    moved = report_with({"identifier": [1]}, embedding_model="text-embedding-3-large")

    failures = check(moved, baseline)

    assert any("embedding model" in failure for failure in failures)


def test_tolerance_is_the_value_the_addendum_settled_on():
    assert TOLERANCE == 0.02


def test_the_baseline_records_which_side_of_the_corpus_each_hit_came_from():
    """Recorded but not gated. Once hybrid search lands, the pre-hybrid split
    cannot be recovered without reverting it, so it is captured now."""
    report = report_with({"conceptual": [1]})

    payload = to_baseline(report, recorded="2026-09-03")

    assert payload["hit_source"] == {"conceptual-0": None}


def test_a_changed_hit_source_does_not_fail_the_gate():
    """A docs-to-code shift is a thing to look at, not a regression."""
    baseline = to_baseline(report_with({"conceptual": [1]}), recorded="2026-09-03")
    moved = report_with({"conceptual": [1]})
    assert check(moved, baseline) == []
