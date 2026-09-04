"""The committed golden set, checked as data.

Safe as a unit test: `evals/` is committed, unlike `data/`, so this file exists
in a clean clone. See the R12 note -- a unit test that reads a path only present
on the author's machine is green here and red everywhere else.
"""

from pathlib import Path

from devagent.eval.dataset import KINDS, load_golden

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden.yaml"

MIN_TOTAL = 60
MIN_PER_KIND = 25
DOCS_PREFIX = "docs/en/docs/"
MIN_CONCEPTUAL_WITH_DOCS = 0.5


def test_the_committed_golden_set_loads():
    assert load_golden(GOLDEN)


def test_the_set_is_large_enough_to_resolve_a_small_change():
    """Below this, recall@5 moves in steps too coarse to show a real gain."""
    questions = load_golden(GOLDEN)
    assert len(questions) >= MIN_TOTAL, f"{len(questions)} questions; need {MIN_TOTAL}"


def test_each_kind_is_large_enough_to_report_on_its_own():
    questions = load_golden(GOLDEN)
    for kind in KINDS:
        count = sum(1 for q in questions if q.kind == kind)
        assert count >= MIN_PER_KIND, f"{kind}: {count} questions; need {MIN_PER_KIND}"


def test_every_docs_label_is_inside_the_ingested_glob():
    """Ingestion only indexes docs/en/docs/**/*.md. A docs label outside that is
    unreachable, and would read as a permanent miss."""
    for question in load_golden(GOLDEN):
        for path in question.expect_docs:
            assert path.startswith(DOCS_PREFIX), f"{path} ({question.question})"


def test_most_conceptual_questions_carry_a_docs_label():
    """Docs are two thirds of the corpus. A set that labels only source scores
    correct docs retrieval as a miss -- and would make hybrid search look like it
    improved conceptual recall while it was suppressing the best answers.

    A floor, not a requirement: some questions are genuinely internal-only and
    the set's header documents which."""
    conceptual = [q for q in load_golden(GOLDEN) if q.kind == "conceptual"]
    with_docs = [q for q in conceptual if q.expect_docs]
    assert len(with_docs) >= MIN_CONCEPTUAL_WITH_DOCS * len(conceptual), (
        f"{len(with_docs)} of {len(conceptual)} conceptual questions carry docs"
    )
