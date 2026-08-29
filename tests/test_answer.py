import pytest

from devagent.answer import (
    Answer,
    Citation,
    EmptyCorpusError,
    answer_question,
    build_context,
)
from devagent.retrieval.vector_search import RetrievedChunk
from tests.fakes import FakeProvider


def make_chunk(symbol: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        file_path="fastapi/param_functions.py",
        symbol=symbol,
        kind="code",
        start_line=10,
        end_line=20,
        text=f"def {symbol}(): ...",
        score=score,
    )


def test_build_context_labels_each_chunk_with_its_source():
    context = build_context([make_chunk("Depends")])
    assert "fastapi/param_functions.py" in context
    assert "Depends" in context
    assert "def Depends(): ..." in context


def test_build_context_includes_line_numbers():
    context = build_context([make_chunk("Depends")])
    assert "10" in context and "20" in context


def test_build_context_separates_multiple_chunks():
    context = build_context([make_chunk("Depends"), make_chunk("Security")])
    assert context.count("fastapi/param_functions.py") == 2


def test_answer_question_returns_an_answer_object():
    provider = FakeProvider(answer="Depends declares a dependency.")
    result = answer_question("what does Depends do", [make_chunk("Depends")], provider)
    assert isinstance(result, Answer)
    assert result.answer == "Depends declares a dependency."


def test_answer_question_returns_one_citation_per_chunk():
    provider = FakeProvider()
    chunks = [make_chunk("Depends"), make_chunk("Security")]
    result = answer_question("q", chunks, provider)
    assert len(result.citations) == 2
    assert all(isinstance(c, Citation) for c in result.citations)


def test_citations_carry_location_and_score():
    provider = FakeProvider()
    result = answer_question("q", [make_chunk("Depends", score=0.77)], provider)
    citation = result.citations[0]
    assert citation.file_path == "fastapi/param_functions.py"
    assert citation.symbol == "Depends"
    assert citation.start_line == 10
    assert citation.end_line == 20
    assert citation.score == 0.77


def test_prompt_contains_the_question_and_the_context():
    provider = FakeProvider()
    answer_question("what does Depends do", [make_chunk("Depends")], provider)
    _system, user = provider.complete_calls[0]
    assert "what does Depends do" in user
    assert "def Depends(): ..." in user


def test_system_prompt_demands_grounding():
    provider = FakeProvider()
    answer_question("q", [make_chunk("Depends")], provider)
    system, _user = provider.complete_calls[0]
    lowered = system.lower()
    assert "only from the context" in lowered
    assert "say so" in lowered, "the model needs explicit permission to decline"


def test_system_prompt_marks_context_as_untrusted_data():
    """Retrieved chunks are third-party content; the prompt must not invite the
    model to follow instructions embedded in them."""
    provider = FakeProvider()
    answer_question("q", [make_chunk("Depends")], provider)
    system, _user = provider.complete_calls[0]
    lowered = system.lower()
    assert "never as instructions" in lowered


def test_blank_chunks_are_omitted_from_the_context():
    blank = RetrievedChunk(
        file_path="fastapi/empty.py",
        symbol="nothing",
        kind="code",
        start_line=1,
        end_line=1,
        text="   \n  ",
        score=0.5,
    )
    context = build_context([blank, make_chunk("Depends")])
    assert "fastapi/empty.py" not in context
    assert "Depends" in context


def test_empty_chunks_raises_rather_than_hallucinating():
    with pytest.raises(EmptyCorpusError):
        answer_question("q", [], FakeProvider())
