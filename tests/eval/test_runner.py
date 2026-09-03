"""The scoring half of the runner, with retrieval replaced by a canned one.

No database and no key: the seam is the `Retriever` callable, so everything
except the SQL round-trip is exercised offline.
"""

from devagent.eval.dataset import GoldenQuestion
from devagent.eval.runner import Report, run_eval, score
from devagent.retrieval.vector_search import RetrievedChunk


def chunk(file_path, score_value=0.5):
    return RetrievedChunk(
        file_path=file_path, symbol=None, kind="code",
        start_line=1, end_line=2, text="body", score=score_value,
    )


def question(text, files, kind="identifier"):
    return GoldenQuestion(
        question=text, expect_files=tuple(files), expect_symbols=(), kind=kind
    )


def retriever_returning(by_question):
    def retrieve(text):
        return [chunk(path) for path in by_question[text]]

    return retrieve


def test_ranks_the_first_expected_file_one_based():
    questions = [question("q1", ["fastapi/params.py"])]
    retrieve = retriever_returning({"q1": ["docs/a.md", "fastapi/params.py"]})

    report = run_eval(questions, retrieve, embedding_model="text-embedding-3-small")

    assert report.results[0].rank == 2


def test_a_question_whose_files_never_appear_has_no_rank():
    questions = [question("q1", ["fastapi/params.py"])]
    retrieve = retriever_returning({"q1": ["docs/a.md", "docs/b.md"]})

    report = run_eval(questions, retrieve, embedding_model="m")

    assert report.results[0].rank is None
    assert report.overall.recall[10] == 0.0


def test_recall_and_mrr_come_from_one_ranked_list():
    """Ranks 1 and 3 over two questions: recall@1 = 0.5, recall@5 = 1.0,
    MRR = (1/1 + 1/3) / 2 = 0.666..."""
    questions = [question("q1", ["a.py"]), question("q2", ["b.py"])]
    retrieve = retriever_returning({
        "q1": ["a.py", "z.py"],
        "q2": ["z.py", "y.py", "b.py"],
    })

    report = run_eval(questions, retrieve, embedding_model="m")

    assert report.overall.n == 2
    assert report.overall.recall[1] == 0.5
    assert report.overall.recall[5] == 1.0
    assert report.overall.mrr == (1.0 + 1.0 / 3) / 2


def test_scores_are_split_by_kind():
    questions = [
        question("q1", ["a.py"], kind="identifier"),
        question("q2", ["b.py"], kind="conceptual"),
    ]
    retrieve = retriever_returning({"q1": ["a.py"], "q2": ["z.py"]})

    report = run_eval(questions, retrieve, embedding_model="m")

    assert report.by_kind["identifier"].n == 1
    assert report.by_kind["identifier"].recall[1] == 1.0
    assert report.by_kind["conceptual"].n == 1
    assert report.by_kind["conceptual"].recall[10] == 0.0


def test_a_kind_with_no_questions_still_appears_with_n_zero():
    """Reporting 0.0 without n beside it is how a thin population gets misread."""
    questions = [question("q1", ["a.py"], kind="identifier")]
    retrieve = retriever_returning({"q1": ["a.py"]})

    report = run_eval(questions, retrieve, embedding_model="m")

    assert report.by_kind["conceptual"].n == 0
    assert report.by_kind["conceptual"].mrr == 0.0


def test_the_report_records_the_embedding_model_and_k():
    """The model is recorded so a red gate can distinguish a retrieval
    regression from an embedding-model snapshot moving underneath it."""
    report = run_eval([], retriever_returning({}), embedding_model="text-embedding-3-small")

    assert isinstance(report, Report)
    assert report.embedding_model == "text-embedding-3-small"
    assert report.k == 10


def test_score_keeps_retrieved_files_for_diagnosis():
    questions = [question("q1", ["a.py"])]
    retrieve = retriever_returning({"q1": ["z.py", "a.py"]})

    report = run_eval(questions, retrieve, embedding_model="m")

    assert report.results[0].retrieved_files == ("z.py", "a.py")


def test_score_over_no_results_is_all_zero():
    scores = score([])
    assert scores.n == 0
    assert scores.mrr == 0.0
    assert scores.recall[5] == 0.0
