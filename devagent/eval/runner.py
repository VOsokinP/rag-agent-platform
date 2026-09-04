"""Running the golden set against retrieval and scoring what comes back.

Retrieval enters as a `Retriever` callable rather than a (provider, session)
pair. That is what lets everything except the SQL round-trip be unit-tested
offline, and it keeps a test-only argument off a production signature.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from devagent.eval.dataset import KINDS, GoldenQuestion
from devagent.eval.metrics import first_hit_rank, mrr, recall_at_k
from devagent.llm.provider import Provider
from devagent.retrieval.vector_search import RetrievedChunk
from devagent.retrieval.vector_search import search as vector_search

KS = (1, 5, 10)
# One retrieval per question, deep enough for the largest k reported. Retrieving
# separately per k would triple the embedding spend and let the three numbers
# disagree with each other.
RETRIEVAL_K = max(KS)

Retriever = Callable[[str], list[RetrievedChunk]]


def retriever(
    provider: Provider, session: Any, repo: str | None = None, k: int = RETRIEVAL_K
) -> Retriever:
    """Bind the real retrieval path into the callable `run_eval` expects."""

    def retrieve(question: str) -> list[RetrievedChunk]:
        return vector_search(question, provider, session, repo=repo, k=k)

    return retrieve


@dataclass(frozen=True)
class QuestionResult:
    """One question's outcome.

    `rank` is the whole point of storing these individually: Milestone 2's
    before/after is a paired comparison over the same questions, which is far
    more sensitive at this sample size than two aggregate means, and a
    regression that names three questions is diagnosable where a moved average
    is not. `hit_source` records which side of the corpus the first hit came
    from -- `"code"`, `"docs"`, or `None`. Recorded because a shift between the
    two is precisely what hybrid search causes, and an aggregate recall number
    hides it entirely.
    """

    question: str
    kind: str
    rank: int | None
    retrieved_files: tuple[str, ...]
    hit_source: str | None


@dataclass(frozen=True)
class Scores:
    """Aggregates over one population. `n` travels with them deliberately --
    a recall figure without its sample size is how a thin population gets
    misread as a result."""

    n: int
    recall: dict[int, float]
    mrr: float


@dataclass(frozen=True)
class Report:
    embedding_model: str
    k: int
    overall: Scores
    by_kind: dict[str, Scores]
    results: tuple[QuestionResult, ...]


def score(results: Sequence[QuestionResult]) -> Scores:
    """Aggregate one population of results."""
    ranks = [result.rank for result in results]
    return Scores(
        n=len(results),
        recall={k: recall_at_k(ranks, k) for k in KS},
        mrr=mrr(ranks),
    )


def run_eval(
    questions: Sequence[GoldenQuestion],
    retrieve: Retriever,
    *,
    embedding_model: str,
    k: int = RETRIEVAL_K,
) -> Report:
    """Retrieve for every question once, then score overall and per kind."""
    results = []
    for question in questions:
        chunks = retrieve(question.question)
        retrieved_files = tuple(chunk.file_path for chunk in chunks)
        expected_code = set(question.expect_files)
        expected_docs = set(question.expect_docs)
        rank = first_hit_rank(retrieved_files, expected_code | expected_docs)
        hit_source = None
        if rank is not None:
            hit = retrieved_files[rank - 1]
            hit_source = "docs" if hit in expected_docs else "code"
        results.append(
            QuestionResult(
                question=question.question,
                kind=question.kind,
                rank=rank,
                retrieved_files=retrieved_files,
                hit_source=hit_source,
            )
        )

    # Every kind gets a row even when empty, so a missing population is visible
    # as n=0 rather than as an absent line nobody notices.
    by_kind = {
        kind: score([r for r in results if r.kind == kind]) for kind in KINDS
    }
    return Report(
        embedding_model=embedding_model,
        k=k,
        overall=score(results),
        by_kind=by_kind,
        results=tuple(results),
    )
