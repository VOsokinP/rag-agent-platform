"""Hand-checked examples. Every expected number here is computed in the comment
beside it, so a change to the arithmetic has to argue with the comment."""

from devagent.eval.metrics import first_hit_rank, mrr, recall_at_k


def test_first_hit_rank_is_one_based():
    assert first_hit_rank(["fastapi/params.py"], {"fastapi/params.py"}) == 1


def test_first_hit_rank_counts_the_misses_before_it():
    retrieved = ["docs/index.md", "fastapi/routing.py", "fastapi/params.py"]
    assert first_hit_rank(retrieved, {"fastapi/params.py"}) == 3


def test_first_hit_rank_returns_the_first_hit_not_the_best_one():
    """Rank is a position, so a later hit must not overwrite an earlier one."""
    retrieved = ["fastapi/params.py", "fastapi/param_functions.py"]
    expected = {"fastapi/params.py", "fastapi/param_functions.py"}
    assert first_hit_rank(retrieved, expected) == 1


def test_first_hit_rank_is_none_when_nothing_expected_was_retrieved():
    assert first_hit_rank(["docs/index.md"], {"fastapi/params.py"}) is None


def test_first_hit_rank_of_an_empty_result_is_none():
    assert first_hit_rank([], {"fastapi/params.py"}) is None


def test_recall_at_k_counts_questions_whose_rank_is_within_k():
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 1) == 0.25  # only rank 1        -> 1/4
    assert recall_at_k(ranks, 5) == 0.5  # ranks 1 and 3      -> 2/4
    assert recall_at_k(ranks, 10) == 0.75  # ranks 1, 3 and 6   -> 3/4


def test_recall_at_k_of_no_questions_is_zero_not_an_error():
    """A kind with no questions must report 0.0, not divide by zero."""
    assert recall_at_k([], 5) == 0.0


def test_mrr_averages_reciprocal_ranks_and_scores_a_miss_zero():
    # 1/1 + 1/3 + 0 + 1/6 = 1.5, over 4 questions -> 0.375
    assert mrr([1, 3, None, 6]) == 0.375


def test_mrr_of_no_questions_is_zero_not_an_error():
    assert mrr([]) == 0.0
