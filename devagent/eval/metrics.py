"""Retrieval metrics: ranked lists in, numbers out.

Nothing here touches the database, the network, or the rest of the project. A
metric is only worth having if a person can check it against an example by
hand, and that is only possible while it stays arithmetic.

A rank is the **1-based** position of the first retrieved chunk from an expected
file, or `None` when no expected file was retrieved at all. `None` is not rank
infinity and not rank 0: it is "absent", and every function here has to say what
it does with an absent answer rather than letting a sentinel decide quietly.
"""

from collections.abc import Collection, Sequence


def first_hit_rank(
    retrieved_files: Sequence[str], expected: Collection[str]
) -> int | None:
    """Return the 1-based rank of the first expected file, or None.

    `retrieved_files` is one entry per retrieved *chunk*, best match first, so
    the same file can appear several times. That is deliberate: rank is a
    position in what retrieval actually returned, and de-duplicating files first
    would flatter every result whose top hits are several chunks of one file.
    """
    for position, file_path in enumerate(retrieved_files, start=1):
        if file_path in expected:
            return position
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """The fraction of questions whose first expected file landed in the top k."""
    if not ranks:
        # A `kind` with no questions is a reportable state, not an error: the
        # report prints n alongside, so 0.0 with n=0 reads correctly.
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank, scoring a question with no hit as zero.

    Catches what recall@10 hides: a corpus where every answer is somewhere in
    the top ten but never near the top has good recall and bad MRR, and the
    difference is exactly what reranking is supposed to fix.
    """
    if not ranks:
        return 0.0
    total = sum(1.0 / rank for rank in ranks if rank is not None)
    return total / len(ranks)
