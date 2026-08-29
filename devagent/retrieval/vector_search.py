"""Top-k cosine similarity retrieval against the chunks table."""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from devagent.db.models import Chunk as ChunkRow
from devagent.llm.provider import Provider


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score.

    `score` is cosine similarity in [0, 1] — higher is a closer match. pgvector's
    `<=>` operator returns cosine *distance*, so it is converted here rather than
    leaking an inverted scale into the API response.
    """

    file_path: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    text: str
    score: float


def search(
    question: str,
    provider: Provider,
    session: Any,
    repo: str | None = None,
    k: int = 8,
) -> list[RetrievedChunk]:
    """Return the k chunks most similar to the question, best match first."""
    query_vector = provider.embed([question])[0]
    distance = ChunkRow.embedding.cosine_distance(query_vector)

    statement = select(ChunkRow, distance.label("distance"))
    if repo is not None:
        statement = statement.where(ChunkRow.repo == repo)
    statement = statement.order_by(distance).limit(k)

    return [
        RetrievedChunk(
            file_path=row.file_path,
            symbol=row.symbol,
            kind=row.kind,
            start_line=row.start_line,
            end_line=row.end_line,
            text=row.text,
            score=1.0 - float(row_distance),
        )
        for row, row_distance in session.execute(statement).all()
    ]
