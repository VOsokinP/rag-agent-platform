"""The chunks table: one row per retrievable unit of a repository."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from devagent.config import get_settings


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    """A chunk of code or documentation, with its embedding.

    Code and docs share one table and are distinguished by `kind`, so retrieval
    can rank both against a single query without a union across tables.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(512), nullable=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=False
    )

    __table_args__ = (
        # NOTE: `symbol` is intentionally excluded. It is nullable (heading-less
        # Markdown files produce symbol=None), and in Postgres NULL != NULL, so a
        # nullable column in a unique key would let such chunks duplicate
        # silently on every re-ingest. A chunk's identity is its location in a
        # file; symbol is derived from that location, not an independent key.
        UniqueConstraint(
            "repo", "file_path", "start_line", "end_line", name="uq_chunk_location"
        ),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
