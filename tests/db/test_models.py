from sqlalchemy import inspect

from devagent.db.models import Base, Chunk


def test_chunk_maps_to_chunks_table():
    assert Chunk.__tablename__ == "chunks"


def test_chunk_has_the_expected_columns():
    columns = {c.name for c in inspect(Chunk).columns}
    assert columns == {
        "id",
        "repo",
        "file_path",
        "symbol",
        "kind",
        "start_line",
        "end_line",
        "text",
        "embedding",
    }


def test_symbol_is_nullable_but_file_path_is_not():
    columns = {c.name: c for c in inspect(Chunk).columns}
    assert columns["symbol"].nullable is True
    assert columns["file_path"].nullable is False


def test_unique_constraint_covers_the_chunk_location():
    constraint = next(
        c for c in Chunk.__table__.constraints if c.name == "uq_chunk_location"
    )
    assert {c.name for c in constraint.columns} == {
        "repo",
        "file_path",
        "start_line",
        "end_line",
    }


def test_unique_constraint_excludes_nullable_symbol():
    """NULL != NULL in Postgres, so a nullable column in a unique key lets
    heading-less doc chunks duplicate silently on every re-ingest."""
    constraint = next(
        c for c in Chunk.__table__.constraints if c.name == "uq_chunk_location"
    )
    assert "symbol" not in {c.name for c in constraint.columns}
    assert all(not c.nullable for c in constraint.columns)


def test_embedding_index_uses_cosine_distance():
    index = next(i for i in Chunk.__table__.indexes if i.name == "ix_chunks_embedding")
    assert index.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}


def test_embedding_index_is_hnsw_not_ivfflat():
    """IVFFlat clusters on the rows present at build time, and `init_db()` runs
    before any ingest. Built empty it has no centroids, so later rows become
    unreachable and retrieval silently returns too few rows or none."""
    index = next(i for i in Chunk.__table__.indexes if i.name == "ix_chunks_embedding")
    assert index.dialect_options["postgresql"]["using"] == "hnsw"


def test_base_metadata_registers_the_table():
    assert "chunks" in Base.metadata.tables
