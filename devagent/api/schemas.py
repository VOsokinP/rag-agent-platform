"""Request and response models for the HTTP API."""

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Optionally override which repository to ingest."""

    repo_url: str | None = None
    repo: str | None = Field(
        default=None, description="Name recorded on each row, e.g. 'fastapi/fastapi'."
    )


class IngestResponse(BaseModel):
    repo: str
    files_processed: int
    chunks_written: int
    files_skipped: list[str]
    batches_failed: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=100)
    repo: str | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class CitationOut(BaseModel):
    file_path: str
    symbol: str | None
    start_line: int
    end_line: int
    score: float = Field(
        description="Cosine similarity in [-1, 1]; not clamped, negative values are legitimate."
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
