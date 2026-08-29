"""Request and response models for the HTTP API."""

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Optionally override the label recorded on the ingested rows.

    Which repository is ingested is configured by `REPO_URL` in the environment,
    not per request. A per-request URL would be a lie: the checkout directory
    comes from settings and `ensure_repo` reuses any directory that already has a
    `.git`, so pointing this at a second repository would ingest the first one's
    files while labelling the rows with the second one's name.
    """

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
