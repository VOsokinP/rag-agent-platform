"""Request and response models for the HTTP API."""

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Optionally override the label recorded on the ingested rows.

    Which repository is ingested is configured by `REPO_URL` in the environment,
    not per request. A per-request URL would have nowhere to put the checkout —
    the directory and the include globs both come from settings — so it could
    only ever mislabel rows from the configured repository.
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


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    patch: str | None = Field(
        default=None, description="Unified diff to apply before running."
    )
    k: int = Field(default=8, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class StepOut(BaseModel):
    tool: str
    input: str
    result: str


class UsageOut(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = Field(description="Estimated from configured rates, not billed.")


class AgentResponse(BaseModel):
    answer: str
    steps: list[StepOut]
    citations: list[CitationOut]
    usage: UsageOut
    budget_exhausted: bool
