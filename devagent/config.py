"""Application configuration, read from environment or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for DevAgent. Values come from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    openai_api_key: str

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536

    agent_step_budget: int = 8
    # USD per 1M tokens. Estimates for the usage report, not billing truth.
    chat_input_cost_per_mtok: float = 0.15
    chat_output_cost_per_mtok: float = 0.60

    repo_url: str = "https://github.com/fastapi/fastapi"
    repo_dir: Path = Path("data/repos/fastapi")
    # Which files of the checkout to ingest. These belong next to REPO_URL:
    # pointing that at another project without them would ingest nothing, which
    # is what made REPO_URL look like a knob without being one. The defaults are
    # FastAPI's layout. The runner image stays pinned to FastAPI regardless — it
    # bakes in that project's test dependencies.
    include_code_glob: str = "fastapi/**/*.py"
    include_docs_glob: str = "docs/en/docs/**/*.md"
    runner_image: str = "devagent-runner:fastapi"

    @property
    def include_globs(self) -> tuple[str, str]:
        return (self.include_code_glob, self.include_docs_glob)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings, constructed once."""
    # Required fields come from the environment, which mypy cannot see.
    return Settings()  # type: ignore[call-arg]
