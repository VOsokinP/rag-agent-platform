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

    repo_url: str = "https://github.com/fastapi/fastapi"
    repo_dir: Path = Path("data/repos/fastapi")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings, constructed once."""
    return Settings()
