from pathlib import Path

from devagent.config import Settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5433/db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings()
    assert settings.database_url == "postgresql+psycopg://u:p@localhost:5433/db"
    assert settings.openai_api_key == "sk-test"


def test_settings_has_model_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5433/db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings()
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chat_model == "gpt-4o-mini"
    assert settings.embedding_dimensions == 1536


def test_repo_dir_is_a_path(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5433/db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings()
    assert isinstance(settings.repo_dir, Path)
    assert settings.repo_url == "https://github.com/fastapi/fastapi"
