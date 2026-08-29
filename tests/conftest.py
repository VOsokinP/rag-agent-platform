"""Test-wide environment setup and fixtures.

`devagent.db.models` reads settings at import time to size the embedding column,
so the required settings must exist before any test imports it. These are dummy
values; no unit test connects to a real database or API.
"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://devagent:devagent@localhost:5433/devagent"
)
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

import pytest  # noqa: E402

from devagent.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Keep `get_settings()`'s lru_cache from leaking between tests.

    The cache is process-wide, so without this the first test to call
    `get_settings()` fixes the settings for every test that follows and
    `monkeypatch.setenv` silently stops working.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
