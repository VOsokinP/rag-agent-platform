"""Test-wide environment setup and fixtures.

`devagent.db.models` reads settings at import time to size the embedding column,
so the required settings must exist before any test imports it.

The order below matters. A real exported environment variable wins, then `.env`,
then a dummy placeholder. Setting the placeholder first -- as this file used to --
looks harmless because `setdefault` does not overwrite, but an environment
variable outranks `.env` in pydantic-settings, so the placeholder shadowed the
real key for the whole session and every integration test failed with a 401.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
FAKE_KEY = "sk-test-not-a-real-key"


def _load_env_file(path: Path) -> None:
    """Apply `KEY=value` lines from `path` without overriding the environment.

    Deliberately minimal rather than a dotenv dependency: this needs to handle
    the handful of plain assignments `.env.example` documents, nothing more.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_file(ENV_FILE)

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://devagent:devagent@localhost:5433/devagent"
)
os.environ.setdefault("OPENAI_API_KEY", FAKE_KEY)

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


@pytest.fixture(autouse=True)
def _fake_key_outside_integration(request, monkeypatch):
    """Give every non-integration test a placeholder key.

    Loading `.env` above puts a real key in the environment for the whole
    session, which integration tests need. Unit tests must not inherit it: they
    are offline by construction (they use FakeProvider), but if one ever reached
    the real provider by mistake, a placeholder turns that into a loud 401
    instead of a silent charge.
    """
    if "integration" not in request.keywords:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
