"""Test-wide fixtures."""

import pytest

from devagent.config import get_settings


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
