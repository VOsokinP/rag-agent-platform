"""Decorated definitions, as they appear throughout FastAPI."""

import functools

app = object()


@app.get("/foo")
@functools.lru_cache
def handler(x: int) -> int:
    """Handle a request."""
    return x + 1


@functools.total_ordering
class Decorated:
    """A decorated class."""

    @property
    def value(self) -> int:
        return 1
