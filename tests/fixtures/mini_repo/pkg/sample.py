"""A sample module for chunker tests."""

import os
from typing import Any

CONSTANT = 42


def top_level_function(x: int) -> int:
    """Double a number."""
    return x * 2


async def async_function(y: int) -> int:
    """Triple a number, asynchronously."""
    return y * 3


class Greeter:
    """Greets people."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}"

    async def greet_async(self) -> str:
        """Return a greeting, asynchronously."""
        return f"Hello again, {self.name}"
