"""The provider interface every LLM and embedding call goes through.

Keeping this narrow is what makes the OpenAI dependency swappable for Anthropic
or a local Ollama model without touching ingestion, retrieval, or the API layer.
"""

from functools import lru_cache
from typing import Protocol


class Provider(Protocol):
    """Embeddings and chat completion."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to a system + user prompt pair."""
        ...


@lru_cache
def get_provider() -> Provider:
    """Return the process-wide provider."""
    from devagent.llm.openai_provider import OpenAIProvider

    return OpenAIProvider()
