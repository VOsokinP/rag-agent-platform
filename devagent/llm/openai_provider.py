"""The OpenAI implementation of the Provider protocol."""

import logging
import time

from openai import OpenAI

from devagent.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Embeddings and completions backed by the OpenAI API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._embedding_model = settings.embedding_model
        self._chat_model = settings.chat_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, retrying once before giving up."""
        for attempt in (1, 2):
            try:
                response = self._client.embeddings.create(
                    model=self._embedding_model, input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as exc:  # noqa: BLE001 - any API failure is retryable here
                if attempt == 2:
                    raise
                logger.warning("Embedding call failed (%s); retrying once", exc)
                time.sleep(1)
        raise AssertionError("unreachable")

    def complete(self, system: str, user: str) -> str:
        """Send a system + user prompt and return the reply text."""
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
