"""Deterministic test doubles. No test outside the `integration` marker may
touch the network, so every test that needs a Provider uses FakeProvider."""

import hashlib


class FakeProvider:
    """A Provider that derives stable vectors from a hash of the input text."""

    def __init__(self, dimensions: int = 1536, answer: str = "fake answer") -> None:
        self.dimensions = dimensions
        self.answer = answer
        self.embed_calls: list[list[str]] = []
        self.complete_calls: list[tuple[str, str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def complete(self, system: str, user: str) -> str:
        self.complete_calls.append((system, user))
        return self.answer

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the digest until it covers the requested dimensionality.
        raw = (digest * (self.dimensions // len(digest) + 1))[: self.dimensions]
        return [byte / 255.0 for byte in raw]
