from pathlib import Path

import devagent.llm.provider
import pytest

from tests.fakes import FakeProvider


def test_fake_provider_returns_one_vector_per_text():
    provider = FakeProvider(dimensions=8)
    vectors = provider.embed(["alpha", "beta", "gamma"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


def test_fake_provider_is_deterministic():
    provider = FakeProvider(dimensions=8)
    assert provider.embed(["alpha"]) == provider.embed(["alpha"])


def test_fake_provider_distinguishes_different_texts():
    provider = FakeProvider(dimensions=8)
    assert provider.embed(["alpha"]) != provider.embed(["beta"])


def test_fake_provider_records_embed_calls():
    provider = FakeProvider(dimensions=8)
    provider.embed(["alpha"])
    provider.embed(["beta", "gamma"])
    assert provider.embed_calls == [["alpha"], ["beta", "gamma"]]


def test_fake_provider_complete_returns_configured_answer():
    provider = FakeProvider(dimensions=8, answer="a grounded answer")
    assert provider.complete("system", "user") == "a grounded answer"
    assert provider.complete_calls == [("system", "user")]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def test_unrelated_texts_are_not_all_clustered_together():
    """Fake vectors must spread, or Task 8's ranking tests measure hash noise."""
    provider = FakeProvider(dimensions=1536)
    texts = [
        "alpha",
        "def foo(): pass",
        "completely unrelated text about databases",
        "the quick brown fox",
        "APIRoute.get_route_handler",
    ]
    vectors = provider.embed(texts)
    similarities = [
        _cosine(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    # Unrelated inputs should sit near-orthogonal, not in a tight positive band.
    assert max(similarities) < 0.5, f"vectors too clustered: {similarities}"


def test_identical_text_still_scores_as_identical():
    provider = FakeProvider(dimensions=1536)
    vector = provider.embed(["alpha"])[0]
    assert _cosine(vector, vector) == pytest.approx(1.0)


def test_provider_module_does_not_import_the_openai_sdk_at_module_level():
    """The lazy import in get_provider() is what keeps API-key-free imports working."""
    source = Path(devagent.llm.provider.__file__).read_text(encoding="utf-8")
    module_level = source.split("def get_provider")[0]
    assert "openai_provider" not in module_level
