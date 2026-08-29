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
