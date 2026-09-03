from devagent.llm.chat import chat_model, estimate_cost


def test_chat_model_uses_the_configured_model(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert "gpt-4o-mini" in repr(chat_model().model_name)


def test_estimate_cost_uses_the_configured_rates(monkeypatch):
    monkeypatch.setenv("CHAT_INPUT_COST_PER_MTOK", "1.0")
    monkeypatch.setenv("CHAT_OUTPUT_COST_PER_MTOK", "2.0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # 1M input at $1 plus 1M output at $2
    assert estimate_cost(1_000_000, 1_000_000) == 3.0


def test_estimate_cost_of_nothing_is_zero(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert estimate_cost(0, 0) == 0.0
