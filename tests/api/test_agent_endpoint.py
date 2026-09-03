import pytest
from fastapi.testclient import TestClient

from devagent.agent.state import AgentResult, Step, Usage
from devagent.api.main import app, get_provider_dep, get_session_dep
from devagent.sandbox.workspace import PatchError
from tests.fakes import FakeProvider


@pytest.fixture(autouse=True)
def _overrides():
    app.dependency_overrides[get_provider_dep] = lambda: FakeProvider()
    app.dependency_overrides[get_session_dep] = lambda: None
    yield
    app.dependency_overrides.clear()


RESULT = AgentResult(
    answer="Yes - 3 tests fail.",
    steps=[Step("run_tests", "target=tests/test_params_repr.py", "3 failed, 39 passed")],
    citations=[],
    usage=Usage(llm_calls=4, prompt_tokens=18000, completion_tokens=340, cost_usd=0.012),
)


def test_agent_returns_answer_steps_and_usage(monkeypatch):
    monkeypatch.setattr("devagent.api.main.run_agent", lambda *a, **k: RESULT)
    monkeypatch.setattr("devagent.api.main.chat_model", lambda: object())

    response = TestClient(app).post("/agent", json={"question": "would this break tests?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Yes - 3 tests fail."
    assert body["steps"][0]["tool"] == "run_tests"
    assert body["usage"]["llm_calls"] == 4
    assert body["budget_exhausted"] is False


def test_the_context_carries_both_roots(monkeypatch):
    """Tools read the throwaway copy; blame reads the checkout that outlives it."""
    seen = {}

    def capture(question, context, model, *args, **kwargs):
        seen["ctx"] = context
        return RESULT

    monkeypatch.setattr("devagent.api.main.run_agent", capture)
    monkeypatch.setattr("devagent.api.main.chat_model", lambda: object())

    TestClient(app).post("/agent", json={"question": "q", "k": 3})
    ctx = seen["ctx"]
    assert ctx.workspace_root != ctx.source_repo
    assert ctx.k == 3


def test_a_patch_that_does_not_apply_is_a_400(monkeypatch):
    def boom(*args, **kwargs):
        raise PatchError("error: patch failed: fastapi/params.py:1")

    monkeypatch.setattr("devagent.api.main.workspace", boom)
    monkeypatch.setattr("devagent.api.main.chat_model", lambda: object())

    response = TestClient(app).post(
        "/agent", json={"question": "q", "patch": "not a patch"}
    )
    assert response.status_code == 400
    assert "patch failed" in response.json()["detail"]


def test_a_blank_question_is_rejected():
    response = TestClient(app).post("/agent", json={"question": "   "})
    assert response.status_code == 422
