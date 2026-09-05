"""The loop, driven by a scripted model so it runs offline."""

from langchain_core.messages import ToolMessage

from devagent.agent import graph
from devagent.agent.graph import bound_tools, run_agent
from devagent.agent.tools import ToolContext
from tests.fakes import FakeProvider


class FakeMessage:
    def __init__(self, content="", tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage or {"input_tokens": 10, "output_tokens": 5}


class FakeModel:
    """Replays a scripted sequence of model replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def make_ctx(tmp_path):
    return ToolContext(
        workspace_root=tmp_path,
        source_repo=tmp_path,
        provider=FakeProvider(),
        session=None,
        runner=None,
        repo="fastapi/fastapi",
    )


def test_answers_without_calling_a_tool(tmp_path):
    model = FakeModel([FakeMessage(content="It declares a dependency.")])
    result = run_agent("what does Depends do?", make_ctx(tmp_path), model)
    assert result.answer == "It declares a dependency."
    assert result.steps == []
    assert result.budget_exhausted is False


def test_records_a_tool_call_in_the_trace(tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\n", encoding="utf-8")
    model = FakeModel(
        [
            FakeMessage(
                tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "1"}]
            ),
            FakeMessage(content="It has two lines."),
        ]
    )
    result = run_agent("how long is a.py?", make_ctx(tmp_path), model)
    assert result.answer == "It has two lines."
    assert [s.tool for s in result.steps] == ["read_file"]
    assert "one" in result.steps[0].result


def test_stops_at_the_budget_and_says_so(tmp_path):
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    call = FakeMessage(
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "1"}]
    )
    model = FakeModel([call] * 10)
    result = run_agent("loop forever", make_ctx(tmp_path), model, budget=3)
    assert result.budget_exhausted is True
    assert len(result.steps) == 3


def test_every_tool_call_is_answered_when_the_budget_runs_out_mid_batch(tmp_path):
    """A model turn's tool calls must all get a reply, budget or not.

    The API rejects an assistant message whose tool_calls are not each answered
    by a ToolMessage, so a budget hit partway through a batch has to decline the
    rest rather than drop them -- the loop goes back to the model afterwards
    and would hand it an unanswerable turn.
    """
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    batch = [
        {"name": "read_file", "args": {"path": "a.py"}, "id": str(n)} for n in (1, 2, 3)
    ]
    model = FakeModel([FakeMessage(tool_calls=batch), FakeMessage(content="done")])
    result = run_agent("q", make_ctx(tmp_path), model, budget=1)

    handed_back = model.seen[1]
    answered = {m.tool_call_id for m in handed_back if isinstance(m, ToolMessage)}
    assert answered == {"1", "2", "3"}
    # The budget still binds: the declined calls are answered, not executed.
    assert len(result.steps) == 1


def test_accumulates_usage_across_calls(tmp_path):
    model = FakeModel(
        [FakeMessage(content="done", usage={"input_tokens": 100, "output_tokens": 20})]
    )
    result = run_agent("q", make_ctx(tmp_path), model)
    assert result.usage.llm_calls == 1
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 20
    assert result.usage.cost_usd >= 0


def test_an_unknown_tool_name_is_reported_not_raised(tmp_path):
    model = FakeModel(
        [
            FakeMessage(tool_calls=[{"name": "rm_rf", "args": {}, "id": "1"}]),
            FakeMessage(content="I could not do that."),
        ]
    )
    result = run_agent("q", make_ctx(tmp_path), model)
    assert "unknown tool" in result.steps[0].result.lower()


def test_the_context_is_never_offered_to_the_model(tmp_path):
    """A security boundary, not a style choice.

    If `ctx` were a model-fillable argument the model could set the workspace
    root and walk straight past the path confinement in read_file and run_tests.
    """
    for tool in bound_tools(make_ctx(tmp_path)):
        assert "ctx" not in tool.args, f"{tool.name} exposes ctx"
    assert {t.name for t in bound_tools(make_ctx(tmp_path))} == {
        "search_code",
        "read_file",
        "run_tests",
        "git_blame",
    }


def test_a_bound_tool_runs_with_only_the_model_facing_arguments(tmp_path):
    """Proves ctx really is closed over rather than merely hidden."""
    (tmp_path / "a.py").write_text("one\ntwo\n", encoding="utf-8")
    read = next(t for t in bound_tools(make_ctx(tmp_path)) if t.name == "read_file")
    assert "one" in read.invoke({"path": "a.py"})


def test_the_model_is_given_the_bound_tools(tmp_path):
    model = FakeModel([FakeMessage(content="hi")])
    run_agent("q", make_ctx(tmp_path), model)
    assert {t.name for t in model.bound} == {
        "search_code",
        "read_file",
        "run_tests",
        "git_blame",
    }


def test_system_prompt_steers_search_away_from_keywords():
    """Guards intent, not behaviour -- the behaviour needs a real model.

    Measured: asking "can I use pydantic.v1 models with latest version of
    FastAPI?", the model searched `pydantic.v1`, which does not retrieve the
    page saying support was removed in 0.128.0, and it answered "yes" from a
    stale chunk. Searching the whole question ranks that page second, and the
    answer flips to the correct one.
    """
    assert "whole question" in graph.SYSTEM
    assert "keywords" in graph.SYSTEM


def test_system_prompt_warns_that_the_corpus_spans_versions():
    """Release notes state things that were true once and are false now."""
    assert "release notes" in graph.SYSTEM
