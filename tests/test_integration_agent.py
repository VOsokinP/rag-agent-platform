"""The Milestone 3 done-when criterion, executed.

Needs Postgres, an OPENAI_API_KEY, network, and Docker, so it carries both
markers and is deselected twice over by default. It spends real money.
"""

import pytest

from devagent.agent.graph import run_agent
from devagent.agent.tools import ToolContext
from devagent.config import get_settings
from devagent.db.session import session_scope
from devagent.llm.chat import chat_model
from devagent.llm.provider import get_provider
from devagent.sandbox.runner import DockerRunner
from devagent.sandbox.workspace import workspace

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

# Breaks Depends at import time, so any test touching it fails loudly.
BREAKING_PATCH = """diff --git a/fastapi/params.py b/fastapi/params.py
index d3f2ae1..0e3a25e 100644
--- a/fastapi/params.py
+++ b/fastapi/params.py
@@ -1,3 +1,4 @@
+raise RuntimeError("deliberately broken by the integration test")
 import warnings
 from collections.abc import Callable, Sequence
 from dataclasses import dataclass
"""

TARGET = "tests/test_params_repr.py"


def test_the_agent_runs_the_tests_and_reports_the_real_failure():
    settings = get_settings()
    with (
        workspace(settings.repo_dir, BREAKING_PATCH) as root,
        session_scope() as session,
    ):
        context = ToolContext(
            workspace_root=root,
            source_repo=settings.repo_dir,
            provider=get_provider(),
            session=session,
            runner=DockerRunner(image=settings.runner_image),
            repo="fastapi/fastapi",
        )
        result = run_agent(
            f"would this change break any tests? check {TARGET}",
            context,
            chat_model(),
        )

    assert any(step.tool == "run_tests" for step in result.steps), (
        f"the agent never ran the tests; steps were {[s.tool for s in result.steps]}"
    )
    ran = next(step for step in result.steps if step.tool == "run_tests")
    assert "could not run" not in ran.result.lower(), ran.result
    assert "failed" in ran.result.lower()
    assert result.usage.llm_calls > 0
    assert not result.budget_exhausted
