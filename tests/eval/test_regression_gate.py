"""The gate, against the real corpus. Needs Postgres and a real key.

Embedding 60-odd short questions is a fraction of a cent. This is the test that
turns the harness from a report into a tool: a retrieval change that quietly
makes things worse cannot land while this is green.
"""

from pathlib import Path

import pytest

from devagent.config import get_settings
from devagent.db.session import session_scope
from devagent.eval.baseline import check, load_baseline
from devagent.eval.dataset import load_golden
from devagent.eval.runner import retriever, run_eval
from devagent.llm.provider import get_provider

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "evals" / "golden.yaml"
BASELINE = ROOT / "evals" / "baseline.json"

pytestmark = pytest.mark.integration


def test_retrieval_has_not_regressed_below_the_recorded_baseline():
    settings = get_settings()
    questions = load_golden(GOLDEN)
    with session_scope() as session:
        report = run_eval(
            questions,
            retriever(get_provider(), session),
            embedding_model=settings.embedding_model,
        )

    failures = check(report, load_baseline(BASELINE))
    assert not failures, "\n".join(failures)
