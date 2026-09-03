"""Proves the container runs the patched copy, not the image's installed package.

This is the assumption the whole milestone rests on. If PYTHONPATH ever stops
shadowing site-packages, every run reports on unpatched code and cheerfully
answers "no, nothing breaks".
"""

import pytest

from devagent.config import get_settings
from devagent.sandbox.runner import DockerRunner
from devagent.sandbox.workspace import workspace

pytestmark = pytest.mark.sandbox

IMAGE = "devagent-runner:fastapi"
# Breaks fastapi.params at import time, so any test importing it fails loudly.
BREAKING_PATCH = """diff --git a/fastapi/params.py b/fastapi/params.py
index d3f2ae1..1ec5681 100644
--- a/fastapi/params.py
+++ b/fastapi/params.py
@@ -1,3 +1,4 @@
+raise RuntimeError("patched")
 import warnings
 from collections.abc import Callable, Sequence
 from dataclasses import dataclass
"""

TARGET = "tests/test_params_repr.py"


def test_the_container_sees_the_patch():
    settings = get_settings()
    runner = DockerRunner(image=IMAGE, timeout=300)
    with workspace(settings.repo_dir, BREAKING_PATCH) as root:
        result = runner.run(root, TARGET)
    assert result.ok, result.error
    assert result.failed > 0 or result.exit_code == 1
    assert "patched" in result.output_tail


def test_a_clean_workspace_reports_the_baseline():
    """Records whether FastAPI's suite is green before any patch.

    If this fails, the milestone's question needs a before/after comparison
    rather than a bare pass/fail -- note the outcome in _notes/TODO.md.
    """
    settings = get_settings()
    runner = DockerRunner(image=IMAGE, timeout=300)
    with workspace(settings.repo_dir, None) as root:
        result = runner.run(root, TARGET)
    assert result.ok, result.error
    assert result.failed == 0, f"baseline is not green: {result.output_tail}"
